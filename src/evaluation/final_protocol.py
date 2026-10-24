"""Freeze (and later verify) the protocol for the one-time final test evaluation.

Writes reports/final_evaluation/protocol.json and protocol.md. Every frozen value
is read from the saved artifacts (checkpoint, calibration summary, degradation
plan), the test-access audit is run, and a manifest of content identities is
recorded. This module never loads the test split, decodes test audio, or runs
inference: the audio manifest hashes file bytes only.

Content identity (what --verify compares):
  * text files that Git tracks or would track (not ignored): the Git blob id of
    the LF-normalized bytes, sha1("blob <len>\\0" + content with CRLF -> LF). This
    equals `git hash-object` under core.autocrlf, so a CRLF checkout of the same
    commit verifies, while any real content change fails. Computed here, so it
    does not depend on the local Git configuration.
  * ignored or binary artifacts (checkpoint, data/metadata.csv, raw WAV files):
    exact SHA-256 of the bytes.
  Raw SHA-256 is recorded for every file as supplementary provenance.

A protocol is never overwritten. Before any test stage has started it can be
superseded explicitly (--write --supersede): the previous version is kept as
protocol_v<N>_superseded.json/.md and referenced from the new one. Once a test
stage marker exists, superseding is refused.

Usage::

    python src/evaluation/final_protocol.py --write               # refuses if protocol.json exists
    python src/evaluation/final_protocol.py --write --supersede   # only before any test stage started
    python src/evaluation/final_protocol.py --verify              # non-zero exit on any change
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
OUT_DIR = REPO_ROOT / "reports" / "final_evaluation"
AUDIO_DIR = REPO_ROOT / "data" / "raw" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database" / "audio_and_txt_files"
MODEL_RUN = "baseline_cnn_v3"

MANIFEST_FILES = [
    "models/baseline_cnn_v3/best.pt",
    "reports/runs/baseline_cnn_v3/run.json", "reports/runs/baseline_cnn_v3/history.json",
    "reports/runs/baseline_cnn_v3/val_predictions.csv", "reports/runs/baseline_cnn_v3/val_analysis.json",
    "data/splits/patient_split.csv", "data/metadata.csv", "data/normalization/logmel_primary_train_band_stats.json",
    "src/data/build_metadata.py", "src/data/create_splits.py", "src/data/length_bucketing.py",
    "src/data/normalization.py", "src/data/respiratory_dataset.py",
    "src/features/audio_preprocessing.py", "src/features/logmel.py", "src/features/degradations.py",
    "src/models/train.py", "src/models/baseline_cnn.py", "src/models/crnn.py", "src/models/mobilenet_transfer.py",
    "src/evaluation/analyze_run.py", "src/evaluation/classification_metrics.py", "src/evaluation/compare_runs.py",
    "src/evaluation/calibration_metrics.py", "src/evaluation/calibration_uncertainty.py",
    "src/evaluation/calibration_report.py", "src/evaluation/noise_robustness.py", "src/evaluation/final_protocol.py",
    "src/evaluation/final_evaluation.py", "tests/test_final_evaluation.py", "tests/test_final_protocol.py",
    "reports/runs/seed_study/plan.json", "reports/runs/seed_study/summary.json",
    "reports/robustness/noise_robustness/plan.json", "reports/robustness/noise_robustness/summary.json",
    "reports/robustness/noise_robustness/clean_check.json",
    "reports/calibration_uncertainty/plan.json", "reports/calibration_uncertainty/summary.json",
    "reports/calibration_uncertainty/lopo_temperatures.csv",
    "README.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, root: Path = REPO_ROOT) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8000]  # Git's own text/binary heuristic


def canonical_blob_id(data: bytes) -> str:
    """Git blob id of the LF-normalized content (what `git hash-object` gives under core.autocrlf)."""
    data = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_external(rel: str, root: Path = REPO_ROOT) -> bool:
    """True for paths Git ignores (frozen external artifacts such as data/metadata.csv or checkpoints)."""
    try:
        return subprocess.run(["git", "check-ignore", "-q", rel], cwd=root, capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def file_entry(rel: str, root: Path = REPO_ROOT) -> dict:
    p = Path(root) / rel
    data = p.read_bytes()
    text_tracked = not is_binary(data) and not is_external(rel, root)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "git_blob": canonical_blob_id(data) if text_tracked else None,
            "identity": "git_blob" if text_tracked else "sha256"}


def entry_matches(entry: dict, path: Path) -> bool:
    if not path.exists():
        return False
    data = path.read_bytes()
    if entry.get("identity") == "git_blob":
        return canonical_blob_id(data) == entry["git_blob"]
    return hashlib.sha256(data).hexdigest() == entry["sha256"]  # external/binary artifacts and v1 entries


def audio_manifest(audio_dir: Path = AUDIO_DIR, root: Path = REPO_ROOT) -> dict:
    """Combined hash of every recording's bytes (sorted by name). Bytes only: nothing is decoded."""
    files = sorted(Path(audio_dir).glob("*.wav"))
    combined = hashlib.sha256()
    for p in files:
        combined.update(f"{p.name}\t{sha256(p)}\n".encode())
    return {"directory": str(Path(audio_dir).relative_to(root)).replace("\\", "/"), "n_wav_files": len(files),
            "combined_sha256": combined.hexdigest(),
            "method": "sha256 over lines '<filename>\\t<sha256 of file bytes>\\n' for all .wav files sorted by name; "
                      "integrity only, no audio decoded, no split-specific selection"}


def build_manifest(files: list[str] = MANIFEST_FILES, audio_dir: Path | None = AUDIO_DIR,
                   root: Path = REPO_ROOT) -> dict:
    out = {"files": {rel: file_entry(rel, root) for rel in files}}
    if audio_dir is not None:
        out["audio"] = audio_manifest(audio_dir, root)
    return out


def verify_manifest(manifest: dict, audio_dir: Path | None = AUDIO_DIR, root: Path = REPO_ROOT) -> list[str]:
    """Paths whose content identity differs from the manifest (empty list = all unchanged)."""
    bad = [rel for rel, e in manifest["files"].items() if not entry_matches(e, Path(root) / rel)]
    if audio_dir is not None and "audio" in manifest and \
            audio_manifest(audio_dir, root)["combined_sha256"] != manifest["audio"]["combined_sha256"]:
        bad.append(manifest["audio"]["directory"])
    return bad


# ---------------------------------------------------------------- audit

def test_access_audit() -> dict:
    """Evidence that no model has evaluated the test split and no test predictions exist."""
    with open(REPO_ROOT / "data" / "splits" / "patient_split.csv", newline="", encoding="utf-8") as f:
        split = list(csv.DictReader(f))
    test = {r["patient_id"] for r in split if r["split"] == "test"}
    with open(REPO_ROOT / "data" / "metadata.csv", newline="", encoding="utf-8") as f:
        test_recs = {r["recording_filename"] for r in csv.DictReader(f) if r["patient_id"] in test}
    text_files = [p for root in ("reports", "models") for p in (REPO_ROOT / root).rglob("*")
                  if p.is_file() and p.suffix in (".csv", ".json", ".md", ".log", ".txt")
                  and "final_evaluation" not in p.parts]
    mentions = sorted(str(p.relative_to(REPO_ROOT)) for p in text_files
                      if any(r in p.read_text(encoding="utf-8", errors="ignore") for r in test_recs))
    test_patient_csvs = []
    for p in (REPO_ROOT / "reports").rglob("*.csv"):
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "patient_id" in rows[0] and {r["patient_id"] for r in rows} & test:
            test_patient_csvs.append(str(p.relative_to(REPO_ROOT)))
    runs = list((REPO_ROOT / "reports").rglob("run.json"))
    flagged, non_val_splits, datasets = [], set(), set()
    for p in (REPO_ROOT / "reports").rglob("*.json"):
        if "final_evaluation" in p.parts:
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(j, dict):
            if j.get("test_split_evaluated") is True:
                flagged.append(str(p.relative_to(REPO_ROOT)))
            if isinstance(j.get("split"), str) and j["split"] != "val":
                non_val_splits.add(j["split"])
    for p in runs:
        datasets |= set(json.loads(p.read_text(encoding="utf-8")).get("data", {}).get("datasets", {}))
    callers = [line for line in (git("grep", "-n", "allow_test=True", "--", "src", "tests") or "").splitlines()
               if "reserved for the final" not in line and "pass allow_test=True only then" not in line]
    history = [p for p in (git("log", "--all", "--name-only", "--format=") or "").splitlines()
               if p and "test" in p.lower() and not p.startswith("tests/") and not p.endswith("post_unittest.log")]
    result = {
        "test_patients": len(test), "test_recordings": len(test_recs),
        "report_or_model_files_mentioning_a_test_recording": mentions,
        "prediction_csvs_containing_test_patients": sorted(test_patient_csvs),
        "json_records_with_test_split_evaluated_true": sorted(flagged),
        "non_validation_split_values_in_report_json": sorted(non_val_splits),
        "datasets_used_by_any_training_run": sorted(datasets),
        "code_calling_evaluate_with_allow_test_true": callers,
        "test_like_files_ever_committed": sorted(set(history)),
        "checkpoint_contains": "model_state, config, epoch, val_metrics (validation only), class_names, data_sha256",
        "n_run_records_checked": len(runs), "n_report_csvs_checked": sum(1 for _ in (REPO_ROOT / "reports").rglob("*.csv")),
    }
    result["passed"] = not (mentions or test_patient_csvs or flagged or non_val_splits or callers or history) \
        and set(datasets) <= {"train", "val"}
    return result


# ---------------------------------------------------------------- protocol

def environment() -> dict:
    import numpy
    import soundfile
    import torch
    import torchaudio
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "torchaudio": torchaudio.__version__, "numpy": numpy.__version__, "soundfile": soundfile.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pip_freeze": freeze}


def build_protocol(audit: dict, manifest: dict, env: dict) -> dict:
    for sub in ("models", "data", "features", "evaluation"):
        sys.path.insert(0, str(SRC_DIR / sub))
    import torch
    import audio_preprocessing as ap
    import final_evaluation as fe
    import degradations as dg
    import logmel as lm
    import respiratory_dataset as rd

    ckpt = torch.load(REPO_ROOT / "models" / MODEL_RUN / "best.pt", map_location="cpu", weights_only=False)
    run = json.loads((REPO_ROOT / "reports" / "runs" / MODEL_RUN / "run.json").read_text(encoding="utf-8"))
    cal = json.loads((REPO_ROOT / "reports" / "calibration_uncertainty" / "summary.json").read_text(encoding="utf-8"))
    cal_plan = json.loads((REPO_ROOT / "reports" / "calibration_uncertainty" / "plan.json").read_text(encoding="utf-8"))
    rob_plan = json.loads((REPO_ROOT / "reports" / "robustness" / "noise_robustness" / "plan.json").read_text(encoding="utf-8"))
    seed = json.loads((REPO_ROOT / "reports" / "runs" / "seed_study" / "summary.json").read_text(encoding="utf-8"))
    with open(REPO_ROOT / "data" / "splits" / "patient_split.csv", newline="", encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    files = manifest["files"]
    temp = cal["temperature_scaling"]["final_validation_temperature"]
    taus = cal["part5_degradation"]["fixed_clean_thresholds"]
    conds = [c.as_dict() for c in dg.PREDECLARED_CONDITIONS]
    if conds != rob_plan["conditions"]:
        raise ValueError("degradation conditions in code differ from the robustness plan")
    bootstrap = {"unit": "patient", "n_resamples": 2000, "seed": 0,
                 "rng": "Python random.Random(seed); each resample draws len(patients) patient ids with "
                        "replacement (rng.choice over sorted patient ids) and includes all cycles of each draw",
                 "interval": "percentile: sorted values at indices int(0.025 n) and int(0.975 n) - 1",
                 "paired": "the same patient draws are used for every method/condition being compared",
                 "implementation": "compare_runs.patient_bootstrap / calibration_metrics.patient_bootstrap_indices"}
    return {
        "protocol": "Frozen final evaluation protocol: one-time test evaluation of the reference model",
        "status": "FROZEN — test split NOT evaluated; no test inference has been run",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head_at_freeze": git("rev-parse", "HEAD"),
        "git_status_at_freeze": (git("status", "--porcelain") or "").splitlines(),
        "1_frozen_model": {
            "architecture": "baseline_cnn", "run": MODEL_RUN, "seed": run["seed"],
            "checkpoint": f"models/{MODEL_RUN}/best.pt", "checkpoint_sha256": files[f"models/{MODEL_RUN}/best.pt"]["sha256"],
            "checkpoint_epoch": ckpt["epoch"], "parameter_count": run["model"]["parameter_count"],
            "config": {k: v for k, v in ckpt["config"].items() if k not in ("checkpoint_dir", "reports_dir")},
            "validation_macro_f1_at_selection": run["best_validation"]["macro_f1"],
            "trained_at_commit": run["code"]["git_commit"],
            "rules": ["no retraining", "no ensembling", "no architecture, hyperparameter or checkpoint changes"],
            "why_seed_42": "baseline_cnn_v3 (seed 42) was the reference run selected during the baseline optimization "
                           "study, before the multi-seed study. The seed study (seeds 42/1337/2026) verified that "
                           "architecture differences did not exceed seed noise and recommended baseline_cnn under its "
                           "predeclared rule; it was used to check stability, not to pick the best seed. Seeds 1337 "
                           "and 2026 are not evaluated on test.",
            "code_compatibility": "the checkpoint predates the config 'model' field; train.build_model treats its "
                                  "absence as baseline_cnn. Current inference code reproduces the recorded validation "
                                  "predictions exactly (reports/robustness/noise_robustness/clean_check.json).",
        },
        "2_frozen_data_pipeline": {
            "patient_split": {"file": "data/splits/patient_split.csv",
                              "sha256": files["data/splits/patient_split.csv"]["sha256"],
                              "patients": {s: sum(r["split"] == s for r in split_rows) for s in ("train", "val", "test")},
                              "method": "patient-balanced, seed 42 (src/data/create_splits.py); no patient in two splits"},
            "metadata": {"file": "data/metadata.csv", "sha256": files["data/metadata.csv"]["sha256"],
                         "test_rows": audit["test_cycles"] if "test_cycles" in audit else None},
            "normalization": {"file": "data/normalization/logmel_primary_train_band_stats.json",
                              "sha256": files["data/normalization/logmel_primary_train_band_stats.json"]["sha256"],
                              "fitted_on": "train patients only (88), per-mel-band mean/std in dB"},
            "checkpoint_data_sha256_must_match": ckpt["data_sha256"],
            "audio_preprocessing": {
                "read": "soundfile, float32 in [-1, 1]; multi-channel averaged to mono",
                "cycle_slice": "[cycle_start_s, cycle_end_s) at the original sample rate; end up to "
                               f"{ap.DURATION_TOLERANCE_S} s past the file is clipped",
                "leading_zero_trimming": f"leading samples exactly 0.0 removed, capped at {ap.MAX_LEADING_SILENCE_S} s, "
                                         "at least one sample kept; applied to every cycle on every device",
                "resampling": f"torchaudio.functional.resample (windowed sinc) to {ap.TARGET_SAMPLE_RATE} Hz",
                "no": "padding, cropping, amplitude normalization or augmentation"},
            "logmel": {**dataclasses.asdict(lm.PRIMARY_CONFIG),
                       "details": "DC removal (subtract waveform mean); Hann window; center=True with zero padding; "
                                  "power 2; HTK mel scale; no filter-area normalization; 10*log10(max(power, floor))"},
            "model_input": "train-band standardization with the frozen statistics; batches padded with 0 and a "
                           "valid-frame mask; sequential (unshuffled) evaluation loader, batch size 32, num_workers 0",
            "label_mapping": {c: i for i, c in enumerate(rd.CLASS_NAMES)},
            "quality_metadata": {
                "original_clipped_fraction": f"share of original-rate samples with |x| >= {ap.CLIP_THRESHOLD}",
                "clipped_plateau_count": f"contiguous runs with |x| >= {ap.CLIP_THRESHOLD} lasting >= "
                                         f"{ap.MIN_CLIPPED_PLATEAU_S * 1000:.0f} ms",
                "measured_on": "the annotated cycle at its original sample rate, before trimming and resampling"},
            "determinism": "train.set_determinism(0): cuDNN deterministic, benchmark off, deterministic algorithms "
                           "(warn_only), CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "rule": "nothing in this pipeline may change after the test evaluation",
        },
        "3_primary_endpoint": {
            "primary_metric": "macro F1 over the 4 classes on all test cycles (classification_metrics."
                              "classification_report; zero division gives 0)",
            "secondary_metrics": ["balanced accuracy", "accuracy", "per-class precision", "per-class recall",
                                  "per-class F1", "confusion matrix"],
            "confidence_intervals": bootstrap,
            "rule": "the primary metric will not change after viewing test results",
        },
        "4_quality_sensitivity_groups": {
            "groups": {"all": "every cycle (primary)",
                       "moderately_clean": f"original_clipped_fraction <= {rd.MODERATELY_CLEAN_MAX_CLIPPED_FRACTION}",
                       "clean": f"original_clipped_fraction <= {rd.CLEAN_MAX_CLIPPED_FRACTION} and "
                                "clipped_plateau_count == 0"},
            "fixed": "README, 2026-10-03, before any model was trained (respiratory_dataset.in_quality_group)",
            "metrics_per_group": ["n cycles and patients", "macro F1 over 4 classes (zero division 0)",
                                  "macro F1 over classes present in the group (analyze_run.group_metrics convention)",
                                  "balanced accuracy (4 classes and classes present)", "accuracy", "per-class F1",
                                  "per-class support", "confusion matrix",
                                  "patient-bootstrap 95% interval of macro F1 (resampling the group's patients)"],
            "role": "sensitivity analyses; groups filter test cycles, they never change the split",
        },
        "5_calibration": {
            "frozen_temperature": temp["temperature"],
            "source": "reports/calibration_uncertainty/summary.json -> temperature_scaling.final_validation_temperature "
                      "(fitted on all 19 validation patients, clean)",
            "application": "p_T = softmax(logits / T), applied once; T is never fitted or modified on test data",
            "report_for": ["raw probabilities", "probabilities scaled with the frozen T"],
            "metrics": {"nll": cal_plan["definitions"]["nll"], "brier": cal_plan["definitions"]["brier"],
                        "ece": cal_plan["definitions"]["ece_primary"],
                        "overconfidence_gap": cal_plan["definitions"]["overconfidence_gap"],
                        "reliability_diagram": cal_plan["definitions"]["reliability_diagram"]},
            "secondary_metrics_same_definitions_as_validation": cal_plan["definitions"]["ece_secondary"],
            "statement": "temperature scaling cannot change any predicted class (T > 0 preserves the argmax), so "
                         "accuracy, macro F1 and the confusion matrix are identical for raw and scaled probabilities",
        },
        "6_uncertainty_error_detection": {
            "scores": {"primary": ["1 - MSP", "predictive entropy (nats)"],
                       "secondary": ["1 - top-2 margin (predeclared in the validation plan)"]},
            "target": "error = wrong prediction (positive class)",
            "metrics": ["AUROC (Mann-Whitney, ties 1/2)", "AUPRC (average precision; tied scores one threshold)",
                        "test error rate as the AUPRC chance reference"],
            "probabilities": "raw (primary) and frozen-T scaled (secondary)",
            "rule": "no new uncertainty score will be introduced after seeing test results",
        },
        "7_selective_prediction": {
            "coverage_levels": cal_plan["part4_selective_prediction"]["coverage_levels"],
            "n_retained": cal_plan["part4_selective_prediction"]["n_retained"],
            "ranking": {"primary": "raw MSP, most confident first, stable tie-break by file order",
                        "secondary": ["lowest raw entropy", "frozen-T scaled MSP"]},
            "primary_analysis": "selective risk (error rate among retained) and the risk-coverage curve; AURC and E-AURC",
            "secondary_metrics": ["macro F1", "balanced accuracy", "accuracy",
                                  "retained / rejected counts by true class and by predicted class"],
            "zero_division": "project-wide convention (classification_metrics._safe_div): any ratio with a zero "
                             "denominator is 0, so a class with no retained true cycles has recall 0 and a class never "
                             "predicted has precision 0, both giving F1 0; such a class still counts in the 4-class "
                             "macro F1 and balanced accuracy and its zero support is shown in the retained counts "
                             "(as in the validation plan, clarification 3)",
            "frozen_validation_thresholds": {"tau_80": taus["tau_80"], "tau_50": taus["tau_50"],
                                             "definition": "raw MSP of the last retained clean-validation cycle at 80% / "
                                                           "50% coverage; applied unchanged to test (retain if MSP >= tau)",
                                             "report": ["realized test coverage", "selective risk among retained",
                                                        "confidently wrong share (wrong and MSP >= tau_50)"]},
            "rule": "no threshold will be chosen using test performance",
        },
        "8_robustness": {
            "primary_final_test_evaluation": "untouched clean test audio through the frozen pipeline",
            "secondary_predeclared_robustness_evaluation": "the 15 frozen conditions below applied to test audio ONLY "
                                                           "after the clean test outputs are written and locked",
            "conditions": conds, "definitions": rob_plan["definitions"],
            "lock": "clean test predictions, logits and metrics are written to disk and their SHA-256 hashes recorded in "
                    "a lock file before any degraded test inference starts; the degraded run's 'clean' condition must "
                    "reproduce the locked clean predictions exactly",
            "reported": ["per condition: macro F1, balanced accuracy, accuracy, per-class F1, confusion matrix, absolute "
                         "and relative macro-F1 change vs clean test, patient-bootstrap interval of the change, share of "
                         "predictions changed", "uncertainty under shift: mean MSP, entropy, overconfidence gap, ECE, NLL, "
                         "error-detection AUROC, raw and frozen-T scaled; tau_80/tau_50 coverage, risk and confidently "
                         "wrong share"],
            "rule": "degradation parameters, seeds and conditions will not change after seeing test performance",
        },
        "9_statistical_rules": {
            "bootstrap": bootstrap,
            "rules": ["patient-level resampling only; cycles of one patient are never treated as independent",
                      "no significance fishing, no post-test model selection, no re-running with changed settings",
                      "predeclared analyses and any later exploratory observations are reported in separate, "
                      "labelled sections",
                      "with 19 test patients intervals are wide; results are estimates with uncertainty, not "
                      "significance claims", "no medical or diagnostic claims"],
            "reference_for_magnitudes": {"seed_study_pooled_seed_sd_macro_f1": seed["pooled_within_architecture_seed_sd"]},
        },
        "10_test_access_audit": audit,
        "11_freeze_manifest": manifest,
        "12_unlock_procedure": [
            "Commit this protocol, src/evaluation/final_evaluation.py and its tests; the working tree must be clean "
            "(the test stages refuse to run otherwise) and the commit is recorded in every test output.",
            "Run `python src/evaluation/final_protocol.py --verify`; abort if any content identity differs.",
            f"Run `{fe.COMMANDS['dry_run']}` (VALIDATION only): it must reproduce "
            "reports/runs/baseline_cnn_v3/val_predictions.csv and run.json's validation metrics exactly.",
            f"Unlock once: `{fe.COMMANDS['clean_test']}`. It verifies everything again, writes a STARTED marker, loads "
            "the test split once, saves per-cycle logits/probabilities/predictions, computes sections 3-7, and writes "
            "clean_test/lock.json with the SHA-256 of every clean output.",
            f"Only then: `{fe.COMMANDS['robustness_test']}`. It refuses unless the clean lock verifies, first reproduces "
            "the locked clean predictions through its own clean condition, then runs the 15 frozen conditions with the "
            "frozen T and thresholds (section 8).",
            "Write the final report with predeclared results first and any exploratory observations separately. Any "
            "bug found after unlock is documented; original outputs are preserved, never silently replaced, and no "
            "stage is re-run because a result looks surprising (STARTED markers block re-runs).",
        ],
        "13_final_evaluator": {
            "script": "src/evaluation/final_evaluation.py",
            "stages": {"verify": "Stage A: protocol manifest, checkpoint/data hashes, frozen values, provenance",
                       "dry-run-validation": "Stage B: exact clean inference path on VALIDATION; must reproduce the "
                                             "frozen reference",
                       "clean-test": "Stages A + C + D: one-time clean test evaluation and its lock",
                       "robustness-test": "Stage E: fixed degradations on test, only after the clean lock verifies"},
            "commands": fe.COMMANDS,
            "unlock_token": fe.UNLOCK_TOKEN,
            "default_behaviour": "no stage runs without an explicit subcommand; test stages additionally require the "
                                 "exact unlock token and a clean Git working tree; validation is the only split the "
                                 "dry run can read",
            "no_test_time_choices": "the temperature, thresholds, coverages, conditions, metrics, checkpoint and "
                                    "bootstrap settings come only from this protocol; the CLI has no flags for them; "
                                    "nothing is trained, fitted or selected on test data",
            "outputs": {"validation_dry_run": "reports/final_evaluation/validation_dry_run/",
                        "clean_test": "reports/final_evaluation/clean_test/ (STARTED.json, predictions.csv, "
                                      "metrics.json, summary.md, reliability_diagram.svg, risk_coverage_curves.csv, "
                                      "lock.json)",
                        "robustness_test": "reports/final_evaluation/robustness_test/ (STARTED.json, conditions/<c>/"
                                           "predictions.csv, summary.json, summary.md, lock.json)"},
        },
        "environment": {k: v for k, v in env.items() if k != "pip_freeze"},
        "environment_pip_freeze_file": "reports/final_evaluation/environment_pip_freeze.txt",
    }


def render_markdown(p: dict) -> str:
    m, d, e = p["1_frozen_model"], p["2_frozen_data_pipeline"], p["3_primary_endpoint"]
    q, c, u, s, r = (p["4_quality_sensitivity_groups"], p["5_calibration"], p["6_uncertainty_error_detection"],
                     p["7_selective_prediction"], p["8_robustness"])
    a = p["10_test_access_audit"]
    lm_cfg = d["logmel"]
    L = ["# Frozen final evaluation protocol", "",
         f"**Status: {p['status']}.** Frozen {p['created']} at commit `{p['git_head_at_freeze']}`. "
         "Machine-readable version with every hash: `protocol.json`. Verify before unlocking with "
         "`python src/evaluation/final_protocol.py --verify`.", "",
         "## 1. Frozen model", "",
         f"- `{m['architecture']}`, run `{m['run']}` (seed {m['seed']}), checkpoint `{m['checkpoint']}` "
         f"(epoch {m['checkpoint_epoch']}, {m['parameter_count']:,} parameters).",
         f"- Checkpoint SHA-256: `{m['checkpoint_sha256']}`.",
         f"- Validation macro F1 at selection: {m['validation_macro_f1_at_selection']:.4f}.",
         "- No retraining, no ensembling, no architecture/hyperparameter/checkpoint changes.",
         f"- Why seed 42: {m['why_seed_42']}", "",
         "## 2. Frozen data pipeline", "",
         f"- Patient split `{d['patient_split']['file']}` (train/val/test patients "
         f"{d['patient_split']['patients']['train']}/{d['patient_split']['patients']['val']}/"
         f"{d['patient_split']['patients']['test']}), SHA-256 `{d['patient_split']['sha256']}`.",
         f"- Metadata SHA-256 `{d['metadata']['sha256']}`; normalization statistics SHA-256 "
         f"`{d['normalization']['sha256']}` ({d['normalization']['fitted_on']}). These three must also match the "
         "hashes stored inside the checkpoint.",
         f"- Audio: {d['audio_preprocessing']['read']}; slice {d['audio_preprocessing']['cycle_slice']}; "
         f"{d['audio_preprocessing']['leading_zero_trimming']}; {d['audio_preprocessing']['resampling']}; no "
         f"{d['audio_preprocessing']['no']}.",
         f"- Log-Mel: {lm_cfg['n_mels']} mels, {lm_cfg['f_min']:g}–{lm_cfg['f_max']:g} Hz, n_fft {lm_cfg['n_fft']}, "
         f"window {lm_cfg['win_length']}, hop {lm_cfg['hop_length']}, {lm_cfg['sample_rate']} Hz, floor "
         f"{lm_cfg['log_floor']:g}; {lm_cfg['details']}.",
         f"- Model input: {d['model_input']}.",
         f"- Labels: {', '.join(f'{k} = {v}' for k, v in d['label_mapping'].items())}.",
         f"- Quality metadata: clipped fraction = {d['quality_metadata']['original_clipped_fraction']}; plateau count "
         f"= {d['quality_metadata']['clipped_plateau_count']}; measured on {d['quality_metadata']['measured_on']}.",
         f"- Determinism: {d['determinism']}.", "",
         "## 3. Primary endpoint", "",
         f"- **Primary metric: {e['primary_metric']}.** It will not change after viewing test results.",
         f"- Secondary: {', '.join(e['secondary_metrics'])}.",
         f"- Intervals: patient-level bootstrap, {e['confidence_intervals']['n_resamples']} resamples, seed "
         f"{e['confidence_intervals']['seed']}; {e['confidence_intervals']['rng']}; {e['confidence_intervals']['interval']}.",
         "", "## 4. Quality sensitivity groups (fixed 2026-10-03)", ""]
    L += [f"- **{g}**: {v}" for g, v in q["groups"].items()]
    L += [f"- Reported per group: {', '.join(q['metrics_per_group'])}.", "",
          "## 5. Calibration", "",
          f"- **Frozen temperature T = {c['frozen_temperature']!r}** ({c['source']}). Applied once; never fitted or "
          "changed on test.",
          "- Reported for raw and frozen-T probabilities: NLL, Brier, 15-bin equal-width ECE, overconfidence gap, "
          "reliability diagram; secondary: equal-mass ECE, classwise ECE, MCE (validation definitions).",
          f"- ECE binning: {c['metrics']['ece']}.",
          f"- {c['statement'][0].upper() + c['statement'][1:]}.", "",
          "## 6. Uncertainty / error detection", "",
          f"- Scores: {', '.join(u['scores']['primary'])} (primary); {', '.join(u['scores']['secondary'])}.",
          f"- {', '.join(u['metrics'])}; {u['probabilities']}. No new score after seeing test results.", "",
          "## 7. Selective prediction", "",
          f"- Coverages {', '.join(f'{x:.0%}' for x in s['coverage_levels'])}; retained n = {s['n_retained']}.",
          f"- Primary: {s['primary_analysis']}; ranking {s['ranking']['primary']}. Secondary: "
          f"{', '.join(s['secondary_metrics'])}.",
          f"- Frozen clean-validation thresholds: **τ80 = {s['frozen_validation_thresholds']['tau_80']!r}**, "
          f"**τ50 = {s['frozen_validation_thresholds']['tau_50']!r}**; {s['frozen_validation_thresholds']['definition']}.",
          f"- Zero division: {s['zero_division']}", "",
          "## 8. Robustness", "",
          f"- **Primary final test evaluation:** {r['primary_final_test_evaluation']}.",
          f"- **Secondary predeclared robustness evaluation:** {r['secondary_predeclared_robustness_evaluation']}.",
          f"- Lock: {r['lock']}.",
          "- Conditions (frozen): " + ", ".join(x["name"] for x in r["conditions"]) + ".", "",
          "## 9. Statistical rules", ""]
    L += [f"- {x}" for x in p["9_statistical_rules"]["rules"]]
    L += ["", "## 10. Test-access audit", "",
          f"- **Passed: {a['passed']}.** {a['test_patients']} test patients, {a['test_recordings']} test recordings.",
          f"- Report/model files mentioning a test recording: {a['report_or_model_files_mentioning_a_test_recording'] or 'none'}; "
          f"prediction CSVs with test patients: {a['prediction_csvs_containing_test_patients'] or 'none'} "
          f"({a['n_report_csvs_checked']} CSVs checked).",
          f"- Records with test_split_evaluated = true: {a['json_records_with_test_split_evaluated_true'] or 'none'}; "
          f"non-validation split values in report JSON: {a['non_validation_split_values_in_report_json'] or 'none'}; "
          f"datasets used by any training run: {a['datasets_used_by_any_training_run']}.",
          f"- Code calling evaluate(..., allow_test=True): {a['code_calling_evaluate_with_allow_test_true'] or 'none'}; "
          f"test-like files ever committed: {a['test_like_files_ever_committed'] or 'none'}.",
          f"- Checkpoint contains: {a['checkpoint_contains']}.", "",
          "## 11. Freeze manifest", "",
          "| File | SHA-256 | Git blob |", "|---|---|---|"]
    L += [f"| `{k}` | `{v['sha256'][:16]}…` | `{(v['git_blob'] or '—')[:12]}` |"
          for k, v in p["11_freeze_manifest"]["files"].items()]
    au = p["11_freeze_manifest"].get("audio")
    if au:
        L.append(f"| `{au['directory']}` ({au['n_wav_files']} .wav, combined) | `{au['combined_sha256'][:16]}…` | — |")
    L += ["", "Full hashes are in `protocol.json`. Git blob ids are stable across LF/CRLF checkouts; raw SHA-256 "
          "values describe this working copy.", "", "## 12. Unlock procedure", ""]
    L += [f"{i}. {x}" for i, x in enumerate(p["12_unlock_procedure"], 1)]
    ev = p["13_final_evaluator"]
    L += ["", "## 13. Final evaluator", "", f"- Script `{ev['script']}`; {ev['default_behaviour']}.",
          f"- {ev['no_test_time_choices'][0].upper() + ev['no_test_time_choices'][1:]}."]
    L += [f"- `{k}`: {v}" for k, v in ev["stages"].items()]
    L += [f"- Unlock token: `{ev['unlock_token']}`."]
    if p.get("supersedes"):
        sp = p["supersedes"]
        L += ["", f"Version {p['version']}; supersedes `{sp['file']}` (version {sp['version']}, created {sp['created']}, "
              f"SHA-256 `{sp['sha256']}`), revised before any test stage started. Changes: {sp['changes']}"]
    L += ["", f"Environment: Python {p['environment']['python']}, torch {p['environment']['torch']}, torchaudio "
          f"{p['environment']['torchaudio']}, numpy {p['environment']['numpy']}, CUDA build "
          f"{p['environment']['cuda_build']}, {p['environment']['gpu']}; full package list in "
          f"`{p['environment_pip_freeze_file']}`.", ""]
    return "\n".join(L)


TEST_STAGE_DIRS = ("clean_test", "robustness_test")


def test_stage_started(out_dir: Path) -> list[str]:
    return [d for d in TEST_STAGE_DIRS if (Path(out_dir) / d).exists()]


def supersede(out_dir: Path, changes: str) -> dict:
    """Move the current protocol aside (never deleted) and describe it for the next version."""
    started = test_stage_started(out_dir)
    if started:
        raise PermissionError(f"test stage(s) {started} already started; the protocol can no longer be revised")
    path = Path(out_dir) / "protocol.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    version = old.get("version", 1)
    kept = Path(out_dir) / f"protocol_v{version}_superseded.json"
    if kept.exists():
        raise FileExistsError(f"{kept} already exists")
    info = {"file": kept.name, "version": version, "created": old["created"], "sha256": sha256(path),
            "changes": changes}
    path.rename(kept)
    md = Path(out_dir) / "protocol.md"
    if md.exists():
        md.rename(Path(out_dir) / f"protocol_v{version}_superseded.md")
    return info


def main(argv: list[str] | None = None) -> int:
    ap_ = argparse.ArgumentParser(description="Freeze or verify the final evaluation protocol.")
    ap_.add_argument("--write", action="store_true")
    ap_.add_argument("--verify", action="store_true")
    ap_.add_argument("--supersede", metavar="CHANGES", help="with --write: replace a pre-final protocol (kept as "
                     "protocol_v<N>_superseded.json) only if no test stage has started; describe the changes")
    ap_.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap_.parse_args(argv)
    path = a.out_dir / "protocol.json"
    if a.verify:
        protocol = json.loads(path.read_text(encoding="utf-8"))
        bad = verify_manifest(protocol["11_freeze_manifest"])
        print("manifest OK: all frozen artifacts unchanged" if not bad else f"CHANGED: {bad}")
        return 1 if bad else 0
    if not a.write:
        ap_.error("pass --write or --verify")
    if path.exists():
        if not a.supersede:
            print(f"{path} exists; a frozen protocol is never overwritten (see --supersede)", file=sys.stderr)
            return 1
        if test_stage_started(a.out_dir):
            print(f"refused: test stage(s) {test_stage_started(a.out_dir)} already started", file=sys.stderr)
            return 1
    audit = test_access_audit()
    if not audit["passed"]:
        print(f"test-access audit FAILED: {audit}", file=sys.stderr)
        return 1
    with open(REPO_ROOT / "data" / "splits" / "patient_split.csv", newline="", encoding="utf-8") as f:
        test = {r["patient_id"] for r in csv.DictReader(f) if r["split"] == "test"}
    with open(REPO_ROOT / "data" / "metadata.csv", newline="", encoding="utf-8") as f:
        audit["test_cycles"] = sum(r["patient_id"] in test for r in csv.DictReader(f))  # row count only
    env = environment()
    manifest = build_manifest()
    protocol = build_protocol(audit, manifest, env)
    superseded = None
    if path.exists():
        try:
            superseded = supersede(a.out_dir, a.supersede)
        except (PermissionError, FileExistsError) as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
    protocol["version"] = 1 if superseded is None else superseded["version"] + 1
    protocol["supersedes"] = superseded
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "environment_pip_freeze.txt").write_text(env["pip_freeze"], encoding="utf-8")
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (a.out_dir / "protocol.md").write_text(render_markdown(protocol), encoding="utf-8")
    print(f"wrote {path} and protocol.md; audit passed; {len(manifest['files'])} files + audio manifest hashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
