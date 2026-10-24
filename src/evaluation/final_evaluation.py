"""Staged, one-time final evaluation of the frozen reference model (implements reports/final_evaluation/protocol.json).

Stages (subcommands; there is no default stage):
  verify              A  verify the frozen protocol manifest, checkpoint/data hashes and frozen values
  dry-run-validation  B  run the exact clean inference path on the VALIDATION split; it must reproduce the frozen
                         baseline_cnn_v3 validation predictions and metrics, otherwise it fails loudly
  clean-test          A+C+D  ONE-TIME clean test evaluation, then a lock of its outputs
  robustness-test     E  the 15 frozen degradations on test, only after the clean lock verifies

Safety:
  * Test stages need the exact unlock token (--unlock-test) and a clean Git working tree; the default of every
    function is to refuse the test split (TestAccessError).
  * Each test stage writes STARTED.json before touching test data; if its directory exists the stage refuses to
    run again (no overwriting, no re-running because a result looks surprising).
  * Temperature, thresholds, coverages, conditions, bootstrap settings and the checkpoint come only from the
    protocol. Nothing is trained, fitted or selected: the model is in eval mode under torch.no_grad with
    gradients disabled, and the CLI exposes no flags for any of these choices.
  * The robustness stage first reproduces the locked clean test predictions through its own clean condition.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
for _sub in ("models", "data", "features", "evaluation"):
    if str(SRC_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(SRC_DIR / _sub))

import analyze_run as ar  # noqa: E402
import calibration_metrics as cmx  # noqa: E402
import calibration_report as crep  # noqa: E402
import calibration_uncertainty as cu  # noqa: E402
import compare_runs as cr  # noqa: E402
import degradations as dg  # noqa: E402
import final_protocol as fp  # noqa: E402
import noise_robustness as nr  # noqa: E402
import train as tr  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402

rd, nz = tr.rd, tr.nz
REPO_ROOT = SRC_DIR.parent
OUT_DIR = REPO_ROOT / "reports" / "final_evaluation"
CLASS_NAMES = rd.CLASS_NAMES
K = len(CLASS_NAMES)
MODEL_RUN = "baseline_cnn_v3"
UNLOCK_TOKEN = "UNLOCK-ONE-TIME-TEST-EVALUATION-baseline_cnn_v3"
_SCRIPT = "python src/evaluation/final_evaluation.py"
COMMANDS = {"verify": f"{_SCRIPT} verify", "dry_run": f"{_SCRIPT} dry-run-validation",
            "clean_test": f"{_SCRIPT} clean-test --unlock-test {UNLOCK_TOKEN}",
            "robustness_test": f"{_SCRIPT} robustness-test --unlock-test {UNLOCK_TOKEN}"}
PROB_TOLERANCE = 1e-6  # reproduction: largest allowed |probability difference|; the actual maximum is always reported


class FrozenArtifactError(RuntimeError):
    """A frozen artifact or value differs from the protocol."""


class TestAccessError(PermissionError):
    """The test split was requested without the explicit one-time unlock."""


class ResultsLockedError(RuntimeError):
    """A test stage already started / a lock is missing or does not verify."""


class ReproductionError(RuntimeError):
    """An inference path did not reproduce its frozen reference."""


@dataclasses.dataclass(frozen=True)
class Context:
    root: Path = REPO_ROOT
    protocol_path: Path = OUT_DIR / "protocol.json"
    out_dir: Path = OUT_DIR
    checkpoint_dir: Path = REPO_ROOT / "models"
    model_run: str = MODEL_RUN
    metadata: Path = rd.DEFAULT_METADATA
    split_file: Path = rd.DEFAULT_SPLIT_FILE
    audio_dir: Path = rd.ap.DEFAULT_AUDIO_DIR
    normalization: Path = nz.DEFAULT_STATS_PATH
    reference_val_predictions: Path = REPO_ROOT / "reports" / "runs" / MODEL_RUN / "val_predictions.csv"
    reference_run_json: Path = REPO_ROOT / "reports" / "runs" / MODEL_RUN / "run.json"
    calibration_summary: Path = REPO_ROOT / "reports" / "calibration_uncertainty" / "summary.json"
    audio_manifest_dir: Path | None = fp.AUDIO_DIR
    device: str = "cuda"
    require_cuda: bool = True
    require_clean_git: bool = True


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    return fp.sha256(Path(path))


def _write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=cu._json_default) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- guards

def check_token(token: str | None) -> None:
    if token != UNLOCK_TOKEN:
        raise TestAccessError("the test split can only be evaluated with the exact one-time unlock token")


def assert_clean_git(root: Path) -> str:
    status = fp.git("status", "--porcelain", root=root)
    if status is None:
        raise FrozenArtifactError("cannot read Git status; test stages require a clean, committed working tree")
    if status.strip():
        raise FrozenArtifactError(f"working tree is not clean; commit the frozen state first:\n{status}")
    return fp.git("rev-parse", "HEAD", root=root)


def frozen_values(protocol: dict) -> dict:
    ci = protocol["3_primary_endpoint"]["confidence_intervals"]
    th = protocol["7_selective_prediction"]["frozen_validation_thresholds"]
    return {"temperature": float(protocol["5_calibration"]["frozen_temperature"]),
            "taus": {"tau_80": float(th["tau_80"]), "tau_50": float(th["tau_50"])},
            "coverages": [float(c) for c in protocol["7_selective_prediction"]["coverage_levels"]],
            "n_boot": int(ci["n_resamples"]), "seed": int(ci["seed"]),
            "conditions": [dg.Condition(**c) for c in protocol["8_robustness"]["conditions"]]}


# ---------------------------------------------------------------- Stage A

def verify(ctx: Context) -> dict:
    """Stage A. Raises FrozenArtifactError on any difference; returns the protocol, frozen values and provenance."""
    protocol = json.loads(Path(ctx.protocol_path).read_text(encoding="utf-8"))
    bad = fp.verify_manifest(protocol["11_freeze_manifest"], ctx.audio_manifest_dir, root=ctx.root)
    if bad:
        raise FrozenArtifactError(f"frozen artifacts changed since the protocol was written: {bad}")
    ckpt_path = Path(ctx.checkpoint_dir) / ctx.model_run / "best.pt"
    ckpt_sha = _sha(ckpt_path)
    if ckpt_sha != protocol["1_frozen_model"]["checkpoint_sha256"]:
        raise FrozenArtifactError("checkpoint hash differs from the protocol")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipe = protocol["2_frozen_data_pipeline"]
    current = {"metadata": _sha(ctx.metadata), "split_file": _sha(ctx.split_file),
               "normalization_stats": _sha(ctx.normalization)}
    recorded = {"metadata": pipe["metadata"]["sha256"], "split_file": pipe["patient_split"]["sha256"],
                "normalization_stats": pipe["normalization"]["sha256"]}
    for k in current:
        if not (current[k] == recorded[k] == ckpt["data_sha256"][k]):
            raise FrozenArtifactError(f"{k}: current, protocol and checkpoint hashes are not all equal")
    fv = frozen_values(protocol)
    if fv["conditions"] != list(dg.PREDECLARED_CONDITIONS):
        raise FrozenArtifactError("protocol degradation conditions differ from degradations.PREDECLARED_CONDITIONS")
    cal = json.loads(Path(ctx.calibration_summary).read_text(encoding="utf-8"))
    if fv["temperature"] != cal["temperature_scaling"]["final_validation_temperature"]["temperature"] or \
            fv["taus"] != {k: float(v) for k, v in cal["part5_degradation"]["fixed_clean_thresholds"].items()}:
        raise FrozenArtifactError("protocol temperature/thresholds differ from the stored validation calibration")
    if not (math.isfinite(fv["temperature"]) and fv["temperature"] > 0) or \
            not all(0 < t <= 1 for t in fv["taus"].values()):
        raise FrozenArtifactError("invalid frozen temperature or thresholds")
    provenance = {"verified": now(), "protocol_sha256": _sha(ctx.protocol_path),
                  "protocol_version": protocol.get("version", 1), "checkpoint_sha256": ckpt_sha,
                  "data_sha256": current, "git_head": fp.git("rev-parse", "HEAD", root=ctx.root),
                  "git_status": (fp.git("status", "--porcelain", root=ctx.root) or "").splitlines(),
                  "python": platform.python_version(), "torch": torch.__version__,
                  "device": ctx.device, "cuda_available": torch.cuda.is_available()}
    return {"protocol": protocol, "frozen": fv, "provenance": provenance}


# ---------------------------------------------------------------- data / inference

class FinalDegradedDataset(nr.DegradedCycleDataset):
    """noise_robustness.DegradedCycleDataset for the validation or (unlocked) test split; identical load_features."""

    def __init__(self, split: str, condition: dg.Condition, allow_test: bool = False,
                 base_seed: int = dg.DEFAULT_BASE_SEED, **kw):
        if split == "test" and not allow_test:
            raise TestAccessError("degraded test data requires the one-time unlock")
        if split not in ("val", "test"):
            raise PermissionError(f"final evaluation reads only val (dry run) or unlocked test, not {split!r}")
        rd.RespiratoryCycleDataset.__init__(self, split, **kw)
        self.condition, self.base_seed, self.degradation_stats = condition, base_seed, {}


def make_dataset(ctx: Context, split: str, allow_test: bool = False, condition: dg.Condition | None = None):
    if split == "test" and not allow_test:
        raise TestAccessError("the test split requires the explicit one-time unlock")
    if split not in ("val", "test"):
        raise PermissionError(f"final evaluation reads only val (dry run) or unlocked test, not {split!r}")
    kw = dict(metadata_path=ctx.metadata, split_path=ctx.split_file, audio_dir=ctx.audio_dir,
              normalizer=nz.load_normalizer(ctx.normalization))
    if condition is None:
        return rd.RespiratoryCycleDataset(split, **kw)
    return FinalDegradedDataset(split, condition, allow_test=allow_test, **kw)


def device_for(ctx: Context) -> torch.device:
    if ctx.require_cuda and not torch.cuda.is_available():
        raise FrozenArtifactError("CUDA is required (the frozen reference was produced on the GPU)")
    return torch.device(ctx.device if ctx.device == "cpu" or torch.cuda.is_available() else "cpu")


def load_model(ctx: Context, device: torch.device):
    tr.set_determinism(0)
    model, ckpt = ar.load_run_model(ctx.model_run, ctx.checkpoint_dir, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


@torch.no_grad()
def infer(model, dataset, device, allow_test: bool = False) -> list[dict]:
    """Per-cycle logits, probabilities and predictions (same computation as analyze_run.predict, plus logits)."""
    if dataset.split == "test" and not allow_test:
        raise TestAccessError("inference on the test split requires the explicit one-time unlock")
    if dataset.split not in ("val", "test"):
        raise PermissionError(f"inference only on val or unlocked test, not {dataset.split!r}")
    if model.training:
        raise RuntimeError("model must be in eval mode")
    stats = getattr(dataset, "degradation_stats", None)
    rows = []
    for batch in tr.make_eval_loader(dataset, 32):
        logits = model(batch["features"].to(device), batch["mask"].to(device)).float().cpu()
        probs = torch.softmax(logits, dim=1)
        for i in range(len(batch["labels"])):
            q = batch["quality"][i]
            row = {"patient_id": batch["patient_id"][i], "recording_filename": batch["recording_filename"][i],
                   "cycle_index": int(batch["cycle_index"][i]), "device": batch["device"][i],
                   "original_sample_rate": int(batch["original_sample_rate"][i]),
                   "cycle_duration_s": float(batch["cycle_duration_s"][i]),
                   "original_clipped_fraction": q["original_clipped_fraction"],
                   "clipped_plateau_count": q["clipped_plateau_count"],
                   "label": int(batch["labels"][i]), "pred": int(probs[i].argmax()),
                   **{f"logit_{c}": float(logits[i, k]) for k, c in enumerate(CLASS_NAMES)},
                   **{f"prob_{c}": float(probs[i, k]) for k, c in enumerate(CLASS_NAMES)}}
            if stats is not None:
                row.update(stats[(row["recording_filename"], row["cycle_index"])])
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arrays(rows: list[dict]) -> dict:
    return {"patient": [str(r["patient_id"]) for r in rows], "labels": np.array([int(r["label"]) for r in rows]),
            "preds": np.array([int(r["pred"]) for r in rows]),
            "logits": np.array([[float(r[f"logit_{c}"]) for c in CLASS_NAMES] for r in rows]),
            "probs": np.array([[float(r[f"prob_{c}"]) for c in CLASS_NAMES] for r in rows]),
            "keys": [(r["recording_filename"], int(r["cycle_index"])) for r in rows],
            "quality": [{"original_clipped_fraction": float(r["original_clipped_fraction"]),
                         "clipped_plateau_count": int(r["clipped_plateau_count"])} for r in rows]}


def report(labels: np.ndarray, preds: np.ndarray) -> dict:
    return classification_report(confusion_matrix(torch.tensor(labels), torch.tensor(preds), K), CLASS_NAMES)


def compare_predictions(reference: list[dict], rows: list[dict], what: str) -> dict:
    """Exact comparison of cycle ids/order, labels and predictions; probability differences are measured."""
    ref, new = arrays_ref(reference), arrays_ref(rows)
    same_keys = ref["keys"] == new["keys"]
    out = {"what": what, "n_reference": len(reference), "n_new": len(rows), "same_cycle_ids_and_order": same_keys}
    if same_keys:
        diff = np.abs(ref["probs"] - new["probs"])
        out.update(labels_identical=bool(np.array_equal(ref["labels"], new["labels"])),
                   predictions_identical=bool(np.array_equal(ref["preds"], new["preds"])),
                   max_abs_probability_difference=float(diff.max()),
                   n_probability_values_bitwise_identical=int((diff == 0).sum()),
                   n_probability_values=int(diff.size))
    out["ok"] = bool(same_keys and out.get("labels_identical") and out.get("predictions_identical")
                     and out.get("max_abs_probability_difference", math.inf) <= PROB_TOLERANCE)
    return out


def arrays_ref(rows: list[dict]) -> dict:
    return {"keys": [(r["recording_filename"], int(r["cycle_index"])) for r in rows],
            "labels": np.array([int(r["label"]) for r in rows]), "preds": np.array([int(r["pred"]) for r in rows]),
            "probs": np.array([[float(r[f"prob_{c}"]) for c in CLASS_NAMES] for r in rows])}


# ---------------------------------------------------------------- Stage B

def dry_run_validation(ctx: Context) -> dict:
    """Stage B: the exact clean inference path on VALIDATION must reproduce the frozen reference."""
    v = verify(ctx)
    device = device_for(ctx)
    model, _ = load_model(ctx, device)
    rows = infer(model, make_dataset(ctx, "val"), device)
    reference = read_rows(ctx.reference_val_predictions)
    cmp = compare_predictions(reference, rows, "validation predictions vs reports/runs/<run>/val_predictions.csv")
    a = arrays(rows)
    logit_consistency = float(np.abs(cmx.softmax(a["logits"]) - a["probs"]).max())
    rep = report(a["labels"], a["preds"])
    best = json.loads(Path(ctx.reference_run_json).read_text(encoding="utf-8"))["best_validation"]
    metrics = {"confusion_matrix_identical": rep["confusion_matrix"] == best["confusion_matrix"],
               **{f"{k}_identical": rep[k] == best[k] for k in ("macro_f1", "balanced_accuracy", "accuracy")},
               **{k: {"dry_run": rep[k], "reference": best[k], "difference": rep[k] - best[k]}
                  for k in ("macro_f1", "balanced_accuracy", "accuracy")}}
    result = {"stage": "B validation dry run", "split": "val", "test_split_evaluated": False,
              "predictions": cmp, "metrics_vs_run_json": metrics,
              "max_abs_softmax_of_saved_logits_minus_saved_probabilities": logit_consistency,
              "provenance": v["provenance"]}
    result["ok"] = bool(cmp["ok"] and all(val for key, val in metrics.items() if key.endswith("_identical"))
                        and logit_consistency <= PROB_TOLERANCE)
    d = Path(ctx.out_dir) / "validation_dry_run"
    d.mkdir(parents=True, exist_ok=True)
    write_rows(d / "predictions.csv", rows)
    _write_json(d / "comparison.json", result)
    if not result["ok"]:
        raise ReproductionError(f"validation dry run does not reproduce the frozen reference: {result}")
    return result


# ---------------------------------------------------------------- metrics (Stage C)

def quality_group_metrics(a: dict, n_boot: int, seed: int) -> dict:
    out = {}
    for g in rd.QUALITY_GROUPS:
        idx = np.array([i for i, q in enumerate(a["quality"]) if rd.in_quality_group(q, g)], dtype=int)
        if len(idx) == 0:
            out[g] = {"n_cycles": 0, "n_patients": 0}
            continue
        y, p = a["labels"][idx], a["preds"][idx]
        s = cmx.classification_summary(y, p, K)
        patients = [a["patient"][i] for i in idx]
        boots = [cmx.classification_summary(y[b], p[b], K) for b in cmx.patient_bootstrap_indices(patients, n_boot, seed)]
        out[g] = {"n_cycles": int(len(idx)), "n_patients": len(set(patients)), **s,
                  "macro_f1_ci95": cmx.percentile_ci([b["macro_f1"] for b in boots]),
                  "macro_f1_present_classes_ci95": cmx.percentile_ci([b["macro_f1_present_classes"] for b in boots])}
    return out


def clean_metrics(a: dict, fv: dict) -> tuple[dict, dict]:
    """All predeclared clean metrics (protocol sections 3-7) from per-cycle arrays; nothing is fitted."""
    y, preds, logits, patients = a["labels"], a["preds"], a["logits"], a["patient"]
    p_raw = cmx.softmax(logits)
    p_t = cmx.apply_temperature(logits, fv["temperature"])
    if not (np.array_equal(p_raw.argmax(1), preds) and np.array_equal(p_t.argmax(1), preds)):
        raise ReproductionError("predictions are not the argmax of the logits, or temperature changed a prediction")
    rep = report(y, preds)
    n_boot, seed, covs, taus = fv["n_boot"], fv["seed"], fv["coverages"], fv["taus"]
    draws = cmx.patient_bootstrap_indices(patients, n_boot, seed)
    boot: dict[str, list[float]] = {}

    def put(k, v):
        boot.setdefault(k, []).append(float(v))

    correct = preds == y
    orders = {"msp_raw": cmx.retention_order(cmx.confidence(p_raw)),
              "entropy_raw": cmx.retention_order(cmx.entropy(p_raw), higher_is_more_confident=False),
              "msp_frozen_t": cmx.retention_order(cmx.confidence(p_t))}
    for idx in draws:
        yb, pb = y[idx], preds[idx]
        s = cmx.classification_summary(yb, pb, K)
        for k in ("macro_f1", "balanced_accuracy", "accuracy"):
            put(f"primary.{k}", s[k])
        for w, p in (("raw", p_raw[idx]), ("frozen_t", p_t[idx])):
            b = cu.calibration_block(p, yb, bins=False)
            for k in ("nll", "brier", "ece", "overconfidence_gap", "ece_equal_mass", "classwise_ece", "mce"):
                put(f"calibration.{w}.{k}", b[k])
            ed = cu.error_detection(p, yb)
            for sc in ("1-msp", "entropy", "1-margin"):
                put(f"error_detection.{w}.{sc}.auroc", ed[sc]["auroc"])
                put(f"error_detection.{w}.{sc}.auprc", ed[sc]["auprc"])
        cb = pb == yb
        o = cmx.retention_order(cmx.confidence(p_raw[idx]))
        rc = cmx.risk_coverage(o, cb)
        put("selective.msp_raw.aurc", rc["aurc"])
        put("selective.msp_raw.e_aurc", rc["e_aurc"])
        for c in covs:
            sel = cmx.selective_at(o, yb, pb, c, K)
            put(f"selective.msp_raw.{c}.selective_risk", sel["selective_risk"])
            put(f"selective.msp_raw.{c}.macro_f1", sel["macro_f1"])
        for name, t in cu.threshold_block(p_raw[idx], yb, taus).items():
            for k in ("realized_coverage", "selective_risk", "confidently_wrong_share"):
                put(f"thresholds.{name}.{k}", t[k])
    ci = {k: cmx.percentile_ci(v) for k, v in boot.items()}

    selective, curves = {}, {"n": np.arange(1, len(y) + 1), "coverage": np.arange(1, len(y) + 1) / len(y)}
    for r, o in orders.items():
        rc = cmx.risk_coverage(o, correct)
        selective[r] = {"aurc": rc["aurc"], "e_aurc": rc["e_aurc"],
                        "levels": [cmx.selective_at(o, y, preds, c, K) for c in covs]}
        curves[f"risk_{r}"] = rc["risk"]
        curves["oracle_risk"] = rc["oracle_risk"]
    metrics = {
        "split": "test", "n_cycles": int(len(y)), "n_patients": len(set(patients)),
        "primary": {"macro_f1": rep["macro_f1"], "macro_f1_ci95": ci["primary.macro_f1"]},
        "secondary": {"balanced_accuracy": rep["balanced_accuracy"], "balanced_accuracy_ci95": ci["primary.balanced_accuracy"],
                      "accuracy": rep["accuracy"], "accuracy_ci95": ci["primary.accuracy"],
                      "per_class": rep["per_class"], "confusion_matrix": rep["confusion_matrix"]},
        "quality_groups": quality_group_metrics(a, n_boot, seed),
        "calibration": {"frozen_temperature": fv["temperature"], "predictions_unchanged_by_temperature": True,
                        "raw": cu.calibration_block(p_raw, y), "frozen_t": cu.calibration_block(p_t, y)},
        "error_detection": {"raw": cu.error_detection(p_raw, y), "frozen_t": cu.error_detection(p_t, y)},
        "selective_prediction": {"coverage_levels": covs, "rankings": selective},
        "frozen_validation_thresholds": {"taus": taus, "raw": cu.threshold_block(p_raw, y, taus)},
        "bootstrap": {"unit": "patient", "n_resamples": n_boot, "seed": seed},
        "ci95": ci,
    }
    return metrics, {"curves": curves, "p_t": p_t}


def clean_summary_md(m: dict, provenance: dict) -> str:
    f = crep.f
    ci = lambda k: f"({m['ci95'][k][0]:.3f}, {m['ci95'][k][1]:.3f})"  # noqa: E731
    pr, se, cal = m["primary"], m["secondary"], m["calibration"]
    L = ["# Final clean test evaluation (one-time) — frozen baseline_cnn_v3", "",
         f"Test split: {m['n_cycles']} cycles, {m['n_patients']} patients. Protocol SHA-256 "
         f"`{provenance['protocol_sha256']}`, checkpoint `{provenance['checkpoint_sha256'][:16]}…`, commit "
         f"`{provenance['git_head']}`. All analyses predeclared in the frozen protocol; intervals are patient-level "
         f"bootstrap 95% ({m['bootstrap']['n_resamples']} resamples, seed {m['bootstrap']['seed']}). No clinical or "
         "diagnostic claim.", "",
         "## Primary endpoint", "", f"**Macro F1 = {pr['macro_f1']:.4f}** {ci('primary.macro_f1')}", "",
         "## Secondary", "",
         f"Balanced accuracy {se['balanced_accuracy']:.4f} {ci('primary.balanced_accuracy')}; accuracy "
         f"{se['accuracy']:.4f} {ci('primary.accuracy')}.", "",
         "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    L += [f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
          for c, v in se["per_class"].items()]
    L += ["", "Confusion matrix (rows = true, columns = predicted; " + " / ".join(CLASS_NAMES) + "):", ""]
    L += ["    " + "  ".join(f"{x:5d}" for x in row) for row in se["confusion_matrix"]]
    L += ["", "## Quality sensitivity groups", "",
          "| Group | Cycles / patients | Macro F1 (4 classes) | Macro F1 (classes present) | Bal. acc. | Accuracy |",
          "|---|---|---|---|---|---|"]
    for g, v in m["quality_groups"].items():
        if not v["n_cycles"]:
            L.append(f"| {g} | 0 / 0 | — | — | — | — |")
            continue
        L.append(f"| {g} | {v['n_cycles']} / {v['n_patients']} | {v['macro_f1']:.4f} ({v['macro_f1_ci95'][0]:.3f}, "
                 f"{v['macro_f1_ci95'][1]:.3f}) | {v['macro_f1_present_classes']:.4f} | {v['balanced_accuracy']:.4f} | "
                 f"{v['accuracy']:.4f} |")
    L += ["", "## Calibration (frozen validation temperature, never fitted on test)", "",
          f"T = {cal['frozen_temperature']!r}. Temperature scaling cannot change any predicted class.", "",
          "![Reliability diagrams](reliability_diagram.svg)", "",
          "| Metric | Raw | Frozen T |", "|---|---|---|"]
    for k, lab in (("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE (15 equal-width)"),
                   ("overconfidence_gap", "Overconfidence gap"), ("ece_equal_mass", "ECE equal-mass (secondary)"),
                   ("classwise_ece", "Classwise ECE (secondary)"), ("mce", "MCE (secondary)")):
        L.append(f"| {lab} | {f(cal['raw'][k])} {ci(f'calibration.raw.{k}')} | {f(cal['frozen_t'][k])} "
                 f"{ci(f'calibration.frozen_t.{k}')} |")
    ed = m["error_detection"]
    L += ["", "## Error detection", "", f"Test error rate {ed['raw']['error_rate']:.3f} (AUPRC chance level).", "",
          "| Score | AUROC raw | AUPRC raw | AUROC frozen T | AUPRC frozen T |", "|---|---|---|---|---|"]
    for sc in ("1-msp", "entropy", "1-margin"):
        L.append(f"| {crep.SCORE_LABEL[sc]} | {f(ed['raw'][sc]['auroc'], 3)} {ci(f'error_detection.raw.{sc}.auroc')} | "
                 f"{f(ed['raw'][sc]['auprc'], 3)} {ci(f'error_detection.raw.{sc}.auprc')} | "
                 f"{f(ed['frozen_t'][sc]['auroc'], 3)} | {f(ed['frozen_t'][sc]['auprc'], 3)} |")
    sp = m["selective_prediction"]["rankings"]
    L += ["", "## Selective prediction (primary ranking: raw MSP)", "",
          f"AURC {sp['msp_raw']['aurc']:.4f} {ci('selective.msp_raw.aurc')}; E-AURC {sp['msp_raw']['e_aurc']:.4f} "
          f"{ci('selective.msp_raw.e_aurc')}.", "",
          "| Coverage | Retained | Selective risk | Macro F1 | Bal. acc. | Accuracy | Retained true (n/c/w/b) | "
          "Rejected true | Retained pred | Rejected pred |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lv in sp["msp_raw"]["levels"]:
        c = lv["coverage"]
        L.append(f"| {c:.0%} | {lv['n_retained']} | {lv['selective_risk']:.4f} "
                 f"{ci(f'selective.msp_raw.{c}.selective_risk')} | {lv['macro_f1']:.4f} | {lv['balanced_accuracy']:.4f}"
                 f" | {lv['accuracy']:.4f} | {'/'.join(map(str, lv['retained_true_counts']))} | "
                 f"{'/'.join(map(str, lv['rejected_true_counts']))} | {'/'.join(map(str, lv['retained_pred_counts']))} | "
                 f"{'/'.join(map(str, lv['rejected_pred_counts']))} |")
    th = m["frozen_validation_thresholds"]
    L += ["", "Frozen clean-validation thresholds applied unchanged:", "",
          "| Threshold | τ | Test coverage | Selective risk | Confidently wrong (share of all) |", "|---|---|---|---|---|"]
    for name, t in th["raw"].items():
        L.append(f"| {name} | {t['tau']!r} | {t['realized_coverage']:.1%} | {f(t['selective_risk'])} | "
                 f"{t['confidently_wrong_share']:.1%} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- Stage C + D

LOCKED_CLEAN_FILES = ("predictions.csv", "metrics.json", "summary.md", "reliability_diagram.svg",
                      "risk_coverage_curves.csv")


def start_stage(d: Path, what: str, provenance: dict) -> None:
    if d.exists():
        raise ResultsLockedError(f"{d} exists: the {what} stage has already been started and is never re-run or "
                                 "overwritten. Document any problem instead of re-running.")
    d.mkdir(parents=True)
    _write_json(d / "STARTED.json", {"stage": what, "started": now(), "provenance": provenance})


def write_lock(d: Path, files: tuple[str, ...], provenance: dict, stage: str) -> dict:
    lock = {"stage": stage, "locked": now(), "protocol_sha256": provenance["protocol_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"], "git_head": provenance["git_head"],
            "files": {name: _sha(d / name) for name in files}}
    _write_json(d / "lock.json", lock)
    return lock


def verify_lock(d: Path, protocol_path: Path) -> dict:
    lock_path = Path(d) / "lock.json"
    if not lock_path.exists():
        raise ResultsLockedError(f"no lock at {lock_path}: the clean test stage has not completed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bad = [n for n, h in lock["files"].items() if not (Path(d) / n).exists() or _sha(Path(d) / n) != h]
    if bad:
        raise ResultsLockedError(f"locked outputs changed: {bad}")
    if _sha(protocol_path) != lock["protocol_sha256"]:
        raise ResultsLockedError("the protocol changed after the clean results were locked")
    return lock


def clean_test(ctx: Context, token: str | None) -> dict:
    """Stages A + C + D: the ONE-TIME clean test evaluation. Never run this except as the final evaluation."""
    check_token(token)
    if ctx.require_clean_git:
        assert_clean_git(ctx.root)
    v = verify(ctx)
    fv, prov = v["frozen"], v["provenance"]
    device = device_for(ctx)
    d = Path(ctx.out_dir) / "clean_test"
    start_stage(d, "clean-test", prov)
    ds = make_dataset(ctx, "test", allow_test=True)  # the only place the clean test split is loaded
    pipe = v["protocol"]["2_frozen_data_pipeline"]
    if len(ds.patients) != pipe["patient_split"]["patients"]["test"] or \
            (pipe["metadata"].get("test_rows") is not None and len(ds) != pipe["metadata"]["test_rows"]):
        raise FrozenArtifactError("test patients/cycles differ from the counts frozen in the protocol")
    model, _ = load_model(ctx, device)
    rows = infer(model, ds, device, allow_test=True)
    a = arrays(rows)
    metrics, extra = clean_metrics(a, fv)
    p_t = extra["p_t"]
    for r, pt in zip(rows, p_t):
        r.update({f"prob_frozen_t_{c}": float(pt[k]) for k, c in enumerate(CLASS_NAMES)})
    metrics["provenance"] = prov
    metrics["test_split_evaluated"] = True
    write_rows(d / "predictions.csv", rows)
    _write_json(d / "metrics.json", metrics)
    (d / "summary.md").write_text(clean_summary_md(metrics, prov), encoding="utf-8")
    cal = metrics["calibration"]
    (d / "reliability_diagram.svg").write_text(crep.reliability_svg(
        None, panels=[("Raw softmax", cal["raw"]), (f"Frozen validation T = {fv['temperature']:.4f}", cal["frozen_t"])],
        title="Reliability diagrams, clean TEST (15 equal-width bins)"), encoding="utf-8")
    c = extra["curves"]
    cols = ["n", "coverage", "risk_msp_raw", "risk_entropy_raw", "risk_msp_frozen_t", "oracle_risk"]
    with open(d / "risk_coverage_curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for i in range(len(c["n"])):
            w.writerow([c[k][i].item() for k in cols])
    write_lock(d, LOCKED_CLEAN_FILES, prov, "clean-test")
    return metrics


# ---------------------------------------------------------------- Stage E

def check_reproduction(locked_rows: list[dict], rows: list[dict]) -> dict:
    cmp = compare_predictions(locked_rows, rows, "robustness clean condition vs locked clean test predictions")
    if not cmp["ok"]:
        raise ReproductionError(f"the robustness clean condition does not reproduce the locked clean test: {cmp}")
    return cmp


def robustness_test(ctx: Context, token: str | None) -> dict:
    """Stage E: the 15 frozen degradations on test, only after the clean lock verifies."""
    check_token(token)
    if ctx.require_clean_git:
        assert_clean_git(ctx.root)
    v = verify(ctx)
    fv, prov = v["frozen"], v["provenance"]
    clean_dir = Path(ctx.out_dir) / "clean_test"
    lock = verify_lock(clean_dir, ctx.protocol_path)
    device = device_for(ctx)
    d = Path(ctx.out_dir) / "robustness_test"
    start_stage(d, "robustness-test", {**prov, "clean_lock": lock})
    locked_rows = read_rows(clean_dir / "predictions.csv")
    model, _ = load_model(ctx, device)
    results, reproduction = {}, None
    for c in fv["conditions"]:
        rows = infer(model, make_dataset(ctx, "test", allow_test=True, condition=c), device, allow_test=True)
        if c.family == "clean":
            reproduction = check_reproduction(locked_rows, rows)
        cdir = d / "conditions" / c.name
        cdir.mkdir(parents=True)
        write_rows(cdir / "predictions.csv", rows)
        results[c.name] = rows
    summary = robustness_summary(results, fv, reproduction, prov, lock)
    _write_json(d / "summary.json", summary)
    (d / "summary.md").write_text(robustness_summary_md(summary), encoding="utf-8")
    files = ("summary.json", "summary.md", *(f"conditions/{n}/predictions.csv" for n in results))
    write_lock(d, files, prov, "robustness-test")
    return summary


def robustness_summary(results: dict[str, list[dict]], fv: dict, reproduction: dict, prov: dict, lock: dict) -> dict:
    clean = arrays(results["clean"])
    y, T, taus = clean["labels"], fv["temperature"], fv["taus"]
    preds_by = {n: {(r["recording_filename"], int(r["cycle_index"])): r for r in rows} for n, rows in results.items()}
    boot = cr.patient_bootstrap(preds_by, "clean", n_boot=fv["n_boot"], seed=fv["seed"])
    draws = cmx.patient_bootstrap_indices(clean["patient"], fv["n_boot"], fv["seed"])
    clean_f1 = report(y, clean["preds"])["macro_f1"]
    per = {}
    for c in fv["conditions"]:
        a = arrays(results[c.name])
        if a["keys"] != clean["keys"] or not np.array_equal(a["labels"], y):
            raise ReproductionError(f"{c.name}: cycles, order or labels differ from the clean condition")
        rep = report(y, a["preds"])
        p_raw, p_t = cmx.softmax(a["logits"]), cmx.apply_temperature(a["logits"], T)
        gaps = {w: [cu.calibration_block(p[idx], y[idx], bins=False)["overconfidence_gap"] for idx in draws]
                for w, p in (("raw", p_raw), ("frozen_t", p_t))}
        per[c.name] = {
            **c.as_dict(), "n": rep["n"], "macro_f1": rep["macro_f1"], "balanced_accuracy": rep["balanced_accuracy"],
            "accuracy": rep["accuracy"], "per_class_f1": {k: v["f1"] for k, v in rep["per_class"].items()},
            "confusion_matrix": rep["confusion_matrix"], "macro_f1_change_abs": rep["macro_f1"] - clean_f1,
            "macro_f1_change_rel": (rep["macro_f1"] - clean_f1) / clean_f1 if clean_f1 else None,
            "macro_f1_ci95": boot["runs"][c.name]["macro_f1_ci95"],
            "macro_f1_change_ci95": boot["runs"][c.name]["diff_vs_clean_ci95"],
            "predictions_changed_vs_clean": float((a["preds"] != clean["preds"]).mean()),
            "uncertainty": {w: {**{k: v for k, v in cu.calibration_block(p, y, bins=False).items()},
                                "error_detection": cu.error_detection(p, y),
                                "overconfidence_gap_ci95": cmx.percentile_ci(gaps[w])}
                            for w, p in (("raw", p_raw), ("frozen_t", p_t))},
            "frozen_validation_thresholds": cu.threshold_block(p_raw, y, taus),
            "degradation_stats": nr.degradation_summary(results[c.name]),
        }
    return {"stage": "E fixed robustness on test", "split": "test", "clean_reproduction": reproduction,
            "frozen_temperature": T, "frozen_thresholds": taus, "conditions": per,
            "bootstrap": {"unit": "patient", "n_resamples": fv["n_boot"], "seed": fv["seed"]},
            "provenance": prov, "clean_lock": lock}


def robustness_summary_md(s: dict) -> str:
    L = ["# Fixed robustness evaluation on test (secondary, predeclared)", "",
         "Run only after the clean test results were locked; the clean condition reproduced the locked predictions "
         f"(max |Δp| = {s['clean_reproduction']['max_abs_probability_difference']:.3g}). Frozen T = "
         f"{s['frozen_temperature']!r}; frozen thresholds {s['frozen_thresholds']}. Intervals: patient-level "
         f"bootstrap ({s['bootstrap']['n_resamples']} resamples, seed {s['bootstrap']['seed']}).", "",
         "| Condition | Macro F1 (95% CI) | Δ vs clean (95% CI) | Bal. acc. | Acc. | Preds changed | Mean MSP raw | "
         "Gap raw | Gap frozen T | AUROC 1 − MSP | Coverage at τ80 | Confidently wrong (τ50) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, c in s["conditions"].items():
        u, t = c["uncertainty"], c["frozen_validation_thresholds"]
        L.append(f"| {crep.COND_LABEL.get(n, n)} | {c['macro_f1']:.4f} ({c['macro_f1_ci95'][0]:.3f}, "
                 f"{c['macro_f1_ci95'][1]:.3f}) | {c['macro_f1_change_abs']:+.4f} ({c['macro_f1_change_ci95'][0]:+.3f}, "
                 f"{c['macro_f1_change_ci95'][1]:+.3f}) | {c['balanced_accuracy']:.4f} | {c['accuracy']:.4f} | "
                 f"{c['predictions_changed_vs_clean']:.1%} | {u['raw']['mean_confidence']:.3f} | "
                 f"{u['raw']['overconfidence_gap']:+.3f} | {u['frozen_t']['overconfidence_gap']:+.3f} | "
                 f"{crep.f(u['raw']['error_detection']['1-msp']['auroc'], 3)} | {t['tau_80']['realized_coverage']:.1%} | "
                 f"{t['tau_50']['confidently_wrong_share']:.1%} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-time final evaluation of the frozen baseline_cnn_v3 (see protocol).")
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("verify", help="Stage A only")
    sub.add_parser("dry-run-validation", help="Stage B: reproduce the frozen validation inference (no test data)")
    for name, hlp in (("clean-test", "ONE-TIME clean test evaluation (Stages A, C, D)"),
                      ("robustness-test", "fixed degradations on test after the clean lock (Stage E)")):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--unlock-test", required=True, metavar="TOKEN", help="exact one-time unlock token")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    ctx = Context()
    try:
        if a.stage == "verify":
            v = verify(ctx)
            print(f"Stage A OK: protocol v{v['provenance']['protocol_version']} verified; frozen T "
                  f"{v['frozen']['temperature']!r}, thresholds {v['frozen']['taus']}")
        elif a.stage == "dry-run-validation":
            r = dry_run_validation(ctx)
            print(f"Stage B OK: validation reproduced; max |dp| = {r['predictions']['max_abs_probability_difference']:.3g}"
                  f" ({r['predictions']['n_probability_values_bitwise_identical']}/{r['predictions']['n_probability_values']}"
                  " bitwise identical)")
        elif a.stage == "clean-test":
            m = clean_test(ctx, a.unlock_test)
            print(f"clean test done and locked: macro F1 {m['primary']['macro_f1']:.4f} {m['primary']['macro_f1_ci95']}")
        elif a.stage == "robustness-test":
            robustness_test(ctx, a.unlock_test)
            print("robustness test done and locked")
    except (FrozenArtifactError, TestAccessError, ResultsLockedError, ReproductionError) as e:
        print(f"REFUSED/FAILED ({type(e).__name__}): {e}".encode("ascii", "replace").decode(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
