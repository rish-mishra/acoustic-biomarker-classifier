"""Validation-only robustness of the frozen baseline CNN to synthetic audio degradations.

The frozen reference model (reports/runs/baseline_cnn_v3, models/baseline_cnn_v3/best.pt)
is evaluated, without retraining, on the VALIDATION split under every
predeclared condition in degradations.PREDECLARED_CONDITIONS. Each degradation
is applied to the 16 kHz cycle waveform between the frozen preprocessing
(audio_preprocessing) and the frozen Log-Mel extractor; the frozen train-band
normalization statistics are then applied as usual. The test split is never
loaded: DegradedCycleDataset and analyze_run.predict both refuse it.

Outputs (reports/robustness/noise_robustness/):
  plan.json                         predeclared conditions, definitions and report rules
                                    (written once with --write-plan, before any result)
  conditions/<name>/predictions.csv per-cycle labels, predictions, probabilities, degradation stats
  conditions/<name>/metrics.json    metrics for that condition
  clean_check.json                  clean condition vs the recorded baseline_cnn_v3 validation results
  summary.json / summary.md         all conditions, changes vs clean, patient-bootstrap intervals,
                                    subgroups, and the predeclared descriptive rules
  macro_f1_by_condition.svg         F1 by degradation severity
  _runner/status.json, runner.log   progress and events (runner.lock is a runtime file)

Conditions are evaluated sequentially in one process. A completed condition is
never re-evaluated or overwritten; rerunning resumes with the missing ones.

Usage::

    python src/evaluation/noise_robustness.py --write-plan
    python src/evaluation/noise_robustness.py              # evaluate + summarize
    python src/evaluation/noise_robustness.py --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
for _sub in ("models", "data", "features", "evaluation"):
    if str(SRC_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(SRC_DIR / _sub))

import analyze_run as ar  # noqa: E402
import compare_runs as cr  # noqa: E402
import degradations as dg  # noqa: E402
import train as tr  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402

rd, nz, ap = tr.rd, tr.nz, tr.rd.ap
CLASS_NAMES = rd.CLASS_NAMES
REPO_ROOT = SRC_DIR.parent
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "robustness" / "noise_robustness"
MODEL_RUN = "baseline_cnn_v3"
SPLIT = "val"
SEED_STUDY_SUMMARY = REPO_ROOT / "reports" / "runs" / "seed_study" / "summary.json"

# Severity order (mild -> severe) used by the descriptive rules. Amplitude is two directions.
SEVERITY_ORDER = {
    "gaussian_noise": ["noise_snr20db", "noise_snr10db", "noise_snr5db", "noise_snr0db"],
    "amplitude_attenuation": ["gain_0.5x", "gain_0.25x"],
    "amplitude_amplification": ["gain_1.5x", "gain_2.0x"],
    "lowpass": ["lowpass_2000hz", "lowpass_1000hz", "lowpass_500hz"],
    "temporal_dropout": ["dropout_10pct", "dropout_25pct", "dropout_50pct"],
}
COLLAPSE_MIN_STEP = 0.05  # macro F1
CLASS_FAIL_DROP = 0.05  # per-class F1
PREDICTION_COLLAPSE_SHARE = 0.80

RULES = {
    "reference_scale": "pooled within-architecture seed SD of validation macro F1 from the seed study "
                       "(reports/runs/seed_study/summary.json); changes smaller than this are of the same "
                       "size as retraining with another seed",
    "negligible": "a family is 'negligible' if |macro-F1 change vs clean| < reference_scale at every level",
    "collapse": f"otherwise, walking clean -> mild -> severe, a family shows a 'collapse threshold' at the level "
                f"reached by its largest single-step macro-F1 drop if that step is >= {COLLAPSE_MIN_STEP} and >= "
                f"twice the sum of all its other positive step drops; otherwise 'gradual'",
    "hurts_most": "lowest macro F1 among degraded conditions; per family, the change at the most severe level",
    "tolerated_best": "family with the smallest mean |macro-F1 change| over its levels",
    "class_fails_first": f"per family, the first level (mild -> severe) at which any class's F1 drops by >= "
                         f"{CLASS_FAIL_DROP} vs clean; the class with the largest drop there",
    "prediction_collapse": f"a condition 'collapses predictions' if >= {PREDICTION_COLLAPSE_SHARE:.0%} of cycles "
                           f"are predicted as one class",
    "uncertainty": "patient-level bootstrap (compare_runs.patient_bootstrap, 2000 resamples, seed 0) of the paired "
                   "macro-F1 difference vs clean; supporting only, not used by the rules",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- dataset / evaluation

class DegradedCycleDataset(rd.RespiratoryCycleDataset):
    """Validation cycles with one degradation applied before Log-Mel extraction.

    Identical to RespiratoryCycleDataset except that each cycle's 16 kHz
    waveform passes through degradations.apply. Per-example degradation stats
    are recorded in ``degradation_stats`` keyed by (recording, cycle_index).
    """

    def __init__(self, split: str, condition: dg.Condition, base_seed: int = dg.DEFAULT_BASE_SEED, **kw):
        if split != SPLIT:
            raise PermissionError(f"robustness evaluation is validation-only; got split {split!r}")
        super().__init__(split, **kw)
        self.condition = condition
        self.base_seed = base_seed
        self.degradation_stats: dict[tuple[str, int], dict] = {}

    def load_features(self, index: int):
        r = self.rows[index]
        waveform, sample_rate = self.cache.get(r["recording_filename"])
        if sample_rate != int(r["sample_rate"]):
            raise ap.AudioLoadError(
                f"{r['recording_filename']}: metadata sample_rate {r['sample_rate']} does not match file ({sample_rate})")
        cycle = ap.cycle_from_waveform(waveform, sample_rate, float(r["cycle_start_s"]),
                                       float(r["cycle_end_s"]), r["recording_filename"])
        degraded, stats = dg.apply(self.condition, cycle.waveform, r["recording_filename"], int(r["cycle_index"]),
                                   self.base_seed, cycle.sample_rate)
        if degraded.shape != cycle.waveform.shape:
            raise RuntimeError(f"{self.condition.name} changed the waveform shape")
        self.degradation_stats[(r["recording_filename"], int(r["cycle_index"]))] = stats
        return self.extractor(dataclasses.replace(cycle, waveform=degraded))


def make_val_dataset(condition: dg.Condition, base_seed: int = dg.DEFAULT_BASE_SEED) -> DegradedCycleDataset:
    return DegradedCycleDataset(SPLIT, condition, base_seed, normalizer=nz.load_normalizer())


def evaluate_condition(model, dataset: DegradedCycleDataset, device) -> tuple[list[dict], dict]:
    """Per-cycle predictions (with degradation stats) and the overall report."""
    rows = ar.predict(model, dataset, device)
    for row in rows:
        row.update(dataset.degradation_stats[(row["recording_filename"], row["cycle_index"])])
    report = classification_report(confusion_matrix(torch.tensor([r["label"] for r in rows]),
                                                    torch.tensor([r["pred"] for r in rows]), len(CLASS_NAMES)),
                                   CLASS_NAMES)
    return rows, report


def _stat_summary(rows: list[dict], key: str) -> dict | None:
    vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None) and math.isfinite(float(r[key]))]
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def degradation_summary(rows: list[dict]) -> dict:
    out = {k: s for k in ("realized_snr_db", "dropped_fraction", "peak_abs_after")
           if (s := _stat_summary(rows, k)) is not None}
    if any("silent_input" in r for r in rows):
        out["silent_inputs"] = sum(str(r["silent_input"]) == "True" for r in rows)
    if "peak_abs_after" in out:
        out["cycles_exceeding_full_scale"] = sum(float(r["peak_abs_after"]) > 1.0 for r in rows)
    return out


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def condition_done(out_dir: Path, name: str) -> bool:
    m = Path(out_dir) / "conditions" / name / "metrics.json"
    return m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "completed" \
        and (m.parent / "predictions.csv").exists()


def check_clean(rows: list[dict], ckpt: dict, reference_csv: Path | None) -> dict:
    """The clean condition must reproduce the checkpoint's logged validation results exactly."""
    cm = confusion_matrix(torch.tensor([r["label"] for r in rows]), torch.tensor([r["pred"] for r in rows]),
                          len(CLASS_NAMES)).tolist()
    res = {"confusion_matrix_matches_checkpoint": cm == ckpt["val_metrics"]["confusion_matrix"]}
    if reference_csv is not None:
        with open(reference_csv, newline="", encoding="utf-8") as f:
            ref = list(csv.DictReader(f))
        same_order = [(r["recording_filename"], int(r["cycle_index"])) for r in ref] == \
                     [(r["recording_filename"], r["cycle_index"]) for r in rows]
        res.update(
            reference_predictions=str(reference_csv.relative_to(REPO_ROOT)) if reference_csv.is_relative_to(REPO_ROOT)
            else str(reference_csv),
            same_cycles_and_order=same_order,
            labels_identical=same_order and all(int(a["label"]) == b["label"] for a, b in zip(ref, rows)),
            predictions_identical=same_order and all(int(a["pred"]) == b["pred"] for a, b in zip(ref, rows)),
            max_abs_probability_difference=max(abs(float(a[f"prob_{c}"]) - b[f"prob_{c}"])
                                               for a, b in zip(ref, rows) for c in CLASS_NAMES) if same_order else None,
        )
    res["ok"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


# ---------------------------------------------------------------- provenance

def frozen_files(model_run: str = MODEL_RUN) -> list[Path]:
    run_dir = tr.DEFAULT_REPORTS_DIR / model_run
    return [tr.DEFAULT_CHECKPOINT_DIR / model_run / "best.pt", *sorted(p for p in run_dir.iterdir() if p.is_file()),
            rd.DEFAULT_METADATA, rd.DEFAULT_SPLIT_FILE, nz.DEFAULT_STATS_PATH]


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(Path(p).relative_to(REPO_ROOT)).replace("\\", "/"): tr.sha256_file(p) for p in paths}


# ---------------------------------------------------------------- plan

def plan_payload() -> dict:
    return {
        "study": "Validation-only robustness of the frozen baseline CNN to synthetic audio degradations",
        "declared": now(),
        "git_commit": tr._run_cmd(["git", "rev-parse", "HEAD"]),
        "model": {"run": MODEL_RUN, "checkpoint": f"models/{MODEL_RUN}/best.pt",
                  "why": "seed-study recommendation: baseline_cnn (no convincing winner; simplest within one pooled "
                         "seed SD); baseline_cnn_v3 is its frozen seed-42 configuration"},
        "split": SPLIT,
        "conditions": [c.as_dict() for c in dg.PREDECLARED_CONDITIONS],
        "definitions": {
            "applied_to": "16 kHz cycle waveform after frozen trimming/resampling, before frozen Log-Mel "
                          "(PRIMARY_CONFIG, 50-2000 Hz) and frozen train-band normalization",
            "gaussian_noise": "white Gaussian, full 0-8 kHz band; SNR from DC-free mean-square powers; noise "
                              "mean-removed and rescaled so the realized SNR is exact",
            "amplitude": "x * gain, no clipping",
            "lowpass": f"Blackman-windowed sinc FIR, {dg.LOWPASS_NUM_TAPS} taps, unit DC gain, centered (zero "
                       f"phase), cutoff = -6 dB point",
            "temporal_dropout": f"{dg.DROPOUT_BLOCK_S * 1000:.0f} ms blocks; round-half-up(fraction x n_blocks) "
                                f"random blocks set to 0",
            "randomness": f"torch.Generator per cycle, seed = sha256(base_seed|family|recording|cycle_index), "
                          f"base_seed {dg.DEFAULT_BASE_SEED}; shared within a family",
        },
        "severity_order": SEVERITY_ORDER,
        "report_rules": RULES,
        "rules": ["no retraining", "test split never loaded or evaluated",
                  "frozen preprocessing, normalization statistics, split, labels and checkpoint unchanged",
                  "no conditions added, removed or tuned based on results"],
    }


def plan_conditions(plan: dict) -> list[dg.Condition]:
    conds = [dg.Condition(**c) for c in plan["conditions"]]
    if conds != list(dg.PREDECLARED_CONDITIONS):
        raise ValueError("plan.json conditions differ from degradations.PREDECLARED_CONDITIONS")
    return conds


# ---------------------------------------------------------------- summary

def load_condition(out_dir: Path, name: str) -> tuple[dict, list[dict]]:
    d = Path(out_dir) / "conditions" / name
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    with open(d / "predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return metrics, rows


def subgroup_metrics(rows: list[dict]) -> dict:
    as_int = [{**r, "label": int(r["label"]), "pred": int(r["pred"])} for r in rows]
    out = {}
    for key in ("original_sample_rate", "device"):
        groups: dict[str, list] = {}
        for r in as_int:
            groups.setdefault(str(r[key]), []).append(r)
        out[key] = {g: {k: v for k, v in ar.group_metrics(m).items()
                        if k in ("n_cycles", "n_patients", "accuracy", "macro_f1_present_classes")}
                    for g, m in sorted(groups.items())}
    return out


def family_shape(changes: list[float], ref_scale: float) -> dict:
    """Descriptive rule (RULES['negligible'] / RULES['collapse']) on macro-F1 changes, mild -> severe."""
    levels = [0.0, *changes]
    steps = [levels[i] - levels[i + 1] for i in range(len(changes))]  # positive = drop
    if all(abs(c) < ref_scale for c in changes):
        return {"shape": "negligible", "step_drops": steps}
    i = max(range(len(steps)), key=lambda j: steps[j])
    others = sum(s for j, s in enumerate(steps) if j != i and s > 0)
    collapse = steps[i] >= COLLAPSE_MIN_STEP and steps[i] >= 2 * others
    return {"shape": "collapse" if collapse else "gradual", "step_drops": steps,
            "largest_step_index": i, "collapse_level_index": i if collapse else None}


def summarize(out_dir: Path, conditions: list[dg.Condition], ref_scale: float, n_boot: int = 2000) -> dict:
    out_dir = Path(out_dir)
    loaded = {c.name: load_condition(out_dir, c.name) for c in conditions}
    clean_m, clean_rows = loaded["clean"]
    clean_f1 = clean_m["overall"]["macro_f1"]
    clean_pred = {(r["recording_filename"], r["cycle_index"]): r["pred"] for r in clean_rows}
    keys = list(clean_pred)
    boot = cr.patient_bootstrap(
        {n: {(r["recording_filename"], int(r["cycle_index"])): r for r in rows} for n, (_, rows) in loaded.items()},
        "clean", n_boot=n_boot)

    per = {}
    for c in conditions:
        m, rows = loaded[c.name]
        ov = m["overall"]
        if [(r["recording_filename"], r["cycle_index"]) for r in rows] != keys or \
                [r["label"] for r in rows] != [r["label"] for r in clean_rows]:
            raise ValueError(f"{c.name}: cycles, order or labels differ from clean")
        pred_counts = {cls: sum(int(r["pred"]) == i for r in rows) for i, cls in enumerate(CLASS_NAMES)}
        top_cls, top_n = max(pred_counts.items(), key=lambda kv: kv[1])
        b = boot["runs"][c.name]
        per[c.name] = {
            **c.as_dict(),
            "n": ov["n"],
            "macro_f1": ov["macro_f1"], "balanced_accuracy": ov["balanced_accuracy"], "accuracy": ov["accuracy"],
            "per_class_f1": {k: v["f1"] for k, v in ov["per_class"].items()},
            "per_class_recall": {k: v["recall"] for k, v in ov["per_class"].items()},
            "confusion_matrix": ov["confusion_matrix"],
            "macro_f1_change_abs": ov["macro_f1"] - clean_f1,
            "macro_f1_change_rel": (ov["macro_f1"] - clean_f1) / clean_f1,
            "balanced_accuracy_change_abs": ov["balanced_accuracy"] - clean_m["overall"]["balanced_accuracy"],
            "accuracy_change_abs": ov["accuracy"] - clean_m["overall"]["accuracy"],
            "per_class_f1_change_abs": {k: ov["per_class"][k]["f1"] - clean_m["overall"]["per_class"][k]["f1"]
                                        for k in CLASS_NAMES},
            "prediction_counts": pred_counts,
            "predictions_changed_vs_clean": sum(r["pred"] != clean_pred[(r["recording_filename"], r["cycle_index"])]
                                                for r in rows) / len(rows),
            "prediction_collapse": {"class": top_cls, "share": top_n / len(rows),
                                    "collapsed": top_n / len(rows) >= PREDICTION_COLLAPSE_SHARE},
            "bootstrap_macro_f1_ci95": b["macro_f1_ci95"],
            "bootstrap_change_vs_clean_ci95": b["diff_vs_clean_ci95"],
            "degradation_stats": m.get("degradation_stats", {}),
            "subgroups": subgroup_metrics(rows),
        }

    families = {}
    for fam, names in SEVERITY_ORDER.items():
        ch = [per[n]["macro_f1_change_abs"] for n in names]
        shape = family_shape(ch, ref_scale)
        if shape.get("collapse_level_index") is not None:
            shape["collapse_level"] = names[shape["collapse_level_index"]]
        first = None
        for n in names:
            drops = {k: -v for k, v in per[n]["per_class_f1_change_abs"].items()}
            worst = max(drops, key=drops.get)
            if drops[worst] >= CLASS_FAIL_DROP:
                first = {"level": n, "class": worst, "f1_drop": drops[worst],
                         "classes_dropping": sorted((k for k, v in drops.items() if v >= CLASS_FAIL_DROP),
                                                    key=lambda k: -drops[k])}
                break
        families[fam] = {"levels": names, "macro_f1": [per[n]["macro_f1"] for n in names], "changes": ch,
                         "mean_abs_change": statistics.fmean(abs(x) for x in ch),
                         "most_severe_change": ch[-1], **shape, "class_fails_first": first}

    degraded = [c.name for c in conditions if c.family != "clean"]
    worst = min(degraded, key=lambda n: per[n]["macro_f1"])
    return {
        "study": "validation-only robustness of the frozen baseline CNN",
        "model_run": MODEL_RUN,
        "split": SPLIT,
        "test_split_evaluated": False,
        "n_cycles": clean_m["overall"]["n"],
        "n_patients": boot["n_patients"],
        "clean_macro_f1": clean_f1,
        "reference_scale_pooled_seed_sd": ref_scale,
        "rules": RULES,
        "conditions": per,
        "families": families,
        "findings": {
            "hurts_most_condition": worst,
            "hurts_most_family_at_most_severe": min(families, key=lambda f: families[f]["most_severe_change"]),
            "tolerated_best_family": min(families, key=lambda f: families[f]["mean_abs_change"]),
            "conditions_within_reference_scale": [n for n in degraded if abs(per[n]["macro_f1_change_abs"]) < ref_scale],
            "prediction_collapses": [n for n in degraded if per[n]["prediction_collapse"]["collapsed"]],
        },
        "bootstrap": {k: boot[k] for k in ("n_bootstrap", "seed", "unit", "n_patients")},
    }


# ---------------------------------------------------------------- report

LABELS = {"gaussian_noise": "Gaussian noise", "amplitude_attenuation": "Amplitude attenuation",
          "amplitude_amplification": "Amplitude amplification", "lowpass": "Low-pass filtering",
          "temporal_dropout": "Temporal dropout"}


def _cond_label(c: dict) -> str:
    if c["family"] == "clean":
        return "Clean"
    if c["family"] == "gaussian_noise":
        return f"Noise {c['param']:g} dB SNR"
    if c["family"] == "amplitude":
        return f"Gain {c['param']:g}x ({20 * math.log10(c['param']):+.1f} dB)"
    if c["family"] == "lowpass":
        return f"Low-pass {c['param']:g} Hz"
    return f"Dropout {c['param']:.0%}"


def render_markdown(s: dict) -> str:
    per, fam, fnd, ref = s["conditions"], s["families"], s["findings"], s["reference_scale_pooled_seed_sd"]
    pct = lambda x: f"{x:+.1%}"  # noqa: E731
    L = ["# Robustness of the baseline CNN to synthetic audio degradations (validation only)", "",
         f"Frozen model `{s['model_run']}` (no retraining), validation split only ({s['n_cycles']} cycles, "
         f"{s['n_patients']} patients). The test split was not loaded or evaluated. Each degradation is applied to "
         "the 16 kHz cycle waveform before the frozen Log-Mel features (50-2000 Hz) and frozen normalization. "
         "Definitions and the descriptive rules below were fixed in `plan.json` before any result was computed.", "",
         "These are synthetic degradations on one validation split; they describe model behaviour, "
         "not clinical or diagnostic performance.", "",
         f"Reference scale: the seed study's pooled within-architecture seed SD of macro F1 is **{ref:.4f}**; "
         "changes smaller than this are about as large as retraining with a different seed. Intervals are "
         f"patient-level bootstrap 95% intervals ({s['bootstrap']['n_bootstrap']} resamples of "
         f"{s['n_patients']} patients) for the paired change vs clean; with 19 patients they are wide.", "",
         "![Macro and per-class F1 by condition](macro_f1_by_condition.svg)", "",
         "## Results by condition", "",
         "| Condition | n | Macro F1 | Δ abs (95% CI) | Δ rel | Bal. acc. | Acc. | " +
         " | ".join(f"F1 {c}" for c in CLASS_NAMES) + " | Preds changed |",
         "|---|---|---|---|---|---|---|" + "---|" * len(CLASS_NAMES) + "---|"]
    for n, c in per.items():
        lo, hi = c["bootstrap_change_vs_clean_ci95"]
        clean = c["family"] == "clean"
        change = "—" if clean else f"{c['macro_f1_change_abs']:+.4f} ({lo:+.3f}, {hi:+.3f})"
        rel = "—" if clean else pct(c["macro_f1_change_rel"])
        flips = "—" if clean else f"{c['predictions_changed_vs_clean']:.1%}"
        L.append(f"| {_cond_label(c)} | {c['n']} | {c['macro_f1']:.4f} | {change} | {rel} | "
                 f"{c['balanced_accuracy']:.4f} | {c['accuracy']:.4f} | "
                 + " | ".join(f"{c['per_class_f1'][k]:.3f}" for k in CLASS_NAMES) + f" | {flips} |")
    L += ["", "## Per-class F1 change vs clean", "", "| Condition | " + " | ".join(CLASS_NAMES) +
          " | Predicted class distribution (normal / crackle / wheeze / both) |",
          "|---|" + "---|" * len(CLASS_NAMES) + "---|"]
    for n, c in per.items():
        pc = c["prediction_counts"]
        L.append(f"| {_cond_label(c)} | " + " | ".join(f"{c['per_class_f1_change_abs'][k]:+.3f}" for k in CLASS_NAMES)
                 + f" | {' / '.join(str(pc[k]) for k in CLASS_NAMES)} |")

    L += ["", "## Degradation families (predeclared descriptive rules)", "",
          "| Family | Levels (mild → severe) | Macro-F1 change per level | Step drops | Shape | First class to drop ≥ "
          f"{CLASS_FAIL_DROP} F1 |", "|---|---|---|---|---|---|"]
    for f, d in fam.items():
        first = d["class_fails_first"]
        shape = d["shape"] + (f" at {_cond_label(per[d['collapse_level']])}" if d.get("collapse_level") else "")
        L.append(f"| {LABELS[f]} | " + ", ".join(_cond_label(per[n]) for n in d["levels"]) + " | "
                 + ", ".join(f"{x:+.4f}" for x in d["changes"]) + " | "
                 + ", ".join(f"{x:+.4f}" for x in d["step_drops"]) + f" | {shape} | "
                 + (f"{first['class']} ({first['f1_drop']:.3f}) at {_cond_label(per[first['level']])}" if first
                    else "none") + " |")

    worst, best = per[fnd["hurts_most_condition"]], fnd["tolerated_best_family"]
    hf = fnd["hurts_most_family_at_most_severe"]
    L += ["", "## Findings", "",
          f"- **Most harmful condition:** {_cond_label(worst)}, macro F1 {worst['macro_f1']:.4f} "
          f"({worst['macro_f1_change_abs']:+.4f}, {pct(worst['macro_f1_change_rel'])} vs clean). At its most severe "
          f"level, the most harmful family is {LABELS[hf].lower()} ({fam[hf]['most_severe_change']:+.4f}).",
          f"- **Best tolerated family:** {LABELS[best].lower()} (mean |change| {fam[best]['mean_abs_change']:.4f}). "
          "Conditions within the seed-SD reference scale: "
          + (", ".join(_cond_label(per[n]) for n in fnd["conditions_within_reference_scale"]) or "none") + ".",
          "- **Classes that fail first:** "
          + "; ".join(f"{LABELS[f].lower()}: " + (f"{d['class_fails_first']['class']} at "
                                                 f"{_cond_label(per[d['class_fails_first']['level']])}"
                                                 if d["class_fails_first"] else "no class drops by "
                                                 f"≥ {CLASS_FAIL_DROP}") for f, d in fam.items()) + ".",
          "- **Gradual vs collapse:** " + "; ".join(
              f"{LABELS[f].lower()}: {d['shape']}" + (f" (threshold at {_cond_label(per[d['collapse_level']])})"
                                                     if d.get("collapse_level") else "") for f, d in fam.items()) + ".",
          "- **Prediction collapse** (≥ 80% of cycles predicted as one class): "
          + (", ".join(f"{_cond_label(per[n])} → {per[n]['prediction_collapse']['class']} "
                       f"({per[n]['prediction_collapse']['share']:.0%})" for n in fnd["prediction_collapses"])
             or "none") + ".", ""]

    L += _sensitivity_sections(s)
    L += _posthoc_section(s)
    L += ["## Subgroups: macro F1 (classes present) by source sample rate", "",
          "| Condition | " + " | ".join(f"{g} Hz (n={per['clean']['subgroups']['original_sample_rate'][g]['n_cycles']})"
                                       for g in per["clean"]["subgroups"]["original_sample_rate"]) + " |",
          "|---|" + "---|" * len(per["clean"]["subgroups"]["original_sample_rate"])]
    for n, c in per.items():
        L.append(f"| {_cond_label(c)} | " + " | ".join(
            f"{v['macro_f1_present_classes']:.3f}" for v in c["subgroups"]["original_sample_rate"].values()) + " |")
    L += ["", "## Subgroups: accuracy by device", "",
          "| Condition | " + " | ".join(f"{g} (n={v['n_cycles']}, {v['n_patients']} pts)"
                                       for g, v in per["clean"]["subgroups"]["device"].items()) + " |",
          "|---|" + "---|" * len(per["clean"]["subgroups"]["device"])]
    for n, c in per.items():
        L.append(f"| {_cond_label(c)} | " + " | ".join(f"{v['accuracy']:.3f}" for v in c["subgroups"]["device"].values())
                 + " |")
    L += ["", "Subgroups have few patients and are noisy; they are descriptive only.", ""]
    return "\n".join(L)


def _sensitivity_sections(s: dict) -> list[str]:
    per, fam = s["conditions"], s["families"]
    sr = lambda n, g: per[n]["subgroups"]["original_sample_rate"][g]["macro_f1_present_classes"]  # noqa: E731
    rates = list(per["clean"]["subgroups"]["original_sample_rate"])
    L = ["## Sensitivity by degradation type", "",
         "**Bandwidth loss.** The model's features cover 50-2000 Hz only, so a 2000 Hz low-pass removes almost "
         "nothing the model sees (only the filter's ~±45 Hz transition band); 1000 Hz and 500 Hz remove the upper "
         "half and three quarters of the feature band. Macro-F1 change: "
         + ", ".join(f"{_cond_label(per[n])} {per[n]['macro_f1_change_abs']:+.4f}" for n in fam["lowpass"]["levels"])
         + ". By source rate (macro F1, classes present), clean → 500 Hz: "
         + ", ".join(f"{g} Hz {sr('clean', g):.3f} → {sr('lowpass_500hz', g):.3f}" for g in rates) + ".", "",
         "**Amplitude changes.** A gain change shifts every Log-Mel band by the same number of dB (20·log10(gain)) "
         "and leaves spectral shape and timing intact; it carries no new respiratory information. Macro-F1 change: "
         + ", ".join(f"{_cond_label(per[n])} {per[n]['macro_f1_change_abs']:+.4f} "
                     f"({per[n]['predictions_changed_vs_clean']:.1%} of predictions changed)"
                     for n in fam["amplitude_attenuation"]["levels"] + fam["amplitude_amplification"]["levels"])
         + ".", "",
         "**Missing temporal audio.** Dropped 20 ms blocks become exact silence, which Log-Mel maps to floor-level "
         "frames. Mean dropped share of samples and macro-F1 change: "
         + ", ".join(f"{_cond_label(per[n])}: {per[n]['degradation_stats'].get('dropped_fraction', {}).get('mean', float('nan')):.1%}"
                     f" dropped, {per[n]['macro_f1_change_abs']:+.4f}" for n in fam["temporal_dropout"]["levels"])
         + ".", "",
         "**Indicators of reliance on recording characteristics.** These are observations that would be worth "
         "investigating, not conclusions:"]
    amp = fam["amplitude_attenuation"]["levels"] + fam["amplitude_amplification"]["levels"]
    worst_amp = min(amp, key=lambda n: per[n]["macro_f1_change_abs"])
    flips = max(per[n]["predictions_changed_vs_clean"] for n in amp)
    L.append(f"- Pure gain changes alter up to {flips:.1%} of predictions and change macro F1 by up to "
             f"{per[worst_amp]['macro_f1_change_abs']:+.4f} ({_cond_label(per[worst_amp])}). Because gain carries no "
             "respiratory information, sensitivity here indicates dependence on absolute recording level, which "
             "varies with device, microphone placement and gain settings."
             + (" This exceeds the seed-SD reference scale." if abs(per[worst_amp]["macro_f1_change_abs"])
                >= s["reference_scale_pooled_seed_sd"] else " This stays within the seed-SD reference scale."))
    shifts = []
    for n in amp:
        pc, pc0 = per[n]["prediction_counts"], per["clean"]["prediction_counts"]
        k = max(CLASS_NAMES, key=lambda c: abs(pc[c] - pc0[c]))
        shifts.append(f"{_cond_label(per[n])}: {k} {pc0[k]} → {pc[k]}")
    L.append("- Largest shift in predicted-class counts under gain changes: " + "; ".join(shifts) + ".")
    lp2k = per["lowpass_2000hz"]["macro_f1_change_abs"]
    L.append(f"- Low-pass 2000 Hz (removes essentially nothing inside the feature band): {lp2k:+.4f}. "
             + ("A change this small is consistent with the model ignoring content outside its band."
                if abs(lp2k) < s["reference_scale_pooled_seed_sd"] else
                "A change this large from the filter's narrow transition band would be surprising and worth checking."))
    L.append("- Differences between source sample rates under the same degradation (tables below) can point to "
             "device- or bandwidth-specific cues; 4 kHz sources already contain nothing above 2 kHz.")
    L.append("")
    return L


def _posthoc_section(s: dict) -> list[str]:
    """Descriptive facts added after inspecting the results (not predeclared, not used by the rules)."""
    per, fam = s["conditions"], s["families"]
    L = ["## Post-hoc observations (added after inspecting results; not predeclared)", "",
         "- **Monotonicity with severity:** " + "; ".join(
             f"{LABELS[f].lower()}: " + ("monotonic" if all(a >= b for a, b in zip(d["macro_f1"], d["macro_f1"][1:]))
                                         else "NOT monotonic (" + ", ".join(f"{x:.3f}" for x in d["macro_f1"]) + ")")
             for f, d in fam.items()) + ". A non-monotonic family means the more severe level hurts less, which "
         "points to the model reacting to an unfamiliar input pattern rather than to gradual information loss.",
         "- **Direction of the prediction shift** (class whose predicted count changes most vs clean):"]
    pc0 = per["clean"]["prediction_counts"]
    for n, c in per.items():
        if c["family"] == "clean":
            continue
        k = max(CLASS_NAMES, key=lambda x: abs(c["prediction_counts"][x] - pc0[x]))
        L.append(f"  - {_cond_label(c)}: {k} {pc0[k]} → {c['prediction_counts'][k]}")
    two = [f for f, d in fam.items() if len(d["levels"]) == 2 and d["shape"] == "collapse"]
    if two:
        L.append("- **Two-level families:** " + ", ".join(LABELS[f].lower() for f in two) + " met the collapse rule "
                 "with only two severity levels, which is weak evidence of a threshold.")
    L.append("")
    return L


# ---------------------------------------------------------------- figure

def render_svg(s: dict) -> str:
    """Small multiples: macro F1 (bold) and per-class F1 by severity, one panel per family."""
    per, fam = s["conditions"], s["families"]
    # Reference palette categorical slots 1-4 (validated light + dark); macro F1 uses primary ink.
    series = [("macro", "macro F1"), *((f"c{i}", c) for i, c in enumerate(CLASS_NAMES))]
    style = """
  .s{fill:var(--surface)} .t1{fill:var(--ink)} .t2{fill:var(--ink2)} .tm{fill:var(--muted)}
  .grid{stroke:var(--grid);stroke-width:1} .base{stroke:var(--muted);stroke-width:1}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round} .ln.macro{stroke-width:3}
  .mk{stroke:var(--surface);stroke-width:2}
  .macro{stroke:var(--ink)} .mk.macro{fill:var(--ink)}
  .c0{stroke:var(--c0)} .mk.c0{fill:var(--c0)} .c1{stroke:var(--c1)} .mk.c1{fill:var(--c1)}
  .c2{stroke:var(--c2)} .mk.c2{fill:var(--c2)} .c3{stroke:var(--c3)} .mk.c3{fill:var(--c3)}
  svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
      --c0:#2a78d6;--c1:#eb6834;--c2:#1baf7a;--c3:#eda100}
  @media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
      --grid:#2c2c2a;--c0:#3987e5;--c1:#d95926;--c2:#199e70;--c3:#c98500}}
"""
    pw, ph, ml, mt, gap = 210, 200, 46, 46, 22
    width = ml + len(fam) * (pw + gap)
    height = mt + ph + 64
    top = 0.8 if all(max(c["macro_f1"], *c["per_class_f1"].values()) <= 0.8 for c in per.values()) else 1.0
    y = lambda v: mt + ph - v * ph / top  # noqa: E731  (F1 0-top mapped to the panel)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="11" role="img" '
           f'aria-label="Validation F1 by degradation severity, one panel per degradation family">',
           f"<style>{style}</style>", f'<rect class="s" width="{width}" height="{height}"/>',
           f'<text class="t1" x="{ml}" y="16" font-size="13" font-weight="bold">Validation F1 by degradation '
           f'severity (frozen {s["model_run"]})</text>']
    for i, (f, d) in enumerate(fam.items()):
        x0 = ml + i * (pw + gap)
        names = ["clean", *d["levels"]]
        xs = [x0 + 14 + j * (pw - 28) / (len(names) - 1) for j in range(len(names))]
        out.append(f'<text class="t1" x="{x0}" y="{mt - 12}" font-weight="bold">{LABELS[f]}</text>')
        for t in [x / 10 for x in range(2, round(top * 10) + 1, 2)]:
            out.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
            if i == 0:
                out.append(f'<text class="tm" x="{x0 - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.1f}</text>')
        out.append(f'<line class="base" x1="{x0}" x2="{x0 + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
        for key, name in reversed(series):  # macro F1 drawn last, on top
            vals = [per[n]["macro_f1"] if key == "macro" else per[n]["per_class_f1"][name] for n in names]
            pts = " ".join(f"{xx:.1f},{y(v):.1f}" for xx, v in zip(xs, vals))
            out.append(f'<polyline class="ln {key}" points="{pts}"/>')
            for xx, v, n in zip(xs, vals, names):
                out.append(f'<circle class="mk {key}" cx="{xx:.1f}" cy="{y(v):.1f}" r="4"><title>'
                           f'{_cond_label(per[n])}: {name} {v:.3f}</title></circle>')
        for xx, n in zip(xs, names):
            lab = "clean" if n == "clean" else _cond_label(per[n]).split(" (")[0].replace("Noise ", "").replace(
                " dB SNR", " dB").replace("Gain ", "").replace("Low-pass ", "").replace("Dropout ", "")
            out.append(f'<text class="t2" x="{xx:.1f}" y="{mt + ph + 16}" text-anchor="middle">{lab}</text>')
    out.append(f'<text class="t2" x="12" y="{mt + ph / 2}" transform="rotate(-90 12 {mt + ph / 2})" '
               f'text-anchor="middle">F1</text>')
    lx = ml
    for key, name in series:
        out.append(f'<line class="ln {key}" x1="{lx}" x2="{lx + 18}" y1="{height - 16}" y2="{height - 16}"/>')
        out.append(f'<text class="t2" x="{lx + 24}" y="{height - 12}">{name}</text>')
        lx += 92
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- runner

class Status:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {"state": "starting", "pid": os.getpid(), "started": now(), "conditions": {}}

    def log(self, msg: str) -> None:
        with open(self.dir / "runner.log", "a", encoding="utf-8") as f:
            f.write(f"[{now()}] {msg}\n")

    def save(self) -> None:
        self.data["updated"] = now()
        _write_json(self.dir / "status.json", self.data)


def run_study(out_dir: Path, model, ckpt: dict, device, conditions: list[dg.Condition], status: Status,
              dataset_factory=make_val_dataset, reference_csv: Path | None = None) -> bool:
    """Evaluate every condition sequentially; returns True if all completed and the clean check passed."""
    out_dir = Path(out_dir)
    all_ok = True
    for i, c in enumerate(conditions, 1):
        st = status.data["conditions"].setdefault(c.name, {})
        if condition_done(out_dir, c.name):
            st.update(state="reused_completed")
            status.log(f"REUSE {c.name} (already completed; not overwritten)")
            status.save()
            if c.family == "clean" and not (out_dir / "clean_check.json").exists():
                raise RuntimeError("clean condition exists without clean_check.json")
            continue
        status.data["current"] = f"{i}/{len(conditions)} {c.name}"
        st.update(state="running", started=now())
        status.save()
        status.log(f"START {c.name}")
        t0 = time.time()
        try:
            ds = dataset_factory(c)
            rows, report = evaluate_condition(model, ds, device)
            d = out_dir / "conditions" / c.name
            write_predictions(d / "predictions.csv", rows)
            if c.family == "clean":
                check = check_clean(rows, ckpt, reference_csv)
                _write_json(out_dir / "clean_check.json", check)
                if not check["ok"]:
                    st.update(state="failed", error="clean condition does not reproduce the recorded results")
                    status.log(f"FAIL clean check: {check}; stopping (pipeline wrapper is not a no-op)")
                    status.save()
                    return False
            _write_json(d / "metrics.json", {
                "condition": c.as_dict(), "status": "completed", "split": SPLIT, "model_run": MODEL_RUN,
                "checkpoint_epoch": ckpt["epoch"], "test_split_evaluated": False, "started": st["started"],
                "finished": now(), "elapsed_s": time.time() - t0, "overall": report,
                "degradation_stats": degradation_summary(rows)})
            st.update(state="completed", finished=now(), elapsed_s=round(time.time() - t0, 1),
                      macro_f1=report["macro_f1"], n=report["n"])
            status.log(f"COMPLETE {c.name}: n={report['n']} macro-F1 {report['macro_f1']:.4f} "
                       f"({time.time() - t0:.0f} s)")
        except Exception as e:  # record, keep the log, continue with the next condition
            all_ok = False
            st.update(state="failed", error=f"{type(e).__name__}: {e}", finished=now())
            status.log(f"FAIL {c.name}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        status.save()
    status.data.pop("current", None)
    return all_ok


def reference_scale() -> float:
    return json.loads(SEED_STUDY_SUMMARY.read_text(encoding="utf-8"))["pooled_within_architecture_seed_sd"]


def write_reports(out_dir: Path, conditions: list[dg.Condition], extra: dict | None = None) -> dict:
    s = summarize(out_dir, conditions, reference_scale())
    s.update(extra or {})
    _write_json(Path(out_dir) / "summary.json", s)
    (Path(out_dir) / "summary.md").write_text(render_markdown(s), encoding="utf-8")
    (Path(out_dir) / "macro_f1_by_condition.svg").write_text(render_svg(s), encoding="utf-8")
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validation-only robustness study of the frozen baseline CNN.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--write-plan", action="store_true", help="write plan.json (refuses to overwrite) and exit")
    p.add_argument("--summarize-only", action="store_true", help="rebuild summary/report from saved outputs")
    p.add_argument("--post-tests", action="store_true", help="run the unit test suite after the study")
    a = p.parse_args(argv)
    out_dir = a.out_dir
    plan_path = out_dir / "plan.json"
    if a.write_plan:
        if plan_path.exists():
            print(f"{plan_path} already exists; not overwritten", file=sys.stderr)
            return 1
        _write_json(plan_path, plan_payload())
        print(f"wrote {plan_path}")
        return 0
    conditions = plan_conditions(json.loads(plan_path.read_text(encoding="utf-8")))
    if a.summarize_only:
        old = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        write_reports(out_dir, conditions, {k: old[k] for k in ("frozen_inputs", "clean_check") if k in old})
        return 0

    import msvcrt
    run_dir = out_dir / "_runner"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = open(run_dir / "runner.lock", "a+")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another robustness runner holds the lock; exiting", file=sys.stderr)
        return 2
    status = Status(run_dir)
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keep the system awake
        frozen = frozen_files()
        before = hash_files(frozen)
        status.data.update(git_head=tr._run_cmd(["git", "rev-parse", "HEAD"]), frozen_sha256_before=before)
        status.log(f"RUNNER START pid {os.getpid()}; git HEAD {status.data['git_head']}; "
                   f"{len(conditions)} predeclared conditions")
        dev = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
        tr.set_determinism(0)
        model, ckpt = ar.load_run_model(MODEL_RUN, tr.DEFAULT_CHECKPOINT_DIR, dev)
        status.data["state"] = "evaluating"
        status.save()
        ok = run_study(out_dir, model, ckpt, dev, conditions, status,
                       reference_csv=tr.DEFAULT_REPORTS_DIR / MODEL_RUN / "val_predictions.csv")
        after = hash_files(frozen)
        frozen_check = {"files": {k: {"sha256": v, "unchanged": after[k] == v} for k, v in before.items()},
                        "all_unchanged": after == before,
                        "checkpoint_data_sha256_matches_inputs": all(
                            ckpt["data_sha256"][k] == tr.sha256_file(path) for k, path in
                            (("metadata", rd.DEFAULT_METADATA), ("split_file", rd.DEFAULT_SPLIT_FILE),
                             ("normalization_stats", nz.DEFAULT_STATS_PATH)))}
        status.data["frozen_inputs_unchanged"] = frozen_check["all_unchanged"]
        if not ok:
            status.data["state"] = "finished_with_failures"
            status.log("EVALUATION FINISHED WITH FAILURES; summary not written")
            status.save()
            return 1
        status.data["state"] = "summarizing"
        status.save()
        clean_check = json.loads((out_dir / "clean_check.json").read_text(encoding="utf-8"))
        s = write_reports(out_dir, conditions, {"frozen_inputs": frozen_check, "clean_check": clean_check})
        status.log(f"SUMMARY written: worst {s['findings']['hurts_most_condition']}, best-tolerated family "
                   f"{s['findings']['tolerated_best_family']}; frozen inputs unchanged={frozen_check['all_unchanged']}")
        state = "all_done"
        if a.post_tests:
            status.data["state"] = "post_tests"
            status.save()
            with open(run_dir / "post_unittest.log", "w", encoding="utf-8") as f:
                rc = subprocess.run([str(Path(sys.executable).with_name("python.exe")), "-m", "unittest", "discover",
                                     "-s", "tests", "-v"], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    creationflags=0x08000000).returncode
            status.data["post_unittest_returncode"] = rc
            status.log(f"{'COMPLETE' if rc == 0 else 'FAIL'} post step unittest (returncode {rc})")
            state = "all_done" if rc == 0 else "done_with_test_failures"
        status.data["state"] = state
        status.log(f"RUNNER END: {state}")
        status.save()
        return 0 if state == "all_done" else 1
    except BaseException:
        status.data["state"] = "runner_crashed"
        status.log("RUNNER CRASHED:\n" + traceback.format_exc())
        status.save()
        raise


if __name__ == "__main__":
    sys.exit(main())
