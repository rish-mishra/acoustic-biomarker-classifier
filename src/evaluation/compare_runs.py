"""Compare trained runs on VALIDATION results only.

Reads, for each run: reports/runs/<run>/run.json, history.json, val_analysis.json
and val_predictions.csv (written by analyze_run.py). Writes:
  <out_dir>/comparison.json   machine-readable comparison
  <out_dir>/comparison.md     human-readable report

Ranking uses validation macro F1 only. As supporting uncertainty (not used
for ranking), a patient-level bootstrap (resampling the validation patients
with replacement, fixed seed) gives 95% intervals for each run's macro F1 and
for paired differences against a reference run.

Usage::

    python src/evaluation/compare_runs.py baseline_cnn_v1 baseline_cnn_v2 ... --reference baseline_cnn_v1 \
        --out-dir reports/runs/lr_dropout_study
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
for _sub in ("models", "data", "features", "evaluation"):
    if str(SRC_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(SRC_DIR / _sub))

import train as tr  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402

CLASS_NAMES = tr.rd.CLASS_NAMES


def macro_f1(labels: list[int], preds: list[int]) -> float:
    cm = confusion_matrix(torch.tensor(labels), torch.tensor(preds), len(CLASS_NAMES))
    return classification_report(cm, CLASS_NAMES)["macro_f1"]


def load_predictions(path: Path) -> dict[tuple[str, int], dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["recording_filename"], int(r["cycle_index"])): r for r in csv.DictReader(f)}


def patient_bootstrap(preds: dict[str, dict], reference: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% percentile intervals of macro F1 per run and paired differences vs ``reference``."""
    keys = sorted(next(iter(preds.values())))
    for run, p in preds.items():
        if sorted(p) != keys:
            raise ValueError(f"{run} predictions cover different validation cycles")
    by_patient: dict[str, list] = {}
    for k in keys:
        by_patient.setdefault(next(iter(preds.values()))[k]["patient_id"], []).append(k)
    patients = sorted(by_patient)
    rng = random.Random(seed)
    samples = {run: [] for run in preds}
    for _ in range(n_boot):
        chosen = [k for pid in (rng.choice(patients) for _ in patients) for k in by_patient[pid]]
        labels = [int(preds[reference][k]["label"]) for k in chosen]
        for run, p in preds.items():
            samples[run].append(macro_f1(labels, [int(p[k]["pred"]) for k in chosen]))

    def ci(values):
        s = sorted(values)
        return [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]

    out = {"n_bootstrap": n_boot, "seed": seed, "unit": "patient", "n_patients": len(patients), "runs": {}}
    for run in preds:
        diff = [a - b for a, b in zip(samples[run], samples[reference])]
        out["runs"][run] = {
            "macro_f1_ci95": ci(samples[run]),
            f"diff_vs_{reference}_mean": statistics.fmean(diff),
            f"diff_vs_{reference}_ci95": ci(diff),
            f"p_better_than_{reference}": sum(d > 0 for d in diff) / n_boot,
        }
    return out


def overfitting_indicators(history: list[dict], best_epoch: int) -> dict:
    best = next(h for h in history if h["epoch"] == best_epoch)
    min_val = min(history, key=lambda h: h["val_loss"])
    last = history[-1]
    f1 = [h["val_macro_f1"] for h in history]
    return {
        "train_loss_at_best": best["train_loss"],
        "val_loss_at_best": best["val_loss"],
        "val_minus_train_loss_at_best": best["val_loss"] - best["train_loss"],
        "min_val_loss": min_val["val_loss"],
        "min_val_loss_epoch": min_val["epoch"],
        "train_loss_last": last["train_loss"],
        "val_loss_last": last["val_loss"],
        "train_loss_drop_best_to_last": best["train_loss"] - last["train_loss"],
        "val_loss_change_best_to_last": last["val_loss"] - best["val_loss"],
        "val_macro_f1_epoch_to_epoch_abs_change_mean": statistics.fmean(
            abs(a - b) for a, b in zip(f1[1:], f1[:-1])) if len(f1) > 1 else 0.0,
    }


def compare(runs: list[str], reference: str, reports_dir: Path, out_dir: Path) -> dict:
    reports_dir, out_dir = Path(reports_dir), Path(out_dir)
    table, preds = {}, {}
    for run in runs:
        d = reports_dir / run
        info = json.loads((d / "run.json").read_text(encoding="utf-8"))
        history = json.loads((d / "history.json").read_text(encoding="utf-8"))
        analysis = json.loads((d / "val_analysis.json").read_text(encoding="utf-8"))
        if info.get("test_split_evaluated") or analysis.get("split") != "val":
            raise ValueError(f"{run}: expected validation-only artifacts")
        best = info["best_validation"]
        table[run] = {
            "lr": info["config"]["lr"],
            "dropout": info["config"]["dropout"],
            "seed": info["seed"],
            "parameter_count": info["model"]["parameter_count"],
            "best_epoch": info["best_epoch"],
            "epochs_completed": info["epochs_completed"],
            "stop_reason": info["stop_reason"],
            "training_time_s": info["total_training_time_s"],
            "val_macro_f1": best["macro_f1"],
            "val_balanced_accuracy": best["balanced_accuracy"],
            "val_accuracy": best["accuracy"],
            "val_loss_at_best": best["loss"],
            "per_class": best["per_class"],
            "confusion_matrix": best["confusion_matrix"],
            "overfitting": overfitting_indicators(history, info["best_epoch"]),
            "curves": {k: [h[k] for h in history] for k in ("train_loss", "val_loss", "val_macro_f1")},
            "subgroups": analysis["subgroups"],
            "analysis_reproduces_logged_metrics": analysis["reproduces_logged_val_confusion_matrix"],
            "git_commit": info["code"]["git_commit"],
            "data_sha256": {k: info["data"][k]["sha256"] for k in ("metadata", "split_file", "normalization_stats")},
        }
        preds[run] = load_predictions(d / "val_predictions.csv")
    ranking = sorted(runs, key=lambda r: -table[r]["val_macro_f1"])
    result = {
        "selection_metric": "validation macro F1 (best epoch)",
        "ranking": ranking,
        "runs": table,
        "bootstrap": patient_bootstrap(preds, reference),
        "same_data_sha256_for_all_runs": len({json.dumps(t["data_sha256"], sort_keys=True) for t in table.values()}) == 1,
        "test_split_evaluated": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tr._write_json(out_dir / "comparison.json", result)
    (out_dir / "comparison.md").write_text(render_markdown(result, reference), encoding="utf-8")
    return result


def render_markdown(res: dict, reference: str) -> str:
    runs, t, boot = res["ranking"], res["runs"], res["bootstrap"]["runs"]
    L = ["# Validation-only comparison", "",
         f"Ranking by validation macro F1 (best epoch). Test split not evaluated. "
         f"Patient bootstrap: {res['bootstrap']['n_bootstrap']} resamples of {res['bootstrap']['n_patients']} "
         f"validation patients (supporting uncertainty only).", "",
         "| Rank | Run | lr | dropout | Best epoch | Epochs | Macro F1 | 95% CI | Δ vs " + reference + " (95% CI) "
         "| Bal. acc. | Acc. | Train time | Params |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs, 1):
        x, b = t[r], boot[r]
        lo, hi = b["macro_f1_ci95"]
        dlo, dhi = b[f"diff_vs_{reference}_ci95"]
        L.append(f"| {i} | {r} | {x['lr']:g} | {x['dropout']:g} | {x['best_epoch']} | {x['epochs_completed']} | "
                 f"**{x['val_macro_f1']:.4f}** | {lo:.3f}-{hi:.3f} | {b[f'diff_vs_{reference}_mean']:+.3f} "
                 f"({dlo:+.3f} to {dhi:+.3f}) | {x['val_balanced_accuracy']:.4f} | {x['val_accuracy']:.4f} | "
                 f"{x['training_time_s'] / 60:.1f} min | {x['parameter_count']:,} |")
    L += ["", "## Per-class validation metrics at best epoch (P / R / F1)", "",
          "| Run | " + " | ".join(CLASS_NAMES) + " |", "|---|" + "---|" * len(CLASS_NAMES)]
    for r in runs:
        pc = t[r]["per_class"]
        L.append(f"| {r} | " + " | ".join(
            f"{pc[c]['precision']:.2f} / {pc[c]['recall']:.2f} / **{pc[c]['f1']:.2f}**" for c in CLASS_NAMES) + " |")
    L += ["", "## Confusion matrices (rows = true N/C/W/B, columns = predicted)", ""]
    for r in runs:
        L.append(f"- **{r}**: " + " ".join(str(row) for row in t[r]["confusion_matrix"]))
    L += ["", "## Overfitting indicators", "",
          "| Run | Train loss @best | Val loss @best | Min val loss (epoch) | Train loss drop best→last | "
          "Val loss change best→last | Mean |ΔF1| per epoch |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        o = t[r]["overfitting"]
        L.append(f"| {r} | {o['train_loss_at_best']:.3f} | {o['val_loss_at_best']:.3f} | "
                 f"{o['min_val_loss']:.3f} ({o['min_val_loss_epoch']}) | {o['train_loss_drop_best_to_last']:+.3f} | "
                 f"{o['val_loss_change_best_to_last']:+.3f} | {o['val_macro_f1_epoch_to_epoch_abs_change_mean']:.3f} |")
    L += ["", "## Subgroup macro F1 (classes present) at best epoch, n cycles / patients", ""]
    for dim in ("device", "original_sample_rate", "quality_group", "duration"):
        groups = list(t[runs[0]]["subgroups"][dim])
        L += [f"**{dim}**", "", "| Run | " + " | ".join(
            f"{g} (n={t[runs[0]]['subgroups'][dim][g]['n_cycles']}/{t[runs[0]]['subgroups'][dim][g]['n_patients']})"
            for g in groups) + " |", "|---|" + "---|" * len(groups)]
        for r in runs:
            L.append(f"| {r} | " + " | ".join(
                f"{t[r]['subgroups'][dim][g]['macro_f1_present_classes']:.3f}" for g in groups) + " |")
        L.append("")
    L += ["## Curves (per epoch: train loss / val loss / val macro F1)", ""]
    for r in runs:
        c = t[r]["curves"]
        L.append(f"<details><summary>{r}</summary>\n\n| Epoch | Train loss | Val loss | Val macro F1 |\n|---|---|---|---|")
        for e, (a, b, f) in enumerate(zip(c["train_loss"], c["val_loss"], c["val_macro_f1"])):
            L.append(f"| {e} | {a:.4f} | {b:.4f} | {f:.4f} |")
        L.append("\n</details>\n")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare runs on validation results.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--reference", required=True)
    p.add_argument("--reports-dir", default=str(tr.DEFAULT_REPORTS_DIR))
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    res = compare(a.runs, a.reference, a.reports_dir, a.out_dir)
    for i, r in enumerate(res["ranking"], 1):
        print(f"{i}. {r}: val macro-F1 {res['runs'][r]['val_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
