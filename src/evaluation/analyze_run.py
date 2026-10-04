"""Validation-only analysis of a trained run's best checkpoint.

Re-predicts the VALIDATION split with models/<run>/best.pt and writes:
  reports/runs/<run>/val_predictions.csv   one row per validation cycle
  reports/runs/<run>/val_analysis.json     reproduction check, subgroup and per-patient metrics

There is no option to analyze the test split: the split is fixed to "val".

Subgroups (fixed definitions; they filter validation cycles, never the split):
  device, original sample rate,
  quality group (frozen README definitions: all / moderately_clean / clean, plus
  the complements not_moderately_clean / not_clean),
  annotated cycle duration bins: <1.5 s, 1.5-2.5 s, 2.5-4 s, >=4 s.
Subgroup macro F1 averages only classes present in that subgroup; small
subgroups (few patients) are noisy and are reported with their n.

Usage::

    python src/evaluation/analyze_run.py baseline_cnn_v1 [baseline_cnn_v2 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
for _sub in ("models", "data", "features", "evaluation"):
    if str(SRC_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(SRC_DIR / _sub))

import train as tr  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402

rd, nz = tr.rd, tr.nz
SPLIT = "val"  # fixed; this module never touches the test split
DURATION_BINS = ((0.0, 1.5, "<1.5s"), (1.5, 2.5, "1.5-2.5s"), (2.5, 4.0, "2.5-4s"), (4.0, float("inf"), ">=4s"))


def duration_bin(seconds: float) -> str:
    for lo, hi, name in DURATION_BINS:
        if lo <= seconds < hi:
            return name
    raise ValueError(f"invalid duration {seconds}")


@torch.no_grad()
def predict(model, dataset, device, batch_size: int = 32) -> list[dict]:
    """Per-cycle predictions for the validation dataset (sequential order)."""
    if dataset.split != SPLIT:
        raise PermissionError(f"analysis is validation-only; got split {dataset.split!r}")
    model.eval()
    rows = []
    for batch in tr.make_eval_loader(dataset, batch_size):
        logits = model(batch["features"].to(device), batch["mask"].to(device)).float().cpu()
        probs = torch.softmax(logits, dim=1)
        for i in range(len(batch["labels"])):
            q = batch["quality"][i]
            rows.append({
                "patient_id": batch["patient_id"][i],
                "recording_filename": batch["recording_filename"][i],
                "cycle_index": int(batch["cycle_index"][i]),
                "device": batch["device"][i],
                "original_sample_rate": int(batch["original_sample_rate"][i]),
                "cycle_duration_s": float(batch["cycle_duration_s"][i]),
                "original_clipped_fraction": q["original_clipped_fraction"],
                "clipped_plateau_count": q["clipped_plateau_count"],
                "label": int(batch["labels"][i]),
                "pred": int(probs[i].argmax()),
                **{f"prob_{c}": float(probs[i, k]) for k, c in enumerate(rd.CLASS_NAMES)},
            })
    return rows


def group_metrics(rows: list[dict]) -> dict:
    k = len(rd.CLASS_NAMES)
    cm = confusion_matrix(torch.tensor([r["label"] for r in rows]), torch.tensor([r["pred"] for r in rows]), k)
    rep = classification_report(cm, rd.CLASS_NAMES)
    present = [c for c in rd.CLASS_NAMES if rep["per_class"][c]["support"] > 0]
    return {
        "n_cycles": len(rows),
        "n_patients": len({r["patient_id"] for r in rows}),
        "label_counts": {c: rep["per_class"][c]["support"] for c in rd.CLASS_NAMES},
        "accuracy": rep["accuracy"],
        "balanced_accuracy_present_classes": sum(rep["per_class"][c]["recall"] for c in present) / len(present),
        "macro_f1_present_classes": sum(rep["per_class"][c]["f1"] for c in present) / len(present),
        "recall": {c: rep["per_class"][c]["recall"] for c in present},
        "confusion_matrix": rep["confusion_matrix"],
    }


def subgroup_analysis(rows: list[dict]) -> dict:
    keys = {
        "device": lambda r: r["device"],
        "original_sample_rate": lambda r: str(r["original_sample_rate"]),
        "duration": lambda r: duration_bin(r["cycle_duration_s"]),
    }
    out = {}
    for name, key in keys.items():
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        out[name] = {g: group_metrics(v) for g, v in sorted(groups.items())}
    quality = {}
    for g in rd.QUALITY_GROUPS:
        members = [r for r in rows if rd.in_quality_group(r, g)]
        quality[g] = group_metrics(members)
        if g != "all":
            rest = [r for r in rows if not rd.in_quality_group(r, g)]
            if rest:
                quality[f"not_{g}"] = group_metrics(rest)
    out["quality_group"] = quality
    return out


def patient_analysis(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["patient_id"]].append(r)
    return {
        pid: {"n_cycles": len(v), "device": Counter(r["device"] for r in v).most_common(1)[0][0],
              "accuracy": sum(r["label"] == r["pred"] for r in v) / len(v),
              "label_counts": dict(Counter(rd.CLASS_NAMES[r["label"]] for r in v)),
              "pred_counts": dict(Counter(rd.CLASS_NAMES[r["pred"]] for r in v))}
        for pid, v in sorted(by.items())
    }


def load_run_model(run_name: str, checkpoint_dir: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(Path(checkpoint_dir) / run_name / "best.pt", map_location=device, weights_only=False)
    model = tr.build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def analyze(run_name: str, checkpoint_dir: Path = tr.DEFAULT_CHECKPOINT_DIR,
            reports_dir: Path = tr.DEFAULT_REPORTS_DIR, dataset=None, device: str = "cuda") -> dict:
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    tr.set_determinism(0)
    model, ckpt = load_run_model(run_name, checkpoint_dir, dev)
    if dataset is None:
        dataset = rd.RespiratoryCycleDataset(SPLIT, normalizer=nz.load_normalizer())
    current_sha = {"metadata": tr.sha256_file(dataset.metadata_path), "split_file": tr.sha256_file(dataset.split_path)}
    sha_ok = all(ckpt.get("data_sha256", {}).get(k) == v for k, v in current_sha.items())

    rows = predict(model, dataset, dev)
    overall = classification_report(
        confusion_matrix(torch.tensor([r["label"] for r in rows]), torch.tensor([r["pred"] for r in rows]),
                         len(rd.CLASS_NAMES)), rd.CLASS_NAMES)
    reproduced = (overall["confusion_matrix"] == ckpt["val_metrics"]["confusion_matrix"])
    result = {
        "run_name": run_name,
        "split": SPLIT,
        "checkpoint_epoch": ckpt["epoch"],
        "data_sha256_matches_checkpoint": sha_ok,
        "reproduces_logged_val_confusion_matrix": reproduced,
        "overall": overall,
        "subgroups": subgroup_analysis(rows),
        "patients": patient_analysis(rows),
    }
    out_dir = Path(reports_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "val_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tr._write_json(out_dir / "val_analysis.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validation-only analysis of trained runs.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    for run in a.runs:
        res = analyze(run, device=a.device)
        print(f"{run}: epoch {res['checkpoint_epoch']} macro-F1 {res['overall']['macro_f1']:.4f} "
              f"reproduced={res['reproduces_logged_val_confusion_matrix']} sha_ok={res['data_sha256_matches_checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
