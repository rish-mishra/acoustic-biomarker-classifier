"""POST-HOC EXPLORATORY — NOT PREDECLARED analysis of the locked final test outputs.

Reads only the already-saved, locked prediction files:
  reports/final_evaluation/clean_test/predictions.csv
  reports/final_evaluation/robustness_test/conditions/<condition>/predictions.csv  (clean, noise, gain)
No model, audio or dataset is loaded; nothing is re-run, trained, fitted, tuned or selected. Both locks are
verified (read-only re-hashing) before and after, and this module never writes inside clean_test/ or
robustness_test/. Results are descriptive associations only: no causal, clinical or diagnostic claims, and
they do not change any confirmatory result.

Outputs (reports/final_evaluation/exploratory/, every file labelled POST-HOC EXPLORATORY — NOT PREDECLARED):
  summary.md, summary.json
  per_patient_clean.csv, per_device_clean.csv, normal_to_crackle_contribution.csv
  noise_by_patient.csv, noise_by_device.csv, gain_by_device.csv, gain_by_patient.csv

Usage::

    python src/evaluation/exploratory_test_analysis.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR / "evaluation") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "evaluation"))

import calibration_metrics as cmx  # noqa: E402

REPO_ROOT = SRC_DIR.parent
FINAL = REPO_ROOT / "reports" / "final_evaluation"
OUT_DIR = FINAL / "exploratory"
LABEL = "POST-HOC EXPLORATORY — NOT PREDECLARED"
CLASS_NAMES = ("normal", "crackle", "wheeze", "both")
NORMAL, CRACKLE = 0, 1
NOISE = ("noise_snr20db", "noise_snr10db", "noise_snr5db", "noise_snr0db")
GAIN = ("gain_0.5x", "gain_0.25x", "gain_1.5x", "gain_2.0x")
MIN_CYCLES_FOR_F1 = 20  # macro F1 only "where meaningful": >= 20 cycles and >= 2 true classes present
MIN_CYCLES_PATIENT_SHARE = 10  # patient-level shares reported for patients with >= 10 cycles


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"], r["pred"] = int(r["label"]), int(r["pred"])
        r["msp"] = max(float(r[f"prob_{c}"]) for c in CLASS_NAMES)
    return rows


def group_summary(rows: list[dict]) -> dict:
    """Descriptive metrics for a set of cycles (zero division gives 0, as project-wide)."""
    y = np.array([r["label"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    s = cmx.classification_summary(y, p, len(CLASS_NAMES))
    n_present = sum(1 for c in s["support"] if c > 0)
    meaningful = len(rows) >= MIN_CYCLES_FOR_F1 and n_present >= 2
    normals = int((y == NORMAL).sum())
    n2c = int(((y == NORMAL) & (p == CRACKLE)).sum())
    return {
        "n_cycles": len(rows), "n_patients": len({r["patient_id"] for r in rows}),
        "true_counts": s["support"], "pred_counts": s["predicted_counts"], "confusion_matrix": s["confusion_matrix"],
        "accuracy": s["accuracy"], "error_rate": 1 - s["accuracy"],
        "balanced_accuracy_present_classes": s["balanced_accuracy_present_classes"] if meaningful else None,
        "macro_f1_present_classes": s["macro_f1_present_classes"] if meaningful else None,
        "macro_f1_4_classes": s["macro_f1"] if meaningful else None,
        "normal_cycles": normals, "normal_to_crackle": n2c,
        "normal_to_crackle_rate": n2c / normals if normals else None,
        "mean_msp": float(np.mean([r["msp"] for r in rows])),
        "mean_msp_correct": float(np.mean([r["msp"] for r in rows if r["label"] == r["pred"]]))
        if (y == p).any() else None,
        "mean_msp_incorrect": float(np.mean([r["msp"] for r in rows if r["label"] != r["pred"]]))
        if (y != p).any() else None,
    }


def by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return dict(sorted(out.items()))


def concentration(counts: dict[str, int], base: dict[str, int]) -> dict:
    """How concentrated a count (e.g. normal->crackle errors) is across units, vs their share of a base count."""
    total, base_total = sum(counts.values()), sum(base.values())
    order = sorted(counts, key=lambda k: -counts[k])
    cum, rows, needed = 0, [], {}
    for i, k in enumerate(order, 1):
        cum += counts[k]
        rows.append({"unit": k, "count": counts[k], "share": counts[k] / total if total else 0.0,
                     "cumulative_share": cum / total if total else 0.0,
                     "base": base.get(k, 0), "base_share": base.get(k, 0) / base_total if base_total else 0.0})
        for q in (0.5, 0.8):
            if q not in needed and total and cum / total >= q:
                needed[q] = i
    return {"total": total, "units": len(order), "units_for_50pct": needed.get(0.5), "units_for_80pct": needed.get(0.8),
            "top1_share": rows[0]["share"] if rows else 0.0,
            "top3_share": sum(r["share"] for r in rows[:3]), "table": rows}


def leave_one_out(rows: list[dict], key: str) -> list[dict]:
    """Overall clean-test metrics recomputed without each unit (descriptive influence, not a re-analysis)."""
    out = []
    for unit, members in by(rows, key).items():
        rest = [r for r in rows if r[key] != unit]
        s = group_summary(rest)
        out.append({"excluded": unit, "n_excluded": len(members), "macro_f1_4_classes": s["macro_f1_4_classes"],
                    "accuracy": s["accuracy"], "normal_to_crackle_rate": s["normal_to_crackle_rate"]})
    return out


def pred_shares(rows: list[dict]) -> list[float]:
    p = np.array([r["pred"] for r in rows])
    return [float((p == k).mean()) for k in range(len(CLASS_NAMES))]


def noise_tables(clean: list[dict], noise: dict[str, list[dict]]) -> tuple[list[dict], list[dict], dict]:
    by_patient, by_device, spread = [], [], {}
    for key, out in (("patient_id", by_patient), ("device", by_device)):
        for unit, members in by(clean, key).items():
            keys = {(r["recording_filename"], r["cycle_index"]) for r in members}
            row = {key: unit, "n_cycles": len(members), "n_patients": len({r["patient_id"] for r in members}),
                   "device": members[0]["device"] if key == "patient_id" else unit,
                   "clean_crackle_share": pred_shares(members)[CRACKLE],
                   "true_crackle_share": float(np.mean([r["label"] == CRACKLE for r in members]))}
            for name, rows in noise.items():
                sub = [r for r in rows if (r["recording_filename"], r["cycle_index"]) in keys]
                row[f"{name}_crackle_share"] = pred_shares(sub)[CRACKLE]
                row[f"{name}_accuracy"] = float(np.mean([r["label"] == r["pred"] for r in sub]))
            out.append(row)
    for name in NOISE:
        vals = [r[f"{name}_crackle_share"] for r in by_patient if r["n_cycles"] >= MIN_CYCLES_PATIENT_SHARE]
        spread[name] = {"patients": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals),
                        "patients_at_least_80pct_crackle": sum(v >= 0.8 for v in vals)}
    return by_patient, by_device, spread


def gain_tables(clean: list[dict], gain: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    clean_pred = {(r["recording_filename"], r["cycle_index"]): r["pred"] for r in clean}
    by_device, by_patient = [], []
    for key, out in (("device", by_device), ("patient_id", by_patient)):
        for unit, members in by(clean, key).items():
            keys = {(r["recording_filename"], r["cycle_index"]) for r in members}
            base = pred_shares(members)
            row = {key: unit, "n_cycles": len(members), "n_patients": len({r["patient_id"] for r in members}),
                   "device": members[0]["device"],
                   **{f"clean_{c}_share": base[k] for k, c in enumerate(CLASS_NAMES)}}
            for name, rows in gain.items():
                sub = [r for r in rows if (r["recording_filename"], r["cycle_index"]) in keys]
                sh = pred_shares(sub)
                row[f"{name}_normal_share_change"] = sh[NORMAL] - base[NORMAL]
                row[f"{name}_crackle_share_change"] = sh[CRACKLE] - base[CRACKLE]
                row[f"{name}_both_share_change"] = sh[3] - base[3]
                row[f"{name}_predictions_changed"] = float(np.mean(
                    [r["pred"] != clean_pred[(r["recording_filename"], r["cycle_index"])] for r in sub]))
                row[f"{name}_accuracy_change"] = float(np.mean([r["label"] == r["pred"] for r in sub])
                                                       - np.mean([r["label"] == r["pred"] for r in members]))
            out.append(row)
    return by_device, by_patient


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {LABEL}\n")
        flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows]
        w = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(flat)


def lock_hashes() -> dict:
    import final_evaluation as fe
    protocol = FINAL / "protocol.json"
    return {"clean": fe.verify_lock(FINAL / "clean_test", protocol),
            "robustness": fe.verify_lock(FINAL / "robustness_test", protocol)}


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def render(s: dict) -> str:
    c, pp, pd_ = s["concentration"], s["per_patient"], s["per_device"]
    L = [f"# {LABEL}", "", "## Exploratory analysis of the locked final test outputs", "",
         f"**{LABEL}.** Generated after the confirmatory final evaluation was complete and locked, from the saved "
         "clean-test and robustness prediction files only (no inference, retraining, fitting, tuning or selection). "
         "These are descriptive associations on 19 test patients; they do not change any confirmatory result and "
         "support no causal, clinical or diagnostic claim. Both locks verified before and after this analysis.", "",
         f"Macro F1 / balanced accuracy are shown only where meaningful (≥ {MIN_CYCLES_FOR_F1} cycles and ≥ 2 true "
         "classes present) and use the classes present in that group.", "",
         "## 1. Per-patient clean-test performance", "",
         "| Patient | Device | Cycles | True n/c/w/b | Pred n/c/w/b | Accuracy | Macro F1 (present) | Normal→crackle "
         "(of normals) | Mean MSP |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(pp, key=lambda r: -r["n_cycles"]):
        L.append(f"| {r['patient_id']} | {r['device']} | {r['n_cycles']} | {'/'.join(map(str, r['true_counts']))} | "
                 f"{'/'.join(map(str, r['pred_counts']))} | {fmt(r['accuracy'])} | {fmt(r['macro_f1_present_classes'])} | "
                 f"{r['normal_to_crackle']} / {r['normal_cycles']}"
                 + (f" ({r['normal_to_crackle_rate']:.0%})" if r["normal_to_crackle_rate"] is not None else "")
                 + f" | {fmt(r['mean_msp'])} |")
    L += ["", "## 2. Per-device clean-test performance", "",
          "| Device | Patients | Cycles | True n/c/w/b | Pred n/c/w/b | Accuracy | Bal. acc. (present) | Macro F1 "
          "(present) | Normal→crackle rate | Error rate | Mean MSP (correct / incorrect) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pd_:
        L.append(f"| {r['device']} | {r['n_patients']} | {r['n_cycles']} | {'/'.join(map(str, r['true_counts']))} | "
                 f"{'/'.join(map(str, r['pred_counts']))} | {fmt(r['accuracy'])} | "
                 f"{fmt(r['balanced_accuracy_present_classes'])} | {fmt(r['macro_f1_present_classes'])} | "
                 f"{fmt(r['normal_to_crackle_rate'])} | {fmt(r['error_rate'])} | {fmt(r['mean_msp'])} "
                 f"({fmt(r['mean_msp_correct'])} / {fmt(r['mean_msp_incorrect'])}) |")
    pc, dc = c["patients"], c["devices"]
    L += ["", "## 3. Contribution to the normal→crackle shift", "",
          f"{pc['total']} normal cycles were predicted as crackle. Patients: top 1 accounts for {pc['top1_share']:.0%}, "
          f"top 3 for {pc['top3_share']:.0%}; {pc['units_for_50pct']} of {pc['units']} patients account for ≥ 50% and "
          f"{pc['units_for_80pct']} for ≥ 80%. Devices: top device accounts for {dc['top1_share']:.0%}.", "",
          "| Patient | Normal→crackle | Share of all | Cumulative | Share of all normal cycles |", "|---|---|---|---|---|"]
    for r in pc["table"]:
        if r["count"]:
            L.append(f"| {r['unit']} | {r['count']} | {r['share']:.1%} | {r['cumulative_share']:.1%} | {r['base_share']:.1%} |")
    L += ["", "| Device | Normal→crackle | Share of all | Share of all normal cycles |", "|---|---|---|---|"]
    L += [f"| {r['unit']} | {r['count']} | {r['share']:.1%} | {r['base_share']:.1%} |" for r in dc["table"]]
    L += ["", "Leave-one-patient-out (overall clean-test values recomputed without each patient; descriptive "
          "influence only):", "", "| Excluded patient | Cycles excluded | Macro F1 (4 classes) | Accuracy | Normal→crackle rate |",
          "|---|---|---|---|---|"]
    for r in sorted(s["leave_one_patient_out"], key=lambda r: -(r["macro_f1_4_classes"] or 0))[:6]:
        L.append(f"| {r['excluded']} | {r['n_excluded']} | {fmt(r['macro_f1_4_classes'], 4)} | {fmt(r['accuracy'])} | "
                 f"{fmt(r['normal_to_crackle_rate'])} |")
    L += [f"(all 19 rows in summary.json; full-test macro F1 is {s['overall']['macro_f1_4_classes']:.4f})", "",
          "## 4. Noise: is the crackle collapse concentrated?", "",
          f"Patients with ≥ {MIN_CYCLES_PATIENT_SHARE} cycles: share of their cycles predicted as crackle.", "",
          "| Condition | Patients | Min | Median | Max | Patients ≥ 80% crackle |", "|---|---|---|---|---|---|"]
    for n, v in s["noise_spread"].items():
        L.append(f"| {n} | {v['patients']} | {v['min']:.0%} | {v['median']:.0%} | {v['max']:.0%} | "
                 f"{v['patients_at_least_80pct_crackle']} |")
    L += ["", "| Device | Patients | Cycles | True crackle share | Clean | 20 dB | 10 dB | 5 dB | 0 dB (predicted crackle share) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in s["noise_by_device"]:
        L.append(f"| {r['device']} | {r['n_patients']} | {r['n_cycles']} | {r['true_crackle_share']:.0%} | "
                 f"{r['clean_crackle_share']:.0%} | " + " | ".join(f"{r[f'{n}_crackle_share']:.0%}" for n in NOISE) + " |")
    L += ["", "## 5. Gain sensitivity by device", "",
          "Change in the share of predictions that are normal (and share of predictions changed) vs clean test:", "",
          "| Device | Patients | Cycles | Clean normal share | 0.5x | 0.25x | 1.5x | 2x |", "|---|---|---|---|---|---|---|---|"]
    for r in s["gain_by_device"]:
        L.append(f"| {r['device']} | {r['n_patients']} | {r['n_cycles']} | {r['clean_normal_share']:.0%} | " + " | ".join(
            f"{r[f'{g}_normal_share_change']:+.0%} ({r[f'{g}_predictions_changed']:.0%})" for g in GAIN) + " |")
    L += ["", "Per-patient gain tables: gain_by_patient.csv. Devices are confounded with patients and with "
          "sample rate here (few patients per device), so device differences cannot be separated from patient "
          "differences.", ""]
    return "\n".join(L)


def main() -> int:
    before = lock_hashes()
    clean = load(FINAL / "clean_test" / "predictions.csv")
    rob = FINAL / "robustness_test" / "conditions"
    rob_clean = load(rob / "clean" / "predictions.csv")
    if [(r["recording_filename"], r["cycle_index"], r["pred"]) for r in rob_clean] != \
            [(r["recording_filename"], r["cycle_index"], r["pred"]) for r in clean]:
        raise RuntimeError("robustness clean condition differs from the locked clean test")
    noise = {n: load(rob / n / "predictions.csv") for n in NOISE}
    gain = {g: load(rob / g / "predictions.csv") for g in GAIN}

    per_patient = [{"patient_id": k, "device": v[0]["device"], "original_sample_rate": v[0]["original_sample_rate"],
                    **group_summary(v)} for k, v in by(clean, "patient_id").items()]
    per_device = [{"device": k, **group_summary(v)} for k, v in by(clean, "device").items()]
    n2c = lambda rows, key: {k: group_summary(v)["normal_to_crackle"] for k, v in by(rows, key).items()}  # noqa: E731
    normals = lambda rows, key: {k: group_summary(v)["normal_cycles"] for k, v in by(rows, key).items()}  # noqa: E731
    conc = {"patients": concentration(n2c(clean, "patient_id"), normals(clean, "patient_id")),
            "devices": concentration(n2c(clean, "device"), normals(clean, "device"))}
    noise_p, noise_d, spread = noise_tables(clean, noise)
    gain_d, gain_p = gain_tables(clean, gain)
    summary = {"label": LABEL, "source": "locked clean_test and robustness_test prediction files only",
               "overall": group_summary(clean), "per_patient": per_patient, "per_device": per_device,
               "concentration": conc, "leave_one_patient_out": leave_one_out(clean, "patient_id"),
               "leave_one_device_out": leave_one_out(clean, "device"),
               "noise_by_patient": noise_p, "noise_by_device": noise_d, "noise_spread": spread,
               "gain_by_device": gain_d, "gain_by_patient": gain_p}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(render(summary), encoding="utf-8")
    write_csv(OUT_DIR / "per_patient_clean.csv", per_patient)
    write_csv(OUT_DIR / "per_device_clean.csv", per_device)
    write_csv(OUT_DIR / "normal_to_crackle_contribution.csv",
              [{"level": lvl, **r} for lvl in ("patients", "devices") for r in conc[lvl]["table"]])
    write_csv(OUT_DIR / "noise_by_patient.csv", noise_p)
    write_csv(OUT_DIR / "noise_by_device.csv", noise_d)
    write_csv(OUT_DIR / "gain_by_device.csv", gain_d)
    write_csv(OUT_DIR / "gain_by_patient.csv", gain_p)
    after = lock_hashes()
    if before != after:
        raise RuntimeError("a lock changed during the exploratory analysis")
    print(f"{LABEL}: wrote {OUT_DIR}; locks verified before and after".encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
