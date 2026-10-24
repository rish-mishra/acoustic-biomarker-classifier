"""Calibration, uncertainty and selective prediction for the frozen baseline CNN (validation only).

Implements reports/calibration_uncertainty/plan.json. Inputs are SAVED validation
softmax probabilities only:
  reports/runs/baseline_cnn_v3/val_predictions.csv                     clean
  reports/robustness/noise_robustness/conditions/<c>/predictions.csv   15 degradation conditions
No dataset, audio or checkpoint is loaded, and nothing is re-run. Every input
must contain exactly the cycles of the frozen validation patients
(data/splits/patient_split.csv); anything else aborts. The test split is never
read.

Temperature scaling: leave-one-patient-out (LOPO) over the 19 validation
patients on CLEAN predictions; each patient's temperature (fitted on the other
18) is applied to that patient's clean and degraded predictions. Calibrated
metrics are out-of-fold development-set estimates. final_validation_temperature
(all 19 patients) is recorded for later one-time use on the test set and is
never used as evidence here.

Outputs (reports/calibration_uncertainty/):
  summary.json, summary.md                  all predeclared results (+ labelled post-hoc section)
  per_example_clean.csv                     raw and LOPO-calibrated probabilities and scores per cycle
  per_example_degraded.csv                  per cycle and condition: raw / calibrated confidence and entropy
  lopo_temperatures.csv                     the 19 fold temperatures
  risk_coverage_curves.csv                  selective risk and macro F1 at every coverage
  reliability_diagram.svg, risk_coverage.svg, degradation_confidence.svg

Usage::

    python src/evaluation/calibration_uncertainty.py               # compute + render
    python src/evaluation/calibration_uncertainty.py --render-only # rebuild summary.md / figures from summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR / "evaluation") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "evaluation"))

import calibration_metrics as cm  # noqa: E402

REPO_ROOT = SRC_DIR.parent
OUT_DIR = REPO_ROOT / "reports" / "calibration_uncertainty"
CLEAN_CSV = REPO_ROOT / "reports" / "runs" / "baseline_cnn_v3" / "val_predictions.csv"
DEGRADED_DIR = REPO_ROOT / "reports" / "robustness" / "noise_robustness" / "conditions"
SPLIT_FILE = REPO_ROOT / "data" / "splits" / "patient_split.csv"
CLASS_NAMES = ("normal", "crackle", "wheeze", "both")
K = len(CLASS_NAMES)
N_EXPECTED = 1077
N_BOOT, BOOT_SEED = 2000, 0  # plan.json "statistics"
RANKINGS = ("msp_raw", "entropy_raw", "msp_lopo")
SCORES = ("1-msp", "entropy", "1-margin")


# ---------------------------------------------------------------- inputs

def validation_patients(split_file: Path = SPLIT_FILE) -> set[str]:
    with open(split_file, newline="", encoding="utf-8") as f:
        return {r["patient_id"] for r in csv.DictReader(f) if r["split"] == "val"}


def load_predictions(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "path": str(Path(path).relative_to(REPO_ROOT)).replace("\\", "/") if Path(path).is_relative_to(REPO_ROOT)
        else str(path),
        "keys": [(r["recording_filename"], int(r["cycle_index"])) for r in rows],
        "patient": np.array([r["patient_id"] for r in rows]),
        "device": [r["device"] for r in rows],
        "labels": np.array([int(r["label"]) for r in rows]),
        "preds": np.array([int(r["pred"]) for r in rows]),
        "probs": np.array([[float(r[f"prob_{c}"]) for c in CLASS_NAMES] for r in rows]),
    }


def check_validation(d: dict, val_patients: set[str], n_expected: int | None = N_EXPECTED) -> None:
    """Abort unless ``d`` is exactly the frozen validation cycles with consistent predictions."""
    if set(d["patient"].tolist()) != val_patients:
        raise ValueError(f"{d['path']}: patients are not exactly the frozen validation patients")
    if n_expected is not None and len(d["labels"]) != n_expected:
        raise ValueError(f"{d['path']}: expected {n_expected} validation cycles, got {len(d['labels'])}")
    if len(set(d["keys"])) != len(d["keys"]):
        raise ValueError(f"{d['path']}: duplicate cycles")
    cm.check_inputs(d["probs"], d["labels"])
    if not np.array_equal(d["probs"].argmax(1), d["preds"]):
        raise ValueError(f"{d['path']}: saved predictions are not the argmax of the saved probabilities")


def same_cycles(a: dict, b: dict) -> None:
    if a["keys"] != b["keys"] or not np.array_equal(a["labels"], b["labels"]) or \
            not np.array_equal(a["patient"], b["patient"]):
        raise ValueError(f"{b['path']}: cycles, order, patients or labels differ from {a['path']}")


# ---------------------------------------------------------------- metric blocks

def calibration_block(p: np.ndarray, y: np.ndarray, bins: bool = True) -> dict:
    conf, pred = cm.confidence(p), cm.predictions(p)
    correct = pred == y
    s = cm.classification_summary(y, pred, K)
    out = {"nll": cm.nll(p, y), "brier": cm.brier(p, y), "ece": cm.ece(p, y),
           "ece_equal_mass": cm.ece_equal_mass(p, y), "classwise_ece": cm.classwise_ece(p, y), "mce": cm.mce(p, y),
           "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "mean_confidence": float(conf.mean()),
           "overconfidence_gap": float(conf.mean() - s["accuracy"]),
           "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else float("nan"),
           "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else float("nan"),
           "mean_entropy": float(cm.entropy(p).mean())}
    if bins:
        out["reliability_bins"] = cm.reliability_bins(conf, correct)
    return out


def uncertainty_scores(p: np.ndarray) -> dict[str, np.ndarray]:
    return {"1-msp": 1 - cm.confidence(p), "entropy": cm.entropy(p), "1-margin": 1 - cm.margin(p)}


def error_detection(p: np.ndarray, y: np.ndarray) -> dict:
    err = cm.predictions(p) != y
    return {"error_rate": float(err.mean()), **{
        s: {"auroc": cm.auroc(v, err), "auprc": cm.average_precision(v, err)}
        for s, v in uncertainty_scores(p).items()}}


def by_class(p: np.ndarray, y: np.ndarray) -> dict:
    pred, conf, ent = cm.predictions(p), cm.confidence(p), cm.entropy(p)
    err = pred != y
    out = {}
    for axis, arr in (("true_class", y), ("predicted_class", pred)):
        out[axis] = {}
        for k, name in enumerate(CLASS_NAMES):
            m = arr == k
            n = int(m.sum())
            out[axis][name] = {
                "n": n, "error_rate": float(err[m].mean()) if n else None,
                "mean_msp": float(conf[m].mean()) if n else None, "mean_entropy": float(ent[m].mean()) if n else None,
                "mean_msp_correct": float(conf[m & ~err].mean()) if (m & ~err).any() else None,
                "mean_msp_incorrect": float(conf[m & err].mean()) if (m & err).any() else None,
                "auroc_1-msp": cm.auroc(1 - conf[m], err[m]) if n and 0 < err[m].sum() < n else None}
    return out


def ranking_orders(p_raw: np.ndarray, p_lopo: np.ndarray) -> dict[str, np.ndarray]:
    return {"msp_raw": cm.retention_order(cm.confidence(p_raw)),
            "entropy_raw": cm.retention_order(cm.entropy(p_raw), higher_is_more_confident=False),
            "msp_lopo": cm.retention_order(cm.confidence(p_lopo))}


def macro_f1_curve(order: np.ndarray, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    conf_m = np.zeros((K, K))
    out = np.empty(len(order))
    for i, j in enumerate(order):
        conf_m[y[j], pred[j]] += 1
        tp, sup, prd = np.diag(conf_m), conf_m.sum(1), conf_m.sum(0)
        prec = np.divide(tp, prd, out=np.zeros(K), where=prd > 0)
        rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
        out[i] = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=prec + rec > 0).mean()
    return out


def clean_thresholds(msp_clean: np.ndarray, coverages=(0.8, 0.5)) -> dict[str, float]:
    order = cm.retention_order(msp_clean)
    return {f"tau_{round(c * 100)}": float(msp_clean[order[cm.n_retained(c, len(order)) - 1]]) for c in coverages}


def threshold_block(p: np.ndarray, y: np.ndarray, taus: dict[str, float]) -> dict:
    conf, wrong = cm.confidence(p), cm.predictions(p) != y
    out = {}
    for name, tau in taus.items():
        keep = conf >= tau
        out[name] = {"tau": tau, "realized_coverage": float(keep.mean()),
                     "selective_risk": float(wrong[keep].mean()) if keep.any() else float("nan"),
                     "confidently_wrong_share": float((wrong & keep).mean()),
                     "share_of_errors_confident": float(keep[wrong].mean()) if wrong.any() else float("nan")}
    return out


def family_rule(gaps: list[float], clean_ci: list[float]) -> str:
    above = any(g > clean_ci[1] for g in gaps)
    below = any(g < clean_ci[0] for g in gaps)
    return "mixed" if above and below else "becomes overconfident" if above else \
        "becomes underconfident" if below else "confidence tracks difficulty"


# ---------------------------------------------------------------- bootstrap

def bootstrap(clean: dict, p_lopo: np.ndarray, degraded: dict[str, dict], n_boot: int, seed: int,
              coverages: list[float]) -> dict[str, list[float]]:
    y, p_raw = clean["labels"], clean["probs"]
    draws = cm.patient_bootstrap_indices(clean["patient"].tolist(), n_boot, seed)
    stats: dict[str, list[float]] = {}

    def put(name, value):
        stats.setdefault(name, []).append(float(value))

    cal_keys = ("nll", "brier", "ece", "ece_equal_mass", "classwise_ece", "mce", "overconfidence_gap",
                "mean_confidence_correct", "mean_confidence_incorrect")
    for idx in draws:
        yb = y[idx]
        blocks = {"raw": calibration_block(p_raw[idx], yb, bins=False),
                  "lopo": calibration_block(p_lopo[idx], yb, bins=False)}
        for key in cal_keys:
            for w, b in blocks.items():
                put(f"calibration.{w}.{key}", b[key])
            put(f"calibration.lopo_minus_raw.{key}", blocks["lopo"][key] - blocks["raw"][key])
        for w, p in (("raw", p_raw[idx]), ("lopo", p_lopo[idx])):
            ed = error_detection(p, yb)
            for s in SCORES:
                put(f"error_detection.{w}.{s}.auroc", ed[s]["auroc"])
                put(f"error_detection.{w}.{s}.auprc", ed[s]["auprc"])
            put(f"error_detection.{w}.error_rate", ed["error_rate"])
        correct = (p_raw[idx].argmax(1) == yb)
        pred = p_raw[idx].argmax(1)
        for r, order in ranking_orders(p_raw[idx], p_lopo[idx]).items():
            rc = cm.risk_coverage(order, correct)
            put(f"selective.{r}.aurc", rc["aurc"])
            put(f"selective.{r}.e_aurc", rc["e_aurc"])
            full_risk = 1 - correct.mean()
            for c in coverages:
                s = cm.selective_at(order, yb, pred, c, K)
                for key in ("selective_risk", "accuracy", "macro_f1", "balanced_accuracy"):
                    put(f"selective.{r}.{c}.{key}", s[key])
                put(f"selective.{r}.{c}.risk_minus_full", s["selective_risk"] - full_risk)
        taus = clean_thresholds(cm.confidence(p_raw[idx]))
        for name, d in degraded.items():
            pd_raw, pd_cal = d["probs"][idx], d["probs_lopo"][idx]
            for w, p in (("raw", pd_raw), ("lopo", pd_cal)):
                b = calibration_block(p, yb, bins=False)
                for key in ("overconfidence_gap", "ece", "nll", "mean_confidence", "accuracy", "mean_entropy"):
                    put(f"degradation.{name}.{w}.{key}", b[key])
                err = p.argmax(1) != yb
                put(f"degradation.{name}.{w}.auroc_1-msp", cm.auroc(1 - cm.confidence(p), err))
                put(f"degradation.{name}.{w}.auroc_entropy", cm.auroc(cm.entropy(p), err))
            for tname, t in threshold_block(pd_raw, yb, taus).items():
                for key in ("realized_coverage", "selective_risk", "confidently_wrong_share"):
                    put(f"degradation.{name}.raw.{tname}.{key}", t[key])
    return stats


# ---------------------------------------------------------------- analysis

def analyze(clean: dict, degraded: dict[str, dict], plan: dict, n_boot: int = 2000, seed: int = 0) -> tuple[dict, dict]:
    y, p_raw, groups = clean["labels"], clean["probs"], clean["patient"]
    z = cm.logits_from_probs(p_raw)
    lopo = cm.lopo_temperatures(z, y, groups)
    t_row = np.array([lopo[g]["temperature"] for g in groups])
    p_lopo = cm.apply_temperature(z, t_row)
    if not np.array_equal(p_lopo.argmax(1), p_raw.argmax(1)):
        raise AssertionError("temperature scaling changed a prediction")
    final = cm.fit_temperature(z, y)
    p_final = cm.apply_temperature(z, final["temperature"])

    coverages = plan["part4_selective_prediction"]["coverage_levels"]
    pred, correct = p_raw.argmax(1), p_raw.argmax(1) == y
    orders = ranking_orders(p_raw, p_lopo)
    selective, curves = {}, {"n": np.arange(1, len(y) + 1)}
    for r, order in orders.items():
        rc = cm.risk_coverage(order, correct)
        selective[r] = {"aurc": rc["aurc"], "e_aurc": rc["e_aurc"],
                        "levels": [cm.selective_at(order, y, pred, c, K) for c in coverages]}
        curves[f"risk_{r}"] = rc["risk"]
        curves[f"macro_f1_{r}"] = macro_f1_curve(order, y, pred)
        curves["oracle_risk"] = rc["oracle_risk"]
    curves["coverage"] = curves["n"] / len(y)

    for name, d in degraded.items():
        same_cycles(clean, d)
        d["probs_lopo"] = cm.apply_temperature(cm.logits_from_probs(d["probs"]), t_row)
        if not np.array_equal(d["probs_lopo"].argmax(1), d["probs"].argmax(1)):
            raise AssertionError(f"{name}: temperature scaling changed a prediction")
    taus = clean_thresholds(cm.confidence(p_raw))
    deg = {}
    for name, d in degraded.items():
        deg[name] = {w: {**calibration_block(p, y, bins=False), "error_detection": error_detection(p, y)}
                     for w, p in (("raw", d["probs"]), ("lopo", d["probs_lopo"]))}
        deg[name]["raw"]["fixed_clean_thresholds"] = threshold_block(d["probs"], y, taus)

    t0 = time.time()
    boot = bootstrap(clean, p_lopo, degraded, n_boot, seed, coverages)
    ci = {k: cm.percentile_ci(v) for k, v in boot.items()}

    families = {}
    for fam, levels in plan["part5_degradation_uncertainty"]["family_order_mild_to_severe"].items():
        families[fam] = {"levels": levels}
        for w in ("raw", "lopo"):
            clean_ci = ci[f"degradation.clean.{w}.overconfidence_gap"]
            gaps = [deg[n][w]["overconfidence_gap"] for n in levels]
            c0 = deg["clean"][w]
            families[fam][w] = {
                "overconfidence_gaps": gaps, "clean_gap": c0["overconfidence_gap"], "clean_gap_ci95": clean_ci,
                "rule": family_rule(gaps, clean_ci),
                "mean_confidence": [deg[n][w]["mean_confidence"] for n in levels],
                "accuracy": [deg[n][w]["accuracy"] for n in levels],
                "confidence_moves_with_accuracy": [
                    bool(np.sign(deg[n][w]["mean_confidence"] - c0["mean_confidence"]) ==
                         np.sign(deg[n][w]["accuracy"] - c0["accuracy"])) for n in levels]}
        families[fam]["confidently_wrong_share_tau50"] = [
            deg[n]["raw"]["fixed_clean_thresholds"]["tau_50"]["confidently_wrong_share"] for n in levels]

    temps = [lopo[g]["temperature"] for g in sorted(lopo)]
    n_by_patient = {g: int((groups == g).sum()) for g in sorted(lopo)}
    summary = {
        "study": plan["study"], "model_run": plan["model"]["run"], "split": "val", "test_split_evaluated": False,
        "inputs": {"clean": clean["path"], "degraded": {n: d["path"] for n, d in degraded.items()}},
        "n_cycles": int(len(y)), "n_patients": len(lopo),
        "bootstrap": {"n_bootstrap": n_boot, "seed": seed, "unit": "patient", "elapsed_s": round(time.time() - t0, 1)},
        "temperature_scaling": {
            "procedure": "leave-one-patient-out on clean validation predictions",
            "label": "LOPO out-of-fold (development-set estimate)",
            "lopo": {g: {**lopo[g], "n_held_out": n_by_patient[g]} for g in sorted(lopo)},
            "lopo_summary": {"min": min(temps), "median": float(np.median(temps)), "max": max(temps),
                             "mean": float(np.mean(temps)), "sd": float(np.std(temps, ddof=1)),
                             "any_at_bound": any(lopo[g]["at_bound"] for g in lopo)},
            "final_validation_temperature": {**final, "purpose": "frozen parameter for a single later application "
                                             "to the untouched test set; not evidence of validation improvement"},
            "predictions_unchanged": True,
        },
        "part1_calibration_raw": calibration_block(p_raw, y),
        "part2_calibration_lopo": calibration_block(p_lopo, y),
        "part2_in_sample_final_temperature_descriptive_only": calibration_block(p_final, y, bins=False),
        "part3_error_detection": {"raw": error_detection(p_raw, y), "lopo": error_detection(p_lopo, y),
                                  "by_class_raw": by_class(p_raw, y)},
        "part4_selective_prediction": {"coverage_levels": coverages, "rankings": selective},
        "part5_degradation": {"fixed_clean_thresholds": taus, "conditions": deg, "families": families},
        "ci95": ci,
    }
    tables = {"p_lopo": p_lopo, "t_row": t_row, "curves": curves, "degraded": degraded}
    return summary, tables


# ---------------------------------------------------------------- outputs

def write_tables(out: Path, clean: dict, tables: dict, summary: dict) -> None:
    y, p_raw, p_lopo, t_row = clean["labels"], clean["probs"], tables["p_lopo"], tables["t_row"]
    ranks = {r: np.empty(len(y), dtype=int) for r in RANKINGS}
    for r, order in ranking_orders(p_raw, p_lopo).items():
        ranks[r][order] = np.arange(1, len(y) + 1)
    with open(out / "per_example_clean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["patient_id", "recording_filename", "cycle_index", "device", "label", "pred", "correct",
                    *(f"prob_{c}" for c in CLASS_NAMES), "lopo_temperature", *(f"prob_lopo_{c}" for c in CLASS_NAMES),
                    "msp", "entropy", "margin", "msp_lopo", "entropy_lopo", "margin_lopo",
                    *(f"retention_rank_{r}" for r in RANKINGS)])
        for i in range(len(y)):
            w.writerow([clean["patient"][i], *clean["keys"][i], clean["device"][i], y[i], clean["preds"][i],
                        int(clean["preds"][i] == y[i]), *p_raw[i].tolist(), t_row[i], *p_lopo[i].tolist(),
                        p_raw[i].max(), cm.entropy(p_raw[i:i + 1])[0], cm.margin(p_raw[i:i + 1])[0],
                        p_lopo[i].max(), cm.entropy(p_lopo[i:i + 1])[0], cm.margin(p_lopo[i:i + 1])[0],
                        *(ranks[r][i] for r in RANKINGS)])
    with open(out / "per_example_degraded.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["condition", "patient_id", "recording_filename", "cycle_index", "label", "pred", "correct",
                    "lopo_temperature", "msp", "entropy", "msp_lopo", "entropy_lopo"])
        for name, d in tables["degraded"].items():
            msp, ent = cm.confidence(d["probs"]), cm.entropy(d["probs"])
            msp_l, ent_l = cm.confidence(d["probs_lopo"]), cm.entropy(d["probs_lopo"])
            for i in range(len(y)):
                w.writerow([name, d["patient"][i], *d["keys"][i], y[i], d["preds"][i], int(d["preds"][i] == y[i]),
                            t_row[i], msp[i], ent[i], msp_l[i], ent_l[i]])
    with open(out / "lopo_temperatures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["held_out_patient", "n_held_out", "n_fit", "temperature", "fit_nll", "at_bound"])
        for g, v in summary["temperature_scaling"]["lopo"].items():
            w.writerow([g, v["n_held_out"], v["n"], v["temperature"], v["nll"], v["at_bound"]])
    c = tables["curves"]
    cols = ["n", "coverage", *(f"risk_{r}" for r in RANKINGS), "oracle_risk", *(f"macro_f1_{r}" for r in RANKINGS)]
    with open(out / "risk_coverage_curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for i in range(len(c["n"])):
            w.writerow([c[k][i].item() for k in cols])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_all(degraded_dir: Path = DEGRADED_DIR, clean_csv: Path = CLEAN_CSV, plan: dict | None = None,
             split_file: Path = SPLIT_FILE, n_expected: int | None = N_EXPECTED) -> tuple[dict, dict[str, dict]]:
    val = validation_patients(split_file)
    clean = load_predictions(clean_csv)
    check_validation(clean, val, n_expected)
    names = ["clean", *[n for levels in plan["part5_degradation_uncertainty"]["family_order_mild_to_severe"].values()
                        for n in levels]]
    degraded = {}
    for n in names:
        d = load_predictions(Path(degraded_dir) / n / "predictions.csv")
        check_validation(d, val, n_expected)
        degraded[n] = d
    if not np.array_equal(degraded["clean"]["probs"], clean["probs"]):
        raise ValueError("degradation-study clean probabilities differ from the recorded validation predictions")
    return clean, degraded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validation-only calibration / uncertainty / selective prediction study.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--render-only", action="store_true")
    a = p.parse_args(argv)
    import calibration_report as rep
    plan = json.loads((a.out_dir / "plan.json").read_text(encoding="utf-8"))
    if not a.render_only:
        clean, degraded = load_all(plan=plan)
        summary, tables = analyze(clean, degraded, plan, N_BOOT, BOOT_SEED)
        write_tables(a.out_dir, clean, tables, summary)
        (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n",
                                                encoding="utf-8")
    summary = json.loads((a.out_dir / "summary.json").read_text(encoding="utf-8"))
    rep.write_report(a.out_dir, summary)
    print(f"wrote {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
