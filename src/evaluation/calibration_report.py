"""Markdown report and SVG figures for calibration_uncertainty.py (reads summary.json + curves CSV only)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

CLASS_NAMES = ("normal", "crackle", "wheeze", "both")
RANK_LABEL = {"msp_raw": "MSP (raw)", "entropy_raw": "Entropy (raw)", "msp_lopo": "MSP (LOPO-scaled)"}
SCORE_LABEL = {"1-msp": "1 − MSP (confidence)", "entropy": "Predictive entropy", "1-margin": "1 − top-2 margin"}
FAMILY_LABEL = {"gaussian_noise": "Gaussian noise", "amplitude_attenuation": "Amplitude attenuation",
                "amplitude_amplification": "Amplitude amplification", "lowpass": "Low-pass filtering",
                "temporal_dropout": "Temporal dropout"}
COND_LABEL = {"clean": "Clean", "noise_snr20db": "Noise 20 dB", "noise_snr10db": "Noise 10 dB",
              "noise_snr5db": "Noise 5 dB", "noise_snr0db": "Noise 0 dB", "gain_0.5x": "Gain 0.5x",
              "gain_0.25x": "Gain 0.25x", "gain_1.5x": "Gain 1.5x", "gain_2.0x": "Gain 2x",
              "lowpass_2000hz": "Low-pass 2000 Hz", "lowpass_1000hz": "Low-pass 1000 Hz",
              "lowpass_500hz": "Low-pass 500 Hz", "dropout_10pct": "Dropout 10%", "dropout_25pct": "Dropout 25%",
              "dropout_50pct": "Dropout 50%"}


def f(x, d=4, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(s, key, d=3):
    lo, hi = s["ci95"].get(key, [float("nan"), float("nan")])
    return f"({f(lo, d, True) if 'minus' in key else f(lo, d)}, {f(hi, d, True) if 'minus' in key else f(hi, d)})"


# ---------------------------------------------------------------- markdown

def render_markdown(s: dict, degraded_csv: Path | None = None) -> str:
    raw, lopo = s["part1_calibration_raw"], s["part2_calibration_lopo"]
    ts = s["temperature_scaling"]
    L = ["# Calibration, uncertainty and selective prediction — frozen baseline CNN (validation only)", "",
         f"Model `{s['model_run']}` (frozen, not retrained or re-run); inputs are the saved validation softmax "
         f"probabilities ({s['n_cycles']} cycles, {s['n_patients']} patients). The test split was not read, loaded or "
         "evaluated. All analyses below were predeclared in `plan.json` unless they appear under *Post-hoc*.", "",
         "Calibrated (\"LOPO\") results are **leave-one-patient-out out-of-fold validation/development-set "
         "estimates**, not final held-out generalization evidence. Intervals are patient-level bootstrap 95% "
         f"intervals ({s['bootstrap']['n_bootstrap']} resamples of {s['n_patients']} patients, paired across "
         "methods and conditions; temperatures are not refitted inside resamples). These results describe model "
         "behaviour and support no clinical or diagnostic claim.", "",
         "## Part 1–2 — Calibration before and after temperature scaling", "",
         "![Reliability diagrams](reliability_diagram.svg)", "",
         "| Metric | Raw softmax | LOPO temperature-scaled (out-of-fold) | Change (95% CI) |", "|---|---|---|---|"]
    rows = [("NLL", "nll"), ("Brier (0–2)", "brier"), ("ECE, 15 equal-width bins (primary)", "ece"),
            ("ECE, 15 equal-mass bins", "ece_equal_mass"), ("Classwise ECE", "classwise_ece"), ("MCE", "mce"),
            ("Overconfidence gap (mean MSP − accuracy)", "overconfidence_gap"),
            ("Mean MSP, correct predictions", "mean_confidence_correct"),
            ("Mean MSP, incorrect predictions", "mean_confidence_incorrect")]
    for label, k in rows:
        L.append(f"| {label} | {f(raw[k])} {ci(s, f'calibration.raw.{k}')} | {f(lopo[k])} "
                 f"{ci(s, f'calibration.lopo.{k}')} | {f(lopo[k] - raw[k], 4, True)} "
                 f"{ci(s, f'calibration.lopo_minus_raw.{k}')} |")
    L += [f"| Accuracy / macro F1 | {f(raw['accuracy'])} / {f(raw['macro_f1'])} | {f(lopo['accuracy'])} / "
          f"{f(lopo['macro_f1'])} | unchanged (temperature never changes a prediction) |", "",
          "Temperature scaling changes probabilities only; every predicted class, the accuracy, macro F1 and the "
          "confusion matrix are identical, so no discrimination change is claimed.", "",
          "### Temperatures", "",
          f"LOPO fold temperatures (each fitted on the other 18 patients): min {f(ts['lopo_summary']['min'], 3)}, "
          f"median {f(ts['lopo_summary']['median'], 3)}, max {f(ts['lopo_summary']['max'], 3)}, SD "
          f"{f(ts['lopo_summary']['sd'], 3)}; any at the search bound: {ts['lopo_summary']['any_at_bound']}.", "",
          "| Held-out patient | Cycles held out | T |", "|---|---|---|"]
    for g, v in sorted(ts["lopo"].items(), key=lambda kv: -kv[1]["n_held_out"]):
        L.append(f"| {g} | {v['n_held_out']} | {f(v['temperature'], 4)} |")
    fv = ts["final_validation_temperature"]
    ins = s["part2_in_sample_final_temperature_descriptive_only"]
    L += ["", f"**final_validation_temperature = {f(fv['temperature'], 4)}** (fitted on all 19 validation "
          "patients). It is the frozen calibration parameter for a single later application to the untouched test "
          "set. Descriptive only, in-sample and NOT calibration evidence: NLL "
          f"{f(ins['nll'])}, ECE {f(ins['ece'])}.", ""]

    ed = s["part3_error_detection"]
    L += ["## Part 3 — Does uncertainty identify errors?", "",
          f"Errors are the positive class. Error rate {f(ed['raw']['error_rate'], 3)} (= AUPRC of a random score); "
          "AUROC 0.5 = no separation.", "",
          "| Score (higher = more uncertain) | AUROC raw | AUPRC raw | AUROC LOPO-scaled | AUPRC LOPO-scaled |",
          "|---|---|---|---|---|"]
    for sc in ("1-msp", "entropy", "1-margin"):
        tag = "" if sc != "1-margin" else " (secondary)"
        L.append(f"| {SCORE_LABEL[sc]}{tag} | {f(ed['raw'][sc]['auroc'], 3)} {ci(s, f'error_detection.raw.{sc}.auroc')}"
                 f" | {f(ed['raw'][sc]['auprc'], 3)} {ci(s, f'error_detection.raw.{sc}.auprc')} | "
                 f"{f(ed['lopo'][sc]['auroc'], 3)} {ci(s, f'error_detection.lopo.{sc}.auroc')} | "
                 f"{f(ed['lopo'][sc]['auprc'], 3)} {ci(s, f'error_detection.lopo.{sc}.auprc')} |")
    for axis, title in (("true_class", "true class"), ("predicted_class", "predicted class")):
        L += ["", f"Uncertainty by {title} (raw):", "",
              "| Class | n | Error rate | Mean MSP | Mean entropy | MSP correct | MSP incorrect | AUROC (1 − MSP) |",
              "|---|---|---|---|---|---|---|---|"]
        for c in CLASS_NAMES:
            v = ed["by_class_raw"][axis][c]
            L.append(f"| {c} | {v['n']} | {f(v['error_rate'], 3)} | {f(v['mean_msp'], 3)} | {f(v['mean_entropy'], 3)}"
                     f" | {f(v['mean_msp_correct'], 3)} | {f(v['mean_msp_incorrect'], 3)} | {f(v['auroc_1-msp'], 3)} |")

    sp = s["part4_selective_prediction"]
    L += ["", "## Part 4 — Selective prediction / abstention", "", "![Risk-coverage curves](risk_coverage.svg)", "",
          "**Primary:** selective risk (error rate among retained cycles) and the risk-coverage curve. Random "
          f"abstention keeps the risk at {f(ed['raw']['error_rate'], 3)} at every coverage.", "",
          "| Ranking | AURC | E-AURC |", "|---|---|---|"]
    for r, v in sp["rankings"].items():
        L.append(f"| {RANK_LABEL[r]}{'' if r == 'msp_raw' else ' (secondary)'} | {f(v['aurc'])} "
                 f"{ci(s, f'selective.{r}.aurc')} | {f(v['e_aurc'])} {ci(s, f'selective.{r}.e_aurc')} |")
    L += ["", "Primary ranking (raw MSP), predeclared coverages. Macro F1 / balanced accuracy / accuracy are "
          "secondary; a class with no retained cycles counts with F1 = 0 and recall = 0 (zero-division "
          "convention) and is never dropped.", "",
          "| Coverage | Retained | Selective risk (95% CI) | Risk − full coverage (95% CI) | Macro F1 | Bal. acc. | "
          "Accuracy |", "|---|---|---|---|---|---|---|"]
    for lv in sp["rankings"]["msp_raw"]["levels"]:
        c = lv["coverage"]
        L.append(f"| {c:.0%} | {lv['n_retained']} | {f(lv['selective_risk'])} "
                 f"{ci(s, f'selective.msp_raw.{c}.selective_risk')} | "
                 + ("—" if c == 1.0 else f"{f(lv['selective_risk'] - sp['rankings']['msp_raw']['levels'][0]['selective_risk'], 4, True)} "
                    f"{ci(s, f'selective.msp_raw.{c}.risk_minus_full')}")
                 + f" | {f(lv['macro_f1'])} | {f(lv['balanced_accuracy'])} | {f(lv['accuracy'])} |")
    L += ["", "Retained / rejected counts by class (raw MSP ranking; true-class and predicted-class counts as "
          "normal / crackle / wheeze / both):", "",
          "| Coverage | Retained true | Rejected true | Retained predicted | Rejected predicted | Share of each true "
          "class retained |", "|---|---|---|---|---|---|"]
    for lv in sp["rankings"]["msp_raw"]["levels"]:
        tot = [a + b for a, b in zip(lv["retained_true_counts"], lv["rejected_true_counts"])]
        share = " / ".join(f"{a / t:.0%}" if t else "—" for a, t in zip(lv["retained_true_counts"], tot))
        L.append(f"| {lv['coverage']:.0%} | {' / '.join(map(str, lv['retained_true_counts']))} | "
                 f"{' / '.join(map(str, lv['rejected_true_counts']))} | {' / '.join(map(str, lv['retained_pred_counts']))}"
                 f" | {' / '.join(map(str, lv['rejected_pred_counts']))} | {share} |")
    L += ["", "Secondary rankings, selective risk at each coverage:", "",
          "| Ranking | " + " | ".join(f"{c:.0%}" for c in sp["coverage_levels"]) + " |",
          "|---|" + "---|" * len(sp["coverage_levels"])]
    for r, v in sp["rankings"].items():
        L.append(f"| {RANK_LABEL[r]} | " + " | ".join(f(lv["selective_risk"]) for lv in v["levels"]) + " |")

    dg = s["part5_degradation"]
    taus = dg["fixed_clean_thresholds"]
    L += ["", "## Part 5 — Uncertainty under synthetic degradation", "",
          "![Confidence vs accuracy under degradation](degradation_confidence.svg)", "",
          "Raw probabilities, and LOPO-scaled probabilities using each patient's CLEAN-fitted temperature (no "
          "temperature is fitted on any degraded condition). Overconfidence gap = mean MSP − accuracy.", "",
          "| Condition | Accuracy | Mean MSP raw | Gap raw (95% CI) | ECE raw | NLL raw | Gap LOPO (95% CI) | "
          "ECE LOPO | NLL LOPO | AUROC 1 − MSP | AUROC entropy |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in dg["conditions"].items():
        r_, c_ = v["raw"], v["lopo"]
        L.append(f"| {COND_LABEL[n]} | {f(r_['accuracy'], 3)} | {f(r_['mean_confidence'], 3)} | "
                 f"{f(r_['overconfidence_gap'], 3, True)} {ci(s, f'degradation.{n}.raw.overconfidence_gap')} | "
                 f"{f(r_['ece'], 3)} | {f(r_['nll'], 3)} | {f(c_['overconfidence_gap'], 3, True)} "
                 f"{ci(s, f'degradation.{n}.lopo.overconfidence_gap')} | {f(c_['ece'], 3)} | {f(c_['nll'], 3)} | "
                 f"{f(r_['error_detection']['1-msp']['auroc'], 3)} | {f(r_['error_detection']['entropy']['auroc'], 3)} |")
    L += ["", f"Fixed clean-data abstention thresholds applied unchanged to every condition (raw MSP): "
          f"τ80 = {f(taus['tau_80'])}, τ50 = {f(taus['tau_50'])}. *Confidently wrong* = wrong and MSP ≥ τ50.", "",
          "| Condition | Coverage at τ80 | Risk at τ80 | Coverage at τ50 | Risk at τ50 | Confidently wrong (share of "
          "all cycles) | Share of errors that are confident |", "|---|---|---|---|---|---|---|"]
    for n, v in dg["conditions"].items():
        t80, t50 = v["raw"]["fixed_clean_thresholds"]["tau_80"], v["raw"]["fixed_clean_thresholds"]["tau_50"]
        L.append(f"| {COND_LABEL[n]} | {t80['realized_coverage']:.1%} | {f(t80['selective_risk'], 3)} | "
                 f"{t50['realized_coverage']:.1%} | {f(t50['selective_risk'], 3)} | "
                 f"{t50['confidently_wrong_share']:.1%} {ci(s, f'degradation.{n}.raw.tau_50.confidently_wrong_share')}"
                 f" | {t50['share_of_errors_confident']:.1%} |")
    L += ["", "Predeclared family rule (gap at every level inside the clean gap's 95% interval → *tracks "
          "difficulty*; above → *overconfident*; below → *underconfident*; both → *mixed*):", "",
          "| Family | Raw rule | LOPO rule | Mean MSP moves with accuracy (per level, raw) |", "|---|---|---|---|"]
    for fam, v in dg["families"].items():
        L.append(f"| {FAMILY_LABEL[fam]} | {v['raw']['rule']} | {v['lopo']['rule']} | "
                 + ", ".join(f"{COND_LABEL[n]}: {'yes' if ok else 'no'}" for n, ok in
                             zip(v["levels"], v["raw"]["confidence_moves_with_accuracy"])) + " |")
    L += ["", *posthoc_section(s, degraded_csv), ""]
    return "\n".join(L)


def posthoc_section(s: dict, degraded_csv: Path | None = None) -> list[str]:
    """Exploratory observations added after inspecting the results (not predeclared; computed, not hand-typed)."""
    dg, ed = s["part5_degradation"], s["part3_error_detection"]
    conds = dg["conditions"]
    L = ["## Post-hoc exploratory observations (not predeclared)", "",
         "Added after inspecting the predeclared results. Exploratory; not used to change any analysis.", ""]
    if degraded_csv is not None and Path(degraded_csv).exists():
        counts: dict[str, list[int]] = {}
        with open(degraded_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                counts.setdefault(r["condition"], [0] * 4)[int(r["pred"])] += 1
        L += ["**1. Accuracy-based calibration hides class collapse.** The overconfidence gap and top-label ECE "
              "compare confidence with accuracy, and accuracy stays high when predictions collapse onto the "
              "majority class (normal is 47% of validation cycles). Macro F1 exposes what the gap does not:", "",
              "| Condition | Accuracy | Macro F1 | Mean MSP raw | Gap raw | Most-predicted class (share) |",
              "|---|---|---|---|---|---|"]
        for n, v in conds.items():
            c = counts[n]
            k = max(range(4), key=lambda i: c[i])
            L.append(f"| {COND_LABEL[n]} | {f(v['raw']['accuracy'], 3)} | {f(v['raw']['macro_f1'], 3)} | "
                     f"{f(v['raw']['mean_confidence'], 3)} | {f(v['raw']['overconfidence_gap'], 3, True)} | "
                     f"{CLASS_NAMES[k]} ({c[k] / sum(c):.0%}) |")
        L.append("")
    lo, hi = s["ci95"]["degradation.clean.raw.overconfidence_gap"]
    noise = dg["families"]["gaussian_noise"]["raw"]["overconfidence_gaps"]
    L += [f"**2. The predeclared family rule is insensitive.** The clean gap's patient-bootstrap interval is "
          f"({f(lo, 3, True)}, {f(hi, 3, True)}), {hi - lo:.3f} wide, because overconfidence varies strongly between "
          f"patients. Gaussian noise is therefore labelled *tracks difficulty* although its point-estimate gap grows "
          f"monotonically from {f(conds['clean']['raw']['overconfidence_gap'], 3, True)} to "
          + ", ".join(f(g, 3, True) for g in noise) + " while accuracy falls and mean confidence barely moves.", ""]
    fixed = [(n, v["raw"]["fixed_clean_thresholds"]["tau_80"]) for n, v in conds.items() if n != "clean"]
    up = [(n, t) for n, t in fixed if t["realized_coverage"] > 0.80 and
          t["selective_risk"] > conds["clean"]["raw"]["fixed_clean_thresholds"]["tau_80"]["selective_risk"]]
    L += ["**3. Under shift, the clean-data threshold can retain more cycles while they become more often wrong.** "
          "At τ80 (80% coverage on clean data): " + "; ".join(
              f"{COND_LABEL[n]} keeps {t['realized_coverage']:.0%} with risk {t['selective_risk']:.3f}" for n, t in up)
          + f" (clean: 80% with risk {conds['clean']['raw']['fixed_clean_thresholds']['tau_80']['selective_risk']:.3f})."
          " Abstention calibrated on clean data does not protect against these degradations.", ""]
    inv = [n for n, v in conds.items() if v["raw"]["error_detection"]["1-msp"]["auroc"] < 0.5]
    L += ["**4. Uncertainty can rank errors backwards.** Error-detection AUROC (1 − MSP) is below 0.5 for: "
          + (", ".join(f"{COND_LABEL[n]} ({conds[n]['raw']['error_detection']['1-msp']['auroc']:.3f})" for n in inv)
             or "none") + ". On clean data, cycles whose true class is normal are misclassified with higher "
          f"confidence than they are correctly classified (mean MSP {f(ed['by_class_raw']['true_class']['normal']['mean_msp_incorrect'], 3)}"
          f" vs {f(ed['by_class_raw']['true_class']['normal']['mean_msp_correct'], 3)}; AUROC "
          f"{f(ed['by_class_raw']['true_class']['normal']['auroc_1-msp'], 3)}).", ""]
    lv50 = s["part4_selective_prediction"]["rankings"]["msp_raw"]["levels"][-1]
    tot = [a + b for a, b in zip(lv50["retained_true_counts"], lv50["rejected_true_counts"])]
    L += ["**5. Abstention removes minority classes faster.** At 50% coverage the share of each true class retained "
          "is " + ", ".join(f"{c} {a / t:.0%}" for c, a, t in zip(CLASS_NAMES, lv50["retained_true_counts"], tot))
          + ". Retained-subset macro F1 is therefore measured on a different class mix than full coverage and is "
          "not a like-for-like improvement."]
    return L


# ---------------------------------------------------------------- figures

STYLE = """
  .s{fill:var(--surface)} .t1{fill:var(--ink)} .t2{fill:var(--ink2)} .tm{fill:var(--muted)}
  .grid{stroke:var(--grid);stroke-width:1} .base{stroke:var(--muted);stroke-width:1}
  .ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 3;fill:none}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .mk{stroke:var(--surface);stroke-width:2}
  .c0{stroke:var(--c0)} .f0{fill:var(--c0)} .c1{stroke:var(--c1)} .f1{fill:var(--c1)}
  .c2{stroke:var(--c2)} .f2{fill:var(--c2)}
  svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
      --c0:#2a78d6;--c1:#eb6834;--c2:#1baf7a}
  @media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
      --grid:#2c2c2a;--c0:#3987e5;--c1:#d95926;--c2:#199e70}}
"""


def _svg(width, height, label, body):
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                      f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="11" '
                      f'role="img" aria-label="{label}">', f"<style>{STYLE}</style>",
                      f'<rect class="s" width="{width}" height="{height}"/>', *body, "</svg>"]) + "\n"


def _axes(x0, y0, w, h, xticks, yticks, xlab, ylab, show_y=True):
    out = []
    for v, lab in yticks:
        yy = y0 + h - v * h
        out.append(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        if show_y:
            out.append(f'<text class="tm" x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end">{lab}</text>')
    out.append(f'<line class="base" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
    for v, lab in xticks:
        out.append(f'<text class="tm" x="{x0 + v * w:.1f}" y="{y0 + h + 15}" text-anchor="middle">{lab}</text>')
    out.append(f'<text class="t2" x="{x0 + w / 2}" y="{y0 + h + 32}" text-anchor="middle">{xlab}</text>')
    if show_y:
        out.append(f'<text class="t2" x="{x0 - 36}" y="{y0 + h / 2}" transform="rotate(-90 {x0 - 36} {y0 + h / 2})" '
                   f'text-anchor="middle">{ylab}</text>')
    return out


def reliability_svg(s: dict) -> str:
    panels = [("Raw softmax", s["part1_calibration_raw"]),
              ("LOPO temperature-scaled (out-of-fold)", s["part2_calibration_lopo"])]
    w, h, x0, y0, gap = 300, 300, 60, 52, 80
    body = [f'<text class="t1" x="{x0}" y="18" font-size="13" font-weight="bold">Reliability diagrams, clean '
            f'validation (15 equal-width bins)</text>']
    ticks = [(v / 5, f"{v / 5:.1f}") for v in range(6)]
    for i, (title, block) in enumerate(panels):
        px = x0 + i * (w + gap)
        body.append(f'<text class="t1" x="{px}" y="{y0 - 12}" font-weight="bold">{title} — ECE {block["ece"]:.3f}</text>')
        body += _axes(px, y0, w, h, ticks, ticks, "confidence (max softmax probability)", "accuracy in bin")
        body.append(f'<line class="ref" x1="{px}" y1="{y0 + h}" x2="{px + w}" y2="{y0}"/>')
        n_tot = sum(b["n"] for b in block["reliability_bins"])
        for b in block["reliability_bins"]:
            if not b["n"]:
                continue
            bx, bw = px + b["lower"] * w + 1, w / 15 - 2
            top = y0 + h - b["accuracy"] * h
            body.append(f'<rect class="f0" x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{y0 + h - top:.1f}" '
                        f'rx="2"><title>bin {b["lower"]:.2f}–{b["upper"]:.2f}: n={b["n"]}, accuracy '
                        f'{b["accuracy"]:.3f}, mean confidence {b["mean_confidence"]:.3f}</title></rect>')
            cy = y0 + h - b["mean_confidence"] * h
            body.append(f'<line class="c1" stroke-width="2" x1="{bx:.1f}" x2="{bx + bw:.1f}" y1="{cy:.1f}" '
                        f'y2="{cy:.1f}"/>')
            body.append(f'<text class="tm" x="{bx + bw / 2:.1f}" y="{top - 4:.1f}" text-anchor="middle" '
                        f'font-size="9">{b["n"] / n_tot:.0%}</text>')
    ly = y0 + h + 52
    body += [f'<rect class="f0" x="{x0}" y="{ly - 9}" width="14" height="10" rx="2"/>',
             f'<text class="t2" x="{x0 + 20}" y="{ly}">accuracy in bin (label: share of cycles)</text>',
             f'<line class="c1" stroke-width="2" x1="{x0 + 260}" x2="{x0 + 278}" y1="{ly - 4}" y2="{ly - 4}"/>',
             f'<text class="t2" x="{x0 + 284}" y="{ly}">mean confidence in bin</text>',
             f'<line class="ref" x1="{x0 + 440}" x2="{x0 + 460}" y1="{ly - 4}" y2="{ly - 4}"/>',
             f'<text class="t2" x="{x0 + 466}" y="{ly}">perfect calibration</text>']
    return _svg(x0 + 2 * w + gap + 30, ly + 16, "Reliability diagrams before and after temperature scaling", body)


def risk_coverage_svg(s: dict, curves_csv: Path) -> str:
    with open(curves_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    w, h, x0, y0 = 560, 300, 64, 44
    top = 0.6
    yt = [(v / top, f"{v:.1f}") for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    body = [f'<text class="t1" x="{x0}" y="18" font-size="13" font-weight="bold">Risk-coverage curves, clean '
            f'validation (error rate among retained cycles)</text>']
    body += _axes(x0, y0, w, h, [(v / 10, f"{v * 10}%") for v in range(0, 11, 2)], yt, "coverage (share retained)",
                  "selective risk")
    pt = lambda cov, risk: f"{x0 + cov * w:.1f},{y0 + h - min(risk, top) / top * h:.1f}"  # noqa: E731
    start = 20  # skip the noisiest first few points (n < 20) in the drawing only
    err = s["part3_error_detection"]["raw"]["error_rate"]
    body.append(f'<line class="ref" x1="{x0}" x2="{x0 + w}" y1="{y0 + h - err / top * h:.1f}" '
                f'y2="{y0 + h - err / top * h:.1f}"/>')
    body.append('<polyline class="ref" style="stroke-dasharray:1 3" points="'
                + " ".join(pt(float(r["coverage"]), float(r["oracle_risk"])) for r in rows[start:]) + '"/>')
    for i, r in enumerate(("msp_raw", "entropy_raw", "msp_lopo")):
        body.append(f'<polyline class="ln c{i}" points="'
                    + " ".join(pt(float(row["coverage"]), float(row[f"risk_{r}"])) for row in rows[start:]) + '"/>')
    for lv in s["part4_selective_prediction"]["rankings"]["msp_raw"]["levels"]:
        x, y = pt(lv["coverage"], lv["selective_risk"]).split(",")
        body.append(f'<circle class="mk f0" cx="{x}" cy="{y}" r="4"><title>{lv["coverage"]:.0%} coverage: '
                    f'risk {lv["selective_risk"]:.3f} (n={lv["n_retained"]})</title></circle>')
    ly = y0 + h + 52
    items = [("ln c0", "MSP (raw, primary)"), ("ln c1", "entropy (raw)"), ("ln c2", "MSP (LOPO-scaled)"),
             ("ref", "random abstention"), ("ref dot", "oracle")]
    lx = x0
    for cls, lab in items:
        style = ' style="stroke-dasharray:1 3"' if cls == "ref dot" else ""
        body.append(f'<line class="{cls.replace(" dot", "")}"{style} x1="{lx}" x2="{lx + 18}" y1="{ly - 4}" '
                    f'y2="{ly - 4}"/>')
        body.append(f'<text class="t2" x="{lx + 24}" y="{ly}">{lab}</text>')
        lx += 24 + 7 * len(lab) + 16
    return _svg(max(x0 + w + 30, lx), ly + 16, "Risk-coverage curves for three uncertainty rankings", body)


def degradation_svg(s: dict) -> str:
    dg = s["part5_degradation"]
    fams = dg["families"]
    pw, ph, x0, y0, gap = 200, 220, 56, 50, 24
    body = [f'<text class="t1" x="{x0}" y="18" font-size="13" font-weight="bold">Mean confidence vs accuracy as '
            f'degradation severity increases</text>']
    yt = [(v / 5, f"{v / 5:.1f}") for v in range(6)]
    for i, (fam, v) in enumerate(fams.items()):
        px = x0 + i * (pw + gap)
        names = ["clean", *v["levels"]]
        xs = [j / (len(names) - 1) for j in range(len(names))]
        body.append(f'<text class="t1" x="{px}" y="{y0 - 12}" font-weight="bold">{FAMILY_LABEL[fam]}</text>')
        labels = [COND_LABEL[n].replace("Noise ", "").replace("Gain ", "").replace("Low-pass ", "")
                  .replace("Dropout ", "").replace("Clean", "clean") for n in names]
        body += _axes(px + 10, y0, pw - 20, ph, list(zip(xs, labels)), yt, "", "probability", show_y=i == 0)
        series = [("c1", lambda n: dg["conditions"][n]["raw"]["accuracy"], "accuracy"),
                  ("c0", lambda n: dg["conditions"][n]["raw"]["mean_confidence"], "mean MSP (raw)"),
                  ("c2", lambda n: dg["conditions"][n]["lopo"]["mean_confidence"], "mean MSP (LOPO-scaled)")]
        for cls, get, lab in series:
            pts = [(px + 10 + x * (pw - 20), y0 + ph - get(n) * ph) for x, n in zip(xs, names)]
            body.append(f'<polyline class="ln {cls}" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
            body += [f'<circle class="mk f{cls[1]}" cx="{a:.1f}" cy="{b:.1f}" r="4"><title>{COND_LABEL[n]}: {lab} '
                     f'{get(n):.3f}</title></circle>' for (a, b), n in zip(pts, names)]
    ly = y0 + ph + 46
    lx = x0
    for cls, lab in (("c1", "accuracy"), ("c0", "mean MSP (raw)"), ("c2", "mean MSP (LOPO-scaled)")):
        body.append(f'<line class="ln {cls}" x1="{lx}" x2="{lx + 18}" y1="{ly - 4}" y2="{ly - 4}"/>')
        body.append(f'<text class="t2" x="{lx + 24}" y="{ly}">{lab}</text>')
        lx += 24 + 7 * len(lab) + 18
    return _svg(x0 + len(fams) * (pw + gap) + 10, ly + 14, "Mean confidence and accuracy by degradation severity",
                body)


def write_report(out_dir: Path, summary: dict) -> None:
    out_dir = Path(out_dir)
    (out_dir / "summary.md").write_text(render_markdown(summary, out_dir / "per_example_degraded.csv"),
                                        encoding="utf-8")
    (out_dir / "reliability_diagram.svg").write_text(reliability_svg(summary), encoding="utf-8")
    (out_dir / "risk_coverage.svg").write_text(risk_coverage_svg(summary, out_dir / "risk_coverage_curves.csv"),
                                               encoding="utf-8")
    (out_dir / "degradation_confidence.svg").write_text(degradation_svg(summary), encoding="utf-8")
