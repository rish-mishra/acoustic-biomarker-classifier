"""Figures for reports/final_report/ built only from already-saved summaries (no inference, no recomputation).

Inputs (read-only):
  reports/runs/seed_study/summary.json
  reports/robustness/noise_robustness/summary.json            validation robustness
  reports/final_evaluation/robustness_test/summary.json        locked test robustness
  reports/final_evaluation/clean_test/metrics.json             locked clean test
  reports/calibration_uncertainty/risk_coverage_curves.csv     validation risk-coverage
  reports/final_evaluation/clean_test/risk_coverage_curves.csv locked test risk-coverage

Outputs: reports/final_report/figures/{seed_comparison,robustness_validation_vs_test,test_confusion_matrix,
risk_coverage_validation_vs_test}.svg

Usage::

    python src/evaluation/final_report_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR / "evaluation") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "evaluation"))

import calibration_report as crep  # noqa: E402

REPO_ROOT = SRC_DIR.parent
OUT_DIR = REPO_ROOT / "reports" / "final_report" / "figures"
CLASS_NAMES = ("normal", "crackle", "wheeze", "both")
FAMILIES = {"gaussian_noise": ["noise_snr20db", "noise_snr10db", "noise_snr5db", "noise_snr0db"],
            "amplitude_attenuation": ["gain_0.5x", "gain_0.25x"],
            "amplitude_amplification": ["gain_1.5x", "gain_2.0x"],
            "lowpass": ["lowpass_2000hz", "lowpass_1000hz", "lowpass_500hz"],
            "temporal_dropout": ["dropout_10pct", "dropout_25pct", "dropout_50pct"]}
# reference sequential blue ramp (light -> dark), steps 100..700
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]


def J(rel: str) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def short(name: str) -> str:
    return crep.COND_LABEL[name].replace("Noise ", "").replace("Gain ", "").replace("Low-pass ", "") \
        .replace("Dropout ", "").replace("Clean", "clean")


def legend(items, x, y):
    out = []
    for cls, lab, dashed in items:
        style = ' style="stroke-dasharray:4 3"' if dashed else ""
        out.append(f'<line class="{cls}"{style} x1="{x}" x2="{x + 18}" y1="{y - 4}" y2="{y - 4}"/>')
        out.append(f'<text class="t2" x="{x + 24}" y="{y}">{lab}</text>')
        x += 24 + 7 * len(lab) + 18
    return out, x


def seed_comparison_svg(seed: dict) -> str:
    arch = list(seed["per_architecture"].items())
    labels = {"baseline_cnn": "Baseline CNN (295k)", "crnn": "CRNN (353k)",
              "mobilenetv3_pretrained": "MobileNetV3, pretrained (1.52M)"}
    x0, y0, w, rowh = 230, 50, 420, 46
    lo, hi = 0.36, 0.46
    xs = lambda v: x0 + (v - lo) / (hi - lo) * w  # noqa: E731
    h = rowh * len(arch)
    body = [f'<text class="t1" x="16" y="20" font-size="13" font-weight="bold">Validation macro F1 by architecture '
            f'and seed (42 / 1337 / 2026)</text>']
    for t in (0.36, 0.38, 0.40, 0.42, 0.44, 0.46):
        body.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{y0}" y2="{y0 + h}"/>')
        body.append(f'<text class="tm" x="{xs(t):.1f}" y="{y0 + h + 16}" text-anchor="middle">{t:.2f}</text>')
    body.append(f'<text class="t2" x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle">validation macro F1 '
                f'(best epoch)</text>')
    for i, (a, p) in enumerate(arch):
        cy = y0 + rowh * i + rowh / 2
        m = p["macro_f1"]
        body.append(f'<text class="t1" x="{x0 - 12}" y="{cy + 4:.1f}" text-anchor="end">{labels.get(a, a)}</text>')
        body.append(f'<line class="base" x1="{xs(m["min"]):.1f}" x2="{xs(m["max"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
        for s, v in zip(seed["seeds"], m["values"]):
            body.append(f'<circle class="mk f0" cx="{xs(v):.1f}" cy="{cy:.1f}" r="5"><title>{a}, seed {s}: '
                        f'{v:.4f}</title></circle>')
        body.append(f'<line class="c1" stroke-width="3" x1="{xs(m["mean"]):.1f}" x2="{xs(m["mean"]):.1f}" '
                    f'y1="{cy - 11:.1f}" y2="{cy + 11:.1f}"><title>mean {m["mean"]:.4f} ± {m["sd"]:.4f} (SD)</title></line>')
        body.append(f'<text class="t2" x="{x0 + w + 12}" y="{cy + 4:.1f}">{m["mean"]:.3f} ± {m["sd"]:.3f}</text>')
    ly = y0 + h + 58
    body += [f'<circle class="mk f0" cx="{x0 + 6}" cy="{ly - 4}" r="5"/>', f'<text class="t2" x="{x0 + 16}" y="{ly}">one seed</text>',
             f'<line class="c1" stroke-width="3" x1="{x0 + 96}" x2="{x0 + 96}" y1="{ly - 12}" y2="{ly + 4}"/>',
             f'<text class="t2" x="{x0 + 106}" y="{ly}">mean (label: mean ± SD over seeds)</text>']
    return crep._svg(x0 + w + 110, ly + 16, "Validation macro F1 by architecture and seed", body)


def robustness_svg(val: dict, test: dict) -> str:
    pw, ph, x0, y0, gap = 200, 210, 56, 52, 24
    body = [f'<text class="t1" x="{x0}" y="20" font-size="13" font-weight="bold">Macro F1 under the 15 frozen '
            f'degradations: validation (development) vs untouched test</text>']
    yt = [(v / 0.5, f"{v:.1f}") for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    for i, (fam, levels) in enumerate(FAMILIES.items()):
        px = x0 + i * (pw + gap)
        names = ["clean", *levels]
        xs = [j / (len(names) - 1) for j in range(len(names))]
        body.append(f'<text class="t1" x="{px}" y="{y0 - 12}" font-weight="bold">{crep.FAMILY_LABEL[fam]}</text>')
        body += crep._axes(px + 10, y0, pw - 20, ph, list(zip(xs, [short(n) for n in names])), yt, "", "macro F1",
                           show_y=i == 0)
        for cls, src, lab in (("c0", val, "validation"), ("c1", test, "test")):
            pts = [(px + 10 + x * (pw - 20), y0 + ph - src["conditions"][n]["macro_f1"] / 0.5 * ph) for x, n in zip(xs, names)]
            body.append(f'<polyline class="ln {cls}" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
            body += [f'<circle class="mk f{cls[1]}" cx="{a:.1f}" cy="{b:.1f}" r="4"><title>{crep.COND_LABEL[n]}, {lab}: '
                     f'{src["conditions"][n]["macro_f1"]:.3f}</title></circle>' for (a, b), n in zip(pts, names)]
    ly = y0 + ph + 48
    items, _ = legend([("ln c0", "validation (development, 19 patients)", False),
                       ("ln c1", "untouched test (19 other patients)", False)], x0, ly)
    return crep._svg(x0 + len(FAMILIES) * (pw + gap) + 10, ly + 14,
                     "Macro F1 under degradations, validation vs test", body + items)


def confusion_svg(metrics: dict) -> str:
    cm = metrics["secondary"]["confusion_matrix"]
    cell, x0, y0 = 74, 150, 92
    body = [f'<text class="t1" x="16" y="20" font-size="13" font-weight="bold">Clean test confusion matrix '
            f'(frozen baseline_cnn_v3, {metrics["n_cycles"]} cycles, {metrics["n_patients"]} patients)</text>',
            f'<text class="t2" x="16" y="40">Cell shade = share of the true class (row); number = cycles. '
            f'Macro F1 {metrics["primary"]["macro_f1"]:.4f}.</text>',
            f'<text class="t2" x="{x0 + 2 * cell}" y="{y0 - 30}" text-anchor="middle">predicted class</text>',
            f'<text class="t2" x="26" y="{y0 + 2 * cell}" transform="rotate(-90 26 {y0 + 2 * cell})" '
            f'text-anchor="middle">true class</text>']
    for j, c in enumerate(CLASS_NAMES):
        body.append(f'<text class="t1" x="{x0 + j * cell + cell / 2}" y="{y0 - 10}" text-anchor="middle">{c}</text>')
    for i, row in enumerate(cm):
        tot = sum(row)
        body.append(f'<text class="t1" x="{x0 - 10}" y="{y0 + i * cell + cell / 2 + 4}" text-anchor="end">{CLASS_NAMES[i]} '
                    f'(n={tot})</text>')
        for j, v in enumerate(row):
            share = v / tot if tot else 0.0
            step = min(len(SEQ) - 1, round(share * (len(SEQ) - 1)))
            ink = "#ffffff" if step >= 6 else "#0b0b0b"
            x, y = x0 + j * cell, y0 + i * cell
            body.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="3" fill="{SEQ[step]}">'
                        f'<title>true {CLASS_NAMES[i]}, predicted {CLASS_NAMES[j]}: {v} ({share:.0%} of row)</title></rect>')
            body.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 2}" text-anchor="middle" fill="{ink}" '
                        f'font-size="14" font-weight="{"bold" if i == j else "normal"}">{v}</text>')
            body.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 18}" text-anchor="middle" fill="{ink}" '
                        f'font-size="10">{share:.0%}</text>')
    return crep._svg(max(x0 + 4 * cell + 30, 600), y0 + 4 * cell + 24, "Clean test confusion matrix", body)


def read_curve(rel: str, col: str) -> list[tuple[float, float]]:
    with open(REPO_ROOT / rel, newline="", encoding="utf-8") as f:
        return [(float(r["coverage"]), float(r[col])) for r in csv.DictReader(f)]


def risk_coverage_svg(val_err: float, test_err: float) -> str:
    val = read_curve("reports/calibration_uncertainty/risk_coverage_curves.csv", "risk_msp_raw")
    test = read_curve("reports/final_evaluation/clean_test/risk_coverage_curves.csv", "risk_msp_raw")
    w, h, x0, y0, top = 560, 300, 64, 46, 0.8
    pt = lambda c, r: f"{x0 + c * w:.1f},{y0 + h - min(r, top) / top * h:.1f}"  # noqa: E731
    body = [f'<text class="t1" x="{x0}" y="20" font-size="13" font-weight="bold">Selective prediction: error rate '
            f'among retained cycles (ranked by raw confidence)</text>']
    body += crep._axes(x0, y0, w, h, [(v / 10, f"{v * 10}%") for v in range(0, 11, 2)],
                       [(v / top, f"{v:.1f}") for v in (0.0, 0.2, 0.4, 0.6, 0.8)], "coverage (share of cycles retained)",
                       "selective risk")
    for cls, err in (("c0", val_err), ("c1", test_err)):
        y = y0 + h - err / top * h
        body.append(f'<line class="{cls}" style="stroke-dasharray:4 3" stroke-width="1.5" x1="{x0}" x2="{x0 + w}" '
                    f'y1="{y:.1f}" y2="{y:.1f}"/>')
    for cls, curve in (("c0", val), ("c1", test)):
        body.append(f'<polyline class="ln {cls}" points="' + " ".join(pt(c, r) for c, r in curve[20:]) + '"/>')
    ly = y0 + h + 52
    items, lx = legend([("ln c0", "validation", False), ("ln c1", "untouched test", False),
                        ("c0", "random abstention (validation)", True), ("c1", "random abstention (test)", True)], x0, ly)
    return crep._svg(max(x0 + w + 30, lx), ly + 16, "Risk-coverage curves, validation vs test", body + items)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = J("reports/runs/seed_study/summary.json")
    val = J("reports/robustness/noise_robustness/summary.json")
    test = J("reports/final_evaluation/robustness_test/summary.json")
    metrics = J("reports/final_evaluation/clean_test/metrics.json")
    cal = J("reports/calibration_uncertainty/summary.json")
    figs = {"seed_comparison.svg": seed_comparison_svg(seed),
            "robustness_validation_vs_test.svg": robustness_svg(val, test),
            "test_confusion_matrix.svg": confusion_svg(metrics),
            "risk_coverage_validation_vs_test.svg": risk_coverage_svg(
                cal["part3_error_detection"]["raw"]["error_rate"], metrics["error_detection"]["raw"]["error_rate"])}
    for name, svg in figs.items():
        (OUT_DIR / name).write_text(svg, encoding="utf-8")
    print(f"wrote {len(figs)} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
