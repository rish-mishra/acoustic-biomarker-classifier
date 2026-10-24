"""Aggregate the predeclared seed-variability study (validation only).

Reads reports/runs/seed_study/plan.json, collects each architecture's run.json
for seeds 42 / 1337 / 2026 (seed 42 from reports/runs/<run>/, the others from
reports/runs/seed_study/<run>/), and writes:
  reports/runs/seed_study/summary.json   machine-readable results
  reports/runs/seed_study/summary.md     human-readable report

Statistics use the best-epoch validation metrics recorded in run.json. SD is
the sample SD (ddof=1) over seeds. The recommendation applies the plan's
predeclared decision_rule exactly.

Usage::

    python src/evaluation/seed_study.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
DEFAULT_STUDY_DIR = REPO_ROOT / "reports" / "runs" / "seed_study"
DEFAULT_RUNS_DIR = REPO_ROOT / "reports" / "runs"
CLASS_NAMES = ("normal", "crackle", "wheeze", "both")


def _stats(values: list[float]) -> dict:
    return {"values": values, "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def run_paths(plan: dict, runs_dir: Path, study_dir: Path) -> dict[str, dict[int, Path]]:
    """architecture -> seed -> run directory, following the plan."""
    archs = {m["frozen_from"]: name for name, m in plan["models"].items()}
    paths: dict[str, dict[int, Path]] = {name: {} for name in plan["models"]}
    for base, name in archs.items():
        paths[name][42] = Path(runs_dir) / base
    for seed, runs in plan["new_runs"].items():
        for run in runs:
            base = run.rsplit("_s", 1)[0]
            paths[archs[base]][int(seed)] = Path(study_dir) / run
    return paths


def load_run(path: Path, expected_seed: int, expected_model: str) -> dict:
    run = json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "completed" or run["test_split_evaluated"]:
        raise ValueError(f"{path}: run not completed or test split evaluated")
    if run["seed"] != expected_seed or run["config"].get("model", "baseline_cnn") != expected_model:
        raise ValueError(f"{path}: unexpected seed/model {run['seed']}/{run['config'].get('model')}")
    best = run["best_validation"]
    return {
        "run_name": run["run_name"], "seed": run["seed"], "parameter_count": run["model"]["parameter_count"],
        "macro_f1": best["macro_f1"], "balanced_accuracy": best["balanced_accuracy"], "accuracy": best["accuracy"],
        "per_class_f1": {c: best["per_class"][c]["f1"] for c in CLASS_NAMES},
        "confusion_matrix": best["confusion_matrix"], "best_epoch": run["best_epoch"],
        "epochs_completed": run["epochs_completed"], "training_time_s": run["total_training_time_s"],
        # runs saved before the "model" config field existed are baseline_cnn (as in train.build_model)
        "config": {"model": "baseline_cnn", **{k: v for k, v in run["config"].items()
                                               if k not in ("run_name", "seed", "checkpoint_dir", "reports_dir")}},
        "data_sha256": {k: run["data"][k]["sha256"] for k in ("metadata", "split_file", "normalization_stats")},
        "data_checks_ok": all(run["data"]["checks"].values()),
    }


def summarize(plan: dict, runs: dict[str, dict[int, dict]]) -> dict:
    seeds = plan["seeds"]
    archs = list(plan["models"])
    per_arch = {}
    for a in archs:
        r = [runs[a][s] for s in seeds]
        configs = {json.dumps(x["config"], sort_keys=True) for x in r}
        per_arch[a] = {
            "runs": {s: runs[a][s]["run_name"] for s in seeds},
            "parameter_count": r[0]["parameter_count"],
            "config_identical_across_seeds": len(configs) == 1,
            "macro_f1": _stats([x["macro_f1"] for x in r]),
            "balanced_accuracy": _stats([x["balanced_accuracy"] for x in r]),
            "accuracy": _stats([x["accuracy"] for x in r]),
            "per_class_f1": {c: _stats([x["per_class_f1"][c] for x in r]) for c in CLASS_NAMES},
            "best_epoch": {"values": [x["best_epoch"] for x in r], "epochs_completed": [x["epochs_completed"] for x in r]},
            "training_time_min": _stats([x["training_time_s"] / 60 for x in r]),
            "confusion_matrices": {s: runs[a][s]["confusion_matrix"] for s in seeds},
        }
    pooled_sd = math.sqrt(statistics.fmean(per_arch[a]["macro_f1"]["sd"] ** 2 for a in archs))

    paired = {}
    for a in archs:
        for b in archs:
            if a != b:
                d = [runs[a][s]["macro_f1"] - runs[b][s]["macro_f1"] for s in seeds]
                paired[f"{a}_minus_{b}"] = {**_stats(d), "seeds_a_better": sum(x > 0 for x in d)}

    winners = [a for a in archs if all(
        paired[f"{a}_minus_{b}"]["seeds_a_better"] == len(seeds) and paired[f"{a}_minus_{b}"]["mean"] > pooled_sd
        for b in archs if b != a)]
    best_mean = max(per_arch[a]["macro_f1"]["mean"] for a in archs)
    within_noise = [a for a in archs if best_mean - per_arch[a]["macro_f1"]["mean"] <= pooled_sd]
    simplicity = [s.split(" ")[0] for s in plan["decision_rule"]["simplicity_order"]]
    if winners:
        recommendation, reason = winners[0], "convincing winner under the predeclared rule"
    else:
        recommendation = next(a for a in simplicity if a in within_noise)
        reason = "no convincing winner; simplest architecture whose mean macro F1 is within one pooled seed SD of the best mean"

    means = [per_arch[a]["macro_f1"]["mean"] for a in archs]
    all_sha = {json.dumps(runs[a][s]["data_sha256"], sort_keys=True) for a in archs for s in seeds}
    return {
        "primary_metric": plan["primary_metric"],
        "seeds": seeds,
        "per_architecture": per_arch,
        "pooled_within_architecture_seed_sd": pooled_sd,
        "range_of_architecture_means": max(means) - min(means),
        "seed_paired_differences": paired,
        "decision_rule": plan["decision_rule"],
        "convincing_winners": winners,
        "within_one_pooled_sd_of_best_mean": within_noise,
        "recommendation": {"architecture": recommendation, "reason": reason},
        "same_data_sha256_all_runs": len(all_sha) == 1,
        "all_data_checks_ok": all(runs[a][s]["data_checks_ok"] for a in archs for s in seeds),
        "test_split_evaluated": False,
    }


def render_markdown(s: dict) -> str:
    seeds, pa = s["seeds"], s["per_architecture"]
    f = lambda st: f"{st['mean']:.3f} ± {st['sd']:.3f}"  # noqa: E731
    L = ["# Seed-variability study (validation only)", "",
         f"Seeds {seeds}; best-epoch validation metrics; SD = sample SD over seeds. Test split not evaluated.", "",
         "| Architecture | Params | " + " | ".join(f"Macro F1 s{x}" for x in seeds)
         + " | Mean ± SD | Min–max | Bal. acc. | Acc. | Best epochs | Train time (min) |",
         "|---|---|" + "---|" * len(seeds) + "---|---|---|---|---|---|"]
    for a, p in pa.items():
        m = p["macro_f1"]
        L.append(f"| {a} | {p['parameter_count']:,} | " + " | ".join(f"{v:.4f}" for v in m["values"])
                 + f" | **{f(m)}** | {m['min']:.3f}–{m['max']:.3f} | {f(p['balanced_accuracy'])} | {f(p['accuracy'])} | "
                 f"{p['best_epoch']['values']} | {f(p['training_time_min'])} |")
    L += ["", "## Per-class F1 (mean ± SD over seeds)", "", "| Architecture | " + " | ".join(CLASS_NAMES) + " |",
          "|---|" + "---|" * len(CLASS_NAMES)]
    for a, p in pa.items():
        L.append(f"| {a} | " + " | ".join(f(p["per_class_f1"][c]) for c in CLASS_NAMES) + " |")
    L += ["", "## Seed-paired macro-F1 differences", "",
          f"Pooled within-architecture seed SD: **{s['pooled_within_architecture_seed_sd']:.4f}**; "
          f"range of architecture means: **{s['range_of_architecture_means']:.4f}**.", "",
          "| Comparison | " + " | ".join(f"s{x}" for x in seeds) + " | Mean ± SD | Seeds A better |",
          "|---|" + "---|" * len(seeds) + "---|---|"]
    for k, d in s["seed_paired_differences"].items():
        L.append(f"| {k.replace('_minus_', ' − ')} | " + " | ".join(f"{v:+.4f}" for v in d["values"])
                 + f" | {d['mean']:+.4f} ± {d['sd']:.4f} | {d['seeds_a_better']}/{len(seeds)} |")
    r = s["recommendation"]
    L += ["", "## Predeclared decision", "",
          f"- Convincing winners: {s['convincing_winners'] or 'none'}",
          f"- Within one pooled SD of the best mean: {s['within_one_pooled_sd_of_best_mean']}",
          f"- **Recommendation: {r['architecture']}** ({r['reason']})", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate the seed-variability study.")
    p.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    a = p.parse_args(argv)
    plan = json.loads((a.study_dir / "plan.json").read_text(encoding="utf-8"))
    paths = run_paths(plan, a.runs_dir, a.study_dir)
    runs = {arch: {seed: load_run(path, seed, arch) for seed, path in seeds.items()} for arch, seeds in paths.items()}
    summary = summarize(plan, runs)
    (a.study_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (a.study_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    for arch, st in summary["per_architecture"].items():
        m = st["macro_f1"]
        print(f"{arch}: macro F1 {m['mean']:.4f} ± {m['sd']:.4f} ({', '.join(f'{v:.4f}' for v in m['values'])})")
    print(f"pooled seed SD {summary['pooled_within_architecture_seed_sd']:.4f}; "
          f"recommendation: {summary['recommendation']['architecture']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
