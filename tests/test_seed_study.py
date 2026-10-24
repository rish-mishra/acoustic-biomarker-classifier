"""Tests for src/evaluation/seed_study.py (synthetic run records)."""

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

import seed_study as ss  # noqa: E402

PLAN = {
    "primary_metric": "best-epoch validation macro F1",
    "seeds": [42, 1337, 2026],
    "models": {"baseline_cnn": {"frozen_from": "baseline_cnn_v3"}, "crnn": {"frozen_from": "crnn_v1"},
               "mobilenetv3_pretrained": {"frozen_from": "mobilenetv3_pretrained_v1"}},
    "new_runs": {s: [f"{b}_s{s}" for b in ("baseline_cnn_v3", "crnn_v1", "mobilenetv3_pretrained_v1")]
                 for s in ("1337", "2026")},
    "decision_rule": {"simplicity_order": ["baseline_cnn (x)", "crnn (y)", "mobilenetv3_pretrained (z)"]},
}
PARAMS = {"baseline_cnn": 294740, "crnn": 353365, "mobilenetv3_pretrained": 1521668}


def write_run(path: Path, name, seed, model, f1, **over):
    cm = [[10, 1, 0, 0], [1, 10, 0, 0], [0, 0, 5, 1], [0, 0, 1, 5]]
    run = {
        "run_name": name, "seed": seed, "status": "completed", "test_split_evaluated": False,
        "config": {"run_name": name, "seed": seed, "lr": 1e-4, "dropout": 0.3,
                   **({} if model == "baseline_cnn" else {"model": model})},
        "model": {"parameter_count": PARAMS[model]}, "best_epoch": 5, "epochs_completed": 15,
        "total_training_time_s": 600.0,
        "best_validation": {"macro_f1": f1, "balanced_accuracy": f1 + 0.01, "accuracy": f1 + 0.05,
                            "per_class": {c: {"f1": f1} for c in ss.CLASS_NAMES}, "confusion_matrix": cm},
        "data": {**{k: {"sha256": "abc"} for k in ("metadata", "split_file", "normalization_stats")},
                 "checks": {"x": True}},
    }
    run.update(over)
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(json.dumps(run))


class TestSeedStudy(unittest.TestCase):
    def build(self, f1s):
        tmp = Path(self.tmp.name)
        runs_dir, study_dir = tmp / "runs", tmp / "runs" / "seed_study"
        paths = ss.run_paths(PLAN, runs_dir, study_dir)
        for arch, seeds in paths.items():
            for seed, path in seeds.items():
                write_run(path, path.name, seed, arch, f1s[arch][PLAN["seeds"].index(seed)])
        runs = {a: {s: ss.load_run(p, s, a) for s, p in seeds.items()} for a, seeds in paths.items()}
        return ss.summarize(PLAN, runs)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_paths(self):
        paths = ss.run_paths(PLAN, Path("R"), Path("R/seed_study"))
        self.assertEqual(paths["crnn"][42], Path("R/crnn_v1"))
        self.assertEqual(paths["crnn"][1337], Path("R/seed_study/crnn_v1_s1337"))
        self.assertEqual(paths["mobilenetv3_pretrained"][2026],
                         Path("R/seed_study/mobilenetv3_pretrained_v1_s2026"))

    def test_statistics_and_no_winner_picks_simplest_within_noise(self):
        f1s = {"baseline_cnn": [0.427, 0.40, 0.41], "crnn": [0.415, 0.43, 0.40],
               "mobilenetv3_pretrained": [0.438, 0.41, 0.42]}
        s = self.build(f1s)
        m = s["per_architecture"]["baseline_cnn"]["macro_f1"]
        self.assertAlmostEqual(m["mean"], statistics.fmean(f1s["baseline_cnn"]))
        self.assertAlmostEqual(m["sd"], statistics.stdev(f1s["baseline_cnn"]))
        self.assertEqual((m["min"], m["max"]), (0.40, 0.427))
        d = s["seed_paired_differences"]["mobilenetv3_pretrained_minus_baseline_cnn"]
        self.assertEqual([round(v, 3) for v in d["values"]], [0.011, 0.01, 0.01])
        self.assertEqual(d["seeds_a_better"], 3)
        self.assertEqual(s["convincing_winners"], [])  # mean advantage 0.010 < pooled SD
        self.assertEqual(s["recommendation"]["architecture"], "baseline_cnn")
        self.assertTrue(s["same_data_sha256_all_runs"] and s["all_data_checks_ok"])
        self.assertTrue(all(p["config_identical_across_seeds"] for p in s["per_architecture"].values()))

    def test_convincing_winner(self):
        f1s = {"baseline_cnn": [0.40, 0.41, 0.405], "crnn": [0.39, 0.40, 0.395],
               "mobilenetv3_pretrained": [0.50, 0.51, 0.505]}
        s = self.build(f1s)
        self.assertEqual(s["convincing_winners"], ["mobilenetv3_pretrained"])
        self.assertEqual(s["recommendation"]["architecture"], "mobilenetv3_pretrained")

    def test_simplest_outside_noise_is_skipped(self):
        f1s = {"baseline_cnn": [0.30, 0.31, 0.305], "crnn": [0.44, 0.40, 0.46],
               "mobilenetv3_pretrained": [0.45, 0.42, 0.44]}
        s = self.build(f1s)
        self.assertEqual(s["convincing_winners"], [])
        self.assertEqual(s["recommendation"]["architecture"], "crnn")

    def test_config_identity_treats_missing_model_as_baseline(self):
        tmp = Path(self.tmp.name)
        write_run(tmp / "a", "a", 42, "baseline_cnn", 0.4)  # pre-"model"-field run: no config["model"]
        write_run(tmp / "b", "b", 1337, "baseline_cnn", 0.4)
        cfg_b = json.loads((tmp / "b" / "run.json").read_text())["config"]
        write_run(tmp / "b", "b", 1337, "baseline_cnn", 0.4, config={**cfg_b, "model": "baseline_cnn"})
        a, b = ss.load_run(tmp / "a", 42, "baseline_cnn"), ss.load_run(tmp / "b", 1337, "baseline_cnn")
        self.assertEqual(a["config"], b["config"])
        write_run(tmp / "b", "b", 1337, "baseline_cnn", 0.4, config={**cfg_b, "model": "baseline_cnn", "lr": 1e-3})
        self.assertNotEqual(a["config"], ss.load_run(tmp / "b", 1337, "baseline_cnn")["config"])

    def test_load_run_guards(self):
        p = Path(self.tmp.name) / "r"
        write_run(p, "r", 42, "crnn", 0.4, test_split_evaluated=True)
        with self.assertRaises(ValueError):
            ss.load_run(p, 42, "crnn")
        write_run(p, "r", 42, "crnn", 0.4, status="running")
        with self.assertRaises(ValueError):
            ss.load_run(p, 42, "crnn")
        write_run(p, "r", 1337, "crnn", 0.4)
        with self.assertRaises(ValueError):
            ss.load_run(p, 42, "crnn")
        with self.assertRaises(ValueError):
            ss.load_run(p, 1337, "baseline_cnn")


if __name__ == "__main__":
    unittest.main()
