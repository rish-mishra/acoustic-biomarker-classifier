"""Tests for src/evaluation/analyze_run.py and compare_runs.py (synthetic data, CPU)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("evaluation", "models", "data"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_run as ar  # noqa: E402
import compare_runs as cr  # noqa: E402
import normalization as nz  # noqa: E402
import train as tr  # noqa: E402
from test_respiratory_dataset import SyntheticICBHI  # noqa: E402


class TestRunAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data = SyntheticICBHI(root / "data")
        norm = nz.compute_band_stats(cls.data.dataset("train"))
        cls.stats = root / "stats.json"
        nz.save_normalizer(norm, cls.stats, cls.data.metadata, cls.data.split_file)
        cls.train = cls.data.dataset("train", normalizer=norm)
        cls.val = cls.data.dataset("val", normalizer=norm)
        cls.test = cls.data.dataset("test", normalizer=norm)
        cls.ckpt_dir, cls.reports = root / "models", root / "reports"
        for name, lr in (("r1", 1e-3), ("r2", 3e-4)):
            cfg = tr.TrainConfig(run_name=name, max_epochs=2, batch_size=4, device="cpu", lr=lr,
                                 checkpoint_dir=str(cls.ckpt_dir), reports_dir=str(cls.reports),
                                 bucket_pool_multiplier=2)
            tr.fit(cfg, cls.train, cls.val, log=lambda *_: None, normalization_path=cls.stats)
        cls.results = {r: ar.analyze(r, cls.ckpt_dir, cls.reports, dataset=cls.val, device="cpu")
                       for r in ("r1", "r2")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_refuses_test_split(self):
        model, _ = ar.load_run_model("r1", self.ckpt_dir, "cpu")
        with self.assertRaises(PermissionError):
            ar.predict(model, self.test, "cpu")

    def test_reproduces_logged_metrics_and_writes_files(self):
        for run, res in self.results.items():
            self.assertTrue(res["reproduces_logged_val_confusion_matrix"], run)
            self.assertTrue(res["data_sha256_matches_checkpoint"], run)
            self.assertEqual(res["split"], "val")
            self.assertTrue((self.reports / run / "val_predictions.csv").exists())
            self.assertTrue((self.reports / run / "val_analysis.json").exists())
            self.assertEqual(res["overall"]["n"], len(self.val))

    def test_subgroups_partition_validation_cycles(self):
        sub = self.results["r1"]["subgroups"]
        for dim in ("device", "original_sample_rate", "duration"):
            self.assertEqual(sum(g["n_cycles"] for g in sub[dim].values()), len(self.val), dim)
        q = sub["quality_group"]
        self.assertEqual(q["all"]["n_cycles"], len(self.val))
        self.assertEqual(set(self.results["r1"]["patients"]), set(self.val.patients))

    def test_duration_bins(self):
        self.assertEqual([ar.duration_bin(x) for x in (0.2, 1.5, 2.49, 2.5, 3.99, 4.0, 16.0)],
                         ["<1.5s", "1.5-2.5s", "1.5-2.5s", "2.5-4s", "2.5-4s", ">=4s", ">=4s"])

    def test_comparison(self):
        out = Path(self.tmp.name) / "cmp"
        res = cr.compare(["r1", "r2"], "r1", self.reports, out)
        self.assertEqual(res["ranking"], sorted(["r1", "r2"], key=lambda r: -res["runs"][r]["val_macro_f1"]))
        self.assertTrue(res["same_data_sha256_for_all_runs"])
        self.assertFalse(res["test_split_evaluated"])
        b = res["bootstrap"]["runs"]
        self.assertEqual(b["r1"]["diff_vs_r1_ci95"], [0.0, 0.0])
        for run in ("r1", "r2"):
            lo, hi = b[run]["macro_f1_ci95"]
            self.assertLessEqual(lo, hi)
            for key in ("best_epoch", "val_macro_f1", "val_balanced_accuracy", "val_accuracy", "per_class",
                        "confusion_matrix", "training_time_s", "parameter_count"):
                self.assertIn(key, res["runs"][run])
        self.assertTrue((out / "comparison.json").exists() and (out / "comparison.md").exists())
        again = cr.compare(["r1", "r2"], "r1", self.reports, Path(self.tmp.name) / "cmp2")
        self.assertEqual(json.dumps(res["bootstrap"]), json.dumps(again["bootstrap"]))


if __name__ == "__main__":
    unittest.main()
