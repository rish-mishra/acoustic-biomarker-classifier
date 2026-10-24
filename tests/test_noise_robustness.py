"""Tests for src/evaluation/noise_robustness.py (synthetic data, CPU)."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("evaluation", "models", "data", "features"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_run as ar  # noqa: E402
import degradations as dg  # noqa: E402
import noise_robustness as nr  # noqa: E402
import normalization as nz  # noqa: E402
import train as tr  # noqa: E402
from test_respiratory_dataset import SyntheticICBHI  # noqa: E402

CONDS = list(dg.PREDECLARED_CONDITIONS)


class TestNoiseRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data = SyntheticICBHI(root / "data")
        cls.norm = nz.compute_band_stats(cls.data.dataset("train"))
        stats = root / "stats.json"
        nz.save_normalizer(cls.norm, stats, cls.data.metadata, cls.data.split_file)
        cfg = tr.TrainConfig(run_name="m", max_epochs=2, batch_size=4, device="cpu",
                             checkpoint_dir=str(root / "models"), reports_dir=str(root / "reports"),
                             bucket_pool_multiplier=2)
        tr.fit(cfg, cls.data.dataset("train", normalizer=cls.norm), cls.data.dataset("val", normalizer=cls.norm),
               log=lambda *_: None, normalization_path=stats)
        ar.analyze("m", root / "models", root / "reports", dataset=cls.data.dataset("val", normalizer=cls.norm),
                   device="cpu")
        cls.model, cls.ckpt = ar.load_run_model("m", root / "models", "cpu")
        cls.reference_csv = root / "reports" / "m" / "val_predictions.csv"
        cls.out = root / "out"
        cls.ok = nr.run_study(cls.out, cls.model, cls.ckpt, "cpu", CONDS, nr.Status(cls.out / "_runner"),
                              cls.factory, cls.reference_csv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def factory(cls, condition):
        return nr.DegradedCycleDataset("val", condition, metadata_path=cls.data.metadata,
                                       split_path=cls.data.split_file, audio_dir=cls.data.audio, normalizer=cls.norm)

    def rows(self, name, out=None):
        with open((out or self.out) / "conditions" / name / "predictions.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_all_conditions_complete_and_clean_reproduces_checkpoint(self):
        self.assertTrue(self.ok)
        check = json.loads((self.out / "clean_check.json").read_text())
        self.assertTrue(check["ok"])
        self.assertTrue(check["predictions_identical"] and check["labels_identical"])
        self.assertEqual(check["max_abs_probability_difference"], 0.0)
        status = json.loads((self.out / "_runner" / "status.json").read_text())
        self.assertEqual({v["state"] for v in status["conditions"].values()}, {"completed"})
        for c in CONDS:
            m = json.loads((self.out / "conditions" / c.name / "metrics.json").read_text())
            self.assertEqual((m["status"], m["split"], m["test_split_evaluated"]), ("completed", "val", False))
            self.assertEqual(m["overall"]["n"], 8)

    def test_labels_and_order_unchanged(self):
        val = self.data.dataset("val")
        expected = [(r["recording_filename"], r["cycle_index"], str(rd_label(r))) for r in val.rows]
        for c in CONDS:
            got = [(r["recording_filename"], r["cycle_index"], r["label"]) for r in self.rows(c.name)]
            self.assertEqual(got, expected, c.name)

    def test_degradations_change_features_but_clean_does_not(self):
        clean = self.factory(CONDS[0])
        plain = self.data.dataset("val", normalizer=self.norm)
        for i in range(len(plain)):
            self.assertTrue(torch.equal(clean[i]["features"], plain[i]["features"]))
        for c in CONDS[1:]:
            ds = self.factory(c)
            self.assertFalse(all(torch.equal(ds[i]["features"], plain[i]["features"]) for i in range(len(ds))), c.name)
            self.assertEqual([ds[i]["num_frames"] for i in range(len(ds))],
                             [plain[i]["num_frames"] for i in range(len(plain))], c.name)

    def test_per_example_stats_recorded(self):
        for r in self.rows("noise_snr5db"):
            self.assertAlmostEqual(float(r["realized_snr_db"]), 5.0, delta=1e-3)
        for r in self.rows("dropout_25pct"):
            self.assertAlmostEqual(float(r["dropped_fraction"]), 0.25, delta=0.1)

    def test_refuses_test_split(self):
        with self.assertRaises(PermissionError):
            nr.DegradedCycleDataset("test", CONDS[0], metadata_path=self.data.metadata,
                                    split_path=self.data.split_file, audio_dir=self.data.audio, normalizer=self.norm)
        with self.assertRaises(PermissionError):
            nr.DegradedCycleDataset("train", CONDS[0], metadata_path=self.data.metadata,
                                    split_path=self.data.split_file, audio_dir=self.data.audio, normalizer=self.norm)

    def test_reproducible_rerun(self):
        out2 = Path(self.tmp.name) / "out2"
        subset = [CONDS[0], CONDS[2], CONDS[11], CONDS[13]]  # clean, 10 dB, low-pass 1000 Hz, dropout 25%
        self.assertTrue(nr.run_study(out2, self.model, self.ckpt, "cpu", subset, nr.Status(out2 / "_runner"),
                                     self.factory, self.reference_csv))
        for c in subset:
            self.assertEqual(self.rows(c.name, out2), self.rows(c.name), c.name)

    def test_resume_does_not_overwrite(self):
        out3 = Path(self.tmp.name) / "out3"
        subset = [CONDS[0], CONDS[5]]
        nr.run_study(out3, self.model, self.ckpt, "cpu", subset, nr.Status(out3 / "_runner"), self.factory,
                     self.reference_csv)
        p = out3 / "conditions" / CONDS[5].name / "metrics.json"
        before = p.read_bytes()
        status = nr.Status(out3 / "_runner")
        self.assertTrue(nr.run_study(out3, self.model, self.ckpt, "cpu", subset, status, self.factory,
                                     self.reference_csv))
        self.assertEqual(p.read_bytes(), before)
        self.assertEqual({v["state"] for v in status.data["conditions"].values()}, {"reused_completed"})

    def test_clean_mismatch_stops_the_study(self):
        out4 = Path(self.tmp.name) / "out4"
        bad = {**self.ckpt, "val_metrics": {**self.ckpt["val_metrics"], "confusion_matrix": [[0] * 4] * 4}}
        status = nr.Status(out4 / "_runner")
        self.assertFalse(nr.run_study(out4, self.model, bad, "cpu", CONDS[:3], status, self.factory,
                                      self.reference_csv))
        self.assertFalse((out4 / "conditions" / CONDS[1].name).exists())
        self.assertEqual(status.data["conditions"]["clean"]["state"], "failed")

    def test_failed_condition_is_recorded_and_others_continue(self):
        out5 = Path(self.tmp.name) / "out5"

        def factory(c):
            if c.name == "gain_0.5x":
                raise RuntimeError("simulated failure")
            return self.factory(c)
        status = nr.Status(out5 / "_runner")
        subset = [CONDS[0], CONDS[5], CONDS[6]]
        self.assertFalse(nr.run_study(out5, self.model, self.ckpt, "cpu", subset, status, factory,
                                      self.reference_csv))
        self.assertEqual(status.data["conditions"]["gain_0.5x"]["state"], "failed")
        self.assertIn("simulated failure", (out5 / "_runner" / "runner.log").read_text())
        self.assertTrue(nr.condition_done(out5, CONDS[6].name))

    def test_summary_and_report(self):
        s = nr.summarize(self.out, CONDS, ref_scale=0.0157, n_boot=50)
        self.assertEqual(set(s["conditions"]), {c.name for c in CONDS})
        self.assertFalse(s["test_split_evaluated"])
        clean = s["conditions"]["clean"]
        self.assertEqual((clean["macro_f1_change_abs"], clean["predictions_changed_vs_clean"]), (0.0, 0.0))
        for name, c in s["conditions"].items():
            self.assertAlmostEqual(c["macro_f1_change_abs"], c["macro_f1"] - clean["macro_f1"])
            if clean["macro_f1"]:
                self.assertAlmostEqual(c["macro_f1_change_rel"], c["macro_f1_change_abs"] / clean["macro_f1"])
            self.assertEqual(sum(map(sum, c["confusion_matrix"])), c["n"])
            self.assertEqual(set(c["per_class_f1"]), set(tr.rd.CLASS_NAMES))
        self.assertEqual(set(s["families"]), set(nr.SEVERITY_ORDER))
        md, svg = nr.render_markdown(s), nr.render_svg(s)
        for text in ("Findings", "Bandwidth loss", "Amplitude changes", "Missing temporal audio",
                     "recording characteristics", "not clinical or diagnostic"):
            self.assertIn(text, md)
        self.assertTrue(svg.startswith("<svg") and "prefers-color-scheme: dark" in svg)

    def test_family_shape_rules(self):
        self.assertEqual(nr.family_shape([-0.001, -0.01], 0.0157)["shape"], "negligible")
        g = nr.family_shape([-0.02, -0.04, -0.06], 0.0157)
        self.assertEqual(g["shape"], "gradual")
        c = nr.family_shape([-0.005, -0.01, -0.20], 0.0157)
        self.assertEqual((c["shape"], c["collapse_level_index"]), ("collapse", 2))
        self.assertEqual([round(x, 3) for x in c["step_drops"]], [0.005, 0.005, 0.19])

    def test_plan_matches_predeclared_conditions(self):
        plan = nr.plan_payload()
        self.assertEqual(nr.plan_conditions(plan), CONDS)
        with self.assertRaises(ValueError):
            nr.plan_conditions({**plan, "conditions": plan["conditions"][:-1]})


def rd_label(row):
    return tr.rd.label_to_index(row["cycle_label"])


if __name__ == "__main__":
    unittest.main()
