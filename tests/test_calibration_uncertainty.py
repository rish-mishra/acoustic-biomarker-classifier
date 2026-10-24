"""Tests for src/evaluation/calibration_uncertainty.py and calibration_report.py (synthetic prediction files)."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

import calibration_metrics as cm  # noqa: E402
import calibration_report as rep  # noqa: E402
import calibration_uncertainty as cu  # noqa: E402

PLAN = json.loads((REPO_ROOT / "reports" / "calibration_uncertainty" / "plan.json").read_text(encoding="utf-8"))
CONDITIONS = ["clean", *[n for lv in PLAN["part5_degradation_uncertainty"]["family_order_mild_to_severe"].values()
                         for n in lv]]
VAL = {"201": 30, "202": 25, "203": 12, "204": 8, "205": 5}
FIELDS = ["patient_id", "recording_filename", "cycle_index", "device", "original_sample_rate", "cycle_duration_s",
          "original_clipped_fraction", "clipped_plateau_count", "label", "pred",
          "prob_normal", "prob_crackle", "prob_wheeze", "prob_both"]


def write_preds(path, rows_meta, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for (pid, rec, idx, label), p in zip(rows_meta, probs):
            w.writerow({"patient_id": pid, "recording_filename": rec, "cycle_index": idx, "device": "D",
                        "original_sample_rate": 44100, "cycle_duration_s": 1.0, "original_clipped_fraction": 0.0,
                        "clipped_plateau_count": 0, "label": label, "pred": int(np.argmax(p)),
                        **{f"prob_{c}": float(v) for c, v in zip(cu.CLASS_NAMES, p)}})


class Synthetic:
    def __init__(self, root: Path, seed=0):
        rng = np.random.default_rng(seed)
        self.root = root
        self.split = root / "patient_split.csv"
        with open(self.split, "w", newline="", encoding="utf-8") as f:
            f.write("patient_id,split\n" + "".join(f"{p},val\n" for p in VAL) + "301,test\n302,train\n")
        self.meta, logits = [], []
        for pid, n in VAL.items():
            for i in range(n):
                true = rng.normal(size=4) * 1.5
                label = int(rng.choice(4, p=cm.softmax(true[None])[0]))
                self.meta.append((pid, f"{pid}_rec.wav", i, label))
                logits.append(true * 2.0)  # overconfident model
        self.logits = np.array(logits)
        self.clean_csv = root / "val_predictions.csv"
        write_preds(self.clean_csv, self.meta, cm.softmax(self.logits))
        self.deg_dir = root / "conditions"
        for k, name in enumerate(CONDITIONS):
            z = self.logits if name == "clean" else self.logits * (1 - 0.04 * k) + rng.normal(size=self.logits.shape)
            write_preds(self.deg_dir / name / "predictions.csv", self.meta, cm.softmax(z))

    def load(self):
        return cu.load_all(self.deg_dir, self.clean_csv, PLAN, self.split, n_expected=None)


class TestCalibrationUncertainty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.syn = Synthetic(Path(cls.tmp.name))
        cls.clean, cls.degraded = cls.syn.load()
        cls.summary, cls.tables = cu.analyze(cls.clean, cls.degraded, PLAN, n_boot=30, seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_guards_reject_non_validation_inputs(self):
        val = cu.validation_patients(self.syn.split)
        self.assertEqual(val, set(VAL))
        d = cu.load_predictions(self.syn.clean_csv)
        bad = dict(d, patient=np.array(["301"] + d["patient"][1:].tolist()))  # a test-split patient
        with self.assertRaises(ValueError):
            cu.check_validation(bad, val, None)
        with self.assertRaises(ValueError):
            cu.check_validation(d, val, 1077)
        flipped = dict(d, preds=(d["preds"] + 1) % 4)
        with self.assertRaises(ValueError):
            cu.check_validation(flipped, val, None)
        missing = {k: (v[:-5] if k in ("patient", "labels", "preds", "probs", "keys", "device") else v)
                   for k, v in d.items()}
        with self.assertRaises(ValueError):
            cu.check_validation(missing, val, None)  # drops patient 205 entirely

    def test_degraded_must_match_clean_cycles(self):
        other = dict(self.degraded["noise_snr20db"], labels=(self.clean["labels"] + 1) % 4)
        with self.assertRaises(ValueError):
            cu.same_cycles(self.clean, other)

    def test_lopo_temperatures_never_see_the_held_out_patient(self):
        z = cm.logits_from_probs(self.clean["probs"])
        y = self.clean["labels"].copy()
        held = self.clean["patient"] == "203"
        y_changed = y.copy()
        y_changed[held] = (y_changed[held] + 1) % 4  # changing the held-out patient's labels ...
        t1 = cm.lopo_temperatures(z, y, self.clean["patient"])["203"]["temperature"]
        t2 = cm.lopo_temperatures(z, y_changed, self.clean["patient"])["203"]["temperature"]
        self.assertEqual(t1, t2)  # ... cannot change its own temperature
        lopo = self.summary["temperature_scaling"]["lopo"]
        self.assertEqual(set(lopo), set(VAL))
        for g, v in lopo.items():
            self.assertEqual(v["n"] + v["n_held_out"], sum(VAL.values()))
        self.assertGreater(self.summary["temperature_scaling"]["final_validation_temperature"]["temperature"], 1.0)

    def test_same_clean_temperature_applied_to_degraded_versions(self):
        t_row = self.tables["t_row"]
        for name, d in self.degraded.items():
            expect = cm.apply_temperature(cm.logits_from_probs(d["probs"]), t_row)
            np.testing.assert_allclose(d["probs_lopo"], expect)
        np.testing.assert_allclose(self.degraded["clean"]["probs_lopo"], self.tables["p_lopo"])
        for g in VAL:
            m = self.clean["patient"] == g
            self.assertTrue(np.all(t_row[m] == self.summary["temperature_scaling"]["lopo"][g]["temperature"]))

    def test_predictions_unchanged_and_calibration_improves_out_of_fold(self):
        raw, lopo = self.summary["part1_calibration_raw"], self.summary["part2_calibration_lopo"]
        self.assertEqual((raw["accuracy"], raw["macro_f1"]), (lopo["accuracy"], lopo["macro_f1"]))
        self.assertLess(lopo["nll"], raw["nll"])  # synthetic model is overconfident by construction
        self.assertGreater(raw["overconfidence_gap"], 0)

    def test_selective_prediction_structure(self):
        sp = self.summary["part4_selective_prediction"]
        n = sum(VAL.values())
        for r in cu.RANKINGS:
            levels = sp["rankings"][r]["levels"]
            self.assertEqual([lv["coverage"] for lv in levels], PLAN["part4_selective_prediction"]["coverage_levels"])
            for lv in levels:
                self.assertEqual(lv["n_retained"] + lv["n_rejected"], n)
                self.assertEqual(sum(lv["retained_true_counts"]) + sum(lv["rejected_true_counts"]), n)
                self.assertEqual(sum(lv["retained_pred_counts"]), lv["n_retained"])
                self.assertEqual(len(lv["per_class_f1"]), 4)
        self.assertAlmostEqual(sp["rankings"]["msp_raw"]["levels"][0]["selective_risk"],
                               self.summary["part3_error_detection"]["raw"]["error_rate"])
        curves = self.tables["curves"]
        self.assertEqual(len(curves["risk_msp_raw"]), n)
        self.assertAlmostEqual(curves["risk_msp_raw"].mean(), sp["rankings"]["msp_raw"]["aurc"])

    def test_fixed_clean_thresholds(self):
        taus = self.summary["part5_degradation"]["fixed_clean_thresholds"]
        msp = cm.confidence(self.clean["probs"])
        self.assertAlmostEqual((msp >= taus["tau_50"]).mean(), 0.5, delta=0.02)
        clean_t = self.summary["part5_degradation"]["conditions"]["clean"]["raw"]["fixed_clean_thresholds"]
        self.assertAlmostEqual(clean_t["tau_80"]["realized_coverage"], 0.8, delta=0.02)

    def test_family_rule(self):
        self.assertEqual(cu.family_rule([0.05, 0.06], [0.04, 0.08]), "confidence tracks difficulty")
        self.assertEqual(cu.family_rule([0.05, 0.10], [0.04, 0.08]), "becomes overconfident")
        self.assertEqual(cu.family_rule([0.01], [0.04, 0.08]), "becomes underconfident")
        self.assertEqual(cu.family_rule([0.01, 0.2], [0.04, 0.08]), "mixed")

    def test_bootstrap_intervals_present_and_ordered(self):
        ci = self.summary["ci95"]
        for key in ("calibration.raw.ece", "calibration.lopo_minus_raw.nll", "error_detection.raw.1-msp.auroc",
                    "selective.msp_raw.0.5.selective_risk", "degradation.noise_snr0db.lopo.overconfidence_gap",
                    "degradation.dropout_50pct.raw.tau_50.confidently_wrong_share"):
            self.assertIn(key, ci)
            self.assertLessEqual(ci[key][0], ci[key][1])

    def test_outputs_and_report(self):
        out = Path(self.tmp.name) / "out"
        out.mkdir()
        cu.write_tables(out, self.clean, self.tables, self.summary)
        js = json.loads(json.dumps(self.summary, default=cu._json_default))
        rep.write_report(out, js)
        for name in ("per_example_clean.csv", "per_example_degraded.csv", "lopo_temperatures.csv",
                     "risk_coverage_curves.csv", "summary.md", "reliability_diagram.svg", "risk_coverage.svg",
                     "degradation_confidence.svg"):
            self.assertTrue((out / name).exists(), name)
        with open(out / "per_example_degraded.csv", newline="", encoding="utf-8") as f:
            self.assertEqual(sum(1 for _ in csv.DictReader(f)), len(CONDITIONS) * sum(VAL.values()))
        md = (out / "summary.md").read_text(encoding="utf-8")
        for text in ("out-of-fold validation/development-set", "final_validation_temperature", "Post-hoc",
                     "test split was not read", "no clinical or diagnostic claim"):
            self.assertIn(text, md)


if __name__ == "__main__":
    unittest.main()
