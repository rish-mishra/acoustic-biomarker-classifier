"""Tests for src/evaluation/exploratory_test_analysis.py helpers (synthetic rows only)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

import exploratory_test_analysis as ex  # noqa: E402


def row(pid, dev, label, pred, i, msp=0.6):
    probs = [0.0] * 4
    probs[pred] = msp
    probs[(pred + 1) % 4] = 1 - msp
    return {"patient_id": pid, "device": dev, "label": label, "pred": pred, "msp": msp,
            "recording_filename": f"{pid}.wav", "cycle_index": i,
            **{f"prob_{c}": p for c, p in zip(ex.CLASS_NAMES, probs)}}


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.rows = ([row("a", "D1", 0, 1, i) for i in range(15)] + [row("a", "D1", 1, 1, 15 + i) for i in range(10)]
                     + [row("b", "D2", 0, 0, i) for i in range(5)] + [row("b", "D2", 0, 1, 5)])

    def test_group_summary(self):
        s = ex.group_summary(self.rows)
        self.assertEqual((s["n_cycles"], s["n_patients"], s["normal_cycles"], s["normal_to_crackle"]), (31, 2, 21, 16))
        self.assertAlmostEqual(s["normal_to_crackle_rate"], 16 / 21)
        self.assertEqual(s["true_counts"], [21, 10, 0, 0])
        self.assertAlmostEqual(s["accuracy"], 15 / 31)
        self.assertIsNotNone(s["macro_f1_present_classes"])

    def test_macro_f1_only_where_meaningful(self):
        small = ex.group_summary([r for r in self.rows if r["patient_id"] == "b"])
        self.assertIsNone(small["macro_f1_present_classes"])  # < 20 cycles
        one_class = ex.group_summary([row("c", "D", 0, 0, i) for i in range(30)])
        self.assertIsNone(one_class["macro_f1_present_classes"])  # one true class

    def test_concentration(self):
        c = ex.concentration({"a": 15, "b": 1, "c": 0}, {"a": 15, "b": 6, "c": 9})
        self.assertEqual((c["total"], c["units_for_50pct"], c["units_for_80pct"]), (16, 1, 1))
        self.assertAlmostEqual(c["top1_share"], 15 / 16)
        self.assertAlmostEqual(c["table"][0]["base_share"], 0.5)
        self.assertAlmostEqual(c["table"][-1]["cumulative_share"], 1.0)

    def test_leave_one_out(self):
        loo = {r["excluded"]: r for r in ex.leave_one_out(self.rows, "patient_id")}
        self.assertEqual(loo["a"]["n_excluded"], 25)
        self.assertAlmostEqual(loo["a"]["normal_to_crackle_rate"], 1 / 6)

    def test_gain_and_noise_tables(self):
        shifted = [dict(r, pred=0) for r in self.rows]
        dev, pat = ex.gain_tables(self.rows, {g: shifted for g in ex.GAIN})
        d1 = next(r for r in dev if r["device"] == "D1")
        self.assertAlmostEqual(d1["gain_0.5x_normal_share_change"], 1.0)
        self.assertAlmostEqual(d1["gain_0.5x_predictions_changed"], 1.0)
        all_crackle = [dict(r, pred=1) for r in self.rows]
        p, d, spread = ex.noise_tables(self.rows, {n: all_crackle for n in ex.NOISE})
        self.assertEqual(spread["noise_snr0db"]["patients_at_least_80pct_crackle"], 1)  # only patient a has >= 10 cycles
        self.assertTrue(all(r["noise_snr0db_crackle_share"] == 1.0 for r in d))


if __name__ == "__main__":
    unittest.main()
