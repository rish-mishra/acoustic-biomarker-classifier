"""Tests for src/evaluation/calibration_metrics.py (synthetic arrays)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

import calibration_metrics as cm  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402


def random_case(n=400, k=4, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * scale
    labels = rng.integers(0, k, size=n)
    return logits, labels


class TestBasics(unittest.TestCase):
    def test_nll_brier_entropy_margin_by_hand(self):
        p = np.array([[0.7, 0.2, 0.1, 0.0], [0.25, 0.25, 0.25, 0.25]])
        y = np.array([0, 3])
        self.assertAlmostEqual(cm.nll(p, y), (-math.log(0.7) - math.log(0.25)) / 2)
        self.assertAlmostEqual(cm.brier(p, y), ((0.09 + 0.04 + 0.01) + (3 * 0.0625 + 0.5625)) / 2)
        self.assertAlmostEqual(cm.entropy(p)[1], math.log(4))
        self.assertAlmostEqual(cm.entropy(p)[0], -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1)))
        np.testing.assert_allclose(cm.margin(p), [0.5, 0.0])

    def test_logits_round_trip(self):
        logits, _ = random_case()
        p = cm.softmax(logits)
        np.testing.assert_allclose(cm.softmax(cm.logits_from_probs(p)), p, atol=1e-12)
        with self.assertRaises(ValueError):
            cm.logits_from_probs(np.array([[1.0, 0.0]]))

    def test_check_inputs(self):
        with self.assertRaises(ValueError):
            cm.check_inputs(np.array([[0.5, 0.6]]), np.array([0]))
        with self.assertRaises(ValueError):
            cm.check_inputs(np.array([[0.5, 0.5]]), np.array([2]))
        cm.check_inputs(np.array([[0.5, 0.5]]), np.array([1]))


class TestCalibrationError(unittest.TestCase):
    def test_bin_edges(self):
        np.testing.assert_array_equal(cm.equal_width_bin(np.array([0.0, 1 / 15, 1 / 15 + 1e-9, 0.5, 14 / 15, 1.0])),
                                      [0, 0, 1, 7, 13, 14])

    def test_ece_by_hand(self):
        # bin 14 (conf 0.95): 2 cycles, 1 correct; bin 7 (conf 0.5): 2 cycles, 2 correct
        p = np.array([[0.95, 0.05], [0.95, 0.05], [0.5, 0.5], [0.5, 0.5]])
        y = np.array([0, 1, 0, 0])  # argmax of a tie is class 0
        self.assertAlmostEqual(cm.ece(p, y), 0.5 * abs(0.5 - 0.95) + 0.5 * abs(1.0 - 0.5))
        self.assertAlmostEqual(cm.mce(p, y), 0.5)

    def test_perfectly_calibrated_bins_give_zero(self):
        p = np.array([[0.75, 0.25]] * 4)
        y = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(cm.ece(p, y), 0.0)
        self.assertAlmostEqual(cm.ece_equal_mass(p, y, n_bins=1), 0.0)

    def test_equal_mass_and_classwise(self):
        logits, y = random_case(n=301)
        p = cm.softmax(logits)
        conf, hit = cm.confidence(p), (cm.predictions(p) == y)
        order = np.argsort(conf, kind="stable")
        groups = np.array_split(order, 15)
        self.assertEqual({len(g) for g in groups}, {20, 21})
        expect = sum(len(g) / 301 * abs(hit[g].mean() - conf[g].mean()) for g in groups)
        self.assertAlmostEqual(cm.ece_equal_mass(p, y), expect)
        manual = np.mean([sum((idx := np.flatnonzero(cm.equal_width_bin(p[:, k]) == b)).size / 301 *
                              abs((y[idx] == k).mean() - p[idx, k].mean()) for b in range(15) if
                              np.any(cm.equal_width_bin(p[:, k]) == b)) for k in range(4)])
        self.assertAlmostEqual(cm.classwise_ece(p, y), manual)

    def test_reliability_bins_partition(self):
        logits, y = random_case()
        p = cm.softmax(logits)
        bins = cm.reliability_bins(cm.confidence(p), cm.predictions(p) == y)
        self.assertEqual(sum(b["n"] for b in bins), len(y))
        ece = sum(b["n"] / len(y) * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["n"])
        self.assertAlmostEqual(ece, cm.ece(p, y))


class TestClassificationSummary(unittest.TestCase):
    def test_matches_project_report(self):
        logits, y = random_case(n=257)
        pred = logits.argmax(1)
        s = cm.classification_summary(y, pred, 4)
        ref = classification_report(confusion_matrix(torch.tensor(y), torch.tensor(pred), 4), list("abcd"))
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            self.assertAlmostEqual(s[key], ref[key])
        self.assertEqual(s["confusion_matrix"], ref["confusion_matrix"])

    def test_zero_division_convention(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 0])  # classes 2 and 3 absent and never predicted
        s = cm.classification_summary(y, pred, 4)
        ref = classification_report(confusion_matrix(torch.tensor(y), torch.tensor(pred), 4), list("abcd"))
        self.assertEqual(s["per_class_f1"][2:], [0.0, 0.0])
        self.assertAlmostEqual(s["macro_f1"], ref["macro_f1"])
        self.assertAlmostEqual(s["macro_f1_present_classes"], np.mean(s["per_class_f1"][:2]))
        self.assertEqual(s["support"], [2, 2, 0, 0])


class TestTemperature(unittest.TestCase):
    def test_fit_matches_grid_search(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(400, 4)) * 3.0
        y = np.array([rng.choice(4, p=row) for row in cm.softmax(logits / 2)])  # informative labels
        fit = cm.fit_temperature(logits, y)
        grid = np.linspace(0.02, 5, 20001)
        best = grid[int(np.argmin([cm.nll_at_inverse_temperature(logits, y, b) for b in grid]))]
        self.assertAlmostEqual(fit["inverse_temperature"], best, delta=5e-4)
        self.assertFalse(fit["at_bound"])

    def test_recovers_known_temperature(self):
        rng = np.random.default_rng(1)
        true = rng.normal(size=(20000, 4)) * 1.5
        y = np.array([rng.choice(4, p=row) for row in cm.softmax(true)])
        fit = cm.fit_temperature(true * 2.5, y)  # overconfident logits by a factor of 2.5
        self.assertAlmostEqual(fit["temperature"], 2.5, delta=0.1)

    def test_temperature_preserves_predictions_and_lowers_nll(self):
        logits, y = random_case(scale=4.0)
        fit = cm.fit_temperature(logits, y)
        p0, p1 = cm.softmax(logits), cm.apply_temperature(logits, fit["temperature"])
        np.testing.assert_array_equal(p0.argmax(1), p1.argmax(1))
        self.assertLessEqual(cm.nll(p1, y), cm.nll(p0, y) + 1e-12)
        rows = cm.apply_temperature(logits, np.full(len(y), fit["temperature"]))
        np.testing.assert_allclose(rows, p1)

    def test_lopo_excludes_held_out_group(self):
        logits, y = random_case(n=120)
        groups = np.repeat(np.array(["a", "b", "c"]), 40)
        temps = cm.lopo_temperatures(logits, y, groups)
        self.assertEqual(sorted(temps), ["a", "b", "c"])
        self.assertAlmostEqual(temps["a"]["temperature"], cm.fit_temperature(logits[40:], y[40:])["temperature"])
        self.assertEqual(temps["b"]["n"], 80)

    def test_invalid(self):
        logits, y = random_case(n=10)
        with self.assertRaises(ValueError):
            cm.fit_temperature(logits[:0], y[:0])
        with self.assertRaises(ValueError):
            cm.fit_temperature(logits, y, beta_bounds=(1.0, 0.5))
        with self.assertRaises(ValueError):
            cm.apply_temperature(logits, 0.0)


class TestErrorDetection(unittest.TestCase):
    def test_auroc_by_hand_and_ties(self):
        s = np.array([0.9, 0.8, 0.8, 0.1])
        pos = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(cm.auroc(s, pos), (1 + 1 + 0.5 + 1) / 4)
        self.assertAlmostEqual(cm.auroc(np.zeros(4), pos), 0.5)
        self.assertTrue(math.isnan(cm.auroc(s, np.ones(4))))

    def test_auroc_matches_pairwise(self):
        rng = np.random.default_rng(3)
        s, pos = rng.integers(0, 6, 300).astype(float), rng.integers(0, 2, 300)
        p, n = s[pos == 1], s[pos == 0]
        pairwise = ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n))
        self.assertAlmostEqual(cm.auroc(s, pos), pairwise)

    def test_average_precision(self):
        s = np.array([0.9, 0.8, 0.7, 0.6])
        self.assertAlmostEqual(cm.average_precision(s, np.array([1, 0, 1, 0])), 0.5 * 1 + 0.5 * (2 / 3))
        self.assertAlmostEqual(cm.average_precision(s, np.array([1, 1, 0, 0])), 1.0)
        # all tied: one threshold, precision = base rate
        self.assertAlmostEqual(cm.average_precision(np.zeros(4), np.array([1, 0, 0, 0])), 0.25)


class TestSelective(unittest.TestCase):
    def test_order_ties_and_counts(self):
        conf = np.array([0.5, 0.9, 0.5, 0.7])
        np.testing.assert_array_equal(cm.retention_order(conf), [1, 3, 0, 2])
        np.testing.assert_array_equal(cm.retention_order(conf, higher_is_more_confident=False), [0, 2, 3, 1])
        self.assertEqual([cm.n_retained(c, 1077) for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)],
                         [1077, 969, 862, 754, 646, 539])
        with self.assertRaises(ValueError):
            cm.n_retained(0.0, 10)

    def test_risk_coverage_and_aurc(self):
        correct = np.array([1, 1, 0, 1, 0])
        rc = cm.risk_coverage(np.arange(5), correct)
        np.testing.assert_allclose(rc["risk"], [0, 0, 1 / 3, 1 / 4, 2 / 5])
        np.testing.assert_allclose(rc["oracle_risk"], [0, 0, 0, 1 / 4, 2 / 5])
        self.assertAlmostEqual(rc["aurc"], np.mean([0, 0, 1 / 3, 1 / 4, 2 / 5]))
        self.assertAlmostEqual(rc["e_aurc"], 1 / 3 / 5)
        oracle = cm.risk_coverage(np.array([0, 1, 3, 2, 4]), correct)
        self.assertAlmostEqual(oracle["e_aurc"], 0.0)

    def test_selective_at_counts_and_metrics(self):
        labels = np.array([0, 1, 2, 3, 0, 1])
        preds = np.array([0, 1, 0, 3, 0, 2])
        order = np.array([0, 1, 3, 4, 2, 5])
        s = cm.selective_at(order, labels, preds, 4 / 6, 4)
        self.assertEqual((s["n_retained"], s["n_rejected"]), (4, 2))
        self.assertEqual(s["selective_risk"], 0.0)
        self.assertEqual(s["retained_true_counts"], [2, 1, 0, 1])
        self.assertEqual(s["rejected_true_counts"], [0, 1, 1, 0])
        self.assertEqual(s["retained_pred_counts"], [2, 1, 0, 1])
        self.assertEqual(s["per_class_f1"][2], 0.0)  # wheeze absent from retained: F1 0, still in macro F1
        self.assertAlmostEqual(s["macro_f1"], 3 / 4)
        self.assertAlmostEqual(s["macro_f1_present_classes"], 1.0)


class TestBootstrap(unittest.TestCase):
    def test_resamples_whole_patients(self):
        groups = ["a"] * 3 + ["b"] * 2 + ["c"]
        draws = cm.patient_bootstrap_indices(groups, n_boot=50, seed=0)
        self.assertEqual(len(draws), 50)
        for idx in draws:
            for g, members in (("a", {0, 1, 2}), ("b", {3, 4}), ("c", {5})):
                picked = [i for i in idx if groups[i] == g]
                self.assertEqual(len(picked) % len(members), 0)
        again = cm.patient_bootstrap_indices(groups, n_boot=50, seed=0)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(draws, again)))

    def test_percentile_ci(self):
        self.assertEqual(cm.percentile_ci(list(range(1000))), [25, 974])
        self.assertTrue(math.isnan(cm.percentile_ci([float("nan")])[0]))


if __name__ == "__main__":
    unittest.main()
