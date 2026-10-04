"""Tests for src/data/create_splits.py (standard-library unittest).

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import csv
import io
import random
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

import create_splits as cs  # noqa: E402

DEVICES = {"AKGC417L": "44100", "Meditron": "44100", "LittC2SE": "44100", "Litt3200": "4000"}
DIAGNOSES = ("COPD", "Healthy", "URTI")
TEST_CANDIDATES = 200  # keeps tests fast; the CLI default is larger


def synthetic_rows(n_patients: int = 60, seed: int = 0) -> list[dict[str, str]]:
    """Per-cycle metadata rows for patients of varied size, device, and label mix."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_patients):
        pid = str(100 + i)
        device, diagnosis = rng.choice(sorted(DEVICES)), rng.choice(DIAGNOSES)
        profile = i % 4  # mix of none / crackle / wheeze / both-type patients
        for rec in range(rng.randint(1, 4)):
            for _ in range(rng.randint(3, 12)):
                label = "normal"
                if profile in (1, 3) and rng.random() < 0.4:
                    label = "crackle"
                if profile in (2, 3) and rng.random() < 0.3:
                    label = "both" if label == "crackle" else "wheeze"
                rows.append({
                    "recording_filename": f"{pid}_{rec}b1_Al_sc_{device}.wav",
                    "patient_id": pid,
                    "diagnosis": diagnosis,
                    "device": device,
                    "sample_rate": DEVICES[device],
                    "cycle_label": label,
                })
    return rows


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_and_assign(rows, method=cs.DEFAULT_METHOD, ratios=cs.DEFAULT_RATIOS, seed=cs.DEFAULT_SEED,
                     candidates=TEST_CANDIDATES):
    patients = cs.summarize_patients(rows)
    return patients, cs.assign_splits(patients, cs.parse_ratios(ratios), seed, method, candidates)


class TestLabelProfile(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(cs.label_profile({"normal": 5}), "none")
        self.assertEqual(cs.label_profile({"normal": 5, "crackle": 1}), "crackle")
        self.assertEqual(cs.label_profile({"wheeze": 2}), "wheeze")
        self.assertEqual(cs.label_profile({"crackle": 1, "wheeze": 1}), "crackle+wheeze")
        self.assertEqual(cs.label_profile({"both": 1}), "crackle+wheeze")

    def test_patient_summary(self):
        rows = [
            {"recording_filename": "a.wav", "patient_id": "1", "cycle_label": "normal",
             "device": "Meditron", "sample_rate": "44100"},
            {"recording_filename": "a.wav", "patient_id": "1", "cycle_label": "crackle",
             "device": "Meditron", "sample_rate": "44100"},
            {"recording_filename": "b.wav", "patient_id": "1", "cycle_label": "normal",
             "device": "Litt3200", "sample_rate": "4000"},
        ]
        p = cs.summarize_patients(rows)["1"]
        self.assertEqual((p.n_cycles, p.n_recordings, p.label_counts), (3, 2, (2, 1, 0, 0)))
        self.assertEqual(p.device_cycles, {"Litt3200": 1, "Meditron": 2})
        self.assertEqual(p.sample_rate_cycles, {"4000": 1, "44100": 2})
        self.assertEqual(p.label_profile, "crackle")


class TestRatiosAndQuotas(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(cs.parse_ratios([0.6, 0.2, 0.2]), {"train": 0.6, "val": 0.2, "test": 0.2})

    def test_invalid(self):
        for bad in ([0.7, 0.3], [0.7, 0.2, 0.2], [1.0, 0.0, 0.0], [0.8, 0.3, -0.1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cs.parse_ratios(bad)

    def test_quotas(self):
        ratios = cs.parse_ratios(cs.DEFAULT_RATIOS)
        self.assertEqual(cs.patient_quotas(126, ratios), {"train": 88, "val": 19, "test": 19})
        self.assertEqual(cs.patient_quotas(100, ratios), {"train": 70, "val": 15, "test": 15})
        for n in range(3, 300):
            self.assertEqual(sum(cs.patient_quotas(n, cs.parse_ratios([0.6, 0.2, 0.2])).values()), n)

    def test_quota_too_small(self):
        with self.assertRaises(ValueError):
            cs.patient_quotas(2, cs.parse_ratios(cs.DEFAULT_RATIOS))


class TestScore(unittest.TestCase):
    def test_tvd(self):
        self.assertEqual(cs._tvd(Counter(a=1, b=1), Counter(a=5, b=5)), 0.0)
        self.assertEqual(cs._tvd(Counter(a=1), Counter(b=1)), 1.0)
        self.assertAlmostEqual(cs._tvd(Counter(a=3, b=1), Counter(a=1, b=1)), 0.25)

    def test_device_skew_is_penalized(self):
        rows = synthetic_rows()
        patients = cs.summarize_patients(rows)
        ratios = cs.parse_ratios(cs.DEFAULT_RATIOS)
        quotas = cs.patient_quotas(len(patients), ratios)
        # Deliberately sort by device so test gets one device only.
        by_device = sorted(patients, key=lambda p: (sorted(patients[p].device_cycles)[0], p))
        skewed = dict(zip(by_device, [s for s in cs.SPLITS for _ in range(quotas[s])]))
        balanced = cs.assign_splits(patients, ratios, 0, "patient-balanced", TEST_CANDIDATES)
        self.assertGreater(cs.score_split(patients, skewed, ratios)["device_tvd"],
                           cs.score_split(patients, balanced, ratios)["device_tvd"])

    def test_diagnosis_does_not_affect_split(self):
        rows = synthetic_rows()
        relabeled = [{**r, "diagnosis": "X" + r["patient_id"]} for r in rows]
        self.assertEqual(split_and_assign(rows)[1], split_and_assign(relabeled)[1])


class TestAssignment(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()

    def test_reproducible(self):
        for method in cs.METHODS:
            with self.subTest(method=method):
                self.assertEqual(split_and_assign(self.rows, method, seed=7)[1],
                                 split_and_assign(self.rows, method, seed=7)[1])

    def test_independent_of_row_order(self):
        shuffled = self.rows[:]
        random.Random(1).shuffle(shuffled)
        for method in cs.METHODS:
            with self.subTest(method=method):
                self.assertEqual(split_and_assign(self.rows, method)[1], split_and_assign(shuffled, method)[1])

    def test_zero_leakage_and_full_coverage(self):
        for method in cs.METHODS:
            with self.subTest(method=method):
                patients, assignment = split_and_assign(self.rows, method)
                groups = {s: {p for p, x in assignment.items() if x == s} for s in cs.SPLITS}
                self.assertFalse(groups["train"] & groups["val"])
                self.assertFalse(groups["train"] & groups["test"])
                self.assertFalse(groups["val"] & groups["test"])
                self.assertEqual(set().union(*groups.values()), set(patients))
                self.assertTrue(all(groups.values()))

    def test_recordings_never_span_splits(self):
        for method in cs.METHODS:
            with self.subTest(method=method):
                _, assignment = split_and_assign(self.rows, method)
                rec_splits = {}
                for r in self.rows:
                    rec_splits.setdefault(r["recording_filename"], set()).add(assignment[r["patient_id"]])
                self.assertTrue(all(len(s) == 1 for s in rec_splits.values()))

    def test_patient_balanced_exact_patient_quotas(self):
        for ratios in ((0.70, 0.15, 0.15), (0.60, 0.20, 0.20)):
            with self.subTest(ratios=ratios):
                patients, assignment = split_and_assign(self.rows, ratios=ratios)
                expected = cs.patient_quotas(len(patients), cs.parse_ratios(ratios))
                self.assertEqual(dict(Counter(assignment.values())), expected)

    def test_patient_balanced_is_best_candidate_seen(self):
        patients = cs.summarize_patients(self.rows)
        ratios = cs.parse_ratios(cs.DEFAULT_RATIOS)
        one = cs.assign_splits(patients, ratios, 5, "patient-balanced", 1)
        many = cs.assign_splits(patients, ratios, 5, "patient-balanced", TEST_CANDIDATES)
        # Same seed: the first candidate is among those searched, so more candidates never score worse.
        self.assertLessEqual(cs.score_split(patients, many, ratios)["total"],
                             cs.score_split(patients, one, ratios)["total"])

    def test_cycle_greedy_cycle_proportions_near_targets(self):
        for ratios in ((0.70, 0.15, 0.15), (0.60, 0.20, 0.20)):
            with self.subTest(ratios=ratios):
                _, assignment = split_and_assign(self.rows, "cycle-greedy", ratios=ratios)
                stats = cs.split_statistics(self.rows, assignment)
                for split, target in zip(cs.SPLITS, ratios):
                    self.assertAlmostEqual(stats[split]["cycle_fraction"], target, delta=0.05)
                self.assertEqual(sum(stats[s]["cycles"] for s in cs.SPLITS), len(self.rows))

    def test_cycle_greedy_largest_patient_goes_to_train(self):
        rows = self.rows + [
            {"recording_filename": "999_1b1_Al_sc_Meditron.wav", "patient_id": "999", "diagnosis": "COPD",
             "device": "Meditron", "sample_rate": "44100", "cycle_label": "crackle"}
        ] * 150
        _, assignment = split_and_assign(rows, "cycle-greedy")
        self.assertEqual(assignment["999"], "train")

    def test_too_few_patients_fails(self):
        rows = [r for r in self.rows if r["patient_id"] in ("100", "101")]
        with self.assertRaises(ValueError):
            split_and_assign(rows, "patient-balanced")
        with self.assertRaises(cs.SplitIntegrityError):
            split_and_assign(rows, "cycle-greedy")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            split_and_assign(self.rows, "random")


class TestVerifySplit(unittest.TestCase):
    expected = {"1", "2", "3", "4"}

    def test_valid(self):
        cs.verify_split([("1", "train"), ("2", "train"), ("3", "val"), ("4", "test")], self.expected)

    def assert_fails(self, pairs, fragment):
        with self.assertRaises(cs.SplitIntegrityError) as ctx:
            cs.verify_split(pairs, self.expected)
        self.assertTrue(any(fragment in e for e in ctx.exception.errors), ctx.exception.errors)

    def test_patient_in_two_splits(self):
        self.assert_fails([("1", "train"), ("1", "test"), ("2", "train"), ("3", "val"), ("4", "test")],
                          "LEAKAGE: patient 1")

    def test_duplicate_listing(self):
        self.assert_fails([("1", "train"), ("1", "train"), ("2", "train"), ("3", "val"), ("4", "test")],
                          "listed 2 times")

    def test_missing_patient(self):
        self.assert_fails([("1", "train"), ("3", "val"), ("4", "test")], "not assigned")

    def test_unexpected_patient(self):
        self.assert_fails([("1", "train"), ("2", "train"), ("3", "val"), ("4", "test"), ("5", "test")],
                          "not in metadata")

    def test_unknown_split_name(self):
        self.assert_fails([("1", "train"), ("2", "holdout"), ("3", "val"), ("4", "test")], "unknown split")

    def test_empty_split(self):
        self.assert_fails([("1", "train"), ("2", "train"), ("3", "train"), ("4", "test")],
                          "'val' has no patients")


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.metadata = self.root / "metadata.csv"
        write_metadata(self.metadata, synthetic_rows())

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cs.main(["--metadata", str(self.metadata), "--candidates", str(TEST_CANDIDATES), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_outputs_are_byte_identical_across_runs(self):
        for method in cs.METHODS:
            with self.subTest(method=method):
                dirs = [self.root / method / "a", self.root / method / "b"]
                for d in dirs:
                    code, stdout, _ = self.run_main("--output-dir", str(d), "--seed", "3", "--method", method)
                    self.assertEqual(code, 0)
                    self.assertIn("Device (% of split cycles)", stdout)
                    self.assertIn("Sample rate Hz", stdout)
                for name in (cs.SPLIT_FILE, cs.SUMMARY_FILE):
                    self.assertEqual((dirs[0] / name).read_bytes(), (dirs[1] / name).read_bytes())
                pairs = cs.read_split_file(dirs[0] / cs.SPLIT_FILE)
                cs.verify_split(pairs, {r["patient_id"] for r in synthetic_rows()})

    def test_invalid_ratios_exit_nonzero(self):
        out_dir = self.root / "bad"
        code, _, stderr = self.run_main("--output-dir", str(out_dir), "--ratios", "0.5", "0.5", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("sum to 1", stderr)
        self.assertFalse(out_dir.exists())


@unittest.skipUnless(cs.DEFAULT_METADATA.exists(), "data/metadata.csv not present")
class TestRealMetadata(unittest.TestCase):
    """Checks against the real metadata table, if it has been generated locally."""

    def test_real_split(self):
        rows = cs.load_metadata(cs.DEFAULT_METADATA)
        patients, assignment = split_and_assign(rows)
        ratios = cs.parse_ratios(cs.DEFAULT_RATIOS)
        self.assertEqual(dict(Counter(assignment.values())), cs.patient_quotas(len(patients), ratios))
        score = cs.score_split(patients, assignment, ratios)
        self.assertLess(score["device_tvd"], 0.15)
        self.assertLess(score["label_tvd"], 0.15)


if __name__ == "__main__":
    unittest.main()
