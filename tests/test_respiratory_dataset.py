"""Tests for src/data/respiratory_dataset.py and src/data/normalization.py (unittest).

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

import normalization as nz  # noqa: E402
import respiratory_dataset as rd  # noqa: E402

ap, lm = rd.ap, rd.lm

# patient -> (split, sample rate, device, noise level)
PATIENTS = {
    "101": ("train", 4000, "Litt3200", 0.05),
    "102": ("train", 44100, "AKGC417L", 0.20),
    "103": ("val", 10000, "Meditron", 0.10),
    "104": ("val", 44100, "LittC2SE", 0.15),
    "105": ("test", 4000, "Meditron", 0.08),
    "106": ("test", 44100, "AKGC417L", 0.25),
}
# (start, end, label) per recording; all four classes, varied durations
CYCLES = [(0.0, 0.6, "normal"), (0.6, 1.5, "crackle"), (1.5, 2.1, "wheeze"), (2.1, 3.4, "both")]


class SyntheticICBHI:
    def __init__(self, root: Path):
        self.root = root
        self.audio = root / "audio"
        self.audio.mkdir(parents=True)
        self.metadata = root / "metadata.csv"
        self.split_file = root / "patient_split.csv"
        rows = []
        for pid, (split, rate, device, level) in PATIENTS.items():
            fn = f"{pid}_1b1_Al_sc_{device}.wav"
            self.write_audio(fn, rate, level, seed=int(pid))
            for i, (start, end, label) in enumerate(CYCLES):
                rows.append({"recording_filename": fn, "cycle_index": i, "patient_id": pid, "diagnosis": "X",
                             "device": device, "sample_rate": rate, "cycle_start_s": start, "cycle_end_s": end,
                             "cycle_duration_s": round(end - start, 6), "cycle_label": label})
        with open(self.metadata, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        self.write_split({pid: v[0] for pid, v in PATIENTS.items()})

    def write_audio(self, filename, rate, level, seed, seconds=3.5):
        rng = np.random.default_rng(seed)
        t = np.arange(int(seconds * rate)) / rate
        x = level * rng.standard_normal(len(t)) + 0.3 * np.sin(2 * np.pi * 300 * t)
        sf.write(str(self.audio / filename), np.clip(x, -1, 1).astype(np.float32), rate, subtype="PCM_24")

    def write_split(self, assignment, extra_rows=()):
        with open(self.split_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["patient_id", "split"])
            for pid, split in list(assignment.items()) + list(extra_rows):
                w.writerow([pid, split])

    def dataset(self, split, normalizer=None, cache=rd.DEFAULT_CACHE_RECORDINGS):
        return rd.RespiratoryCycleDataset(split, self.metadata, self.split_file, self.audio,
                                          normalizer=normalizer, cache_recordings=cache)


class SyntheticTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = SyntheticICBHI(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def fit(self):
        return nz.compute_band_stats(self.data.dataset("train"))


class TestClassMapping(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(rd.CLASS_NAMES, ("normal", "crackle", "wheeze", "both"))
        self.assertEqual([rd.label_to_index(c) for c in rd.CLASS_NAMES], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            rd.label_to_index("stridor")


class TestDataset(SyntheticTestCase):
    def test_lengths_and_classes(self):
        for split in rd.SPLITS:
            with self.subTest(split=split):
                ds = self.data.dataset(split)
                self.assertEqual(len(ds), 8)
                self.assertEqual(ds.label_counts(), {c: 2 for c in rd.CLASS_NAMES})
                self.assertEqual(sorted(ds[i]["label"] for i in range(len(ds))), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_item_contents(self):
        ds = self.data.dataset("train")
        item = ds[1]
        self.assertEqual(item["features"].dtype, torch.float32)
        self.assertEqual(tuple(item["features"].shape[:2]), (1, 64))
        self.assertEqual(item["num_frames"], item["features"].shape[-1])
        self.assertEqual((item["label"], item["patient_id"], item["cycle_index"]), (1, "101", 1))
        self.assertEqual(item["recording_filename"], "101_1b1_Al_sc_Litt3200.wav")
        self.assertEqual((item["original_sample_rate"], item["device"], item["split"]), (4000, "Litt3200", "train"))
        self.assertAlmostEqual(item["cycle_duration_s"], 0.9)
        self.assertFalse(item["normalized"])
        for key in ("original_clipped_fraction", "clipped_plateau_count", "leading_samples_removed",
                    "dc_offset", "effective_bandwidth_hz"):
            self.assertIn(key, item["quality"])

    def test_variable_lengths_preserved(self):
        ds = self.data.dataset("val")
        frames = [ds[i]["num_frames"] for i in range(4)]
        expected = [1 + round((e - s) * 16000) // 160 for s, e, _ in CYCLES]
        self.assertEqual(frames, expected)

    def test_matches_direct_pipeline(self):
        ds = self.data.dataset("test")
        for i in range(len(ds)):
            r = ds.rows[i]
            direct = lm.LogMelExtractor()(ap.load_cycle(self.data.audio / r["recording_filename"],
                                                        float(r["cycle_start_s"]), float(r["cycle_end_s"])))
            self.assertTrue(torch.equal(ds[i]["features"], direct.logmel))

    def test_deterministic_regardless_of_cache_and_order(self):
        a = self.data.dataset("train", cache=1)
        b = self.data.dataset("train", cache=32)
        order = [7, 0, 5, 2, 6, 1, 4, 3]
        first = {i: a[i]["features"] for i in order}
        for i in range(len(b)):
            self.assertTrue(torch.equal(first[i], b[i]["features"]))
            self.assertTrue(torch.equal(first[i], a[i]["features"]))

    def test_cache_is_bounded(self):
        cache = rd.RecordingCache(self.data.audio, max_recordings=2)
        for fn in sorted(p.name for p in self.data.audio.glob("*.wav")):
            cache.get(fn)
        self.assertEqual(len(cache), 2)
        cache.get(sorted(p.name for p in self.data.audio.glob("*.wav"))[-1])
        self.assertEqual((cache.hits, cache.misses), (1, 6))
        with self.assertRaises(ValueError):
            rd.RecordingCache(self.data.audio, max_recordings=0)

    def test_sample_rate_mismatch_fails(self):
        with open(self.data.metadata, newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["sample_rate"] = "44100"
        with open(self.data.metadata, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        with self.assertRaises(ap.AudioLoadError):
            self.data.dataset("train")[0]

    def test_invalid_split_and_label(self):
        with self.assertRaises(ValueError):
            self.data.dataset("holdout")
        text = self.data.metadata.read_text().replace(",normal\n", ",stridor\n", 1)
        self.data.metadata.write_text(text)
        with self.assertRaises(ValueError):
            self.data.dataset("train")


class TestLeakage(SyntheticTestCase):
    def test_each_split_has_only_its_patients(self):
        expected = {s: {p for p, v in PATIENTS.items() if v[0] == s} for s in rd.SPLITS}
        for split in rd.SPLITS:
            ds = self.data.dataset(split)
            self.assertEqual(ds.patients, expected[split])
            self.assertEqual({ds[i]["patient_id"] for i in range(len(ds))}, expected[split])

    def test_patient_sets_disjoint(self):
        sets = [self.data.dataset(s).patients for s in rd.SPLITS]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sets[0] | sets[1] | sets[2], set(PATIENTS))

    def test_patient_in_two_splits_fails(self):
        assignment = {pid: v[0] for pid, v in PATIENTS.items()}
        self.data.write_split(assignment, extra_rows=[("101", "test")])
        with self.assertRaises(rd.SplitIntegrityError):
            self.data.dataset("train")

    def test_unassigned_patient_fails(self):
        assignment = {pid: v[0] for pid, v in PATIENTS.items() if pid != "106"}
        self.data.write_split(assignment)
        with self.assertRaises(Exception) as ctx:
            self.data.dataset("test")
        self.assertIn("not assigned", str(ctx.exception))


class TestNormalization(SyntheticTestCase):
    def test_stats_equal_manual_train_computation(self):
        norm = self.fit()
        train = self.data.dataset("train")
        frames = torch.cat([train[i]["features"][0] for i in range(len(train))], dim=1).double()
        self.assertTrue(torch.allclose(norm.mean.double(), frames.mean(dim=1), atol=1e-4))
        self.assertTrue(torch.allclose(norm.std.double(), frames.std(dim=1, unbiased=False), atol=1e-3))
        self.assertEqual(norm.train_patients, ("101", "102"))
        self.assertEqual((norm.n_cycles, norm.n_frames), (8, frames.shape[1]))
        self.assertGreaterEqual(float(norm.std.min()), nz.MIN_BAND_STD_DB)

    def test_normalized_train_frames_are_standardized(self):
        norm = self.fit()
        train = self.data.dataset("train", normalizer=norm)
        frames = torch.cat([train[i]["features"][0] for i in range(len(train))], dim=1).double()
        self.assertTrue(torch.allclose(frames.mean(dim=1), torch.zeros(64, dtype=torch.float64), atol=1e-4))
        self.assertTrue(torch.allclose(frames.std(dim=1, unbiased=False), torch.ones(64, dtype=torch.float64), atol=1e-3))
        self.assertTrue(train[0]["normalized"])

    def test_formula(self):
        norm = self.fit()
        raw, normed = self.data.dataset("val")[2], self.data.dataset("val", normalizer=norm)[2]
        expected = (raw["features"] - norm.mean[:, None]) / norm.std[:, None]
        self.assertTrue(torch.equal(normed["features"], expected))

    def test_only_train_split_can_be_fitted(self):
        for split in ("val", "test"):
            with self.assertRaises(ValueError):
                nz.compute_band_stats(self.data.dataset(split))
        with self.assertRaises(ValueError):
            nz.compute_band_stats(self.data.dataset("train", normalizer=self.fit()))

    def test_val_and_test_audio_cannot_change_stats(self):
        before = self.fit()
        for pid in ("103", "104", "105", "106"):  # replace every val/test recording with loud, different audio
            _, rate, device, _ = PATIENTS[pid]
            self.data.write_audio(f"{pid}_1b1_Al_sc_{device}.wav", rate, level=0.9, seed=999)
        after = self.fit()
        self.assertTrue(torch.equal(before.mean, after.mean))
        self.assertTrue(torch.equal(before.std, after.std))

    def test_changing_train_audio_changes_stats(self):
        before = self.fit()
        _, rate, device, _ = PATIENTS["101"]
        self.data.write_audio(f"101_1b1_Al_sc_{device}.wav", rate, level=0.9, seed=999)
        self.assertFalse(torch.equal(before.mean, self.fit().mean))

    def test_save_load_roundtrip_and_reproducibility(self):
        a, b = self.data.root / "a.json", self.data.root / "b.json"
        nz.save_normalizer(self.fit(), a, self.data.metadata, self.data.split_file)
        nz.save_normalizer(self.fit(), b, self.data.metadata, self.data.split_file)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        loaded = nz.load_normalizer(a)
        self.assertTrue(torch.equal(loaded.mean, self.fit().mean))
        payload = json.loads(a.read_text())
        self.assertEqual(payload["train_patients"], ["101", "102"])
        self.assertEqual(len(payload["metadata_sha256"]), 64)

    def test_normalizer_must_match_frozen_train_patients(self):
        norm = self.fit()
        bad = nz.BandNormalizer(norm.mean, norm.std, norm.config, ("101", "103"), norm.n_cycles, norm.n_frames)
        with self.assertRaises(ValueError):
            self.data.dataset("val", normalizer=bad)
        other_config = lm.MEL128_ABLATION_CONFIG
        with self.assertRaises(ValueError):
            rd.RespiratoryCycleDataset("val", self.data.metadata, self.data.split_file, self.data.audio,
                                       config=other_config, normalizer=norm)

    def test_rejects_tiny_or_invalid_std(self):
        mean = torch.zeros(64)
        for std in (torch.full((64,), 0.01), torch.full((64,), float("nan")), torch.ones(32)):
            with self.assertRaises(ValueError):
                nz.BandNormalizer(mean, std, lm.PRIMARY_CONFIG, ("1",), 1, 1)


class TestCollate(SyntheticTestCase):
    def test_padding_lengths_and_mask(self):
        ds = self.data.dataset("train", normalizer=self.fit())
        items = [ds[i] for i in range(4)]
        stored = [item["features"].clone() for item in items]
        batch = rd.collate_cycles(items)
        lengths = [item["num_frames"] for item in items]
        T = max(lengths)
        self.assertEqual(tuple(batch["features"].shape), (4, 1, 64, T))
        self.assertEqual(batch["lengths"].tolist(), lengths)
        self.assertEqual(batch["mask"].dtype, torch.bool)
        self.assertEqual(batch["mask"].sum(dim=1).tolist(), lengths)
        for i, n in enumerate(lengths):
            self.assertTrue(torch.equal(batch["features"][i, ..., :n], stored[i]))
            self.assertTrue(torch.all(batch["features"][i, ..., n:] == 0))
            self.assertTrue(batch["mask"][i, :n].all() and not batch["mask"][i, n:].any())
            self.assertTrue(torch.equal(items[i]["features"], stored[i]))  # item not modified
        self.assertEqual(batch["labels"].tolist(), [0, 1, 2, 3])
        self.assertEqual(batch["patient_id"], ["101"] * 4)
        self.assertEqual(batch["cycle_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(batch["original_sample_rate"].tolist(), [4000] * 4)

    def test_unnormalized_padding_refused_by_default(self):
        ds = self.data.dataset("train")
        items = [ds[0], ds[1]]
        with self.assertRaises(ValueError):
            rd.collate_cycles(items)
        batch = rd.collate_cycles(items, pad_value=-100.0, require_normalized=False)
        self.assertTrue(torch.all(batch["features"][0, ..., items[0]["num_frames"]:] == -100.0))

    def test_dataloader(self):
        ds = self.data.dataset("val", normalizer=self.fit())
        loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=rd.collate_cycles)
        batches = list(loader)
        self.assertEqual([b["features"].shape[0] for b in batches], [3, 3, 2])
        self.assertEqual(sum(int(b["lengths"].sum()) for b in batches), sum(ds[i]["num_frames"] for i in range(len(ds))))


class TestQualityGroups(unittest.TestCase):
    def q(self, clipped, plateaus):
        return {"original_clipped_fraction": clipped, "clipped_plateau_count": plateaus}

    def test_definitions(self):
        cases = [
            (self.q(0.0, 0), {"all", "moderately_clean", "clean"}),
            (self.q(0.01, 0), {"all", "moderately_clean", "clean"}),  # boundaries are inclusive
            (self.q(0.0101, 0), {"all", "moderately_clean"}),
            (self.q(0.005, 1), {"all", "moderately_clean"}),  # any >= 32 ms plateau excludes from clean
            (self.q(0.10, 3), {"all", "moderately_clean"}),
            (self.q(0.1001, 0), {"all"}),
        ]
        for quality, expected in cases:
            with self.subTest(quality=quality):
                self.assertEqual({g for g in rd.QUALITY_GROUPS if rd.in_quality_group(quality, g)}, expected)
        with self.assertRaises(ValueError):
            rd.in_quality_group(self.q(0, 0), "very_clean")

    def test_batch_mask(self):
        batch = {"quality": [self.q(0.0, 0), self.q(0.05, 0), self.q(0.5, 2)]}
        self.assertEqual(rd.quality_group_mask(batch, "all").tolist(), [True, True, True])
        self.assertEqual(rd.quality_group_mask(batch, "moderately_clean").tolist(), [True, True, False])
        self.assertEqual(rd.quality_group_mask(batch, "clean").tolist(), [True, False, False])


@unittest.skipUnless(rd.DEFAULT_METADATA.exists() and rd.DEFAULT_SPLIT_FILE.exists()
                     and ap.DEFAULT_AUDIO_DIR.exists(), "ICBHI data, metadata, or split not present")
class TestRealData(unittest.TestCase):
    """Real-data checks. Only train/val audio is ever loaded; the test split is
    checked through the metadata and split files alone (no audio, no features)."""

    DEVELOPMENT_SPLITS = ("train", "val")

    def test_splits_cover_metadata_without_overlap(self):
        rows = rd.read_metadata(rd.DEFAULT_METADATA)
        frozen = rd.read_split_assignments(rd.DEFAULT_SPLIT_FILE)
        rd.verify_split(frozen.items(), {r["patient_id"] for r in rows})
        per_split = {s: [r for r in rows if frozen[r["patient_id"]] == s] for s in rd.SPLITS}
        self.assertEqual(sum(len(v) for v in per_split.values()), len(rows))
        for split in self.DEVELOPMENT_SPLITS:
            ds = rd.RespiratoryCycleDataset(split)
            self.assertEqual(len(ds), len(per_split[split]))
            self.assertEqual(ds.patients, {p for p, s in frozen.items() if s == split})
            self.assertTrue(all(n > 0 for n in ds.label_counts().values()))
            item = ds[0]
            self.assertEqual(item["features"].shape[:2], (1, 64))
            self.assertTrue(torch.isfinite(item["features"]).all())

    @unittest.skipUnless(nz.DEFAULT_STATS_PATH.exists(), "normalization statistics not generated")
    def test_saved_stats_match_frozen_train_split(self):
        norm = nz.load_normalizer()
        self.assertEqual(frozenset(norm.train_patients), rd.train_patients_from_split_file())
        self.assertEqual(norm.config, lm.PRIMARY_CONFIG)
        self.assertGreaterEqual(float(norm.std.min()), nz.MIN_BAND_STD_DB)
        ds = rd.RespiratoryCycleDataset("val", normalizer=norm)
        batch = rd.collate_cycles([ds[i] for i in range(4)])
        self.assertTrue(torch.isfinite(batch["features"]).all())


if __name__ == "__main__":
    unittest.main()
