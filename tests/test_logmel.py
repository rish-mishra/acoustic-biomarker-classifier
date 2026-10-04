"""Tests for src/features/logmel.py (unittest).

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import csv
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "features"))

import audio_preprocessing as ap  # noqa: E402
import logmel as lm  # noqa: E402

METADATA = REPO_ROOT / "data" / "metadata.csv"
EXTRACTOR = lm.LogMelExtractor()
EXTRACTOR_128 = lm.LogMelExtractor(lm.MEL128_ABLATION_CONFIG)
ALL_CONFIGS = (lm.PRIMARY_CONFIG, lm.MEL128_ABLATION_CONFIG, lm.BANDWIDTH_ABLATION_CONFIG)


def sine(freq_hz, seconds, sample_rate=16000, amplitude=0.5):
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_cycle(waveform, original_sample_rate=16000, source="synthetic.wav"):
    w = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
    return ap.RespiratoryCycle(
        waveform=w, sample_rate=16000, original_sample_rate=original_sample_rate,
        original_num_samples=w.shape[-1], cycle_start_s=0.0, cycle_end_s=w.shape[-1] / 16000, source=source,
        original_peak_abs=float(w.abs().max()) if w.numel() else 0.0, original_clipped_fraction=0.0,
    )


class TestConfig(unittest.TestCase):
    def test_primary(self):
        c = lm.PRIMARY_CONFIG
        self.assertEqual((c.win_length, c.hop_length, c.n_fft, c.n_mels), (512, 160, 512, 64))
        self.assertEqual((c.f_min, c.f_max), (50.0, 2000.0))

    def test_ablations_change_one_thing(self):
        self.assertEqual(lm.MEL128_ABLATION_CONFIG, replace(lm.PRIMARY_CONFIG, n_mels=128, n_fft=1024))
        self.assertEqual(lm.BANDWIDTH_ABLATION_CONFIG, replace(lm.PRIMARY_CONFIG, f_max=8000.0))

    def test_invalid_configs(self):
        for kwargs in ({"f_max": 9000.0}, {"f_min": 2000.0}, {"win_length": 4096}, {"hop_length": 0},
                       {"n_mels": 0}, {"log_floor": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lm.LogMelConfig(**kwargs)

    def test_no_empty_filters_in_shipped_configs(self):
        for config in ALL_CONFIGS:
            with self.subTest(config=config):
                self.assertEqual(lm.empty_mel_filters(config), 0)
                fb = lm.LogMelExtractor(config).mel.mel_scale.fb
                self.assertEqual(int((fb.sum(0) == 0).sum()), 0)

    def test_primary_n_fft_is_minimal(self):
        # n_fft cannot be below win_length, so the primary uses no zero-padding at all.
        self.assertEqual(lm.PRIMARY_CONFIG.n_fft, lm.PRIMARY_CONFIG.win_length)
        self.assertEqual(lm.BANDWIDTH_ABLATION_CONFIG.n_fft, lm.BANDWIDTH_ABLATION_CONFIG.win_length)

    def test_mel128_needs_zero_padding(self):
        for n_fft in (512, 768, 938):
            with self.subTest(n_fft=n_fft):
                self.assertGreater(lm.empty_mel_filters(replace(lm.MEL128_ABLATION_CONFIG, n_fft=n_fft)), 0)

    def test_empty_mel_filters_rejected(self):
        with self.assertRaises(ValueError):
            lm.LogMelExtractor(replace(lm.MEL128_ABLATION_CONFIG, n_fft=512))


class TestDCRemoval(unittest.TestCase):
    def test_remove_dc(self):
        w = torch.from_numpy(sine(300, 1.0)).unsqueeze(0) + 0.3
        self.assertAlmostEqual(float(lm.remove_dc(w).mean()), 0.0, places=6)

    def test_features_invariant_to_dc_offset(self):
        base = sine(300, 1.0)
        for extractor in (EXTRACTOR, EXTRACTOR_128):
            a = extractor(make_cycle(base)).logmel
            b = extractor(make_cycle(base + 0.25))
            # Only float32 rounding differs: tiny in strong cells, up to a few
            # hundredths of a dB in cells 70+ dB below the peak.
            strong = a > -60
            self.assertTrue(torch.allclose(a[strong], b.logmel[strong], atol=0.005))
            self.assertTrue(torch.allclose(a, b.logmel, atol=0.05))
            self.assertAlmostEqual(b.quality.dc_offset, 0.25, places=4)


class TestShapesAndValues(unittest.TestCase):
    def test_output_dimensions(self):
        for extractor, n_mels in ((EXTRACTOR, 64), (EXTRACTOR_128, 128)):
            for n in (1, 159, 160, 3200, 16000, 40123):
                with self.subTest(n_mels=n_mels, n=n):
                    out = extractor(make_cycle(sine(300, n / 16000) if n > 1 else [0.1])).logmel
                    self.assertEqual(tuple(out.shape), (1, n_mels, 1 + n // 160))
                    self.assertEqual(out.dtype, torch.float32)

    def test_variable_duration_is_preserved(self):
        frames = [EXTRACTOR(make_cycle(sine(300, s))).logmel.shape[-1] for s in (0.5, 2.0, 7.3)]
        self.assertEqual(frames, [51, 201, 731])

    def test_finite_values(self):
        rng = np.random.default_rng(0)
        for config in ALL_CONFIGS:
            extractor = lm.LogMelExtractor(config)
            for w in (sine(300, 1.0), rng.standard_normal(16000).astype(np.float32) * 2.0,
                      np.zeros(1600, np.float32)):
                self.assertTrue(torch.isfinite(extractor(make_cycle(w)).logmel).all())

    def test_silence_hits_floor_exactly(self):
        for extractor in (EXTRACTOR, EXTRACTOR_128):
            out = extractor(make_cycle(np.zeros(4000, np.float32))).logmel
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.all(out == -100.0))

    def test_constant_signal_is_silence_after_dc_removal(self):
        out = EXTRACTOR(make_cycle(np.full(4000, 0.4, np.float32))).logmel
        self.assertLessEqual(float(out.max()), -90.0)

    def test_very_short_cycles(self):
        for n in (1, 2, 50, 3200):  # 3200 samples = 0.2 s, the shortest real ICBHI cycle
            with self.subTest(n=n):
                rng = np.random.default_rng(n)
                out = EXTRACTOR(make_cycle(rng.standard_normal(n).astype(np.float32) * 0.1)).logmel
                self.assertEqual(out.shape[-1], 1 + n // 160)
                self.assertTrue(torch.isfinite(out).all())

    def test_deterministic(self):
        w = np.random.default_rng(1).standard_normal(24000).astype(np.float32)
        for config in ALL_CONFIGS:
            a = lm.LogMelExtractor(config)(make_cycle(w)).logmel
            b = lm.LogMelExtractor(config)(make_cycle(w)).logmel
            self.assertTrue(torch.equal(a, b))

    def test_rejects_wrong_input(self):
        with self.assertRaises(ValueError):
            EXTRACTOR.logmel(torch.zeros(2, 100))
        with self.assertRaises(ValueError):
            EXTRACTOR.logmel(torch.zeros(1, 0))
        cycle = replace(make_cycle(sine(300, 0.5)), sample_rate=8000)
        with self.assertRaises(ValueError):
            EXTRACTOR(cycle)


class TestFrequencyLimits(unittest.TestCase):
    @staticmethod
    def mel_center_hz(config, index):
        hz2mel = lambda f: 2595 * math.log10(1 + f / 700)
        lo, hi = hz2mel(config.f_min), hz2mel(config.f_max)
        m = lo + (index + 1) * (hi - lo) / (config.n_mels + 1)
        return 700 * (10 ** (m / 2595) - 1)

    def band_peak(self, extractor, freq):
        out = extractor(make_cycle(sine(freq, 1.0))).logmel[0].mean(dim=-1)
        return int(out.argmax()), float(out.max())

    def test_in_band_tone_lands_in_matching_mel_bin(self):
        for extractor in (EXTRACTOR, EXTRACTOR_128):
            for freq in (100, 400, 1000, 1800):
                with self.subTest(n_mels=extractor.config.n_mels, freq=freq):
                    idx, _ = self.band_peak(extractor, freq)
                    self.assertAlmostEqual(self.mel_center_hz(extractor.config, idx), freq, delta=40)

    def test_out_of_band_tones_are_excluded(self):
        for extractor in (EXTRACTOR, EXTRACTOR_128):
            _, in_band = self.band_peak(extractor, 1000)
            for freq in (3000, 6000):
                with self.subTest(n_mels=extractor.config.n_mels, freq=freq):
                    _, out_band = self.band_peak(extractor, freq)
                    self.assertLess(out_band, in_band - 60)

    def test_below_f_min_leaks_only_into_lowest_bands(self):
        # f_min is a filterbank edge, not a steep high-pass: with a 32 ms window
        # (~31 Hz resolution) a 15 Hz tone leaks into the lowest bands but is
        # >= 60 dB below an in-band tone from ~250 Hz upward.
        _, in_band = self.band_peak(EXTRACTOR, 1000)
        per_band = EXTRACTOR(make_cycle(sine(15, 1.0))).logmel[0].mean(dim=-1)
        centers = torch.tensor([self.mel_center_hz(lm.PRIMARY_CONFIG, i) for i in range(64)])
        self.assertLess(float(per_band[centers > 250].max()), in_band - 60)

    def test_bandwidth_ablation_covers_up_to_8khz(self):
        ablation = lm.LogMelExtractor(lm.BANDWIDTH_ABLATION_CONFIG)
        idx, level = self.band_peak(ablation, 6000)
        _, ref = self.band_peak(ablation, 1000)
        self.assertAlmostEqual(self.mel_center_hz(lm.BANDWIDTH_ABLATION_CONFIG, idx), 6000, delta=200)
        self.assertGreater(level, ref - 10)


class TestSourceRatesAndQuality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cycle_from_file(self, data, rate, start=0.0, end=None, **kwargs):
        path = self.dir / f"x{rate}.wav"
        sf.write(str(path), data, rate, subtype="PCM_24")
        return ap.load_cycle(path, start, end if end is not None else len(data) / rate, **kwargs)

    def test_source_rates(self):
        ablation = lm.LogMelExtractor(lm.BANDWIDTH_ABLATION_CONFIG)
        for rate, bw, limited_in_ablation in ((4000, 2000, True), (10000, 5000, True), (44100, 8000, False)):
            with self.subTest(rate=rate):
                cycle = self.cycle_from_file(sine(500, 2.0, rate) + 0.1, rate, 0.25, 1.75)
                for extractor, n_mels in ((EXTRACTOR, 64), (EXTRACTOR_128, 128)):
                    feats = extractor(cycle)
                    self.assertEqual(tuple(feats.logmel.shape), (1, n_mels, 1 + cycle.num_samples // 160))
                    self.assertTrue(torch.isfinite(feats.logmel).all())
                q = EXTRACTOR(cycle).quality
                self.assertEqual((q.original_sample_rate, q.effective_bandwidth_hz), (rate, bw))
                self.assertFalse(q.band_limited)
                self.assertEqual(ablation(cycle).quality.band_limited, limited_in_ablation)
                self.assertAlmostEqual(q.duration_s, 1.5, delta=1e-3)
                self.assertAlmostEqual(q.dc_offset, 0.1, delta=0.01)
                self.assertEqual(q.leading_samples_removed, 0)

    def test_leading_silence_removed_before_features(self):
        for rate in (4000, 10000, 44100):
            with self.subTest(rate=rate):
                body = np.random.default_rng(rate).uniform(0.05, 0.3, rate).astype(np.float32)
                body *= np.where(np.arange(rate) % 2, 1, -1).astype(np.float32)  # non-zero, sign-alternating
                data = np.concatenate([np.zeros(round(0.09 * rate), np.float32), body])
                trimmed = EXTRACTOR(self.cycle_from_file(data, rate))
                untrimmed = EXTRACTOR(self.cycle_from_file(data, rate, trim_leading_zeros=False))
                self.assertEqual(trimmed.quality.leading_samples_removed, round(0.09 * rate))
                self.assertAlmostEqual(trimmed.quality.leading_silence_removed_ms, 90.0, delta=1000 / rate)
                self.assertEqual(trimmed.logmel.shape[-1], untrimmed.logmel.shape[-1] - 9)
                self.assertEqual(untrimmed.quality.leading_samples_removed, 0)
                # Untrimmed features start with near-floor frames; trimmed ones do not.
                self.assertLess(float(untrimmed.logmel[0, :, 2].max()), -60)
                self.assertGreater(float(trimmed.logmel[0, :, 2].max()), -40)
                self.assertEqual((trimmed.cycle_start_s, trimmed.cycle_end_s),
                                 (untrimmed.cycle_start_s, untrimmed.cycle_end_s))

    def test_clipping_metrics(self):
        raw = 1.5 * np.sin(2 * np.pi * 200 * np.arange(44100) / 44100)
        clipped = np.clip(raw, -1.0, 1.0).astype(np.float32)
        q = EXTRACTOR(self.cycle_from_file(clipped, 44100)).quality
        expected = float(np.mean(np.abs(raw) >= ap.CLIP_THRESHOLD))
        self.assertAlmostEqual(q.original_clipped_fraction, expected, delta=0.01)
        self.assertGreater(q.original_clipped_fraction, 0.4)
        self.assertGreaterEqual(q.original_peak_abs, ap.CLIP_THRESHOLD)

    def test_unclipped_signal(self):
        q = EXTRACTOR(self.cycle_from_file(sine(200, 1.0, 44100, amplitude=0.3), 44100)).quality
        self.assertEqual(q.original_clipped_fraction, 0.0)
        self.assertAlmostEqual(q.original_peak_abs, 0.3, delta=1e-3)
        self.assertAlmostEqual(q.peak_abs, 0.3, delta=0.01)

    def test_quality_as_dict(self):
        q = EXTRACTOR(self.cycle_from_file(sine(200, 1.0, 4000), 4000)).quality.as_dict()
        self.assertEqual(set(q), {
            "dc_offset", "peak_abs", "original_peak_abs", "original_clipped_fraction",
            "clipped_plateau_count", "max_clipped_plateau_ms", "clipped_plateau_fraction",
            "leading_samples_removed", "leading_silence_removed_ms", "original_sample_rate",
            "effective_bandwidth_hz", "duration_s", "band_limited",
        })


SPLIT_FILE = REPO_ROOT / "data" / "splits" / "patient_split.csv"


@unittest.skipUnless(METADATA.exists() and SPLIT_FILE.exists() and ap.DEFAULT_AUDIO_DIR.exists(),
                     "ICBHI data, metadata, or split not present")
class TestRealDataset(unittest.TestCase):
    """Real cycles from train/val patients only; the test split's audio is never loaded."""

    def test_real_cycles_each_rate(self):
        with open(SPLIT_FILE, newline="", encoding="utf-8") as f:
            split = {r["patient_id"]: r["split"] for r in csv.DictReader(f)}
        with open(METADATA, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if split[r["patient_id"]] in ("train", "val")]
        picks = {}
        for r in rows:
            picks.setdefault(r["sample_rate"], r)
        # An AKGC417L cycle starting within the recording's leading digital silence.
        picks["akgc"] = next(r for r in rows if r["device"] == "AKGC417L" and float(r["cycle_start_s"]) < 0.05)
        for key, row in picks.items():
            with self.subTest(key=key):
                cycle = ap.load_cycle_from_metadata(row)
                for extractor, n_mels in ((EXTRACTOR, 64), (EXTRACTOR_128, 128)):
                    feats = extractor(cycle)
                    self.assertEqual(feats.logmel.shape[:2], (1, n_mels))
                    self.assertTrue(torch.isfinite(feats.logmel).all())
                    self.assertEqual(feats.quality.original_sample_rate, int(row["sample_rate"]))
                    self.assertTrue(0.0 <= feats.quality.original_clipped_fraction <= 1.0)
        self.assertGreater(ap.load_cycle_from_metadata(picks["akgc"]).leading_zeros_removed, 0)


if __name__ == "__main__":
    unittest.main()
