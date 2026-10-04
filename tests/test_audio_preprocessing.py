"""Tests for src/features/audio_preprocessing.py (unittest).

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "features"))

import audio_preprocessing as ap  # noqa: E402

SOURCE_RATES = (4000, 10000, 44100)
METADATA = REPO_ROOT / "data" / "metadata.csv"


def sine(freq_hz: float, seconds: float, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def peak_frequency(waveform: torch.Tensor, sample_rate: int) -> float:
    spectrum = torch.fft.rfft(waveform[0].double()).abs()
    freqs = torch.fft.rfftfreq(waveform.shape[-1], 1 / sample_rate)
    return float(freqs[int(spectrum.argmax())])


def band_energy_fraction(waveform: torch.Tensor, sample_rate: int, low_hz: float) -> float:
    power = torch.fft.rfft(waveform[0].double()).abs() ** 2
    freqs = torch.fft.rfftfreq(waveform.shape[-1], 1 / sample_rate)
    return float(power[freqs > low_hz].sum() / power.sum())


class TempAudioMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data, sample_rate, subtype="PCM_24"):
        path = self.dir / name
        sf.write(str(path), data, sample_rate, subtype=subtype)
        return path


class TestLoadAudio(TempAudioMixin, unittest.TestCase):
    def test_mono_float32_shape_and_values(self):
        for rate, subtype in ((4000, "PCM_16"), (10000, "PCM_24"), (44100, "PCM_24")):
            with self.subTest(rate=rate, subtype=subtype):
                data = sine(440, 1.0, rate)
                waveform, sr = ap.load_audio(self.write(f"{rate}.wav", data, rate, subtype))
                self.assertEqual(sr, rate)
                self.assertEqual(waveform.dtype, torch.float32)
                self.assertEqual(tuple(waveform.shape), (1, len(data)))
                tol = 1e-4 if subtype == "PCM_16" else 1e-6
                self.assertTrue(torch.allclose(waveform[0], torch.from_numpy(data), atol=tol))

    def test_stereo_is_downmixed_to_mono(self):
        left, right = sine(440, 0.5, 8000), sine(880, 0.5, 8000)
        waveform, _ = ap.load_audio(self.write("stereo.wav", np.stack([left, right], axis=1), 8000))
        self.assertEqual(waveform.shape[0], 1)
        self.assertTrue(torch.allclose(waveform[0], torch.from_numpy((left + right) / 2), atol=1e-6))

    def test_missing_file(self):
        with self.assertRaises(ap.AudioLoadError):
            ap.load_audio(self.dir / "missing.wav")

    def test_unreadable_file(self):
        path = self.dir / "garbage.wav"
        path.write_bytes(b"RIFF\x00\x00not really audio")
        with self.assertRaises(ap.AudioLoadError):
            ap.load_audio(path)

    def test_empty_file(self):
        path = self.write("empty.wav", np.zeros(0, dtype=np.float32), 4000)
        with self.assertRaises(ap.AudioLoadError):
            ap.load_audio(path)


class TestCycleRange(unittest.TestCase):
    def test_sample_indices(self):
        self.assertEqual(ap.cycle_sample_range(0.5, 1.25, 4000, 8000), (2000, 5000))
        self.assertEqual(ap.cycle_sample_range(0.0, 2.0, 4000, 8000), (0, 8000))

    def test_end_within_tolerance_is_clipped(self):
        self.assertEqual(ap.cycle_sample_range(1.0, 2.005, 4000, 8000), (4000, 8000))

    def test_invalid_ranges(self):
        cases = {
            "negative start": (-0.1, 1.0),
            "end before start": (1.0, 0.5),
            "zero length": (1.0, 1.0),
            "past end": (1.0, 2.5),
            "nan": (float("nan"), 1.0),
            "inf": (0.0, math.inf),
            "starts at end": (2.0, 2.005),
        }
        for name, (start, end) in cases.items():
            with self.subTest(name), self.assertRaises(ap.CycleRangeError):
                ap.cycle_sample_range(start, end, 4000, 8000)

    def test_cycle_range_error_is_value_error(self):
        self.assertTrue(issubclass(ap.CycleRangeError, ValueError))


class TestExtractAndResample(TempAudioMixin, unittest.TestCase):
    def test_extract_matches_slice(self):
        waveform = torch.arange(8000, dtype=torch.float32).unsqueeze(0)
        cycle = ap.extract_cycle(waveform, 4000, 0.5, 1.25)
        self.assertTrue(torch.equal(cycle, waveform[:, 2000:5000]))
        cycle[0, 0] = -1  # must be a copy, not a view
        self.assertEqual(float(waveform[0, 2000]), 2000.0)

    def test_resampled_length_and_frequency(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                path = self.write(f"{rate}.wav", sine(440, 3.0, rate), rate)
                cycle = ap.load_cycle(path, 0.5, 2.0)
                self.assertEqual(cycle.sample_rate, 16000)
                self.assertEqual(cycle.original_sample_rate, rate)
                self.assertEqual(cycle.original_num_samples, round(1.5 * rate))
                # The sine crosses zero exactly at 0.5 s, so at most that one sample is trimmed.
                self.assertLessEqual(cycle.leading_zeros_removed, 1)
                kept = cycle.original_num_samples - cycle.leading_zeros_removed
                self.assertEqual(cycle.num_samples, math.ceil(kept * 16000 / rate))
                self.assertAlmostEqual(cycle.duration_s, 1.5, delta=1 / rate + 1 / 16000)
                self.assertEqual(cycle.waveform.dtype, torch.float32)
                self.assertEqual(cycle.waveform.shape[0], 1)
                self.assertAlmostEqual(peak_frequency(cycle.waveform, 16000), 440, delta=2)

    def test_target_rate_input_is_unchanged(self):
        data = sine(440, 1.0, 16000)
        cycle = ap.load_cycle(self.write("16k.wav", data, 16000), 0.0, 1.0)
        self.assertEqual(cycle.leading_zeros_removed, 1)  # sin(0) == 0 exactly
        self.assertTrue(torch.allclose(cycle.waveform[0], torch.from_numpy(data[1:]), atol=1e-6))

    def test_upsampled_4khz_has_no_content_above_2khz(self):
        # Documented limitation: upsampling cannot create content above the source Nyquist.
        rng = np.random.default_rng(0)
        noise = (0.3 * rng.standard_normal(8000)).astype(np.float32)
        cycle = ap.load_cycle(self.write("noise4k.wav", noise, 4000), 0.0, 2.0)
        self.assertEqual(cycle.effective_bandwidth_hz, 2000)
        # The sinc filter's transition band leaks a little energy just above 2 kHz
        # (~0.5% above 2.1 kHz for white noise); above 2.5 kHz there is effectively none.
        self.assertLess(band_energy_fraction(cycle.waveform, 16000, 2500), 1e-4)

    def test_effective_bandwidth(self):
        for rate, expected in ((4000, 2000), (10000, 5000), (44100, 8000)):
            path = self.write(f"bw{rate}.wav", sine(440, 1.0, rate), rate)
            self.assertEqual(ap.load_cycle(path, 0.0, 1.0).effective_bandwidth_hz, expected)

    def test_no_normalization(self):
        path = self.write("quiet.wav", sine(440, 1.0, 44100, amplitude=0.01), 44100)
        peak = float(ap.load_cycle(path, 0.0, 1.0).waveform.abs().max())
        self.assertAlmostEqual(peak, 0.01, delta=0.002)

    def test_deterministic(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                path = self.write(f"det{rate}.wav", sine(523, 2.0, rate), rate)
                a = ap.load_cycle(path, 0.25, 1.75).waveform
                b = ap.load_cycle(path, 0.25, 1.75).waveform
                self.assertTrue(torch.equal(a, b))

    def test_load_cycle_invalid_range(self):
        path = self.write("short.wav", sine(440, 1.0, 4000), 4000)
        with self.assertRaises(ap.CycleRangeError):
            ap.load_cycle(path, 0.5, 1.5)

    def test_invalid_resample_rate(self):
        with self.assertRaises(ValueError):
            ap.resample(torch.zeros(1, 10), 0)

    def test_metadata_row(self):
        path = self.write("101_1b1_Al_sc_Meditron.wav", sine(440, 2.0, 4000), 4000)
        row = {"recording_filename": path.name, "cycle_start_s": "0.5", "cycle_end_s": "1.0", "sample_rate": "4000"}
        cycle = ap.load_cycle_from_metadata(row, self.dir)
        self.assertEqual((cycle.source, cycle.original_num_samples), (path.name, 2000))
        with self.assertRaises(ap.AudioLoadError):
            ap.load_cycle_from_metadata({**row, "sample_rate": "44100"}, self.dir)


class TestLeadingSilence(TempAudioMixin, unittest.TestCase):
    def noise(self, seconds, rate, amplitude=0.2, seed=0):
        x = np.random.default_rng(seed).uniform(-amplitude, amplitude, int(round(seconds * rate)))
        x[x == 0] = amplitude / 2  # guarantee no accidental exact zeros
        return x.astype(np.float32)

    def test_count_leading_zeros(self):
        w = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0, 0.2]])
        self.assertEqual(ap.count_leading_zeros(w, 10), 3)
        self.assertEqual(ap.count_leading_zeros(w, 2), 2)
        self.assertEqual(ap.count_leading_zeros(torch.tensor([[0.3, 0.0]]), 10), 0)
        self.assertEqual(ap.count_leading_zeros(torch.zeros(1, 5), 10), 4)  # keeps one sample

    def test_leading_zeros_trimmed_at_every_rate(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                n_zero = round(0.09 * rate)
                body = self.noise(1.0, rate)
                path = self.write(f"z{rate}.wav", np.concatenate([np.zeros(n_zero, np.float32), body]), rate)
                cycle = ap.load_cycle(path, 0.0, 1.09)
                self.assertEqual(cycle.leading_zeros_removed, n_zero)
                self.assertAlmostEqual(cycle.leading_silence_removed_ms, 90.0, delta=1000 / rate)
                expected = ap.resample(ap.load_audio(path)[0][:, n_zero:round(1.09 * rate)], rate)
                self.assertTrue(torch.equal(cycle.waveform, expected))

    def test_low_amplitude_audio_is_not_trimmed(self):
        for rate, subtype in ((4000, "PCM_16"), (44100, "PCM_24")):
            with self.subTest(rate=rate):
                quiet = self.noise(0.1, rate, amplitude=2e-4)  # well above 1 LSB, never exactly zero
                data = np.concatenate([quiet, self.noise(0.5, rate)])
                path = self.write(f"q{rate}.wav", data, rate, subtype)
                self.assertEqual(ap.load_cycle(path, 0.0, 0.6).leading_zeros_removed, 0)

    def test_only_leading_zeros_are_removed(self):
        rate = 44100
        zeros = np.zeros(round(0.05 * rate), np.float32)
        data = np.concatenate([zeros, self.noise(0.3, rate), zeros, self.noise(0.3, rate, seed=1)])
        cycle = ap.load_cycle(self.write("mid.wav", data, rate), 0.0, len(data) / rate)
        self.assertEqual(cycle.leading_zeros_removed, len(zeros))

    def test_trim_cap_150ms(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                data = np.concatenate([np.zeros(round(0.3 * rate), np.float32), self.noise(0.5, rate)])
                cycle = ap.load_cycle(self.write(f"cap{rate}.wav", data, rate), 0.0, 0.8)
                self.assertEqual(cycle.leading_zeros_removed, round(0.150 * rate))
                self.assertAlmostEqual(cycle.leading_silence_removed_ms, 150.0, delta=1000 / rate)

    def test_all_zero_cycle_keeps_one_sample(self):
        path = self.write("silent.wav", np.zeros(400, np.float32), 4000)  # 100 ms, below the cap
        cycle = ap.load_cycle(path, 0.0, 0.1)
        self.assertEqual(cycle.leading_zeros_removed, 399)
        self.assertGreaterEqual(cycle.num_samples, 1)

    def test_original_metadata_unchanged(self):
        rate = 10000
        data = np.concatenate([np.zeros(round(0.2 * rate), np.float32), self.noise(1.0, rate)])
        path = self.write("meta.wav", data, rate)
        raw_before = path.read_bytes()
        trimmed = ap.load_cycle(path, 0.1, 0.9)
        untrimmed = ap.load_cycle(path, 0.1, 0.9, trim_leading_zeros=False)
        self.assertEqual(untrimmed.leading_zeros_removed, 0)
        self.assertEqual(trimmed.leading_zeros_removed, round(0.1 * rate))
        for c in (trimmed, untrimmed):
            self.assertEqual((c.cycle_start_s, c.cycle_end_s), (0.1, 0.9))
            self.assertEqual(c.original_num_samples, round(0.8 * rate))
        self.assertEqual(trimmed.original_clipped_fraction, untrimmed.original_clipped_fraction)
        self.assertEqual(trimmed.original_peak_abs, untrimmed.original_peak_abs)
        self.assertEqual(path.read_bytes(), raw_before)  # raw file is never modified

    def test_trimming_is_deterministic(self):
        data = np.concatenate([np.zeros(500, np.float32), self.noise(0.5, 44100)])
        path = self.write("det.wav", data, 44100)
        self.assertTrue(torch.equal(ap.load_cycle(path, 0.0, 0.5).waveform, ap.load_cycle(path, 0.0, 0.5).waveform))


class TestClippedPlateaus(TempAudioMixin, unittest.TestCase):
    def signal(self, rate, plateaus_ms, seconds=1.0, gap_ms=50):
        """Quiet non-clipped noise with full-scale plateaus of the given lengths inserted."""
        x = np.random.default_rng(rate).uniform(-0.3, 0.3, int(round(seconds * rate))).astype(np.float32)
        pos, sign = round(0.1 * rate), 1.0
        for ms in plateaus_ms:
            n = round(ms * rate / 1000)
            x[pos:pos + n] = sign
            pos += n + round(gap_ms * rate / 1000)
            sign = -sign
        return x

    def test_min_plateau_samples(self):
        self.assertEqual(ap.min_plateau_samples(4000), 128)
        self.assertEqual(ap.min_plateau_samples(10000), 320)
        self.assertEqual(ap.min_plateau_samples(44100), 1412)  # 32 ms = 1411.2 samples

    def test_clipped_runs(self):
        w = torch.tensor([0.0, 1.0, 1.0, 0.5, -0.9995, -1.0, -1.0, 0.2, 1.0])
        self.assertEqual(ap.clipped_runs(w), [(1, 2), (4, 3), (8, 1)])
        self.assertEqual(ap.clipped_runs(torch.zeros(5)), [])

    def test_single_plateau_at_every_rate(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                x = torch.from_numpy(self.signal(rate, [50]))
                s = ap.clipping_stats(x, rate)
                self.assertEqual(s.clipped_plateau_count, 1)
                self.assertAlmostEqual(s.max_clipped_plateau_ms, 50.0, delta=1000 / rate)
                self.assertAlmostEqual(s.clipped_plateau_fraction, 0.05, delta=2 / rate)
                self.assertAlmostEqual(s.clipped_fraction, 0.05, delta=2 / rate)

    def test_multiple_plateaus(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                s = ap.clipping_stats(torch.from_numpy(self.signal(rate, [40, 100, 35])), rate)
                self.assertEqual(s.clipped_plateau_count, 3)
                self.assertAlmostEqual(s.max_clipped_plateau_ms, 100.0, delta=1000 / rate)
                self.assertAlmostEqual(s.clipped_plateau_fraction, 0.175, delta=4 / rate)

    def test_short_clipping_does_not_count(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                s = ap.clipping_stats(torch.from_numpy(self.signal(rate, [5, 20, 31])), rate)
                self.assertEqual(s.clipped_plateau_count, 0)
                self.assertEqual(s.clipped_plateau_fraction, 0.0)
                self.assertAlmostEqual(s.max_clipped_plateau_ms, 31.0, delta=1000 / rate)
                self.assertGreater(s.clipped_fraction, 0.05)  # still counted as clipped samples

    def test_threshold_boundary(self):
        rate = 4000
        n = ap.min_plateau_samples(rate)
        for length, expected in ((n - 1, 0), (n, 1)):
            with self.subTest(length=length):
                x = torch.zeros(1000)
                x[100:100 + length] = 0.999
                self.assertEqual(ap.clipping_stats(x, rate).clipped_plateau_count, expected)
        x = torch.full((1000,), 0.9989)
        self.assertEqual(ap.clipping_stats(x, rate).clipped_plateau_count, 0)

    def test_no_clipping(self):
        s = ap.clipping_stats(torch.from_numpy(self.signal(44100, [])), 44100)
        self.assertEqual((s.clipped_plateau_count, s.max_clipped_plateau_ms, s.clipped_plateau_fraction,
                          s.clipped_fraction), (0, 0.0, 0.0, 0.0))

    def test_load_cycle_reports_plateaus_from_original_audio(self):
        for rate in SOURCE_RATES:
            with self.subTest(rate=rate):
                data = self.signal(rate, [60, 10])
                path = self.write(f"clip{rate}.wav", data, rate)
                raw_before = path.read_bytes()
                cycle = ap.load_cycle(path, 0.0, 1.0)
                self.assertEqual(cycle.clipped_plateau_count, 1)
                self.assertAlmostEqual(cycle.max_clipped_plateau_ms, 60.0, delta=1000 / rate)
                self.assertAlmostEqual(cycle.original_clipped_fraction, 0.07, delta=3 / rate)
                self.assertEqual(path.read_bytes(), raw_before)
                # Audio is not repaired: the waveform equals plain resampling of the original.
                original, _ = ap.load_audio(path)
                self.assertTrue(torch.equal(cycle.waveform, ap.resample(original[:, :rate], rate)))

    def test_deterministic(self):
        for rate in SOURCE_RATES:
            path = self.write(f"d{rate}.wav", self.signal(rate, [45, 80]), rate)
            a, b = ap.load_cycle(path, 0.0, 1.0), ap.load_cycle(path, 0.0, 1.0)
            self.assertEqual((a.clipped_plateau_count, a.max_clipped_plateau_ms, a.clipped_plateau_fraction),
                             (b.clipped_plateau_count, b.max_clipped_plateau_ms, b.clipped_plateau_fraction))


@unittest.skipUnless(METADATA.exists() and ap.DEFAULT_AUDIO_DIR.exists(), "ICBHI data or metadata not present")
class TestRealDataset(unittest.TestCase):
    """Smoke test on real ICBHI cycles: the first and last cycle of each source rate."""

    @classmethod
    def setUpClass(cls):
        with open(METADATA, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_rate = {}
        for r in rows:
            by_rate.setdefault(int(r["sample_rate"]), []).append(r)
        cls.samples = [r for rate in sorted(by_rate) for r in (by_rate[rate][0], by_rate[rate][-1])]

    def test_real_cycles(self):
        self.assertEqual({int(r["sample_rate"]) for r in self.samples}, set(SOURCE_RATES))
        for row in self.samples:
            with self.subTest(file=row["recording_filename"], start=row["cycle_start_s"]):
                cycle = ap.load_cycle_from_metadata(row)
                rate = int(row["sample_rate"])
                self.assertEqual(cycle.original_sample_rate, rate)
                self.assertEqual(cycle.waveform.dtype, torch.float32)
                self.assertEqual(cycle.waveform.ndim, 2)
                self.assertEqual(cycle.waveform.shape[0], 1)
                kept = cycle.original_num_samples - cycle.leading_zeros_removed
                self.assertEqual(cycle.num_samples, math.ceil(kept * 16000 / rate))
                self.assertAlmostEqual(cycle.duration_s + cycle.leading_silence_removed_ms / 1000,
                                       float(row["cycle_duration_s"]), delta=0.002)
                self.assertTrue(torch.isfinite(cycle.waveform).all())
                self.assertTrue(torch.equal(cycle.waveform, ap.load_cycle_from_metadata(row).waveform))


if __name__ == "__main__":
    unittest.main()
