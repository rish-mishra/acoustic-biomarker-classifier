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
                self.assertEqual(cycle.num_samples, math.ceil(cycle.original_num_samples * 16000 / rate))
                self.assertAlmostEqual(cycle.duration_s, 1.5, delta=1 / 16000)
                self.assertEqual(cycle.waveform.dtype, torch.float32)
                self.assertEqual(cycle.waveform.shape[0], 1)
                self.assertAlmostEqual(peak_frequency(cycle.waveform, 16000), 440, delta=2)

    def test_target_rate_input_is_unchanged(self):
        data = sine(440, 1.0, 16000)
        cycle = ap.load_cycle(self.write("16k.wav", data, 16000), 0.0, 1.0)
        self.assertTrue(torch.allclose(cycle.waveform[0], torch.from_numpy(data), atol=1e-6))

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
                self.assertEqual(cycle.num_samples, math.ceil(cycle.original_num_samples * 16000 / rate))
                self.assertAlmostEqual(cycle.duration_s, float(row["cycle_duration_s"]), delta=0.002)
                self.assertTrue(torch.isfinite(cycle.waveform).all())
                self.assertTrue(torch.equal(cycle.waveform, ap.load_cycle_from_metadata(row).waveform))


if __name__ == "__main__":
    unittest.main()
