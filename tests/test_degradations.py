"""Tests for src/features/degradations.py (synthetic waveforms, CPU)."""

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "features"))

import degradations as dg  # noqa: E402

SR = dg.SAMPLE_RATE


def tone(freq_hz, seconds=1.0, amp=0.3, dc=0.0):
    t = torch.arange(int(seconds * SR), dtype=torch.float64) / SR
    return (amp * torch.sin(2 * math.pi * freq_hz * t) + dc).to(torch.float32).unsqueeze(0)


def breathing_like(seed=0, seconds=1.3):
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    return (0.1 * torch.randn(1, n, generator=g) + tone(300, seconds, 0.2)).to(torch.float32)


def gain_db(x, y):
    return 10 * math.log10(dg.centered_power(y) / dg.centered_power(x))


class TestPredeclaredConditions(unittest.TestCase):
    def test_exact_condition_set(self):
        got = [(c.family, c.param) for c in dg.PREDECLARED_CONDITIONS]
        self.assertEqual(got, [("clean", None),
                               *(("gaussian_noise", s) for s in (20.0, 10.0, 5.0, 0.0)),
                               *(("amplitude", g) for g in (0.5, 0.25, 1.5, 2.0)),
                               *(("lowpass", c) for c in (2000.0, 1000.0, 500.0)),
                               *(("temporal_dropout", p) for p in (0.10, 0.25, 0.50))])
        self.assertEqual(len({c.name for c in dg.PREDECLARED_CONDITIONS}), 15)


class TestShapeAndPurity(unittest.TestCase):
    def test_every_condition_preserves_shape_dtype_and_input(self):
        for n in (1, 7, 319, 320, 321, 3888, 16000):
            x = breathing_like(seconds=n / SR) if n > 1 else torch.tensor([[0.25]])
            orig = x.clone()
            for c in dg.PREDECLARED_CONDITIONS:
                y, _ = dg.apply(c, x, "rec.wav", 3)
                self.assertEqual(y.shape, x.shape, (c.name, n))
                self.assertEqual(y.dtype, torch.float32)
                self.assertTrue(torch.isfinite(y).all(), (c.name, n))
                self.assertTrue(torch.equal(x, orig), f"{c.name} modified its input")

    def test_clean_is_identity(self):
        x = breathing_like()
        y, stats = dg.apply(dg.PREDECLARED_CONDITIONS[0], x, "r", 0)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(stats, {})
        self.assertIsNot(x, y)


class TestDeterminism(unittest.TestCase):
    def test_same_cycle_same_output_regardless_of_order(self):
        xs = {("a.wav", i): breathing_like(seed=i) for i in range(4)}
        for c in dg.PREDECLARED_CONDITIONS:
            first = {k: dg.apply(c, x, *k)[0] for k, x in xs.items()}
            second = {k: dg.apply(c, xs[k], *k)[0] for k in reversed(list(xs))}
            for k in xs:
                self.assertTrue(torch.equal(first[k], second[k]), (c.name, k))

    def test_seed_depends_on_cycle_identity_and_base_seed(self):
        s = dg.example_seed(42, "gaussian_noise", "a.wav", 0)
        self.assertEqual(s, dg.example_seed(42, "gaussian_noise", "a.wav", 0))
        others = {dg.example_seed(42, "gaussian_noise", "a.wav", 1), dg.example_seed(42, "gaussian_noise", "b.wav", 0),
                  dg.example_seed(43, "gaussian_noise", "a.wav", 0), dg.example_seed(42, "temporal_dropout", "a.wav", 0)}
        self.assertNotIn(s, others)
        self.assertTrue(0 <= s < 2 ** 63)

    def test_different_cycles_get_different_noise(self):
        x = breathing_like()
        c = dg.Condition("n", "gaussian_noise", 10.0, "dB SNR")
        a, _ = dg.apply(c, x, "a.wav", 0)
        b, _ = dg.apply(c, x, "a.wav", 1)
        self.assertFalse(torch.equal(a, b))

    def test_noise_levels_share_one_realization(self):
        x = breathing_like()
        n20 = dg.apply(dg.Condition("a", "gaussian_noise", 20.0, ""), x, "r", 0)[0].double() - x.double()
        n0 = dg.apply(dg.Condition("b", "gaussian_noise", 0.0, ""), x, "r", 0)[0].double() - x.double()
        self.assertTrue(torch.allclose(n0, n20 * 10, atol=1e-5))


class TestGaussianNoise(unittest.TestCase):
    def test_requested_snr_is_realized(self):
        for seed, dc in ((0, 0.0), (1, 0.3)):  # DC offset must not inflate the signal power
            x = breathing_like(seed) + dc
            for snr in (20.0, 10.0, 5.0, 0.0, -3.0):
                y = dg.add_gaussian_noise(x, snr, dg.generator_for(seed))
                self.assertAlmostEqual(dg.snr_db(x, y), snr, delta=1e-3, msg=(seed, snr))

    def test_apply_reports_realized_snr(self):
        x = breathing_like()
        for c in dg.PREDECLARED_CONDITIONS:
            if c.family == "gaussian_noise":
                _, stats = dg.apply(c, x, "r", 0)
                self.assertAlmostEqual(stats["realized_snr_db"], c.param, delta=1e-3)
                self.assertFalse(stats["silent_input"])

    def test_noise_is_white_and_zero_mean(self):
        x = tone(300, seconds=4.0)
        y = dg.add_gaussian_noise(x, 0.0, dg.generator_for(0))
        noise = (y.double() - x.double())[0]
        self.assertAlmostEqual(float(noise.mean()), 0.0, delta=1e-6)
        spec = torch.fft.rfft(noise).abs() ** 2
        low, high = spec[1: len(spec) // 2].mean(), spec[len(spec) // 2:].mean()
        self.assertAlmostEqual(float(low / high), 1.0, delta=0.1)

    def test_silent_input_is_unchanged(self):
        x = torch.full((1, 800), 0.2)
        y, stats = dg.apply(dg.Condition("n", "gaussian_noise", 10.0, ""), x, "r", 0)
        self.assertTrue(torch.equal(x, y))
        self.assertTrue(stats["silent_input"])


class TestAmplitude(unittest.TestCase):
    def test_gain_is_exact_and_unclipped(self):
        x = tone(300, amp=0.9)
        for g in (0.5, 0.25, 1.5, 2.0):
            y = dg.scale_amplitude(x, g)
            self.assertTrue(torch.allclose(y, x * g, atol=1e-7))
            self.assertAlmostEqual(gain_db(x, y), 20 * math.log10(g), delta=1e-4)
        y, stats = dg.apply(dg.Condition("g", "amplitude", 2.0, "x"), x, "r", 0)
        self.assertGreater(stats["peak_abs_after"], 1.7)  # 0.9 * 2: not clipped to 1


class TestLowpass(unittest.TestCase):
    def test_passband_and_stopband(self):
        for cutoff in (2000.0, 1000.0, 500.0):
            passed = dg.lowpass(tone(cutoff * 0.6, 2.0), cutoff)
            stopped = dg.lowpass(tone(cutoff * 1.4, 2.0), cutoff)
            mid = slice(SR // 4, -SR // 4)  # ignore edge effects
            self.assertAlmostEqual(gain_db(tone(cutoff * 0.6, 2.0)[:, mid], passed[:, mid]), 0.0, delta=0.05)
            self.assertLess(gain_db(tone(cutoff * 1.4, 2.0)[:, mid], stopped[:, mid]), -60)

    def test_zero_phase_and_unit_dc(self):
        x = torch.zeros(1, 4001)
        x[0, 2000] = 1.0
        y = dg.lowpass(x, 1000.0)[0]
        self.assertEqual(int(y.abs().argmax()), 2000)  # no delay
        self.assertTrue(torch.allclose(y[2000 - 300:2000], y[2001:2301].flip(0), atol=1e-7))  # symmetric
        self.assertAlmostEqual(float(dg.lowpass_kernel(1000.0).sum()), 1.0, places=12)

    def test_cutoff_is_minus_6db(self):
        h = dg.lowpass_kernel(1000.0)
        n = torch.arange(len(h), dtype=torch.float64)
        resp = abs(complex(float((h * torch.cos(2 * math.pi * 1000 / SR * n)).sum()),
                           float((h * torch.sin(2 * math.pi * 1000 / SR * n)).sum())))
        self.assertAlmostEqual(20 * math.log10(resp), -6.02, delta=0.1)


class TestTemporalDropout(unittest.TestCase):
    def test_deterministic_and_exact_block_count(self):
        x = breathing_like(seconds=1.0)  # 16000 samples = 50 blocks of 320
        for p in (0.10, 0.25, 0.50):
            a = dg.temporal_dropout(x, p, dg.generator_for(7))
            b = dg.temporal_dropout(x, p, dg.generator_for(7))
            self.assertTrue(torch.equal(a, b))
            zeroed = (a == 0) & (x != 0)
            self.assertEqual(int(zeroed.sum()), math.floor(p * 50 + 0.5) * 320)  # round half up: 12.5 -> 13
            blocks = zeroed[0].reshape(50, 320)
            self.assertTrue(((blocks.all(1)) | (~blocks.any(1))).all(), "drops whole 20 ms blocks only")
            kept = ~zeroed
            self.assertTrue(torch.equal(a[kept], x[kept]))

    def test_levels_are_nested_and_stats_match(self):
        x = breathing_like(seconds=0.97)
        masks = {}
        for c in dg.PREDECLARED_CONDITIONS:
            if c.family == "temporal_dropout":
                y, stats = dg.apply(c, x, "r", 5)
                masks[c.param] = (y == 0) & (x != 0)
                self.assertAlmostEqual(stats["dropped_fraction"], float(masks[c.param].double().mean()), places=12)
                self.assertAlmostEqual(stats["dropped_fraction"], c.param, delta=320 / x.shape[-1])
        self.assertTrue(torch.equal(masks[0.10] & masks[0.25], masks[0.10]))
        self.assertTrue(torch.equal(masks[0.25] & masks[0.50], masks[0.25]))

    def test_round_half_up_block_count(self):
        self.assertEqual(int(dg.dropout_mask(320 * 10, 0.25, dg.generator_for(0)).sum()), 3 * 320)  # 2.5 -> 3
        self.assertEqual(int(dg.dropout_mask(100, 0.5, dg.generator_for(0)).sum()), 100)  # 1 block, 0.5 -> 1


class TestInvalidParameters(unittest.TestCase):
    def test_rejected(self):
        x = breathing_like()
        g = dg.generator_for(0)
        bad = [
            lambda: dg.add_gaussian_noise(x, float("nan"), g),
            lambda: dg.add_gaussian_noise(x, float("inf"), g),
            lambda: dg.add_gaussian_noise(x, "10", g),
            lambda: dg.scale_amplitude(x, 0.0),
            lambda: dg.scale_amplitude(x, -1.0),
            lambda: dg.scale_amplitude(x, float("nan")),
            lambda: dg.scale_amplitude(x, True),
            lambda: dg.lowpass(x, 0.0),
            lambda: dg.lowpass(x, 8000.0),
            lambda: dg.lowpass(x, -5.0),
            lambda: dg.lowpass_kernel(1000.0, num_taps=1024),
            lambda: dg.temporal_dropout(x, 0.0, g),
            lambda: dg.temporal_dropout(x, 1.0, g),
            lambda: dg.temporal_dropout(x, -0.1, g),
            lambda: dg.temporal_dropout(x, 0.1, g, block_s=0.0),
            lambda: dg.apply(dg.Condition("x", "reverb", 1.0, ""), x, "r", 0),
        ]
        for i, f in enumerate(bad):
            with self.assertRaises(ValueError, msg=i):
                f()

    def test_bad_waveforms_rejected(self):
        for w in (torch.zeros(5), torch.zeros(2, 5), torch.zeros(1, 0), torch.tensor([[0.1, float("nan")]])):
            for c in dg.PREDECLARED_CONDITIONS:
                with self.assertRaises(ValueError, msg=(c.name, tuple(w.shape))):
                    dg.apply(c, w, "r", 0)


if __name__ == "__main__":
    unittest.main()
