"""Deterministic waveform degradations for the validation robustness study.

Every degradation acts on a 16 kHz mono cycle waveform of shape (1, n), i.e.
the output of audio_preprocessing (after leading-zero trimming and
resampling) and before Log-Mel extraction. The frozen preprocessing, Log-Mel
config, and normalization statistics are untouched. Each function returns a
new float32 tensor with the input's shape; inputs are never modified.

Predeclared conditions (PREDECLARED_CONDITIONS):
  clean                      identity (reference)
  gaussian noise             20, 10, 5, 0 dB SNR
  amplitude scaling          0.5x, 0.25x, 1.5x, 2.0x
  low-pass filtering         2000, 1000, 500 Hz cutoff
  temporal dropout           10%, 25%, 50% of 20 ms blocks

Definitions:
  Gaussian noise   White Gaussian noise over the full 0-8 kHz band. Signal and
                   noise powers are mean squares of the mean-removed (DC-free)
                   waveforms, because Log-Mel removes DC; the noise is
                   mean-removed and rescaled so the realized SNR equals the
                   requested SNR exactly. Only ~24% of white-noise power falls
                   inside the 50-2000 Hz feature band, so the in-band SNR is
                   higher than the nominal broadband SNR.
  Amplitude        x * gain. No clipping: the frozen pipeline already leaves
                   values outside [-1, 1] unchanged, and clipping would mix a
                   second, non-linear distortion into a pure gain change.
  Low-pass         Linear-phase windowed-sinc FIR (Blackman window,
                   LOWPASS_NUM_TAPS taps, unit DC gain), applied centered
                   ("same" convolution) so there is no delay and the length is
                   unchanged. The cutoff is the sinc's -6 dB point; the
                   transition band is ~+-45 Hz around it. Samples beyond the
                   cycle are treated as zeros (edge effect over the first and
                   last 32 ms).
  Temporal dropout The cycle is cut into consecutive DROPOUT_BLOCK_S (20 ms)
                   blocks (the last may be shorter); round-half-up(fraction x
                   n_blocks) randomly chosen blocks are set to exactly zero
                   (silence, no concealment).

Determinism: randomness comes only from a torch.Generator seeded by
example_seed(base_seed, family, recording, cycle_index), so results do not
depend on evaluation order or batching. The seed is shared within a family:
the four noise levels scale the same noise realization, and the dropout
levels use one block permutation, so a 25% dropout removes a superset of the
10% blocks.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
import torchaudio.functional as AF

SAMPLE_RATE = 16_000
LOWPASS_NUM_TAPS = 1025  # 64 ms at 16 kHz; odd, so the filter is centered with zero delay
DROPOUT_BLOCK_S = 0.020
DEFAULT_BASE_SEED = 42

FAMILIES = ("clean", "gaussian_noise", "amplitude", "lowpass", "temporal_dropout")


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    param: float | None  # snr_db / gain / cutoff_hz / fraction; None for clean
    unit: str

    def as_dict(self) -> dict:
        return asdict(self)


PREDECLARED_CONDITIONS: tuple[Condition, ...] = (
    Condition("clean", "clean", None, ""),
    *(Condition(f"noise_snr{s}db", "gaussian_noise", float(s), "dB SNR") for s in (20, 10, 5, 0)),
    *(Condition(f"gain_{g}x", "amplitude", g, "x") for g in (0.5, 0.25, 1.5, 2.0)),
    *(Condition(f"lowpass_{c}hz", "lowpass", float(c), "Hz") for c in (2000, 1000, 500)),
    *(Condition(f"dropout_{p}pct", "temporal_dropout", p / 100, "fraction") for p in (10, 25, 50)),
)


def _check_waveform(waveform: torch.Tensor) -> None:
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2 or waveform.shape[0] != 1 \
            or waveform.shape[-1] < 1:
        raise ValueError(f"expected a non-empty (1, num_samples) tensor, got "
                         f"{tuple(waveform.shape) if isinstance(waveform, torch.Tensor) else type(waveform)}")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite values")


def _finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def example_seed(base_seed: int, family: str, recording: str, cycle_index: int) -> int:
    """Seed for one cycle's degradation, independent of order and batching."""
    key = f"{base_seed}|{family}|{recording}|{cycle_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & (2 ** 63 - 1)


def generator_for(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def centered_power(x: torch.Tensor) -> float:
    x = x.to(torch.float64)
    return float(((x - x.mean()) ** 2).mean())


def snr_db(clean: torch.Tensor, degraded: torch.Tensor) -> float:
    """Realized SNR of ``degraded`` relative to ``clean`` (DC-free powers)."""
    noise = degraded.to(torch.float64) - clean.to(torch.float64)
    return 10 * math.log10(centered_power(clean) / centered_power(noise))


def add_gaussian_noise(waveform: torch.Tensor, snr: float, generator: torch.Generator) -> torch.Tensor:
    """Add white Gaussian noise at exactly ``snr`` dB (DC-free powers).

    A cycle with zero DC-free power has no defined SNR; it is returned unchanged.
    """
    _check_waveform(waveform)
    snr = _finite("snr_db", snr)
    x = waveform.to(torch.float64)
    p_signal = centered_power(x)
    if p_signal == 0.0:
        return waveform.to(torch.float32).clone()
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float64)
    noise = noise - noise.mean()
    noise = noise * math.sqrt(p_signal / 10 ** (snr / 10) / float((noise ** 2).mean()))
    return (x + noise).to(torch.float32)


def scale_amplitude(waveform: torch.Tensor, gain: float) -> torch.Tensor:
    """Multiply by ``gain`` (> 0). No clipping."""
    _check_waveform(waveform)
    gain = _finite("gain", gain)
    if gain <= 0:
        raise ValueError(f"gain must be > 0, got {gain}")
    return (waveform.to(torch.float64) * gain).to(torch.float32)


def lowpass_kernel(cutoff_hz: float, sample_rate: int = SAMPLE_RATE, num_taps: int = LOWPASS_NUM_TAPS) -> torch.Tensor:
    """Blackman-windowed sinc low-pass FIR with unit DC gain (float64, symmetric)."""
    cutoff_hz = _finite("cutoff_hz", cutoff_hz)
    if not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError(f"cutoff_hz must be in (0, {sample_rate / 2}), got {cutoff_hz}")
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError(f"num_taps must be odd and >= 3, got {num_taps}")
    n = torch.arange(num_taps, dtype=torch.float64) - (num_taps - 1) / 2
    fc = cutoff_hz / sample_rate
    h = 2 * fc * torch.sinc(2 * fc * n) * torch.blackman_window(num_taps, periodic=False, dtype=torch.float64)
    return h / h.sum()


def lowpass(waveform: torch.Tensor, cutoff_hz: float, sample_rate: int = SAMPLE_RATE,
            num_taps: int = LOWPASS_NUM_TAPS) -> torch.Tensor:
    """Zero-phase (centered linear-phase FIR) low-pass; output length equals input length."""
    _check_waveform(waveform)
    h = lowpass_kernel(cutoff_hz, sample_rate, num_taps)
    x = waveform.to(torch.float64)
    full = AF.fftconvolve(x, h.unsqueeze(0))  # (1, n + num_taps - 1)
    half = (num_taps - 1) // 2
    return full[..., half: half + x.shape[-1]].to(torch.float32)


def dropout_blocks(num_samples: int, sample_rate: int = SAMPLE_RATE, block_s: float = DROPOUT_BLOCK_S) -> int:
    block = round(block_s * sample_rate)
    if block < 1:
        raise ValueError(f"block_s {block_s} is shorter than one sample")
    return math.ceil(num_samples / block)


def dropout_mask(num_samples: int, fraction: float, generator: torch.Generator,
                 sample_rate: int = SAMPLE_RATE, block_s: float = DROPOUT_BLOCK_S) -> torch.Tensor:
    """Boolean (num_samples,) mask of the samples temporal_dropout zeroes."""
    fraction = _finite("fraction", fraction)
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    block = round(block_s * sample_rate)
    n_blocks = dropout_blocks(num_samples, sample_rate, block_s)
    k = math.floor(fraction * n_blocks + 0.5)
    mask = torch.zeros(num_samples, dtype=torch.bool)
    for b in torch.randperm(n_blocks, generator=generator)[:k].tolist():
        mask[b * block: (b + 1) * block] = True
    return mask


def temporal_dropout(waveform: torch.Tensor, fraction: float, generator: torch.Generator,
                     sample_rate: int = SAMPLE_RATE, block_s: float = DROPOUT_BLOCK_S) -> torch.Tensor:
    """Zero round-half-up(fraction * n_blocks) randomly chosen ``block_s`` blocks."""
    _check_waveform(waveform)
    mask = dropout_mask(waveform.shape[-1], fraction, generator, sample_rate, block_s)
    out = waveform.to(torch.float32).clone()
    out[..., mask] = 0.0
    return out


def apply(condition: Condition, waveform: torch.Tensor, recording: str, cycle_index: int,
          base_seed: int = DEFAULT_BASE_SEED, sample_rate: int = SAMPLE_RATE) -> tuple[torch.Tensor, dict]:
    """Degrade one cycle; returns (waveform, per-example stats)."""
    _check_waveform(waveform)
    f, p = condition.family, condition.param
    if f == "clean":
        return waveform.to(torch.float32).clone(), {}
    if f == "amplitude":
        out = scale_amplitude(waveform, p)
        return out, {"peak_abs_after": float(out.abs().max())}
    if f == "lowpass":
        return lowpass(waveform, p, sample_rate), {}
    g = generator_for(example_seed(base_seed, f, recording, cycle_index))
    if f == "gaussian_noise":
        out = add_gaussian_noise(waveform, p, g)
        silent = centered_power(waveform) == 0.0
        return out, {"realized_snr_db": float("nan") if silent else snr_db(waveform, out), "silent_input": silent}
    if f == "temporal_dropout":
        mask = dropout_mask(waveform.shape[-1], p, g, sample_rate)
        out = waveform.to(torch.float32).clone()
        out[..., mask] = 0.0
        return out, {"dropped_fraction": float(mask.double().mean())}
    raise ValueError(f"unknown degradation family {f!r}; expected one of {FAMILIES}")
