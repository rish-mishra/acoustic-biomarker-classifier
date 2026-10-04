"""Log-Mel features for respiratory cycles.

Input is a RespiratoryCycle from audio_preprocessing (16 kHz mono float32,
leading exact digital zeros already trimmed at the original sample rate).
Per cycle:
  1. Record quality metrics (DC offset, peak, clipping, trimmed leading
     silence, bandwidth, duration).
  2. Subtract the waveform mean (DC removal). No peak or RMS normalization,
     no high-pass filter.
  3. STFT -> mel power spectrogram -> 10*log10(max(power, floor)) in dB.

Output shape is (1, n_mels, n_frames). The number of frames follows the
cycle's natural duration: n_frames = 1 + num_samples // hop_length. Nothing
is padded, cropped, or resized.

Configurations:
  PRIMARY_CONFIG              64 mels, 50-2000 Hz, n_fft 512
  MEL128_ABLATION_CONFIG     128 mels, 50-2000 Hz, n_fft 1024
  BANDWIDTH_ABLATION_CONFIG   64 mels, 50-8000 Hz, n_fft 512

STFT parameters:
  win_length = 512 samples (32 ms, Hann)
      Compromise between time and frequency resolution. Crackles are brief
      transients (roughly 5-25 ms) that longer windows would smear; wheezes
      are sustained tones (> ~100 ms, mostly 100-1000+ Hz) that need
      frequency resolution. True frequency resolution is ~31 Hz (16000/512),
      and no choice of n_fft changes that.
  hop_length = 160 samples (10 ms)
      Fine enough to localize crackles in time; 100 frames per second.
  n_fft
      The smallest FFT size that leaves no mel filter empty. n_fft cannot be
      smaller than win_length, so 512 is the minimum possible.
        * Primary (64 mels, 50-2000 Hz): n_fft = 512 = win_length, so no
          zero-padding at all; 31.25 Hz bin spacing; no empty filters. Mel
          centers are 15-53 Hz apart, so below ~600 Hz the spacing is finer
          than the bin spacing: 22 low bands cover a single FFT bin each,
          and 5 of those share a bin with another band (scaled duplicates).
          Zero-padding to 640 would make them distinct interpolations, but
          would add no real information, so the minimal n_fft is kept.
        * 128-mel ablation: 1024 (zero-padded, 15.6 Hz bin spacing). The
          bare minimum without empty filters is 940; 1024 is the nearest
          standard size, still ~31 Hz true resolution. Many low bands are
          interpolated near-duplicates; this is the point of the ablation.
        * Bandwidth ablation (64 mels, 50-8000 Hz): 512 suffices.
      Zero-padding (n_fft > win_length) only interpolates the spectrum; it
      does NOT increase true frequency resolution.
  f_min = 50 Hz, f_max = 2000 Hz (primary)
      2 kHz is the Nyquist limit of the 4 kHz recordings, so every cycle can
      fill the whole band and band-limited sources cannot be identified from
      empty upper bands. 50 Hz excludes the lowest band, which holds residual
      offset, handling noise, and heart sounds. f_min is a filterbank edge,
      not a steep high-pass: with ~31 Hz resolution, strong content below
      50 Hz still leaks into the lowest mel bands.
  HTK mel scale, no filter-area normalization.
  log floor = 1e-10 power (-100 dB)
      An absolute floor, not relative to each cycle's maximum, so silence
      gives exactly -100 dB and values stay comparable across cycles.
  center = True with zero (constant) padding
      Frames are centered on hop positions; zero padding (instead of
      reflection) works for cycles of any length, including very short ones.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, replace

import torch
import torchaudio.functional as AF
import torchaudio.transforms as T

from audio_preprocessing import TARGET_SAMPLE_RATE, RespiratoryCycle


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = TARGET_SAMPLE_RATE
    win_length: int = 512
    hop_length: int = 160
    n_fft: int = 512
    f_min: float = 50.0
    f_max: float = 2000.0
    n_mels: int = 64
    log_floor: float = 1e-10

    def __post_init__(self):
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError(f"need 0 < win_length <= n_fft, got {self.win_length}, {self.n_fft}")
        if self.hop_length <= 0 or self.n_mels <= 0 or self.log_floor <= 0:
            raise ValueError("hop_length, n_mels, and log_floor must be positive")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError(
                f"need 0 <= f_min < f_max <= {self.sample_rate / 2} Hz, got {self.f_min}-{self.f_max} Hz"
            )

    def num_frames(self, num_samples: int) -> int:
        return 1 + num_samples // self.hop_length


PRIMARY_CONFIG = LogMelConfig()
MEL128_ABLATION_CONFIG = replace(PRIMARY_CONFIG, n_mels=128, n_fft=1024)
BANDWIDTH_ABLATION_CONFIG = replace(PRIMARY_CONFIG, f_max=8000.0)


@dataclass(frozen=True)
class CycleQuality:
    dc_offset: float  # waveform mean before centering (16 kHz waveform, after trimming)
    peak_abs: float  # max |x| of the 16 kHz waveform before centering
    original_peak_abs: float  # max |x| of the annotated cycle at the original sample rate
    original_clipped_fraction: float  # fraction of original-rate samples with |x| >= CLIP_THRESHOLD
    clipped_plateau_count: int  # clipped runs >= 32 ms at the original rate
    max_clipped_plateau_ms: float  # longest clipped run of any length
    clipped_plateau_fraction: float  # share of original samples inside >= 32 ms clipped runs
    leading_samples_removed: int  # leading exact-zero samples trimmed, at the original rate
    leading_silence_removed_ms: float
    original_sample_rate: int
    effective_bandwidth_hz: float
    duration_s: float  # analyzed duration (after trimming)
    band_limited: bool  # True if the source cannot fill the configured f_max

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LogMelFeatures:
    logmel: torch.Tensor  # (1, n_mels, n_frames), float32, dB
    quality: CycleQuality
    config: LogMelConfig
    source: str
    cycle_start_s: float  # annotated times, unchanged by trimming
    cycle_end_s: float


def remove_dc(waveform: torch.Tensor) -> torch.Tensor:
    """Subtract the per-channel mean."""
    return waveform - waveform.mean(dim=-1, keepdim=True)


def cycle_quality(cycle: RespiratoryCycle, config: LogMelConfig) -> CycleQuality:
    w = cycle.waveform
    return CycleQuality(
        dc_offset=float(w.mean()),
        peak_abs=float(w.abs().max()),
        original_peak_abs=cycle.original_peak_abs,
        original_clipped_fraction=cycle.original_clipped_fraction,
        clipped_plateau_count=cycle.clipped_plateau_count,
        max_clipped_plateau_ms=cycle.max_clipped_plateau_ms,
        clipped_plateau_fraction=cycle.clipped_plateau_fraction,
        leading_samples_removed=cycle.leading_zeros_removed,
        leading_silence_removed_ms=cycle.leading_silence_removed_ms,
        original_sample_rate=cycle.original_sample_rate,
        effective_bandwidth_hz=cycle.effective_bandwidth_hz,
        duration_s=cycle.duration_s,
        band_limited=cycle.effective_bandwidth_hz < config.f_max,
    )


def empty_mel_filters(config: LogMelConfig) -> int:
    """Number of mel filters with no FFT bin inside them for this config."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torchaudio warns about the same condition
        fb = AF.melscale_fbanks(config.n_fft // 2 + 1, config.f_min, config.f_max, config.n_mels,
                                config.sample_rate, norm=None, mel_scale="htk")
    return int((fb.sum(dim=0) == 0).sum())


class LogMelExtractor:
    """Computes Log-Mel features; the mel filterbank is built once per config."""

    def __init__(self, config: LogMelConfig = PRIMARY_CONFIG):
        empty = empty_mel_filters(config)
        if empty:
            raise ValueError(f"{empty} mel filter(s) are empty for {config}; increase n_fft or reduce n_mels")
        self.config = config
        self.mel = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            center=True,
            pad_mode="constant",
            norm=None,
            mel_scale="htk",
        )

    def logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Waveform (1, n) at config.sample_rate -> DC removed -> (1, n_mels, n_frames) dB."""
        if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[-1] < 1:
            raise ValueError(f"expected a non-empty (1, num_samples) waveform, got {tuple(waveform.shape)}")
        power = self.mel(remove_dc(waveform.to(torch.float32)))
        return 10.0 * torch.log10(power.clamp_min(self.config.log_floor))

    def __call__(self, cycle: RespiratoryCycle) -> LogMelFeatures:
        if cycle.sample_rate != self.config.sample_rate:
            raise ValueError(f"cycle is {cycle.sample_rate} Hz but config expects {self.config.sample_rate} Hz")
        return LogMelFeatures(
            logmel=self.logmel(cycle.waveform),
            quality=cycle_quality(cycle, self.config),
            config=self.config,
            source=cycle.source,
            cycle_start_s=cycle.cycle_start_s,
            cycle_end_s=cycle.cycle_end_s,
        )
