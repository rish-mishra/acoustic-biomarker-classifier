"""Load ICBHI recordings and extract respiratory cycles as 16 kHz mono tensors.

Pipeline for one cycle:
  1. Read the WAV with soundfile as float32 in [-1, 1] (16- and 24-bit PCM).
  2. Downmix to mono by averaging channels (ICBHI files are already mono).
  3. Slice [cycle_start_s, cycle_end_s) at the *original* sample rate.
  4. Trim leading exact digital zeros (see below).
  5. Resample to 16 kHz with torchaudio.functional.resample
     (windowed-sinc, deterministic).

Leading digital silence:
  Most AKGC417L recordings (and no other device's) begin with ~90 ms of
  samples that are exactly 0.0, which Log-Mel turns into floor-level frames
  that would identify the device. Before resampling (while zeros are still
  exact), every cycle, on every device, has its contiguous run of leading
  samples equal to exactly 0.0 removed, capped at MAX_LEADING_SILENCE_S
  (150 ms) and always keeping at least one sample. Non-zero samples, however
  quiet, are never removed, and zeros after the first non-zero sample are
  kept. cycle_start_s / cycle_end_s / original_num_samples still describe the
  annotated cycle; ``leading_zeros_removed`` records the trimmed samples.

No padding, cropping, normalization, augmentation, or caching is done here.
Resampling filters can produce values slightly outside [-1, 1]; these are
left unchanged.

Bandwidth limitation:
  Resampling changes the sample rate, not the information content. A cycle
  recorded at 4 kHz contains no frequencies above 2 kHz (its Nyquist limit),
  and upsampling it to 16 kHz does NOT restore them: the 2-8 kHz band of the
  output is essentially empty (only the resampling filter's narrow transition
  band, roughly 2.0-2.5 kHz, holds a little leaked energy from below 2 kHz).
  Likewise 10 kHz audio carries nothing above
  5 kHz. 44.1 kHz audio is low-pass filtered to 8 kHz by downsampling. Every
  RespiratoryCycle records ``original_sample_rate`` and
  ``effective_bandwidth_hz`` so later stages can account for this, e.g. by
  restricting features to 0-2 kHz or analyzing results by source rate.

Edge effect:
  Each cycle is resampled on its own, so the sinc filter sees zeros beyond
  the slice boundaries. This slightly affects the first and last few
  milliseconds of each resampled cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import soundfile as sf
import torch
import torchaudio.functional as AF

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_DIR = (
    REPO_ROOT / "data" / "raw" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"
    / "audio_and_txt_files"
)
TARGET_SAMPLE_RATE = 16_000

# Same tolerance build_metadata.py allows for annotations ending past the audio.
DURATION_TOLERANCE_S = 0.01

# |x| at or above this (on the original-rate float samples, full scale = 1.0)
# counts as near-full-scale / likely clipped.
CLIP_THRESHOLD = 0.999

# Upper limit on leading exact-zero samples removed from a cycle.
MAX_LEADING_SILENCE_S = 0.150

# A clipped plateau is a contiguous run of samples with |x| >= CLIP_THRESHOLD
# lasting at least this long at the original sample rate. 32 ms matches the
# Log-Mel analysis window: a plateau this long can fill a whole frame, which
# then looks like near-silence after DC removal.
MIN_CLIPPED_PLATEAU_S = 0.032


class AudioLoadError(Exception):
    """Raised when an audio file is missing, unreadable, or empty."""


class CycleRangeError(ValueError):
    """Raised when requested cycle times do not fit inside the recording."""


@dataclass(frozen=True)
class RespiratoryCycle:
    waveform: torch.Tensor  # shape (1, num_samples), float32, mono
    sample_rate: int  # rate of ``waveform`` (the target rate)
    original_sample_rate: int
    original_num_samples: int  # samples in the annotated cycle at the original rate (before trimming)
    cycle_start_s: float  # annotated times, unchanged by trimming
    cycle_end_s: float
    source: str  # recording filename
    # Measured on the annotated original-rate cycle, before resampling smears clipped samples.
    original_peak_abs: float = float("nan")
    original_clipped_fraction: float = float("nan")  # fraction of samples with |x| >= CLIP_THRESHOLD
    leading_zeros_removed: int = 0  # leading exact-zero samples trimmed, at the original rate
    # Clipped plateaus (see ClippingStats), measured on the annotated original-rate cycle.
    clipped_plateau_count: int = 0
    max_clipped_plateau_ms: float = 0.0
    clipped_plateau_fraction: float = 0.0

    @property
    def leading_silence_removed_ms(self) -> float:
        return 1000.0 * self.leading_zeros_removed / self.original_sample_rate

    @property
    def num_samples(self) -> int:
        return self.waveform.shape[-1]

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    @property
    def effective_bandwidth_hz(self) -> float:
        """Highest frequency that can carry real content: min(original, target) / 2."""
        return min(self.original_sample_rate, self.sample_rate) / 2


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Read an audio file as a mono float32 tensor of shape (1, num_samples)."""
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"audio file not found: {path}")
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, OSError) as e:
        raise AudioLoadError(f"could not read audio file {path}: {e}") from e
    if data.shape[0] == 0:
        raise AudioLoadError(f"audio file contains no samples: {path}")
    if sample_rate <= 0:
        raise AudioLoadError(f"invalid sample rate {sample_rate} in {path}")

    mono = data.mean(axis=1, dtype="float32") if data.shape[1] > 1 else data[:, 0]
    return torch.from_numpy(mono.copy()).unsqueeze(0), int(sample_rate)


def cycle_sample_range(start_s: float, end_s: float, sample_rate: int, num_samples: int) -> tuple[int, int]:
    """Convert cycle times to a [start, end) sample-index range, validating bounds.

    An end time up to DURATION_TOLERANCE_S past the recording is clipped to
    the last sample; anything further is rejected.
    """
    start_s, end_s = float(start_s), float(end_s)
    duration_s = num_samples / sample_rate
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise CycleRangeError(f"cycle times must be finite, got start={start_s}, end={end_s}")
    if start_s < 0:
        raise CycleRangeError(f"cycle start {start_s}s is negative")
    if end_s <= start_s:
        raise CycleRangeError(f"cycle end {end_s}s is not after start {start_s}s")
    if end_s > duration_s + DURATION_TOLERANCE_S:
        raise CycleRangeError(f"cycle end {end_s}s is past the recording end {duration_s:.4f}s")

    start = round(start_s * sample_rate)
    end = min(round(end_s * sample_rate), num_samples)
    if end - start < 1:
        raise CycleRangeError(
            f"cycle [{start_s}s, {end_s}s) contains no samples at {sample_rate} Hz"
        )
    return start, end


def extract_cycle(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float) -> torch.Tensor:
    """Slice a (1, num_samples) waveform to one cycle at its original rate."""
    start, end = cycle_sample_range(start_s, end_s, sample_rate, waveform.shape[-1])
    return waveform[..., start:end].clone()


@dataclass(frozen=True)
class ClippingStats:
    """Clipping metrics for one original-rate waveform. The audio is never modified.

    clipped_fraction          share of samples with |x| >= CLIP_THRESHOLD
    clipped_plateau_count     number of clipped runs lasting >= MIN_CLIPPED_PLATEAU_S
    max_clipped_plateau_ms    longest clipped run of any length (0 if none), so
                              "no plateau >= 32 ms" is max_clipped_plateau_ms < 32
    clipped_plateau_fraction  share of samples inside runs lasting >= MIN_CLIPPED_PLATEAU_S
    """
    peak_abs: float
    clipped_fraction: float
    clipped_plateau_count: int
    max_clipped_plateau_ms: float
    clipped_plateau_fraction: float


def clipped_runs(waveform: torch.Tensor, threshold: float = CLIP_THRESHOLD) -> list[tuple[int, int]]:
    """(start, length) of each contiguous run of samples with |x| >= threshold."""
    mask = (waveform.reshape(-1).abs() >= threshold).to(torch.int8)
    edges = torch.diff(torch.cat([mask.new_zeros(1), mask, mask.new_zeros(1)]))
    starts = torch.nonzero(edges == 1).flatten()
    ends = torch.nonzero(edges == -1).flatten()
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def min_plateau_samples(sample_rate: int, min_duration_s: float = MIN_CLIPPED_PLATEAU_S) -> int:
    """Smallest run length (samples) lasting at least ``min_duration_s``."""
    return math.ceil(min_duration_s * sample_rate - 1e-9)


def clipping_stats(waveform: torch.Tensor, sample_rate: int) -> ClippingStats:
    """Clipping and clipped-plateau metrics for an original-rate waveform."""
    n = waveform.numel()
    runs = clipped_runs(waveform)
    min_len = min_plateau_samples(sample_rate)
    plateaus = [length for _, length in runs if length >= min_len]
    return ClippingStats(
        peak_abs=float(waveform.abs().max()),
        clipped_fraction=sum(length for _, length in runs) / n,
        clipped_plateau_count=len(plateaus),
        max_clipped_plateau_ms=1000.0 * max((length for _, length in runs), default=0) / sample_rate,
        clipped_plateau_fraction=sum(plateaus) / n,
    )


def count_leading_zeros(waveform: torch.Tensor, max_samples: int) -> int:
    """Length of the leading run of exact 0.0 samples, capped at ``max_samples``.

    Never counts the final sample, so trimming always leaves at least one.
    """
    num_samples = waveform.shape[-1]
    nonzero = torch.nonzero(waveform[0] != 0)
    run = int(nonzero[0, 0]) if len(nonzero) else num_samples
    return max(0, min(run, max_samples, num_samples - 1))


def resample(waveform: torch.Tensor, orig_sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Resample with torchaudio's windowed-sinc resampler. Returns a new tensor."""
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_sample_rate} -> {target_sample_rate}")
    if orig_sample_rate == target_sample_rate:
        return waveform.clone()
    return AF.resample(waveform, orig_sample_rate, target_sample_rate)


def load_cycle(path: str | Path, start_s: float, end_s: float,
               target_sample_rate: int = TARGET_SAMPLE_RATE,
               trim_leading_zeros: bool = True) -> RespiratoryCycle:
    """Load one respiratory cycle, trim leading digital zeros, and resample it."""
    path = Path(path)
    waveform, sample_rate = load_audio(path)
    return cycle_from_waveform(waveform, sample_rate, start_s, end_s, path.name,
                               target_sample_rate, trim_leading_zeros)


def cycle_from_waveform(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float, source: str,
                        target_sample_rate: int = TARGET_SAMPLE_RATE,
                        trim_leading_zeros: bool = True) -> RespiratoryCycle:
    """Same as load_cycle, but from an already-loaded (1, n) recording.

    The input waveform is not modified, so one loaded recording can be reused
    for all of its cycles.
    """
    cycle = extract_cycle(waveform, sample_rate, start_s, end_s)
    trimmed = count_leading_zeros(cycle, round(MAX_LEADING_SILENCE_S * sample_rate)) if trim_leading_zeros else 0
    clipping = clipping_stats(cycle, sample_rate)  # annotated cycle, before trimming and resampling
    return RespiratoryCycle(
        waveform=resample(cycle[..., trimmed:], sample_rate, target_sample_rate),
        sample_rate=target_sample_rate,
        original_sample_rate=sample_rate,
        original_num_samples=cycle.shape[-1],
        cycle_start_s=float(start_s),
        cycle_end_s=float(end_s),
        source=source,
        original_peak_abs=clipping.peak_abs,
        original_clipped_fraction=clipping.clipped_fraction,
        leading_zeros_removed=trimmed,
        clipped_plateau_count=clipping.clipped_plateau_count,
        max_clipped_plateau_ms=clipping.max_clipped_plateau_ms,
        clipped_plateau_fraction=clipping.clipped_plateau_fraction,
    )


def load_cycle_from_metadata(row: Mapping[str, object], audio_dir: str | Path = DEFAULT_AUDIO_DIR,
                             target_sample_rate: int = TARGET_SAMPLE_RATE,
                             trim_leading_zeros: bool = True) -> RespiratoryCycle:
    """Load the cycle described by one data/metadata.csv row.

    If the row has a ``sample_rate`` field, it must match the file on disk.
    """
    cycle = load_cycle(Path(audio_dir) / str(row["recording_filename"]),
                       float(row["cycle_start_s"]), float(row["cycle_end_s"]), target_sample_rate,
                       trim_leading_zeros)
    expected = row.get("sample_rate")
    if expected not in (None, "") and int(expected) != cycle.original_sample_rate:
        raise AudioLoadError(
            f"{cycle.source}: metadata sample_rate {expected} does not match file ({cycle.original_sample_rate})"
        )
    return cycle
