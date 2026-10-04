"""Load ICBHI recordings and extract respiratory cycles as 16 kHz mono tensors.

Pipeline for one cycle:
  1. Read the WAV with soundfile as float32 in [-1, 1] (16- and 24-bit PCM).
  2. Downmix to mono by averaging channels (ICBHI files are already mono).
  3. Slice [cycle_start_s, cycle_end_s) at the *original* sample rate.
  4. Resample the slice to 16 kHz with torchaudio.functional.resample
     (windowed-sinc, deterministic).

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


class AudioLoadError(Exception):
    """Raised when an audio file is missing, unreadable, or empty."""


class CycleRangeError(ValueError):
    """Raised when requested cycle times do not fit inside the recording."""


@dataclass(frozen=True)
class RespiratoryCycle:
    waveform: torch.Tensor  # shape (1, num_samples), float32, mono
    sample_rate: int  # rate of ``waveform`` (the target rate)
    original_sample_rate: int
    original_num_samples: int  # samples in the cycle before resampling
    cycle_start_s: float
    cycle_end_s: float
    source: str  # recording filename

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


def resample(waveform: torch.Tensor, orig_sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Resample with torchaudio's windowed-sinc resampler. Returns a new tensor."""
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_sample_rate} -> {target_sample_rate}")
    if orig_sample_rate == target_sample_rate:
        return waveform.clone()
    return AF.resample(waveform, orig_sample_rate, target_sample_rate)


def load_cycle(path: str | Path, start_s: float, end_s: float,
               target_sample_rate: int = TARGET_SAMPLE_RATE) -> RespiratoryCycle:
    """Load one respiratory cycle from a recording and resample it."""
    path = Path(path)
    waveform, sample_rate = load_audio(path)
    cycle = extract_cycle(waveform, sample_rate, start_s, end_s)
    return RespiratoryCycle(
        waveform=resample(cycle, sample_rate, target_sample_rate),
        sample_rate=target_sample_rate,
        original_sample_rate=sample_rate,
        original_num_samples=cycle.shape[-1],
        cycle_start_s=float(start_s),
        cycle_end_s=float(end_s),
        source=path.name,
    )


def load_cycle_from_metadata(row: Mapping[str, object], audio_dir: str | Path = DEFAULT_AUDIO_DIR,
                             target_sample_rate: int = TARGET_SAMPLE_RATE) -> RespiratoryCycle:
    """Load the cycle described by one data/metadata.csv row.

    If the row has a ``sample_rate`` field, it must match the file on disk.
    """
    cycle = load_cycle(Path(audio_dir) / str(row["recording_filename"]),
                       float(row["cycle_start_s"]), float(row["cycle_end_s"]), target_sample_rate)
    expected = row.get("sample_rate")
    if expected not in (None, "") and int(expected) != cycle.original_sample_rate:
        raise AudioLoadError(
            f"{cycle.source}: metadata sample_rate {expected} does not match file ({cycle.original_sample_rate})"
        )
    return cycle
