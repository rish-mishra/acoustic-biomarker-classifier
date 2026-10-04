"""Per-mel-band standardization fitted on TRAINING data only.

normalized = (logmel - train_band_mean) / train_band_std

Mean and (population) standard deviation of each mel band are computed over
every Log-Mel frame of every training cycle at its natural length (no
padding). Statistics are therefore frame-weighted: longer cycles and patients
with more cycles contribute more frames. Validation and test cycles are never
read. Accumulation runs in float64 in a fixed order, so results are
reproducible.

The statistics file records the Log-Mel config, the exact train patient IDs,
and SHA-256 hashes of the metadata and split files. A dataset refuses a
normalizer whose config or train patients do not match the frozen split.

Usage (writes DEFAULT_STATS_PATH)::

    python src/data/normalization.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import respiratory_dataset as rd  # noqa: E402

lm = rd.lm
DEFAULT_STATS_PATH = rd.REPO_ROOT / "data" / "normalization" / "logmel_primary_train_band_stats.json"

# A band std below this (dB) would blow up normalized values; treat as an error.
MIN_BAND_STD_DB = 0.1


@dataclass(frozen=True)
class BandNormalizer:
    mean: torch.Tensor  # (n_mels,) float32, dB
    std: torch.Tensor  # (n_mels,) float32, dB
    config: lm.LogMelConfig
    train_patients: tuple[str, ...]  # sorted
    n_cycles: int
    n_frames: int

    def __post_init__(self):
        if self.mean.shape != (self.config.n_mels,) or self.std.shape != (self.config.n_mels,):
            raise ValueError(f"expected ({self.config.n_mels},) mean/std, got {tuple(self.mean.shape)}, {tuple(self.std.shape)}")
        if not (torch.isfinite(self.mean).all() and torch.isfinite(self.std).all()):
            raise ValueError("normalization statistics contain non-finite values")
        if float(self.std.min()) < MIN_BAND_STD_DB:
            raise ValueError(f"band std {float(self.std.min()):.4g} dB is below the {MIN_BAND_STD_DB} dB minimum")

    def __call__(self, logmel: torch.Tensor) -> torch.Tensor:
        """(1, n_mels, n_frames) dB -> standardized; returns a new tensor."""
        return (logmel - self.mean[:, None]) / self.std[:, None]

    def check_compatible(self, config: lm.LogMelConfig, train_patients: frozenset[str]) -> None:
        if config != self.config:
            raise ValueError(f"normalizer was fitted with {self.config}, dataset uses {config}")
        if frozenset(self.train_patients) != train_patients:
            raise ValueError("normalizer was not fitted on exactly the frozen split's train patients")


def compute_band_stats(dataset: rd.RespiratoryCycleDataset) -> BandNormalizer:
    """Fit per-band mean/std on a TRAIN dataset (un-normalized features)."""
    if dataset.split != "train":
        raise ValueError(f"normalization statistics must come from the train split, not {dataset.split!r}")
    if dataset.normalizer is not None:
        raise ValueError("fit statistics on un-normalized features (dataset.normalizer must be None)")
    n_mels = dataset.config.n_mels
    total = torch.zeros(n_mels, dtype=torch.float64)
    total_sq = torch.zeros(n_mels, dtype=torch.float64)
    n_frames = 0
    for i in range(len(dataset)):
        x = dataset.load_features(i).logmel[0].to(torch.float64)  # (n_mels, n_frames)
        total += x.sum(dim=1)
        total_sq += (x * x).sum(dim=1)
        n_frames += x.shape[1]
    if n_frames == 0:
        raise ValueError("train split produced no frames")
    mean = total / n_frames
    var = (total_sq / n_frames - mean * mean).clamp_min(0.0)
    return BandNormalizer(
        mean=mean.to(torch.float32), std=var.sqrt().to(torch.float32), config=dataset.config,
        train_patients=tuple(sorted(dataset.patients)), n_cycles=len(dataset), n_frames=n_frames,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_normalizer(normalizer: BandNormalizer, path: Path, metadata_path: Path, split_path: Path) -> None:
    payload = {
        "description": "Per-mel-band Log-Mel mean/std (dB) over all frames of TRAIN-split cycles only.",
        "config": asdict(normalizer.config),
        "metadata_sha256": _sha256(metadata_path),
        "split_file_sha256": _sha256(split_path),
        "train_patients": list(normalizer.train_patients),
        "n_cycles": normalizer.n_cycles,
        "n_frames": normalizer.n_frames,
        "mean": [float(v) for v in normalizer.mean],
        "std": [float(v) for v in normalizer.std],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_normalizer(path: Path = DEFAULT_STATS_PATH) -> BandNormalizer:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return BandNormalizer(
        mean=torch.tensor(payload["mean"], dtype=torch.float32),
        std=torch.tensor(payload["std"], dtype=torch.float32),
        config=lm.LogMelConfig(**payload["config"]),
        train_patients=tuple(payload["train_patients"]),
        n_cycles=payload["n_cycles"],
        n_frames=payload["n_frames"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit train-only per-band Log-Mel normalization statistics.")
    parser.add_argument("--metadata", type=Path, default=rd.DEFAULT_METADATA)
    parser.add_argument("--split-file", type=Path, default=rd.DEFAULT_SPLIT_FILE)
    parser.add_argument("--audio-dir", type=Path, default=rd.ap.DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args(argv)

    train = rd.RespiratoryCycleDataset("train", args.metadata, args.split_file, args.audio_dir)
    normalizer = compute_band_stats(train)
    save_normalizer(normalizer, args.output, args.metadata, args.split_file)
    print(f"Fitted on {normalizer.n_cycles} train cycles / {normalizer.n_frames} frames "
          f"from {len(normalizer.train_patients)} patients")
    print(f"band mean {float(normalizer.mean.min()):.2f} to {float(normalizer.mean.max()):.2f} dB; "
          f"band std {float(normalizer.std.min()):.2f} to {float(normalizer.std.max()):.2f} dB")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
