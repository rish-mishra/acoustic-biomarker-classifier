"""PyTorch Dataset over ICBHI respiratory cycles for the frozen patient split.

Each item is one annotated respiratory cycle, loaded through the existing
pipeline (audio_preprocessing -> logmel PRIMARY_CONFIG) at its natural
duration. Target classes: normal, crackle, wheeze, both.

Leakage safeguards:
  * Patients come from the frozen data/splits/patient_split.csv. Every
    metadata patient must be in exactly one split (create_splits.verify_split).
  * A dataset holds only the cycles of its own split.
  * A BandNormalizer is accepted only if it was computed from exactly the
    frozen split's train patients with the same Log-Mel config.

Efficiency: recordings are kept in a small in-memory LRU cache
(RecordingCache), so the cycles of one recording are cut from one decoded
waveform. The cache holds raw decoded audio only; it never changes outputs.
Items are ordered by recording, so sequential passes reuse the cache well.

Quality groups (frozen README definitions) are evaluation filters only; the
dataset never drops cycles.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
for _sub in ("features", "data"):
    if str(SRC_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(SRC_DIR / _sub))

import audio_preprocessing as ap  # noqa: E402
import logmel as lm  # noqa: E402
from create_splits import SPLITS, SplitIntegrityError, verify_split  # noqa: E402, F401

REPO_ROOT = SRC_DIR.parent
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata.csv"
DEFAULT_SPLIT_FILE = REPO_ROOT / "data" / "splits" / "patient_split.csv"
DEFAULT_CACHE_RECORDINGS = 32  # worst case ~32 x 3.8 MB for 20 s of 44.1 kHz float32

CLASS_NAMES = ("normal", "crackle", "wheeze", "both")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

REQUIRED_COLUMNS = ("recording_filename", "cycle_index", "patient_id", "device", "sample_rate",
                    "cycle_start_s", "cycle_end_s", "cycle_duration_s", "cycle_label")

# Predeclared sensitivity groups (README, fixed 2026-10-03). Evaluation filters only.
QUALITY_GROUPS = ("all", "moderately_clean", "clean")
MODERATELY_CLEAN_MAX_CLIPPED_FRACTION = 0.10
CLEAN_MAX_CLIPPED_FRACTION = 0.01


def label_to_index(label: str) -> int:
    try:
        return CLASS_TO_INDEX[label]
    except KeyError:
        raise ValueError(f"unknown cycle label {label!r}; expected one of {CLASS_NAMES}") from None


def in_quality_group(quality: Mapping[str, object], group: str) -> bool:
    """Whether a cycle's quality metadata places it in a predeclared group."""
    if group == "all":
        return True
    clipped = float(quality["original_clipped_fraction"])
    if group == "moderately_clean":
        return clipped <= MODERATELY_CLEAN_MAX_CLIPPED_FRACTION
    if group == "clean":
        return clipped <= CLEAN_MAX_CLIPPED_FRACTION and int(quality["clipped_plateau_count"]) == 0
    raise ValueError(f"unknown quality group {group!r}; expected one of {QUALITY_GROUPS}")


def quality_group_mask(batch: Mapping[str, object], group: str) -> torch.Tensor:
    """Boolean mask over a collated batch selecting cycles in ``group``."""
    return torch.tensor([in_quality_group(q, group) for q in batch["quality"]], dtype=torch.bool)


def read_metadata(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        rows = list(reader)
    for r in rows:
        label_to_index(r["cycle_label"])
    return rows


def read_split_assignments(path: Path) -> dict[str, str]:
    """patient_id -> split from the frozen split file.

    Fails loudly (SplitIntegrityError) on a patient listed twice or in two
    splits, an unknown split name, or an empty split.
    """
    with open(path, newline="", encoding="utf-8") as f:
        pairs = [(r["patient_id"], r["split"]) for r in csv.DictReader(f)]
    verify_split(pairs, {p for p, _ in pairs})
    return dict(pairs)


def train_patients_from_split_file(path: Path = DEFAULT_SPLIT_FILE) -> frozenset[str]:
    return frozenset(p for p, s in read_split_assignments(path).items() if s == "train")


class RecordingCache:
    """Bounded in-memory LRU cache of decoded recordings: filename -> (waveform, sample_rate)."""

    def __init__(self, audio_dir: Path, max_recordings: int = DEFAULT_CACHE_RECORDINGS):
        if max_recordings < 1:
            raise ValueError("max_recordings must be >= 1")
        self.audio_dir = Path(audio_dir)
        self.max_recordings = max_recordings
        self._items: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, filename: str) -> tuple[torch.Tensor, int]:
        if filename in self._items:
            self.hits += 1
            self._items.move_to_end(filename)
            return self._items[filename]
        self.misses += 1
        item = ap.load_audio(self.audio_dir / filename)
        self._items[filename] = item
        while len(self._items) > self.max_recordings:
            self._items.popitem(last=False)
        return item


class RespiratoryCycleDataset(Dataset):
    """Respiratory cycles of one split, as Log-Mel features at natural duration.

    Items are dicts with: features (1, n_mels, n_frames) float32, label (int),
    num_frames, normalized (bool), patient_id, recording_filename,
    cycle_index, original_sample_rate, device, cycle_duration_s (annotated),
    split, and quality (CycleQuality fields, see logmel.py).
    """

    def __init__(self, split: str, metadata_path: Path = DEFAULT_METADATA,
                 split_path: Path = DEFAULT_SPLIT_FILE, audio_dir: Path = ap.DEFAULT_AUDIO_DIR,
                 config: lm.LogMelConfig = lm.PRIMARY_CONFIG, normalizer=None,
                 cache_recordings: int = DEFAULT_CACHE_RECORDINGS):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        rows = read_metadata(Path(metadata_path))
        assignments = read_split_assignments(Path(split_path))
        verify_split(assignments.items(), {r["patient_id"] for r in rows})

        self.split = split
        self.metadata_path = Path(metadata_path)
        self.split_path = Path(split_path)
        self.audio_dir = Path(audio_dir)
        self.rows = [r for r in rows if assignments[r["patient_id"]] == split]
        self.patients = frozenset(r["patient_id"] for r in self.rows)
        self.config = config
        self.extractor = lm.LogMelExtractor(config)
        self.cache = RecordingCache(audio_dir, cache_recordings)

        if normalizer is not None:
            train = frozenset(p for p, s in assignments.items() if s == "train")
            normalizer.check_compatible(config, train)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.rows)

    def label_counts(self) -> dict[str, int]:
        counts = Counter(r["cycle_label"] for r in self.rows)
        return {name: counts.get(name, 0) for name in CLASS_NAMES}

    def load_features(self, index: int) -> lm.LogMelFeatures:
        """Un-normalized Log-Mel features and quality metadata for one cycle."""
        r = self.rows[index]
        waveform, sample_rate = self.cache.get(r["recording_filename"])
        if sample_rate != int(r["sample_rate"]):
            raise ap.AudioLoadError(
                f"{r['recording_filename']}: metadata sample_rate {r['sample_rate']} does not match file ({sample_rate})"
            )
        cycle = ap.cycle_from_waveform(waveform, sample_rate, float(r["cycle_start_s"]),
                                       float(r["cycle_end_s"]), r["recording_filename"])
        return self.extractor(cycle)

    def __getitem__(self, index: int) -> dict:
        r = self.rows[index]
        feats = self.load_features(index)
        x = feats.logmel if self.normalizer is None else self.normalizer(feats.logmel)
        return {
            "features": x,
            "label": label_to_index(r["cycle_label"]),
            "num_frames": x.shape[-1],
            "normalized": self.normalizer is not None,
            "patient_id": r["patient_id"],
            "recording_filename": r["recording_filename"],
            "cycle_index": int(r["cycle_index"]),
            "original_sample_rate": feats.quality.original_sample_rate,
            "device": r["device"],
            "cycle_duration_s": float(r["cycle_duration_s"]),
            "split": self.split,
            "quality": feats.quality.as_dict(),
        }


def collate_cycles(batch: Sequence[dict], pad_value: float = 0.0, require_normalized: bool = True) -> dict:
    """Pad a batch to its longest cycle; return lengths and a valid-frame mask.

    With train-band standardization, 0.0 equals the training-set mean, so it
    is the default pad value. Padding un-normalized dB features with 0 would
    look like a loud signal, so that is refused unless explicitly allowed.
    Items' own feature tensors are copied, never modified.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")
    if require_normalized and not all(item["normalized"] for item in batch):
        raise ValueError("batch contains un-normalized features; normalize them or pass require_normalized=False")
    lengths = torch.tensor([item["features"].shape[-1] for item in batch], dtype=torch.long)
    channels, n_mels = batch[0]["features"].shape[:2]
    features = torch.full((len(batch), channels, n_mels, int(lengths.max())), pad_value, dtype=torch.float32)
    for i, item in enumerate(batch):
        features[i, ..., : lengths[i]] = item["features"]
    mask = torch.arange(features.shape[-1]).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "features": features,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "lengths": lengths,
        "mask": mask,
        "patient_id": [item["patient_id"] for item in batch],
        "recording_filename": [item["recording_filename"] for item in batch],
        "cycle_index": torch.tensor([item["cycle_index"] for item in batch], dtype=torch.long),
        "original_sample_rate": torch.tensor([item["original_sample_rate"] for item in batch], dtype=torch.long),
        "device": [item["device"] for item in batch],
        "cycle_duration_s": torch.tensor([item["cycle_duration_s"] for item in batch], dtype=torch.float32),
        "quality": [item["quality"] for item in batch],
    }
