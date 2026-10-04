"""Deterministic length-aware batch sampler for TRAINING batches.

Each epoch:
  1. Shuffle all indices with an RNG seeded by (seed, epoch).
  2. Cut the shuffled list into pools of batch_size * pool_multiplier.
  3. Sort each pool by frame length (ties by index) and cut it into batches.
  4. Shuffle the order of all batches with the same RNG.

Every index appears exactly once per epoch, so the cycles (and patients) seen
in an epoch are unchanged; only which cycles share a batch changes. Pools
keep batches random across epochs while grouping similar lengths.

Lengths are approximated from the annotated duration (1 + samples // hop at
16 kHz); leading-silence trimming shortens ~2% of cycles by at most a few
frames. The approximation only affects padding efficiency, never the data.

For validation / test use plain sequential batching (not this sampler).
"""

from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


def approximate_frame_lengths(rows: Sequence[dict], sample_rate: int = 16000, hop_length: int = 160) -> list[int]:
    return [1 + round(float(r["cycle_duration_s"]) * sample_rate) // hop_length for r in rows]


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, pool_multiplier: int = 50):
        if batch_size < 1 or pool_multiplier < 1:
            raise ValueError("batch_size and pool_multiplier must be >= 1")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.pool_size = batch_size * pool_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches(self, epoch: int | None = None) -> list[list[int]]:
        rng = random.Random(self.seed * 1_000_003 + (self.epoch if epoch is None else epoch))
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.pool_size):
            pool = sorted(order[start:start + self.pool_size], key=lambda i: (self.lengths[i], i))
            batches += [pool[j:j + self.batch_size] for j in range(0, len(pool), self.batch_size)]
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches())

    def __len__(self) -> int:
        n = len(self.lengths)
        full, rest = divmod(n, self.pool_size)
        return full * math.ceil(self.pool_size / self.batch_size) + math.ceil(rest / self.batch_size)
