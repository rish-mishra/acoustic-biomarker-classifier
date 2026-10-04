"""Padding-safe CRNN for 4-class respiratory-cycle classification (architecture comparison v1).

Research question: does explicitly modeling temporal structure improve
respiratory-cycle classification over the frozen CNN baseline (baseline_cnn_v3)?

Input:  features (B, 1, 64, T) standardized Log-Mel (10 ms frames), mask (B, T) bool
Output: logits (B, 4) for normal / crackle / wheeze / both

Architecture (fixed before training; not tuned on validation results):
  CNN front end, 3 blocks [Conv3x3 -> MaskedGN -> ReLU -> Conv3x3 -> MaskedGN -> ReLU -> MaxPool]
    channels 16 -> 32 -> 64 (the baseline's first three widths)
    pooling (freq x time): (2, 2), (2, 1), (2, 1)
      frequency 64 -> 8; time downsampled only once -> 20 ms steps
      (the baseline pools time 4x down to 160 ms steps; crackles last ~5-25 ms)
  Frequency collapse: flatten 64 channels x 8 bins per step -> Linear(512, 128) -> ReLU -> Dropout(0.3)
  Temporal model: 1-layer bidirectional GRU, hidden 128 per direction, on packed sequences
  Pooling: masked additive attention over time (Linear 256->64 -> tanh -> Linear 64->1, softmax over valid steps)
  Head: Dropout(0.3) -> Linear(256, 4)
  353,365 parameters (baseline: 294,740; ~1.2x), so the comparison is about temporal modeling, not size.

Padding safety (output for a cycle is independent of batch padding):
  * same masked GroupNorm / zeroed padded positions / ceil-mode pooling as baseline_cnn
  * the GRU sees packed sequences, so padded steps never enter it and the backward
    direction starts at each cycle's true last step
  * attention scores of padded steps are -inf before the softmax
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from baseline_cnn import MaskedGroupNorm, lengths_to_mask


class CRNNBlock(nn.Module):
    """Conv-GN-ReLU x2 then max pooling with a configurable (freq, time) kernel."""

    def __init__(self, in_channels: int, out_channels: int, groups: int, pool: tuple[int, int]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = MaskedGroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = MaskedGroupNorm(groups, out_channels)
        self.time_pool = pool[1]
        self.pool = nn.MaxPool2d(pool, ceil_mode=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = lengths_to_mask(lengths, x.shape[-1])
        x = torch.relu(self.norm1(self.conv1(x), mask))
        x = torch.relu(self.norm2(self.conv2(x), mask))
        x = self.pool(x)
        return x, (lengths + self.time_pool - 1) // self.time_pool


class CRNN(nn.Module):
    def __init__(self, n_classes: int = 4, n_mels: int = 64, channels: tuple[int, ...] = (16, 32, 64),
                 pools: tuple[tuple[int, int], ...] = ((2, 2), (2, 1), (2, 1)), groups: int = 8,
                 rnn_input: int = 128, rnn_hidden: int = 128, attention_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        if len(channels) != len(pools):
            raise ValueError("channels and pools must have the same length")
        blocks, in_ch, freq = [], 1, n_mels
        for ch, pool in zip(channels, pools):
            blocks.append(CRNNBlock(in_ch, ch, min(groups, ch), pool))
            in_ch, freq = ch, -(-freq // pool[0])
        self.blocks = nn.ModuleList(blocks)
        self.project = nn.Linear(in_ch * freq, rnn_input)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(rnn_input, rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(2 * rnn_hidden, attention_hidden), nn.Tanh(),
                                       nn.Linear(attention_hidden, 1))
        self.classifier = nn.Linear(2 * rnn_hidden, n_classes)
        self.n_mels = n_mels

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step bidirectional GRU outputs (B, T', 2H) and valid lengths (B,)."""
        if features.ndim != 4 or features.shape[1] != 1 or features.shape[2] != self.n_mels:
            raise ValueError(f"expected (B, 1, {self.n_mels}, T) features, got {tuple(features.shape)}")
        if mask.shape != (features.shape[0], features.shape[-1]):
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match features {tuple(features.shape)}")
        lengths = mask.sum(dim=1)
        if bool((lengths < 1).any()):
            raise ValueError("every sample needs at least one valid frame")
        x = torch.where(mask.view(mask.shape[0], 1, 1, -1), features,
                        torch.zeros((), dtype=features.dtype, device=features.device))
        for block in self.blocks:
            x, lengths = block(x, lengths)
        b, c, f, t = x.shape
        seq = self.dropout(torch.relu(self.project(x.permute(0, 3, 1, 2).reshape(b, t, c * f))))
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=t)
        return out, lengths

    def attention_weights(self, out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Softmax attention over valid steps; exactly 0 on padded steps. Shape (B, T')."""
        scores = self.attention(out).squeeze(-1)
        valid = lengths_to_mask(lengths, out.shape[1])
        return torch.softmax(scores.masked_fill(~valid, float("-inf")), dim=1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, lengths = self.encode(features, mask)
        weights = self.attention_weights(out, lengths)
        pooled = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        return self.classifier(self.dropout(pooled))
