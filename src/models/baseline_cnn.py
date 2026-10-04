"""Small padding-safe 2D CNN baseline for 4-class respiratory-cycle classification.

Input:  features (B, 1, 64, T) standardized Log-Mel, mask (B, T) bool (True = real frame)
Output: logits (B, 4) for normal / crackle / wheeze / both

Padding safety: a cycle's output does not depend on how much padding its
batch needs.
  * MaskedGroupNorm computes each sample's group statistics over real frames
    only (plain GroupNorm would include padded frames; BatchNorm would also
    mix samples).
  * After every normalization/activation, padded positions are set to 0, so
    the next convolution sees exactly the zeros a lone, unpadded cycle would
    get from the convolution's own zero padding.
  * Max pooling runs after ReLU (values >= 0) with ceil_mode, so a padded 0
    never wins over a real value, and the mask is downsampled to ceil(T / 2).
  * The head averages over frequency, then takes a masked mean and a masked
    max over time.

Architecture (default channels 16-32-64-128, 294,740 parameters):
  4 x [Conv3x3 -> MaskedGN -> ReLU -> Conv3x3 -> MaskedGN -> ReLU -> MaxPool 2x2]
  -> mean over frequency -> [masked mean, masked max] over time -> Dropout -> Linear(2C, 4)
"""

from __future__ import annotations

import torch
from torch import nn


def downsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Valid lengths after a stride-2 ceil-mode pool."""
    return (lengths + 1) // 2


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose statistics use only valid time frames of each sample."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"num_channels {num_channels} not divisible by num_groups {num_groups}")
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        g = self.num_groups
        m = mask.to(x.dtype).view(b, 1, 1, 1, t)
        x = torch.where(mask.view(b, 1, 1, t), x, torch.zeros((), dtype=x.dtype, device=x.device))
        xg = x.view(b, g, c // g, f, t)
        count = m.sum(dim=(2, 3, 4), keepdim=True) * (c // g) * f
        mean = (xg * m).sum(dim=(2, 3, 4), keepdim=True) / count
        var = (((xg - mean) * m) ** 2).sum(dim=(2, 3, 4), keepdim=True) / count
        x = ((xg - mean) / torch.sqrt(var + self.eps)).view(b, c, f, t)
        x = x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)
        return torch.where(mask.view(b, 1, 1, t), x, torch.zeros((), dtype=x.dtype, device=x.device))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = MaskedGroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = MaskedGroupNorm(groups, out_channels)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = lengths_to_mask(lengths, x.shape[-1])
        x = torch.relu(self.norm1(self.conv1(x), mask))
        x = torch.relu(self.norm2(self.conv2(x), mask))
        x = self.pool(x)
        return x, downsample_lengths(lengths)


class BaselineCNN(nn.Module):
    def __init__(self, n_classes: int = 4, channels: tuple[int, ...] = (16, 32, 64, 128),
                 groups: int = 8, dropout: float = 0.3):
        super().__init__()
        blocks, in_ch = [], 1
        for ch in channels:
            blocks.append(ConvBlock(in_ch, ch, min(groups, ch)))
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * in_ch, n_classes)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError(f"expected (B, 1, n_mels, T) features, got {tuple(features.shape)}")
        if mask.shape != (features.shape[0], features.shape[-1]):
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match features {tuple(features.shape)}")
        lengths = mask.sum(dim=1)
        if bool((lengths < 1).any()):
            raise ValueError("every sample needs at least one valid frame")
        # torch.where (not multiplication) so even NaN/inf padding cannot leak in.
        x = torch.where(mask.view(mask.shape[0], 1, 1, -1), features,
                        torch.zeros((), dtype=features.dtype, device=features.device))
        for block in self.blocks:
            x, lengths = block(x, lengths)
        x = x.mean(dim=2)  # (B, C, T') average over frequency
        m = lengths_to_mask(lengths, x.shape[-1]).unsqueeze(1)  # (B, 1, T')
        mean = (x * m).sum(dim=-1) / lengths.unsqueeze(1).to(x.dtype)
        peak = x.masked_fill(~m, float("-inf")).amax(dim=-1)
        return self.classifier(self.dropout(torch.cat([mean, peak], dim=1)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
