"""ImageNet-pretrained MobileNetV3-Small adapted to padded Log-Mel input (transfer-learning comparison).

Research question: can pretrained visual feature representations applied to
Log-Mel spectrograms improve respiratory-sound classification over models
trained from scratch?

Input:  features (B, 1, 64, T) standardized Log-Mel (frozen pipeline; no resizing), mask (B, T) bool
Output: logits (B, 4)

Starting point: torchvision.models.mobilenet_v3_small with weights
MobileNet_V3_Small_Weights.IMAGENET1K_V1 (sha256 047dcff4..., ImageNet acc@1 67.7%).

Adaptations (all pretrained weights kept except where stated):
  1. First convolution 3 -> 1 input channel. The new 1-channel kernel is the
     SUM of the pretrained kernel over its RGB input channels:
         W_gray[o, 0] = W_rgb[o, R] + W_rgb[o, G] + W_rgb[o, B]
     For a 1-channel input x this gives exactly the response the pretrained
     filter would give to the 3-channel image (x, x, x), i.e. the function of
     "grayscale replicated to RGB", without tripling the input or the
     first-layer compute. ImageNet per-channel input normalization is not
     applied: inputs are already standardized per mel band with train-only
     statistics (mean 0, std 1), the same scale as ImageNet-normalized pixels.
  2. Classifier: the pretrained Linear(576, 1024) + Hardswish is kept; dropout
     uses the project's protocol value (0.3, torchvision default 0.2); the final
     Linear(1024, 1000) is replaced by a new Linear(1024, 4) (torchvision init).
  3. Padding awareness (needed because MobileNetV3 uses BatchNorm and
     squeeze-excitation, which average over every position):
       * the network is run block by block with a time mask; padded positions
         are reset to 0 after every layer, so each convolution sees the same
         zeros an unpadded cycle gets from its own zero padding;
       * the valid time length is updated after every strided convolution
         (L_out = floor((L + 2p - d(k-1) - 1) / s) + 1, as for an unpadded input);
       * BatchNorm -> MaskedBatchNorm2d: in training, batch statistics use valid
         positions only; running statistics start from ImageNet and are updated
         (torchvision momentum 0.01); inference uses running statistics;
       * squeeze-excitation and the final global pooling are masked means
         over valid positions.
     Consequences: in eval mode each cycle's output is independent of padding
     and batch composition; in training mode, padding never affects outputs
     (BatchNorm still couples real frames within a batch, as usual for BN).

Downsampling is 32x in both axes (64 mel bands -> 2; T -> ~T/32, >= 1 step).
"""

from __future__ import annotations

import torch
import torchvision
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops import SqueezeExcitation
from torchvision.ops.misc import Conv2dNormActivation

from baseline_cnn import lengths_to_mask

PRETRAINED_WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def _zero_padding(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = lengths_to_mask(lengths, x.shape[-1]).view(x.shape[0], 1, 1, -1)
    return torch.where(valid, x, torch.zeros((), dtype=x.dtype, device=x.device))


def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean over frequency and valid time steps: (B, C, F, T) -> (B, C, 1, 1)."""
    x = _zero_padding(x, lengths)
    denom = (lengths.to(x.dtype) * x.shape[2]).view(-1, 1, 1, 1)
    return x.sum(dim=(2, 3), keepdim=True) / denom


def conv_out_lengths(conv: nn.Conv2d, lengths: torch.Tensor) -> torch.Tensor:
    k, s, p, d = conv.kernel_size[1], conv.stride[1], conv.padding[1], conv.dilation[1]
    return torch.div(lengths + 2 * p - d * (k - 1) - 1, s, rounding_mode="floor") + 1


class MaskedBatchNorm2d(nn.Module):
    """BatchNorm2d whose training statistics use only valid (unpadded) time positions."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.eps, self.momentum = bn.eps, bn.momentum
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())
        self.register_buffer("num_batches_tracked", bn.num_batches_tracked.detach().clone())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.training:
            m = lengths_to_mask(lengths, x.shape[-1]).view(x.shape[0], 1, 1, -1).to(x.dtype)
            n = m.sum() * x.shape[2]
            mean = (x * m).sum(dim=(0, 2, 3)) / n
            var = (((x - mean.view(1, -1, 1, 1)) * m) ** 2).sum(dim=(0, 2, 3)) / n
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var * n / max(float(n) - 1, 1.0))
                self.num_batches_tracked.add_(1)
        else:
            mean, var = self.running_mean, self.running_var
        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return _zero_padding(x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1), lengths)


def _convert_batchnorms(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, MaskedBatchNorm2d(child))
        else:
            _convert_batchnorms(child)


def rgb_to_gray_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """1-input-channel copy of a 3-channel conv; kernel summed over the RGB axis."""
    gray = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                     conv.groups, conv.bias is not None)
    with torch.no_grad():
        gray.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            gray.bias.copy_(conv.bias)
    return gray


class MobileNetV3Transfer(nn.Module):
    def __init__(self, n_classes: int = 4, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        net = torchvision.models.mobilenet_v3_small(weights=PRETRAINED_WEIGHTS if pretrained else None)
        net.features[0][0] = rgb_to_gray_conv(net.features[0][0])
        _convert_batchnorms(net)
        self.features = net.features
        last = net.classifier[3]
        self.classifier = nn.Sequential(net.classifier[0], net.classifier[1], nn.Dropout(dropout),
                                        nn.Linear(last.in_features, n_classes))
        nn.init.normal_(self.classifier[3].weight, 0, 0.01)  # torchvision's init for its classifier
        nn.init.zeros_(self.classifier[3].bias)
        self.pretrained = pretrained

    def _conv_norm_act(self, layer: Conv2dNormActivation, x, lengths):
        conv, *rest = list(layer)
        x = conv(_zero_padding(x, lengths))
        lengths = conv_out_lengths(conv, lengths)
        for m in rest:
            x = m(x, lengths) if isinstance(m, MaskedBatchNorm2d) else m(x)
        return _zero_padding(x, lengths), lengths

    def _squeeze_excite(self, se: SqueezeExcitation, x, lengths):
        s = se.scale_activation(se.fc2(se.activation(se.fc1(_masked_mean(x, lengths)))))
        return _zero_padding(x * s, lengths)

    def _run(self, layer: nn.Module, x, lengths):
        if isinstance(layer, Conv2dNormActivation):
            return self._conv_norm_act(layer, x, lengths)
        if isinstance(layer, SqueezeExcitation):
            return self._squeeze_excite(layer, x, lengths), lengths
        if isinstance(layer, InvertedResidual):
            out, out_len = x, lengths
            for sub in layer.block:
                out, out_len = self._run(sub, out, out_len)
            return (out + x, lengths) if layer.use_res_connect else (out, out_len)
        raise TypeError(f"unexpected layer {type(layer).__name__}")

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError(f"expected (B, 1, n_mels, T) features, got {tuple(features.shape)}")
        if mask.shape != (features.shape[0], features.shape[-1]):
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match features {tuple(features.shape)}")
        lengths = mask.sum(dim=1)
        if bool((lengths < 1).any()):
            raise ValueError("every sample needs at least one valid frame")
        x = features
        for layer in self.features:
            x, lengths = self._run(layer, x, lengths)
        return self.classifier(_masked_mean(x, lengths).flatten(1))
