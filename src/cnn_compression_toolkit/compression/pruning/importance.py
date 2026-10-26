"""Channel-importance calculations."""

from __future__ import annotations

import torch
from torch import nn


def calculate_channel_importance(layer: nn.Conv2d, p: int = 1) -> torch.Tensor:
    """Return one Lp importance score for each convolution output channel."""

    if p <= 0:
        raise ValueError("p must be positive")
    weights = layer.weight.detach().abs().flatten(start_dim=1)
    return weights.pow(p).sum(dim=1).pow(1.0 / p)


def rank_channels(layer: nn.Conv2d, p: int = 1) -> torch.Tensor:
    """Return channel indices from least to most important."""

    return torch.argsort(calculate_channel_importance(layer, p=p), stable=True)