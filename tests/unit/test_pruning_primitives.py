import pytest
import torch
from torch import nn

from cnn_compression_toolkit.compression.pruning.alignment import (
    aligned_retained_width,
    resolve_alignment,
)
from cnn_compression_toolkit.compression.pruning.importance import calculate_channel_importance


def test_l1_importance_matches_channel_sums() -> None:
    layer = nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]], [[[5.0]], [[6.0]]]])
        )

    assert torch.equal(calculate_channel_importance(layer), torch.tensor([3.0, 7.0, 11.0]))


def test_alignment_clamps_ratio_and_preserves_valid_width() -> None:
    retained, alignment, effective_ratio = aligned_retained_width(64, 0.95, 16)

    assert retained == 16
    assert alignment == 16
    assert effective_ratio == 0.75


def test_narrow_width_falls_back_to_alignment_one() -> None:
    assert resolve_alignment(3, 32) == 1


def test_alignment_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        aligned_retained_width(0, 0.5, 8)
    with pytest.raises(ValueError):
        aligned_retained_width(16, 0.5, 3)