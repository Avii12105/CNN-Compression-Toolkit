"""Dependency-aware structured channel pruning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch_pruning as tp
from torch import nn

from .alignment import aligned_retained_width
from .importance import calculate_channel_importance


@dataclass(frozen=True)
class PruningRecord:
    layer: str
    original_width: int
    retained_width: int
    requested_ratio: float
    effective_ratio: float
    alignment: int


def prune_model(
    model: nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
    requested_alignment: int = 1,
    maximum_ratio: float = 0.9,
    layer_ratios: Mapping[str, float] | None = None,
) -> list[PruningRecord]:
    """Prune convolution output channels while honoring Torch-Pruning dependencies."""

    if not 0 <= pruning_ratio <= 1:
        raise ValueError("pruning_ratio must be between 0 and 1")
    if tuple(example_inputs.shape) != (1, 3, 32, 32):
        raise ValueError("example_inputs must have shape (1, 3, 32, 32)")

    records: list[PruningRecord] = []
    layer_names = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and module.out_channels > 1
    ]
    for layer_name, layer in layer_names:
        if layer_name.startswith("shortcut"):
            continue
        original_width = layer.out_channels
        layer_ratio = (
            pruning_ratio if layer_ratios is None else layer_ratios.get(layer_name, pruning_ratio)
        )
        if not 0 <= layer_ratio <= 1:
            raise ValueError(f"pruning ratio for {layer_name} must be between 0 and 1")
        retained_width, alignment, effective_ratio = aligned_retained_width(
            original_width, layer_ratio, requested_alignment, maximum_ratio
        )
        prune_count = original_width - retained_width
        if prune_count <= 0:
            records.append(
                PruningRecord(
                    layer_name,
                    original_width,
                    original_width,
                    layer_ratio,
                    0.0,
                    alignment,
                )
            )
            continue

        importance = calculate_channel_importance(layer)
        prune_indices = torch.argsort(importance, stable=True)[:prune_count].tolist()
        dependency = tp.DependencyGraph().build_dependency(
            model, example_inputs=example_inputs, verbose=False
        )
        group = dependency.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=prune_indices)
        if not dependency.check_pruning_group(group):
            raise RuntimeError(f"Torch-Pruning rejected group for layer {layer_name}")
        group.prune()
        records.append(
            PruningRecord(
                layer_name,
                original_width,
                retained_width,
                pruning_ratio,
                effective_ratio,
                alignment,
            )
        )
    return records


def pruning_records_to_dict(records: list[PruningRecord]) -> list[dict[str, Any]]:
    """Serialize pruning records for experiment artifacts."""

    return [
        {
            "layer": record.layer,
            "original_width": record.original_width,
            "retained_width": record.retained_width,
            "requested_ratio": record.requested_ratio,
            "effective_ratio": record.effective_ratio,
            "alignment": record.alignment,
        }
        for record in records
    ]