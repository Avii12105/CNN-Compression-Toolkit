"""Weights-only per-output-channel sensitivity analysis."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class SensitivityResult:
    layer_name: str
    layer_type: str
    mse: float
    sqnr: float
    bit_width: int
    weights_only_proxy: bool = True
    protected: bool = False


def analyze_sensitivity(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    bit_width: int = 8,
) -> list[SensitivityResult]:
    """Rank weighted layers by logits perturbation after fake weight quantization."""

    if bit_width < 2:
        raise ValueError("bit_width must be at least 2")
    was_training = model.training
    model.eval()
    try:
        reference_logits = _collect_logits(model, calibration_loader, device)
        results: list[SensitivityResult] = []
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                continue
            perturbed = copy.deepcopy(model).to(device).eval()
            target = dict(perturbed.named_modules())[name]
            with torch.no_grad():
                target.weight.copy_(_fake_quantize_weight(target.weight, bit_width))
            perturbed_logits = _collect_logits(perturbed, calibration_loader, device)
            difference = reference_logits - perturbed_logits
            mse = float(torch.mean(difference.square()))
            signal_power = float(torch.sum(reference_logits.square()))
            error_power = float(torch.sum(difference.square()))
            sqnr = math.inf if error_power == 0 else 10.0 * math.log10(signal_power / error_power)
            results.append(
                SensitivityResult(
                    layer_name=name,
                    layer_type=type(module).__name__,
                    mse=mse,
                    sqnr=sqnr,
                    bit_width=bit_width,
                )
            )
        return sorted(results, key=lambda result: (result.sqnr, result.layer_name))
    finally:
        model.train(was_training)


def _fake_quantize_weight(weight: torch.Tensor, bit_width: int) -> torch.Tensor:
    quantization_max = 2 ** (bit_width - 1) - 1
    flattened = weight.detach().reshape(weight.shape[0], -1)
    scale = (flattened.abs().amax(dim=1) / quantization_max).clamp_min(1.0e-12)
    quantized = (flattened / scale.unsqueeze(1)).round().clamp(
        -quantization_max - 1, quantization_max
    )
    return (quantized * scale.unsqueeze(1)).reshape_as(weight)


def _collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            outputs.append(model(images.to(device)).detach().cpu())
    if not outputs:
        raise ValueError("calibration_loader produced no samples")
    return torch.cat(outputs, dim=0)


def sensitivity_results_to_dict(results: list[SensitivityResult]) -> list[dict[str, Any]]:
    """Serialize sensitivity results for experiment artifacts."""

    return [asdict(result) for result in results]


def protected_pruning_ratios(
    results: list[SensitivityResult],
    pruning_ratio: float,
    protected_count: int,
    protection_factor: float = 0.5,
) -> tuple[dict[str, float], list[SensitivityResult]]:
    """Protect the lowest-SQNR layers with a reduced pruning ratio."""

    if not 0 <= pruning_ratio <= 1:
        raise ValueError("pruning_ratio must be between 0 and 1")
    if protected_count < 0 or protected_count > len(results):
        raise ValueError("protected_count must be within the sensitivity result count")
    if not 0 < protection_factor <= 1:
        raise ValueError("protection_factor must be greater than zero and at most one")
    protected_names = {result.layer_name for result in results[:protected_count]}
    ratios = {
        result.layer_name: pruning_ratio * protection_factor
        for result in results[:protected_count]
    }
    marked = [
        SensitivityResult(
            layer_name=result.layer_name,
            layer_type=result.layer_type,
            mse=result.mse,
            sqnr=result.sqnr,
            bit_width=result.bit_width,
            weights_only_proxy=result.weights_only_proxy,
            protected=result.layer_name in protected_names,
        )
        for result in results
    ]
    return ratios, marked