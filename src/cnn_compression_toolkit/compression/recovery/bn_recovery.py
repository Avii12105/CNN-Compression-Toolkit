"""Calibration-only BatchNorm running-statistics recovery."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class BNRecoveryResult:
    batch_norm_layers: tuple[str, ...]
    sample_count: int


def recover_batch_norm(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    expected_samples: int | None = None,
) -> BNRecoveryResult:
    """Re-estimate BN running statistics without changing trainable parameters."""

    batch_norm_layers = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    if not batch_norm_layers:
        return BNRecoveryResult(batch_norm_layers=(), sample_count=0)

    original_modes = {module: module.training for module in model.modules()}
    for module in model.modules():
        module.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.train()

    sample_count = 0
    try:
        with torch.no_grad():
            for batch in calibration_loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(images.to(device))
                sample_count += images.shape[0]
    finally:
        for module, was_training in original_modes.items():
            module.train(was_training)

    if expected_samples is not None and sample_count != expected_samples:
        raise ValueError(
            f"Calibration loader yielded {sample_count} samples, expected {expected_samples}"
        )
    return BNRecoveryResult(batch_norm_layers=batch_norm_layers, sample_count=sample_count)