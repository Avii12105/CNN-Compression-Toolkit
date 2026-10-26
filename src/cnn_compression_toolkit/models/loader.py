"""Strict checkpoint loading and model ownership boundaries."""

from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .resnet import resnet56
from .vgg import vgg16_bn


@dataclass(frozen=True)
class LoadedModel:
    architecture: str
    reference: nn.Module
    working: nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    input_shape: tuple[int, int, int, int]
    num_classes: int


def load_model_checkpoint(
    architecture: str,
    checkpoint_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    num_classes: int = 10,
    device: str | torch.device = "cpu",
) -> LoadedModel:
    """Load a raw state_dict and return isolated reference and working models."""

    path = Path(checkpoint_path)
    model_factories = {"resnet56": resnet56, "vgg16_bn": vgg16_bn}
    if architecture not in model_factories:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if input_shape != (1, 3, 32, 32):
        raise ValueError("Phase 1 expects input_shape (1, 3, 32, 32)")
    if num_classes != 10:
        raise ValueError("Phase 1 expects 10 output classes")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        raise TypeError("Checkpoint must be a raw state_dict mapping names to tensors") from exc
    if not isinstance(checkpoint, dict) or not all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        raise TypeError("Checkpoint must be a raw state_dict mapping names to tensors")
    if not all(torch.isfinite(value).all() for value in checkpoint.values()):
        raise ValueError("Checkpoint contains non-finite tensor values")

    model = model_factories[architecture](num_classes=num_classes)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    reference = copy.deepcopy(model).eval()
    working = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        output = reference(torch.zeros(input_shape, device=device))
    if tuple(output.shape) != (input_shape[0], num_classes):
        expected_shape = (input_shape[0], num_classes)
        raise ValueError(f"Model output shape is {tuple(output.shape)}, expected {expected_shape}")
    if not torch.isfinite(output).all():
        raise ValueError("Model produced non-finite output")
    return LoadedModel(
        architecture=architecture,
        reference=reference,
        working=working,
        checkpoint_path=path,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_shape=input_shape,
        num_classes=num_classes,
    )