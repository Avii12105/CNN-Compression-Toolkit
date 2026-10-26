"""Model parameter, serialized size, and checksum metrics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from torch import nn


def serialized_file_metrics(path: str | Path) -> dict[str, Any]:
    """Measure actual serialized bytes and SHA-256 for an artifact."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")
    return {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def model_size_metrics(model: nn.Module, onnx_path: str | Path) -> dict[str, Any]:
    """Combine parameter count with actual ONNX serialized size."""

    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "onnx": serialized_file_metrics(onnx_path),
    }