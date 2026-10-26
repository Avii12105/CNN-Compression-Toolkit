"""Runtime and environment metadata collection."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "pandas",
    "torch",
    "torch-pruning",
    "torchvision",
    "PyYAML",
)


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility metadata without requiring CUDA or model assets."""

    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    metadata: dict[str, Any] = {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda": {"available": False, "version": None, "device_count": 0},
    }
    try:
        import onnxruntime

        metadata["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        metadata["onnxruntime_providers"] = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        flags = next(
            (
                line.split(":", 1)[1].strip().split()
                for line in cpuinfo.splitlines()
                if line.startswith("flags")
            ),
            [],
        )
        metadata["cpu_flags"] = flags
    except OSError:
        metadata["cpu_flags"] = []
    try:
        import torch

        metadata["torch_device"] = {"cuda_available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            metadata["cuda"] = {
                "available": True,
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
    except (ImportError, RuntimeError):
        metadata["torch_device"] = {"cuda_available": False, "torch_available": False}
    return metadata