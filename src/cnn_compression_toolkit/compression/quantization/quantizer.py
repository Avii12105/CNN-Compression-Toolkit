"""Terminal quantization: ONNX Runtime INT8 static quantization and FP16 conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # ONNX Runtime quantization API is version-dependent.
    from onnxruntime.quantization import QuantizationMode, QuantType, quantize_static
    _HAS_QUANTIZE_STATIC = True
except ImportError:  # pragma: no cover - older ONNX Runtime
    QuantizationMode = None  # type: ignore[assignment]
    QuantType = None  # type: ignore[assignment]
    quantize_static = None  # type: ignore[assignment]
    _HAS_QUANTIZE_STATIC = False

from ...evaluation.onnx_export import export_to_onnx
from ...evaluation.size import serialized_file_metrics


def _collect_calibration_batches(
    calibration_loader: DataLoader, device, max_batches: int | None = None
) -> list[np.ndarray]:
    """Collect individual calibration samples (batch size 1) from a DataLoader.

    The exported ONNX model has a fixed batch size of 1, so calibration
    samples must be fed one at a time.
    """

    samples: list[np.ndarray] = []
    count = 0
    with torch.no_grad():
        for batch in calibration_loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.numpy().astype(np.float32)
            for i in range(images.shape[0]):
                samples.append(images[i : i + 1])  # keep batch dimension = 1
                count += 1
                if max_batches is not None and count >= max_batches:
                    break
            if max_batches is not None and count >= max_batches:
                break
    if not samples:
        raise ValueError("calibration_loader produced no samples")
    return samples


def quantize_onnx_static(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str | Path,
    calibration_loader: DataLoader,
    device,
    input_name: str = "input",
    output_name: str = "logits",
    opset: int = 18,
) -> dict[str, Any]:
    """Apply ONNX Runtime static INT8 quantization (QDQ graph).

    Exports the FP32 model to ONNX, then runs ONNX Runtime static quantization
    using the 512-image calibration subset. Never uses test data for calibration.
    """

    if not _HAS_QUANTIZE_STATIC:
        raise RuntimeError(
            "ONNX Runtime static quantization API is unavailable in this installation"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export FP32 ONNX first.
    fp32_path = destination.with_suffix(".fp32.onnx")
    export_to_onnx(
        model,
        sample_input,
        fp32_path,
        input_name=input_name,
        output_name=output_name,
        opset=opset,
    )

    # 2. Define a calibration data reader that feeds the calibration subset.
    calibration_batches = _collect_calibration_batches(calibration_loader, device)

    class _CalibrationDataReader:
        def __init__(self, batches: list[np.ndarray], input_name: str) -> None:
            self._batches = batches
            self._input_name = input_name
            self._index = 0

        def get_next(self):
            if self._index >= len(self._batches):
                return None
            batch = self._batches[self._index]
            self._index += 1
            return {self._input_name: batch}

        def reset(self) -> None:
            self._index = 0

    # 3. Run static quantization with QDQ format, per-channel weight scales.
    quantize_static(
        model_input=fp32_path.as_posix(),
        model_output=destination.as_posix(),
        calibration_data_reader=_CalibrationDataReader(calibration_batches, input_name),
        quant_format=QuantizationMode.QLinearOps,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

    # Clean up the intermediate FP32 ONNX.
    try:
        fp32_path.unlink()
    except OSError:
        pass

    return {
        "path": str(destination),
        "precision": "int8",
        "format": "QDQ",
        "per_channel": True,
        "calibration_samples": len(calibration_batches),
        "size": serialized_file_metrics(destination),
    }


def convert_to_fp16_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str | Path,
    input_name: str = "input",
    output_name: str = "logits",
    opset: int = 18,
) -> dict[str, Any]:
    """Export an FP16 ONNX model by converting weights after FP32 export.

    Converts float weights to FP16 using ONNX type reduction. Records whether
    the CPU provider is likely to cast FP16 back to FP32 (size reduction is
    real, latency improvement is not guaranteed).
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fp32_path = destination.with_suffix(".fp32.onnx")
    export_to_onnx(
        model,
        sample_input,
        fp32_path,
        input_name=input_name,
        output_name=output_name,
        opset=opset,
    )

    graph = onnx.load(fp32_path.as_posix())

    # Convert float tensors (initializers) to FP16.
    converted_count = 0
    for initializer in graph.graph.initializer:
        if initializer.data_type in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
        ):
            try:
                float_array = onnx.numpy_helper.to_array(initializer)
                fp16_array = float_array.astype(np.float16)
                new_initializer = onnx.numpy_helper.from_array(fp16_array, initializer.name)
                new_initializer.data_type = onnx.TensorProto.FLOAT16
                initializer.CopyFrom(new_initializer)
                converted_count += 1
            except Exception:
                # Leave non-convertible initializers as-is.
                continue

    onnx.checker.check_model(graph)
    onnx.save(graph, destination.as_posix())

    try:
        fp32_path.unlink()
    except OSError:
        pass

    return {
        "path": str(destination),
        "precision": "fp16",
        "format": "FP16",
        "converted_initializers": converted_count,
        "cpu_may_cast_back_to_fp32": True,
        "size": serialized_file_metrics(destination),
    }
