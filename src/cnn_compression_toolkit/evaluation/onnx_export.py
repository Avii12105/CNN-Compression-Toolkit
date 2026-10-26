"""Fixed-signature ONNX export, graph validation, and output equivalence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


@dataclass(frozen=True)
class ONNXExportResult:
    path: Path
    input_name: str
    output_name: str
    opset: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str | Path,
    input_name: str = "input",
    output_name: str = "logits",
    opset: int = 18,
) -> ONNXExportResult:
    """Export and validate a fixed-shape ONNX graph."""

    if tuple(sample_input.shape) != (1, 3, 32, 32):
        raise ValueError("Phase 2 requires representative input shape (1, 3, 32, 32)")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            torch_output = model(sample_input)
        torch.onnx.export(
            model,
            sample_input,
            destination,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    finally:
        model.train(was_training)

    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    graph_input_names = [value.name for value in graph.graph.input]
    graph_output_names = [value.name for value in graph.graph.output]
    if graph_input_names != [input_name] or graph_output_names != [output_name]:
        raise ValueError(
            f"Unexpected ONNX names: inputs={graph_input_names}, outputs={graph_output_names}"
        )
    return ONNXExportResult(
        path=destination,
        input_name=input_name,
        output_name=output_name,
        opset=opset,
        input_shape=tuple(sample_input.shape),
        output_shape=tuple(torch_output.shape),
    )


def compare_pytorch_onnx(
    model: nn.Module,
    export: ONNXExportResult,
    sample_input: torch.Tensor,
    rtol: float = 1.0e-3,
    atol: float = 1.0e-4,
) -> dict[str, Any]:
    """Compare PyTorch and CPU ONNX Runtime outputs for one representative input."""

    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            pytorch_output = model(sample_input).detach().cpu().numpy()
    finally:
        model.train(was_training)
    session = ort.InferenceSession(export.path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_output = np.asarray(
        session.run(
            [export.output_name], {export.input_name: sample_input.detach().cpu().numpy()}
        )[0]
    )
    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch={pytorch_output.shape}, ONNX={onnx_output.shape}"
        )
    if not np.isfinite(onnx_output).all():
        raise ValueError("ONNX output contains non-finite values")
    absolute_error = np.abs(pytorch_output - onnx_output)
    relative_error = absolute_error / np.maximum(np.abs(pytorch_output), 1.0e-12)
    passed = bool(np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol))
    return {
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(absolute_error.max(initial=0.0)),
        "max_relative_error": float(relative_error.max(initial=0.0)),
        "pytorch_shape": list(pytorch_output.shape),
        "onnx_shape": list(onnx_output.shape),
        "finite_output": True,
    }