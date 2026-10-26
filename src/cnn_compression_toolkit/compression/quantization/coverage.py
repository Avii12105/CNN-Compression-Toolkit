"""Quantization operator coverage audit for QDQ ONNX graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx

# ONNX node types treated as eligible compute operations for quantization.
# Non-compute nodes (input/output, shape, control flow, etc.) and intentionally
# excluded nodes are not part of the denominator.
ELIGIBLE_COMPUTE_OP_TYPES = {
    "Conv",
    "ConvTranspose",
    "Gemm",
    "MatMul",
    "Add",
    "Mul",
    "Sub",
    "Div",
    "AveragePool",
    "GlobalAveragePool",
    "MaxPool",
    "GlobalMaxPool",
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Tanh",
    "BatchNormalization",
    "Clip",
    "Pad",
    "Resize",
    "Upsample",
    "Softmax",
    "LogSoftmax",
    "ThresholdedRelu",
    "PRelu",
    "Elu",
    "Selu",
    "HardSigmoid",
    "HardSwish",
    "Mish",
    "Gelu",
    "Clip",
    "InstanceNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "ConstantFill",
    "ConvInteger",
    "MatMulInteger",
}


@dataclass(frozen=True)
class CoverageReport:
    eligible_nodes: int
    quantized_nodes: int
    coverage: float
    per_operator: dict[str, dict[str, int]]
    unquantized_eligible: list[str]
    method: str = "weights_and_activations_qdq"


def audit_coverage(model_path: str | Path) -> CoverageReport:
    """Inspect an ONNX graph and compute QDQ operator coverage.

    Coverage = |V_q| / |V| where V is the set of eligible compute nodes and
    V_q is the subset flanked by QuantizeLinear/DequantizeLinear pairs.
    """

    graph = onnx.load(str(model_path)).graph

    # Build a map of node output tensor names -> producer node.
    output_producer: dict[str, Any] = {}
    for node in graph.node:
        for output in node.output:
            output_producer[output] = node

    # QuantizeLinear/DequantizeLinear nodes indicate quantized tensors.
    quantized_tensor_names: set[str] = set()
    for node in graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            for output in node.output:
                quantized_tensor_names.add(output)
            for input_name in node.input:
                quantized_tensor_names.add(input_name)

    eligible: list[Any] = []
    for node in graph.node:
        if node.op_type in ELIGIBLE_COMPUTE_OP_TYPES:
            eligible.append(node)

    quantized_count = 0
    per_operator: dict[str, dict[str, int]] = {}
    unquantized_eligible: list[str] = []

    for node in eligible:
        op_type = node.op_type
        stats = per_operator.setdefault(op_type, {"eligible": 0, "quantized": 0})
        stats["eligible"] += 1

        # A node is considered quantized when its inputs and outputs are
        # quantized tensors (flanked by Q/DQ pairs).
        node_inputs_quantized = all(
            name in quantized_tensor_names for name in node.input
        )
        node_outputs_quantized = all(
            name in quantized_tensor_names for name in node.output
        )
        if node_inputs_quantized and node_outputs_quantized:
            quantized_count += 1
            stats["quantized"] += 1
        else:
            unquantized_eligible.append(node.name or node.op_type)

    eligible_count = len(eligible)
    coverage = quantized_count / eligible_count if eligible_count else 0.0

    return CoverageReport(
        eligible_nodes=eligible_count,
        quantized_nodes=quantized_count,
        coverage=coverage,
        per_operator=per_operator,
        unquantized_eligible=unquantized_eligible,
    )


def coverage_report_to_dict(report: CoverageReport) -> dict[str, Any]:
    """Serialize a coverage report for experiment artifacts."""

    return {
        "method": report.method,
        "eligible_nodes": report.eligible_nodes,
        "quantized_nodes": report.quantized_nodes,
        "coverage": report.coverage,
        "per_operator": report.per_operator,
        "unquantized_eligible": report.unquantized_eligible,
    }
