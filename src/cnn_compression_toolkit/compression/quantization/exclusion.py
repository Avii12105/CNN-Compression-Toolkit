"""Residual quantization exclusion propagation for QDQ graphs.

When an excluded node feeds an ``Add`` operation, the exclusion is propagated
one hop so that residual skip connections remain numerically valid:

    E' = E U { m : exists n in E, m in consumers(n), op_type(m) == Add }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx


@dataclass
class ExclusionResult:
    excluded_nodes: list[str]
    propagated_add_nodes: list[str]
    total_excluded: int
    applied: bool


def collect_excluded_nodes(
    model_path: str | Path, excluded_op_types: set[str] | None = None
) -> list[str]:
    """Collect node names that should be excluded from quantization."""

    if excluded_op_types is None:
        excluded_op_types = {"Shape", "Gather", "Scatter", "Concat", "Split", "Slice"}

    graph = onnx.load(str(model_path)).graph
    excluded: list[str] = []
    for node in graph.node:
        if node.op_type in excluded_op_types:
            excluded.append(node.name or f"{node.op_type}:{id(node)}")
    return excluded


def build_consumer_map(graph) -> dict[str, list[str]]:
    """Map each tensor name to the list of consumer node names."""

    consumers: dict[str, list[str]] = {}
    for node in graph.node:
        node_name = node.name or f"{node.op_type}:{id(node)}"
        for output in node.output:
            consumers.setdefault(output, []).append(node_name)
    return consumers


def propagate_exclusion_to_adds(
    model_path: str | Path,
    excluded_node_names: list[str],
) -> ExclusionResult:
    """Propagate exclusions one hop: excluded nodes feeding an Add pull the Add in."""

    graph = onnx.load(str(model_path)).graph
    consumers = build_consumer_map(graph)

    # Map producer node name -> node for op_type lookup.
    producer_by_output: dict[str, Any] = {}
    for node in graph.node:
        for output in node.output:
            producer_by_output[output] = node

    propagated: list[str] = []
    excluded_set = set(excluded_node_names)

    for excluded_name in excluded_node_names:
        # Find the tensor outputs of the excluded node.
        excluded_node = None
        for node in graph.node:
            if node.name == excluded_name or (
                not node.name and f"{node.op_type}:{id(node)}" == excluded_name
            ):
                excluded_node = node
                break
        if excluded_node is None:
            continue
        for output in excluded_node.output:
            for consumer_name in consumers.get(output, []):
                consumer_node = None
                for node in graph.node:
                    if node.name == consumer_name or (
                        not node.name and f"{node.op_type}:{id(node)}" == consumer_name
                    ):
                        consumer_node = node
                        break
                if consumer_node is None:
                    continue
                if consumer_node.op_type == "Add" and consumer_name not in excluded_set:
                    propagated.append(consumer_name)
                    excluded_set.add(consumer_name)

    return ExclusionResult(
        excluded_nodes=list(excluded_set),
        propagated_add_nodes=propagated,
        total_excluded=len(excluded_set),
        applied=len(propagated) > 0 or len(excluded_node_names) > 0,
    )


def exclusion_result_to_dict(result: ExclusionResult) -> dict[str, Any]:
    """Serialize an exclusion result for experiment artifacts."""

    return {
        "excluded_nodes": result.excluded_nodes,
        "propagated_add_nodes": result.propagated_add_nodes,
        "total_excluded": result.total_excluded,
        "applied": result.applied,
    }
