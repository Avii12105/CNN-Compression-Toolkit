"""Pareto frontier analysis for compression experiment results.

Computes Pareto-optimal candidates across accuracy, latency, and size
objectives, and generates plot data for visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParetoPoint:
    """One candidate point in the Pareto analysis."""

    label: str
    accuracy: float
    latency_ms: float
    size_bytes: int
    pruning_ratio: float
    precision: str
    distillation: bool
    satisfies_budget: bool = False
    pareto_optimal: bool = False
    experiment_id: str | None = None


def is_pareto_optimal(
    points: list[ParetoPoint],
) -> list[bool]:
    """Determine which points are Pareto-optimal.

    A point is Pareto-optimal if no other point is at least as good in
    all objectives and strictly better in one. Accuracy is maximized;
    latency and size are minimized.
    """

    n = len(points)
    optimal = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = points[i], points[j]
            # b dominates a if b is >= in all objectives and > in at least one
            b_at_least_as_good = (
                b.accuracy >= a.accuracy
                and b.latency_ms <= a.latency_ms
                and b.size_bytes <= a.size_bytes
            )
            b_strictly_better = (
                b.accuracy > a.accuracy
                or b.latency_ms < a.latency_ms
                or b.size_bytes < a.size_bytes
            )
            if b_at_least_as_good and b_strictly_better:
                optimal[i] = False
                break
    return optimal


def compute_pareto_frontier(
    results: list[dict[str, Any]],
    budget_satisfying: set[str] | None = None,
) -> dict[str, Any]:
    """Compute Pareto frontier from experiment matrix results.

    Returns a dict with:
      - points: list of ParetoPoint dicts with pareto_optimal flag
      - frontier: list of points on the Pareto frontier
      - plot_data: data for accuracy-latency, accuracy-size, latency-size plots
    """

    points: list[ParetoPoint] = []
    for result in results:
        if not result.get("succeeded"):
            continue
        stages = result.get("stages", [])
        if not stages:
            continue
        final = stages[-1]
        config = result.get("config", {})
        points.append(
            ParetoPoint(
                label=(
                    f"r{config.get('pruning_ratio', 0)}-"
                    f"{config.get('precision', 'fp32')}-"
                    f"{'distill' if config.get('distillation') else 'nodistill'}"
                ),
                accuracy=final["accuracy"],
                latency_ms=final["median_latency_ms"],
                size_bytes=final["onnx_size_bytes"],
                pruning_ratio=float(config.get("pruning_ratio", 0)),
                precision=str(config.get("precision", "fp32")),
                distillation=bool(config.get("distillation", False)),
                satisfies_budget=(
                    result.get("experiment_id") in (budget_satisfying or set())
                ),
                experiment_id=result.get("experiment_id"),
            )
        )

    optimal_flags = is_pareto_optimal(points)
    for point, is_optimal in zip(points, optimal_flags):
        point.pareto_optimal = is_optimal

    frontier = [p for p, opt in zip(points, optimal_flags) if opt]

    # Plot data for the three pairwise comparisons.
    plot_data = {
        "accuracy_vs_latency": [
            {"x": p.latency_ms, "y": p.accuracy, "label": p.label, "pareto": p.pareto_optimal}
            for p in points
        ],
        "accuracy_vs_size": [
            {"x": p.size_bytes, "y": p.accuracy, "label": p.label, "pareto": p.pareto_optimal}
            for p in points
        ],
        "latency_vs_size": [
            {"x": p.size_bytes, "y": p.latency_ms, "label": p.label, "pareto": p.pareto_optimal}
            for p in points
        ],
    }

    return {
        "points": [
            {
                "label": p.label,
                "accuracy": p.accuracy,
                "latency_ms": p.latency_ms,
                "size_bytes": p.size_bytes,
                "pruning_ratio": p.pruning_ratio,
                "precision": p.precision,
                "distillation": p.distillation,
                "satisfies_budget": p.satisfies_budget,
                "pareto_optimal": p.pareto_optimal,
                "experiment_id": p.experiment_id,
            }
            for p in points
        ],
        "frontier": [
            {
                "label": p.label,
                "accuracy": p.accuracy,
                "latency_ms": p.latency_ms,
                "size_bytes": p.size_bytes,
                "pruning_ratio": p.pruning_ratio,
                "precision": p.precision,
                "distillation": p.distillation,
                "satisfies_budget": p.satisfies_budget,
                "experiment_id": p.experiment_id,
            }
            for p in frontier
        ],
        "plot_data": plot_data,
        "total_points": len(points),
        "frontier_size": len(frontier),
    }
