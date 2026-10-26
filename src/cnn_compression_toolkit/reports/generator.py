"""Final report generation: Markdown/HTML report with tables and visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_report(
    output_dir: str | Path,
    stage_summary: list[dict[str, Any]] | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    pareto_data: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
    configuration: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate a comprehensive final report.

    Writes:
      - report.md: human-readable Markdown report
      - report.json: machine-readable JSON report
      - plot_data.json: data for regenerating charts

    Returns a dict of output file paths.
    """

    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_md = _build_markdown(
        stage_summary=stage_summary,
        baseline_metrics=baseline_metrics,
        pareto_data=pareto_data,
        validation_result=validation_result,
        configuration=configuration,
        environment=environment,
    )

    report_path = reports_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    report_json = {
        "stage_summary": stage_summary or [],
        "baseline_metrics": baseline_metrics or {},
        "pareto_data": pareto_data or {},
        "validation_result": validation_result or {},
        "configuration": configuration or {},
        "environment": environment or {},
    }
    json_path = reports_dir / "report.json"
    json_path.write_text(json.dumps(report_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Save plot data separately for regeneration.
    if pareto_data:
        plot_data_path = reports_dir / "plot_data.json"
        plot_data_path.write_text(
            json.dumps(pareto_data.get("plot_data", {}), indent=2) + "\n",
            encoding="utf-8",
        )

    return {"markdown": report_path, "json": json_path}


def _build_markdown(
    stage_summary: list[dict[str, Any]] | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    pareto_data: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
    configuration: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> str:
    """Build the Markdown report content."""

    lines: list[str] = []
    lines.append("# CNN Compression Toolkit — Final Report")
    lines.append("")
    lines.append(f"Generated: {_utc_now()}")
    lines.append("")

    # Baseline table.
    lines.append("## Baseline (M0)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    if baseline_metrics:
        lines.append(f"| Top-1 Accuracy | {baseline_metrics.get('accuracy', 'N/A'):.4f} |")
        lines.append(f"| Macro Precision | {baseline_metrics.get('macro_precision', 'N/A'):.4f} |")
        lines.append(f"| Macro Recall | {baseline_metrics.get('macro_recall', 'N/A'):.4f} |")
        lines.append(f"| Macro F1 | {baseline_metrics.get('macro_f1', 'N/A'):.4f} |")
        lines.append(f"| Parameter Count | {baseline_metrics.get('parameter_count', 'N/A')} |")
        lines.append(f"| Device | {baseline_metrics.get('device', 'N/A')} |")
    else:
        lines.append("| Top-1 Accuracy | N/A |")
        lines.append("| Parameter Count | N/A |")
    lines.append("")

    # Stage comparison table.
    lines.append("## Stage Comparison (M0 → M1 → M2 → M3 → M4)")
    lines.append("")
    lines.append(
        "| State | Accuracy | Median Latency (ms) | P95 Latency (ms) | Size (bytes) | Parameters |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    state_labels = {
        "M0": "M0 — Original",
        "M1": "M1 — Pruned",
        "M2": "M2 — BN-recovered",
        "M3": "M3 — Distilled",
        "M4": "M4 — Quantized",
    }
    if stage_summary:
        for entry in stage_summary:
            state = entry.get("state", "?")
            label = state_labels.get(state, state)
            lines.append(
                f"| {label} | {entry.get('accuracy', 'N/A'):.4f} | "
                f"{entry.get('median_latency_ms', 'N/A'):.3f} | "
                f"{entry.get('p95_latency_ms', 'N/A'):.3f} | "
                f"{entry.get('onnx_size_bytes', 'N/A')} | "
                f"{entry.get('parameter_count', 'N/A')} |"
            )
    else:
        lines.append("| No stage data available | | | | | |")
    lines.append("")

    # Pareto frontier.
    if pareto_data:
        lines.append("## Pareto Frontier")
        lines.append("")
        lines.append(f"Total points: {pareto_data.get('total_points', 0)}")
        lines.append(f"Pareto-optimal: {pareto_data.get('frontier_size', 0)}")
        lines.append("")
        frontier = pareto_data.get("frontier", [])
        if frontier:
            lines.append("### Pareto-Optimal Configurations")
            lines.append("")
            lines.append(
                "| Label | Accuracy | Latency (ms) | Size (bytes) |"
                " Precision | Distillation |"
            )
            lines.append("|---|---:|---:|---:|---|---|")
            for point in frontier:
                lines.append(
                    f"| {point['label']} | {point['accuracy']:.4f} | "
                    f"{point['latency_ms']:.3f} | {point['size_bytes']} | "
                    f"{point['precision']} | {point['distillation']} |"
                )
        lines.append("")

    # Final validation.
    if validation_result:
        lines.append("## Final Validation")
        lines.append("")
        lines.append("| Check | Status |")
        lines.append("|---|---|")
        lines.append(
            f"| ONNX graph valid | "
            f"{'PASS' if validation_result.get('equivalence_passed') else 'FAIL'} |"
        )
        lines.append(
            f"| Fresh-process validation | "
            f"{'PASS' if validation_result.get('fresh_process_passed') else 'FAIL'} |"
        )
        lines.append(
            f"| Overall validation | "
            f"{'PASS' if validation_result.get('validation_passed') else 'FAIL'} |"
        )
        lines.append(f"| Test-set accuracy | {validation_result.get('test_accuracy', 0):.4f} |")
        lines.append(f"| Median latency | {validation_result.get('median_latency_ms', 0):.3f} ms |")
        lines.append(f"| Serialized size | {validation_result.get('onnx_size_bytes', 0)} bytes |")
        if validation_result.get("coverage_report"):
            cov = validation_result["coverage_report"]
            lines.append(f"| Quantization coverage | {cov.get('coverage', 0):.2%} |")
        lines.append("")

    # Configuration.
    if configuration:
        lines.append("## Configuration")
        lines.append("")
        lines.append("```yaml")
        lines.append(_yaml_like(configuration))
        lines.append("```")
        lines.append("")

    # Environment.
    if environment:
        lines.append("## Environment")
        lines.append("")
        packages = environment.get("packages", {})
        lines.append("| Package | Version |")
        lines.append("|---|---|")
        for name, version in sorted(packages.items()):
            lines.append(f"| {name} | {version} |")
        lines.append("")

    return "\n".join(lines)


def _yaml_like(data: Any, indent: int = 0) -> str:
    """Simple YAML-like serialization for display."""

    if isinstance(data, dict):
        lines = []
        for key, value in sorted(data.items()):
            lines.append("  " * indent + f"{key}:")
            lines.append(_yaml_like(value, indent + 1))
        return "\n".join(lines)
    if isinstance(data, list):
        return "\n".join("  " * indent + f"- {item}" for item in data)
    return "  " * indent + str(data)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
