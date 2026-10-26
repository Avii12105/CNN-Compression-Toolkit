"""Charts screen — Pareto frontier and sensitivity visualizations."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static


class ChartsScreen(Screen):
    """Display Pareto frontier and sensitivity visualizations."""

    BINDINGS = [
        ("b", "back", "Back to dashboard"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="charts-scroll"):
            with Vertical(id="charts-screen"):
                yield Label("Charts & Pareto Frontier", classes="section-title")
                yield Static("Pareto-Optimal Configurations", classes="subsection-title")
                yield DataTable(id="pareto-table", zebra_stripes=True)
                yield Static(
                    "Plot data saved to: artifacts/runs/*/reports/plot_data.json",
                    classes="note",
                )
                yield Button("Back", id="btn-back")

    def on_mount(self) -> None:
        """Populate Pareto table from experiment results."""

        table = self.query_one("#pareto-table", DataTable)
        table.add_columns("Label", "Accuracy", "Latency (ms)", "Size (bytes)", "Pareto Optimal")
        config_path = self.app.config_path
        if config_path is None:
            artifact_root = Path("artifacts")
        else:
            artifact_root = config_path.parent.parent / "artifacts"
        runs_dir = artifact_root / "runs"
        points = []
        if runs_dir.is_dir():
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                summary_path = run_dir / "metrics" / "stage_summary.json"
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    states = summary.get("states", [])
                    if states:
                        final = states[-1]
                        points.append(
                            (
                                run_dir.name,
                                final.get("accuracy", 0),
                                final.get("median_latency_ms", 0),
                                final.get("onnx_size_bytes", 0),
                            )
                        )
        # Simple Pareto dominance (accuracy maximized, latency/size minimized)
        n = len(points)
        optimal = [True] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = points[i], points[j]
                if (
                    b[1] >= a[1]
                    and b[2] <= a[2]
                    and b[3] <= a[3]
                    and (b[1] > a[1] or b[2] < a[2] or b[3] < a[3])
                ):
                    optimal[i] = False
                    break
        for i, (label, acc, lat, size) in enumerate(points):
            table.add_row(
                label,
                f"{acc:.4f}",
                f"{lat:.3f}",
                str(size),
                "Yes" if optimal[i] else "No",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        if event.button.id == "btn-back":
            self.app._show_screen("tab-dashboard")

    def action_back(self) -> None:
        """Navigate back to dashboard."""
        self.app._show_screen("tab-dashboard")
