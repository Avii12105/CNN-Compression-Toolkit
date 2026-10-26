"""Results screen — stage comparison table."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Select, Static


class ResultsScreen(Screen):
    """Display stage comparison results from experiment runs."""

    BINDINGS = [
        ("b", "back", "Back to dashboard"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="results-scroll"):
            with Vertical(id="results-screen"):
                yield Label("Results & Stage Comparison", classes="section-title")
                yield Static("Select Experiment", classes="subsection-title")
                yield Select(
                    [],
                    id="experiment-select",
                    allow_blank=True,
                )
                yield Static("Stage Comparison", classes="subsection-title")
                yield DataTable(id="results-table", zebra_stripes=True)
                yield Button("Back", id="btn-back")

    def on_mount(self) -> None:
        """Populate experiment list."""

        select = self.query_one("#experiment-select", Select)
        options = [("(none)", "")]
        config_path = self.app.config_path
        if config_path is None:
            artifact_root = Path("artifacts")
        else:
            artifact_root = config_path.parent.parent / "artifacts"
        runs_dir = artifact_root / "runs"
        if runs_dir.is_dir():
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                manifest_path = run_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    eid = manifest.get("experiment_id", run_dir.name)
                    status = manifest.get("status", "")
                    options.append((f"{eid} ({status})", eid))
        select.set_options(options)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Load results for the selected experiment."""

        if event.select.id != "experiment-select" or not event.value:
            return
        experiment_id = event.value
        config_path = self.app.config_path
        if config_path is None:
            artifact_root = Path("artifacts")
        else:
            artifact_root = config_path.parent.parent / "artifacts"
        runs_dir = artifact_root / "runs"
        summary_path = runs_dir / experiment_id / "metrics" / "stage_summary.json"
        if not summary_path.is_file():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        states = summary.get("states", [])

        table = self.query_one("#results-table", DataTable)
        table.clear()
        table.add_columns(
            "State", "Accuracy", "Median Lat (ms)", "P95 Lat (ms)", "Size (bytes)", "Params"
        )
        state_labels = {
            "M0": "M0 — Original",
            "M1": "M1 — Pruned",
            "M2": "M2 — BN-recovered",
            "M3": "M3 — Distilled",
            "M4": "M4 — Quantized",
        }
        for entry in states:
            state = entry.get("state", "?")
            label = state_labels.get(state, state)
            table.add_row(
                label,
                f"{entry.get('accuracy', 0):.4f}",
                f"{entry.get('median_latency_ms', 0):.3f}",
                f"{entry.get('p95_latency_ms', 0):.3f}",
                str(entry.get("onnx_size_bytes", 0)),
                str(entry.get("parameter_count", 0)),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        if event.button.id == "btn-back":
            self.app._show_screen("tab-dashboard")

    def action_back(self) -> None:
        """Navigate back to dashboard."""
        self.app._show_screen("tab-dashboard")
