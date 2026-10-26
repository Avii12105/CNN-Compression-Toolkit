"""Dashboard screen — recent runs and quick actions."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static


class DashboardScreen(Screen):
    """Dashboard showing recent experiment runs and quick actions."""

    BINDINGS = [
        ("n", "new_experiment", "New experiment"),
        ("r", "refresh_runs", "Refresh runs"),
        ("a", "open_artifacts", "Open artifacts"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="dashboard-scroll"):
            with Vertical(id="dashboard"):
                yield Label("Dashboard", classes="section-title")
                yield Static("Recent Experiment Runs", classes="subsection-title")
                yield DataTable(id="runs-table", zebra_stripes=True)
                yield Horizontal(
                    Button("New Experiment", id="btn-new-experiment", variant="primary"),
                    Button("Refresh", id="btn-refresh"),
                    Button("Open Artifacts Dir", id="btn-open-artifacts"),
                    id="action-buttons",
                )

    def on_mount(self) -> None:
        """Populate the runs table."""

        table = self.query_one("#runs-table", DataTable)
        table.add_columns("Experiment ID", "Command", "Status", "Started")

        config_path = self.app.config_path
        if config_path is None:
            artifact_root = Path("artifacts")
        else:
            artifact_root = config_path.parent.parent / "artifacts"
        runs_dir = artifact_root / "runs"

        if not runs_dir.is_dir():
            self.notify(f"Artifacts directory not found: {runs_dir}", severity="warning")
            return

        count = 0
        for run_dir in sorted(runs_dir.iterdir(), reverse=True)[:10]:
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    table.add_row(
                        manifest.get("experiment_id", run_dir.name),
                        manifest.get("command", ""),
                        manifest.get("status", ""),
                        manifest.get("started_at", ""),
                    )
                    count += 1
                except Exception as exc:
                    self.notify(
                        f"Failed to read {manifest_path}: {exc}",
                        severity="error",
                    )

        if count == 0:
            self.notify("No experiment runs found", severity="information")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        if event.button.id == "btn-refresh":
            self._refresh_runs()
        elif event.button.id == "btn-new-experiment":
            self.app._show_screen("tab-model")
        elif event.button.id == "btn-open-artifacts":
            self.app._show_screen("tab-artifact")

    def action_new_experiment(self) -> None:
        """Navigate to model selection screen."""
        self.app._show_screen("tab-model")

    def action_refresh_runs(self) -> None:
        """Refresh the runs table."""
        self._refresh_runs()

    def action_open_artifacts(self) -> None:
        """Navigate to artifact screen."""
        self.app._show_screen("tab-artifact")

    def _refresh_runs(self) -> None:
        """Refresh the dashboard."""

        table = self.query_one("#runs-table", DataTable)
        table.clear()

        config_path = self.app.config_path
        if config_path is None:
            artifact_root = Path("artifacts")
        else:
            artifact_root = config_path.parent.parent / "artifacts"
        runs_dir = artifact_root / "runs"

        if not runs_dir.is_dir():
            self.notify(f"Artifacts directory not found: {runs_dir}", severity="warning")
            return

        count = 0
        for run_dir in sorted(runs_dir.iterdir(), reverse=True)[:10]:
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    table.add_row(
                        manifest.get("experiment_id", run_dir.name),
                        manifest.get("command", ""),
                        manifest.get("status", ""),
                        manifest.get("started_at", ""),
                    )
                    count += 1
                except Exception as exc:
                    self.notify(
                        f"Failed to read {manifest_path}: {exc}",
                        severity="error",
                    )

        if count == 0:
            self.notify("No experiment runs found", severity="information")
