"""Artifact screen — final ONNX artifact details."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Select, Static


class ArtifactScreen(Screen):
    """Display final ONNX artifact information and validation status."""

    BINDINGS = [
        ("b", "back", "Back to dashboard"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="artifact-scroll"):
            with Vertical(id="artifact-screen"):
                yield Label("Final ONNX Artifact", classes="section-title")
                yield Static("Select Experiment", classes="subsection-title")
                yield Select(
                    [],
                    id="experiment-select",
                    allow_blank=True,
                )
                yield Static("Artifact Details", classes="subsection-title")
                yield DataTable(id="artifact-table", zebra_stripes=True)
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
        """Load artifact details for the selected experiment."""

        if event.select.id != "experiment-select" or not event.value:
            return
        experiment_id = event.value
        config_path = self.app.config_path
        if config_path is None:
            artifact_root = Path("artifacts")
        else:
            artifact_root = config_path.parent.parent / "artifacts"
        runs_dir = artifact_root / "runs"
        manifest_path = runs_dir / experiment_id / "manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})

        table = self.query_one("#artifact-table", DataTable)
        table.clear()
        table.add_columns("Artifact", "Path")

        # Find ONNX artifact
        onnx_key = None
        for key in ("m4_onnx", "m3_onnx", "m2_onnx", "m1_onnx", "onnx_model"):
            if key in artifacts:
                onnx_key = key
                break

        if onnx_key:
            onnx_path = runs_dir / experiment_id / artifacts[onnx_key]
            table.add_row("ONNX Model", str(onnx_path))
            if onnx_path.is_file():
                table.add_row("Size (bytes)", str(onnx_path.stat().st_size))
                table.add_row("SHA-256", _sha256(onnx_path))

        # Fresh process validation
        fp_path = runs_dir / experiment_id / "metrics" / "fresh_process_validation.json"
        if fp_path.is_file():
            fp = json.loads(fp_path.read_text(encoding="utf-8"))
            table.add_row("Fresh Process", "PASS" if fp.get("passed") else "FAIL")

        # Stage summary
        summary_path = runs_dir / experiment_id / "metrics" / "stage_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            final_state = summary.get("final_state", "unknown")
            table.add_row("Final State", final_state)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        if event.button.id == "btn-back":
            self.app._show_screen("tab-dashboard")

    def action_back(self) -> None:
        """Navigate back to dashboard."""
        self.app._show_screen("tab-dashboard")


def _sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
