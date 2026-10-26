"""Main Textual application for the CNN Compression Toolkit TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Tab, Tabs

from .screens.artifact import ArtifactScreen
from .screens.charts import ChartsScreen
from .screens.config import ConfigScreen
from .screens.dashboard import DashboardScreen
from .screens.model_select import ModelSelectScreen
from .screens.results import ResultsScreen
from .screens.run_status import RunStatusScreen
from .screens.shortcuts import ShortcutsScreen

# Map tab IDs to screen classes
_TAB_SCREEN_MAP: dict[str, type] = {
    "tab-dashboard": DashboardScreen,
    "tab-model": ModelSelectScreen,
    "tab-config": ConfigScreen,
    "tab-run": RunStatusScreen,
    "tab-results": ResultsScreen,
    "tab-charts": ChartsScreen,
    "tab-artifact": ArtifactScreen,
}


class CCTApp(App[None]):
    """CNN Compression Toolkit Terminal User Interface."""

    CSS = """
    #screen-container {
        height: 1fr;
        padding: 1 2;
    }

    #main-tabs {
        height: auto;
    }
    """

    TITLE = "CNN Compression Toolkit"
    SUB_TITLE = "Model Compression Experimentation — Terminal UI"

    BINDINGS = [
        ("question_mark", "toggle_shortcuts", "Toggle shortcuts help"),
        ("q", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.config_path = Path(config_path) if config_path else None
        self.shared_config: dict[str, Any] = {
            "architecture": "resnet56",
            "checkpoint_path": "",
            "pruning_ratio": 0.5,
            "alignment": 8,
            "precision": "auto",
            "distillation": False,
            "distillation_epochs": 5,
            "distillation_target_accuracy": 0.90,
            "distillation_alpha": 0.3,
            "distillation_temperature": 4.0,
            "distillation_learning_rate": 0.001,
            "protected_count": 4,
            "protection_gamma": 0.5,
            "simd_alignment": 8,
            "ratio_clamp_max": 0.9,
            "budget_latency_ms": None,
            "budget_max_accuracy_drop": None,
            "budget_min_accuracy": None,
            "budget_max_size_bytes": None,
        }
        self._shortcuts_visible = False

    def compose(self) -> ComposeResult:
        """Compose the app layout."""

        yield Header()
        yield Tabs(
            Tab("Dashboard", id="tab-dashboard"),
            Tab("Model / Checkpoint", id="tab-model"),
            Tab("Experiment Config", id="tab-config"),
            Tab("Run Status", id="tab-run"),
            Tab("Results", id="tab-results"),
            Tab("Charts", id="tab-charts"),
            Tab("Artifact", id="tab-artifact"),
            id="main-tabs",
            active="tab-dashboard",
        )
        yield Container(id="screen-container")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app and show the dashboard."""

        self.push_screen(DashboardScreen())

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle tab switching."""

        tab_id = str(event.tab.id)
        self._show_screen(tab_id)

    def _show_screen(self, tab_id: str) -> None:
        """Switch to the named screen using switch_screen."""

        tab_id = tab_id.strip().lstrip("#")

        screen_cls = _TAB_SCREEN_MAP.get(tab_id)
        if screen_cls is None:
            self.notify(f"Unknown screen for tab: {tab_id}", severity="error")
            return
        self.switch_screen(screen_cls())

    def action_toggle_shortcuts(self) -> None:
        """Toggle the shortcuts help screen."""

        if self._shortcuts_visible:
            self.pop_screen()
            self._shortcuts_visible = False
        else:
            self.push_screen(ShortcutsScreen())
            self._shortcuts_visible = True


def run(config_path: str | Path | None = None) -> None:
    """Run the TUI application."""

    app = CCTApp(config_path=config_path)
    app.run()
