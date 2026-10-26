"""Run status screen — start experiments and view live logs."""

from __future__ import annotations

import threading
from pathlib import Path
from subprocess import PIPE, Popen

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static


class RunStatusScreen(Screen):
    """Run compression experiments and view live logs."""

    BINDINGS = [
        ("r", "run_command", "Run command"),
        ("s", "stop_process", "Stop process"),
        ("b", "back", "Back to dashboard"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._process: Popen | None = None
        self._reader_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="run-status-scroll"):
            with Vertical(id="run-status"):
                yield Label("Run Status & Logs", classes="section-title")

                yield Static("Command", classes="subsection-title")
                yield Select(
                    [
                        ("compress — Full M0 → M4 pipeline", "compress"),
                        ("baseline — M0 baseline only", "baseline"),
                        ("measure — ONNX export & benchmark", "measure"),
                        ("search — Budget-driven candidate search", "search"),
                        ("matrix — Experiment matrix", "matrix"),
                        ("pareto — Pareto frontier analysis", "pareto"),
                        ("validate-final — Final validation", "validate-final"),
                        ("report — Generate report", "report"),
                    ],
                    value="compress",
                    id="run-command",
                )

                yield Static("Config File", classes="subsection-title")
                yield Input(
                    placeholder="configs/baseline.yaml",
                    value=str(self.app.config_path or "configs/baseline.yaml"),
                    id="run-config-path",
                )

                yield Static("Extra Arguments", classes="subsection-title")
                yield Input(
                    placeholder="--pruning-ratio 0.3 --alignment 8 ...",
                    value="",
                    id="run-extra-args",
                )

                yield Horizontal(
                    Button("Run", id="btn-run", variant="primary"),
                    Button("Stop", id="btn-stop", variant="error"),
                    Button("Back", id="btn-back"),
                )
                yield Static("Live Log", classes="subsection-title")
                yield ScrollableContainer(Static(id="log-view", classes="log-content"))

    def on_mount(self) -> None:
        """Pre-fill extra args from shared config for compress command."""
        sc = self.app.shared_config
        args = []
        if sc.get("pruning_ratio") is not None:
            args.append(f"--pruning-ratio {sc['pruning_ratio']}")
        if sc.get("simd_alignment") is not None:
            args.append(f"--alignment {sc['simd_alignment']}")
        if sc.get("precision") is not None:
            args.append(f"--precision {sc['precision']}")
        if sc.get("protected_count") is not None:
            args.append(f"--protected-count {sc['protected_count']}")
        if sc.get("distillation_epochs") and sc.get("distillation_epochs") > 0:
            args.append(f"--distillation-epochs {sc['distillation_epochs']}")
        if sc.get("distillation_target_accuracy") is not None:
            args.append(
                f"--distillation-target-accuracy {sc['distillation_target_accuracy']}"
            )
        if sc.get("budget_latency_ms") is not None:
            args.append(f"--budget-latency-ms {sc['budget_latency_ms']}")
        if sc.get("budget_max_accuracy_drop") is not None:
            args.append(f"--budget-max-accuracy-drop {sc['budget_max_accuracy_drop']}")
        if sc.get("budget_min_accuracy") is not None:
            args.append(f"--budget-min-accuracy {sc['budget_min_accuracy']}")
        if sc.get("budget_max_size_bytes") is not None:
            args.append(f"--budget-max-size-bytes {sc['budget_max_size_bytes']}")

        self.query_one("#run-extra-args", Input).value = " ".join(args)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        log = self.query_one("#log-view", Static)
        if event.button.id == "btn-run":
            config = (
                self.query_one("#run-config-path", Input).value
                or "configs/baseline.yaml"
            )
            extra = self.query_one("#run-extra-args", Input).value or ""
            cmd = self.query_one("#run-command", Select).value
            self.app.config_path = Path(config)
            log.update(f"Starting {cmd} with config: {config}\n")
            if extra:
                log.update(log.renderable + f"Extra args: {extra}\n")
            self._start_process(cmd, config, extra, log)
        elif event.button.id == "btn-stop":
            if self._process:
                self._process.terminate()
                log.update(log.renderable + "\nProcess stopped by user.")
            self.app._show_screen("tab-dashboard")
        elif event.button.id == "btn-back":
            self.app._show_screen("tab-dashboard")

    def action_run_command(self) -> None:
        """Run the compression command."""
        log = self.query_one("#log-view", Static)
        config = (
            self.query_one("#run-config-path", Input).value
            or "configs/baseline.yaml"
        )
        extra = self.query_one("#run-extra-args", Input).value or ""
        cmd = self.query_one("#run-command", Select).value
        self.app.config_path = Path(config)
        log.update(f"Starting {cmd} with config: {config}\n")
        if extra:
            log.update(log.renderable + f"Extra args: {extra}\n")
        self._start_process(cmd, config, extra, log)

    def action_stop_process(self) -> None:
        """Stop the running process."""
        log = self.query_one("#log-view", Static)
        if self._process:
            self._process.terminate()
            log.update(log.renderable + "\nProcess stopped by user.")
        self.app._show_screen("tab-dashboard")

    def action_back(self) -> None:
        """Navigate back to dashboard."""
        self.app._show_screen("tab-dashboard")

    def _start_process(
        self, command: str, config: str, extra_args: str, log
    ) -> None:
        """Start the compression process in a background thread."""

        import sys

        lines: list[str] = []

        def run_and_read():
            cmd_list = [
                sys.executable,
                "-m",
                "cnn_compression_toolkit.cli",
                command,
                "--config",
                config,
            ]
            if extra_args:
                cmd_list.extend(extra_args.split())

            self._process = Popen(
                cmd_list,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                lines.append(line.rstrip())
                self.app.call_from_thread(self._update_log, log, lines)
            self._process.wait()
            lines.append(f"Process exited with code {self._process.returncode}")
            self.app.call_from_thread(self._update_log, log, lines)

        def _update_log(log_widget, line_list):
            log_widget.update("\n".join(line_list[-100:]))

        self._update_log = _update_log
        self._reader_thread = threading.Thread(target=run_and_read, daemon=True)
        self._reader_thread.start()
