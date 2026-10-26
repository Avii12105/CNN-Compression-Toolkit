"""Experiment configuration screen — full configuration editor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static


class ConfigScreen(Screen):
    """Configure experiment parameters for compression."""

    BINDINGS = [
        ("s", "save_config", "Save config YAML"),
        ("l", "load_config", "Load config YAML"),
        ("r", "run_compression", "Run compression"),
        ("b", "back", "Back to dashboard"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="config-scroll"):
            with Vertical(id="config-screen"):
                yield Label("Experiment Configuration", classes="section-title")

                # --- Model ---
                yield Static("Model", classes="subsection-title")
                yield Select(
                    [
                        ("ResNet-56 (CIFAR-10)", "resnet56"),
                        ("VGG-16-BN (CIFAR-10)", "vgg16_bn"),
                    ],
                    value=self.app.shared_config.get("architecture", "resnet56"),
                    id="config-architecture",
                )
                yield Input(
                    placeholder="../checkpoints/resnet56_cifar10.pt",
                    value=self.app.shared_config.get("checkpoint_path", ""),
                    id="config-checkpoint",
                )

                # --- Compression ---
                yield Static("Compression", classes="subsection-title")
                yield Input(
                    placeholder="0.5",
                    value=str(self.app.shared_config.get("pruning_ratio", 0.5)),
                    id="config-pruning-ratio",
                )
                yield Select(
                    [
                        ("1", "1"),
                        ("2", "2"),
                        ("4", "4"),
                        ("8", "8"),
                        ("16", "16"),
                        ("32", "32"),
                    ],
                    value=str(self.app.shared_config.get("simd_alignment", 8)),
                    id="config-alignment",
                )
                yield Select(
                    [
                        ("Auto (host gate decides)", "auto"),
                        ("INT8", "int8"),
                        ("FP16", "fp16"),
                        ("FP32", "fp32"),
                    ],
                    value=self.app.shared_config.get("precision", "auto"),
                    id="config-precision",
                )
                yield Input(
                    placeholder="4",
                    value=str(self.app.shared_config.get("protected_count", 4)),
                    id="config-protected-count",
                )
                yield Select(
                    [
                        ("Disabled", "false"),
                        ("Enabled (if M2 misses target)", "true"),
                    ],
                    value="true" if self.app.shared_config.get("distillation", False) else "false",
                    id="config-distillation",
                )
                yield Input(
                    placeholder="5",
                    value=str(self.app.shared_config.get("distillation_epochs", 5)),
                    id="config-distillation-epochs",
                )
                yield Input(
                    placeholder="0.90",
                    value=str(
                        self.app.shared_config.get(
                            "distillation_target_accuracy", 0.90
                        )
                    ),
                    id="config-distillation-target",
                )

                # --- Budget (Search) ---
                yield Static("Budget Constraints (Search)", classes="subsection-title")
                yield Input(
                    placeholder="e.g. 5.0",
                    value=str(self.app.shared_config.get("budget_latency_ms", "") or ""),
                    id="config-budget-latency",
                )
                yield Input(
                    placeholder="e.g. 0.05",
                    value=str(
                        self.app.shared_config.get("budget_max_accuracy_drop", "") or ""
                    ),
                    id="config-budget-acc-drop",
                )
                yield Input(
                    placeholder="e.g. 0.80",
                    value=str(self.app.shared_config.get("budget_min_accuracy", "") or ""),
                    id="config-budget-min-acc",
                )
                yield Input(
                    placeholder="e.g. 2000000",
                    value=str(
                        self.app.shared_config.get("budget_max_size_bytes", "") or ""
                    ),
                    id="config-budget-max-size",
                )

                yield Horizontal(
                    Button("Save Config YAML", id="btn-save", variant="primary"),
                    Button("Load Config YAML", id="btn-load"),
                    Button("Run Compression", id="btn-run", variant="success"),
                    Button("Back", id="btn-back"),
                )
                yield Label("", id="status-message")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        status = self.query_one("#status-message", Label)

        if event.button.id == "btn-save":
            self._save_config(status)
        elif event.button.id == "btn-load":
            self._load_config(status)
        elif event.button.id == "btn-run":
            self._run_compression(status)
        elif event.button.id == "btn-back":
            self.app._show_screen("tab-dashboard")

    def action_save_config(self) -> None:
        """Save config YAML."""
        status = self.query_one("#status-message", Label)
        self._save_config(status)

    def action_load_config(self) -> None:
        """Load config YAML."""
        status = self.query_one("#status-message", Label)
        self._load_config(status)

    def action_run_compression(self) -> None:
        """Save config and switch to Run Status screen."""
        status = self.query_one("#status-message", Label)
        self._run_compression(status)

    def action_back(self) -> None:
        """Navigate back to dashboard."""
        self.app._show_screen("tab-dashboard")

    def _collect_config(self) -> dict[str, Any]:
        """Collect all config values from widgets into shared_config."""

        sc = self.app.shared_config
        sc["architecture"] = self.query_one("#config-architecture", Select).value
        sc["checkpoint_path"] = self.query_one("#config-checkpoint", Input).value or ""

        try:
            sc["pruning_ratio"] = float(
                self.query_one("#config-pruning-ratio", Input).value or 0.5
            )
        except ValueError:
            sc["pruning_ratio"] = 0.5
        sc["simd_alignment"] = int(self.query_one("#config-alignment", Select).value)
        sc["precision"] = self.query_one("#config-precision", Select).value

        try:
            sc["protected_count"] = int(
                self.query_one("#config-protected-count", Input).value or 0
            )
        except ValueError:
            sc["protected_count"] = 0
        sc["distillation"] = (
            self.query_one("#config-distillation", Select).value == "true"
        )
        try:
            sc["distillation_epochs"] = int(
                self.query_one("#config-distillation-epochs", Input).value or 0
            )
        except ValueError:
            sc["distillation_epochs"] = 0
        try:
            sc["distillation_target_accuracy"] = float(
                self.query_one("#config-distillation-target", Input).value or 0.9
            )
        except ValueError:
            sc["distillation_target_accuracy"] = 0.9

        # Budget constraints
        sc["budget_latency_ms"] = self._parse_float(
            self.query_one("#config-budget-latency", Input).value
        )
        sc["budget_max_accuracy_drop"] = self._parse_float(
            self.query_one("#config-budget-acc-drop", Input).value
        )
        sc["budget_min_accuracy"] = self._parse_float(
            self.query_one("#config-budget-min-acc", Input).value
        )
        sc["budget_max_size_bytes"] = self._parse_int(
            self.query_one("#config-budget-max-size", Input).value
        )

        return sc

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if not value or not value.strip():
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if not value or not value.strip():
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _save_config(self, status) -> None:
        """Save shared config to a YAML file."""

        config = self._collect_config()

        yaml_config = {
            "project": {
                "name": "cnn-compression-toolkit",
                "seed": 20261024,
            },
            "model": {
                "architecture": config["architecture"],
                "checkpoint": config["checkpoint_path"],
                "checkpoint_format": "state_dict",
                "num_classes": 10,
                "input_shape": [1, 3, 32, 32],
            },
            "dataset": {
                "root": "../data/cifar10",
                "calibration_size": 512,
                "calibration_sampling": "fixed-seed",
                "calibration_seed": 20261024,
                "validation_size": 5000,
                "download": False,
                "normalization": {
                    "mean": [0.4914, 0.4822, 0.4465],
                    "std": [0.2470, 0.2435, 0.2616],
                },
                "train_split": "train",
                "test_split": "test",
            },
            "evaluation": {
                "batch_size": 128,
                "num_workers": 2,
            },
            "runtime": {
                "device": "auto",
                "allow_cpu_fallback": True,
            },
            "benchmark": {
                "provider": "CPUExecutionProvider",
                "batch_size": 1,
                "intra_op_threads": 1,
                "inter_op_threads": 1,
                "warmup_runs": 50,
                "timed_runs": 200,
            },
            "onnx": {
                "opset": 18,
                "export_mode": "torch_onnx",
                "input_name": "input",
                "output_name": "logits",
                "input_shape": [1, 3, 32, 32],
                "rtol": 1.0e-03,
                "atol": 1.0e-04,
            },
            "output": {
                "artifact_root": "../artifacts",
            },
            "logging": {
                "level": "INFO",
                "capture_environment": True,
            },
            "compression": {
                "default_pruning_ratio": config["pruning_ratio"],
                "default_precision": config["precision"],
                "run_distillation": config["distillation"],
                "distillation_target_accuracy": config["distillation_target_accuracy"],
                "distillation_alpha": config["distillation_alpha"],
                "distillation_temperature": config["distillation_temperature"],
                "distillation_learning_rate": config["distillation_learning_rate"],
                "protection_gamma": config["protection_gamma"],
                "protected_layers_count": config["protected_count"],
                "simd_alignment": config["simd_alignment"],
                "ratio_clamp_max": config["ratio_clamp_max"],
            },
            "quantization": {
                "precision": config["precision"],
                "calibration_size": 512,
                "qdq": True,
                "per_channel": True,
            },
            "search": {
                "budget_latency_ms": config["budget_latency_ms"],
                "budget_max_accuracy_drop": config["budget_max_accuracy_drop"],
                "budget_min_accuracy": config["budget_min_accuracy"],
                "budget_max_size_bytes": config["budget_max_size_bytes"],
            },
        }

        out_path = Path("configs") / "tui_config.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_path.write_text(
                yaml.safe_dump(yaml_config, sort_keys=False), encoding="utf-8"
            )
            self.app.config_path = out_path
            status.update(f"Config saved to {out_path}")
        except Exception as exc:
            status.update(f"Failed to save config: {exc}")

    def _load_config(self, status) -> None:
        """Load config from a YAML file."""

        load_path = self.app.config_path or Path("configs/baseline.yaml")

        if not load_path.is_file():
            status.update(f"Config file not found: {load_path}")
            return

        try:
            raw = yaml.safe_load(load_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status.update(f"Failed to load config: {exc}")
            return

        model = raw.get("model", {})
        compression = raw.get("compression", {})
        search = raw.get("search", {})

        self.app.shared_config["architecture"] = model.get(
            "architecture", "resnet56"
        )
        self.app.shared_config["checkpoint_path"] = model.get("checkpoint", "")
        self.app.shared_config["pruning_ratio"] = compression.get(
            "default_pruning_ratio", 0.5
        )
        self.app.shared_config["simd_alignment"] = compression.get(
            "simd_alignment", 8
        )
        self.app.shared_config["precision"] = compression.get(
            "default_precision", "fp32"
        )
        self.app.shared_config["protected_count"] = compression.get(
            "protected_layers_count", 4
        )
        self.app.shared_config["distillation"] = compression.get(
            "run_distillation", False
        )
        self.app.shared_config["distillation_epochs"] = compression.get(
            "distillation_epochs", 5
        )
        self.app.shared_config["distillation_target_accuracy"] = compression.get(
            "distillation_target_accuracy"
        )
        self.app.shared_config["distillation_alpha"] = compression.get(
            "distillation_alpha", 0.3
        )
        self.app.shared_config["distillation_temperature"] = compression.get(
            "distillation_temperature", 4.0
        )
        self.app.shared_config["distillation_learning_rate"] = compression.get(
            "distillation_learning_rate", 0.001
        )
        self.app.shared_config["protection_gamma"] = compression.get(
            "protection_gamma", 0.5
        )
        self.app.shared_config["ratio_clamp_max"] = compression.get(
            "ratio_clamp_max", 0.9
        )
        self.app.shared_config["budget_latency_ms"] = search.get(
            "budget_latency_ms"
        )
        self.app.shared_config["budget_max_accuracy_drop"] = search.get(
            "budget_max_accuracy_drop"
        )
        self.app.shared_config["budget_min_accuracy"] = search.get(
            "budget_min_accuracy"
        )
        self.app.shared_config["budget_max_size_bytes"] = search.get(
            "budget_max_size_bytes"
        )

        self.app.config_path = load_path
        status.update(f"Config loaded from {load_path}")

    def _run_compression(self, status) -> None:
        """Save config and switch to Run Status screen."""

        self._save_config(status)
        if not self.app.config_path:
            status.update("Cannot run: config not saved.")
            return
        self.app._show_screen("tab-run")
