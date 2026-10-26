"""Model / checkpoint selection screen."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static


class ModelSelectScreen(Screen):
    """Select model architecture and checkpoint for compression experiments."""

    BINDINGS = [
        ("b", "back", "Back to dashboard"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="model-select-scroll"):
            with Vertical(id="model-select"):
                yield Label("Model / Checkpoint Selection", classes="section-title")

                yield Static("Architecture", classes="subsection-title")
                yield Select(
                    [
                        ("ResNet-56 (CIFAR-10)", "resnet56"),
                        ("VGG-16-BN (CIFAR-10)", "vgg16_bn"),
                    ],
                    value=self.app.shared_config.get("architecture", "resnet56"),
                    id="architecture-select",
                )

                yield Static("Checkpoint Path", classes="subsection-title")
                yield Input(
                    placeholder="../checkpoints/resnet56_cifar10.pt",
                    value=self.app.shared_config.get("checkpoint_path", ""),
                    id="checkpoint-path",
                )

                yield Horizontal(
                    Button("Detect Checkpoints", id="btn-detect", variant="default"),
                    Button("Load & Validate", id="btn-load", variant="primary"),
                    Button("Back", id="btn-back"),
                )
                yield Label("", id="status-message")
                yield Static("", id="checkpoint-info", classes="info-box")

    def on_mount(self) -> None:
        """Populate checkpoint info on mount."""
        self._refresh_checkpoint_info()

    def _refresh_checkpoint_info(self) -> None:
        """Show available checkpoints in the checkpoints directory."""
        info = self.query_one("#checkpoint-info", Static)
        ckpt_dir = Path("checkpoints")
        if ckpt_dir.is_dir():
            checkpoints = sorted(ckpt_dir.glob("*.pt"))
            if checkpoints:
                lines = ["Available checkpoints:"]
                for cp in checkpoints:
                    lines.append(f"  • {cp.name} ({cp.stat().st_size / 1024:.1f} KB)")
                info.update("\n".join(lines))
            else:
                info.update("No checkpoints found in checkpoints/")
        else:
            info.update("checkpoints/ directory not found")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle architecture selection changes."""
        if event.select.id == "architecture-select":
            self.app.shared_config["architecture"] = event.value
            input_w = self.query_one("#checkpoint-path", Input)
            if event.value == "resnet56":
                input_w.placeholder = "../checkpoints/resnet56_cifar10.pt"
            elif event.value == "vgg16_bn":
                input_w.placeholder = "../checkpoints/vgg16_bn_cifar10.pt"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        status = self.query_one("#status-message", Label)
        if event.button.id == "btn-detect":
            self._refresh_checkpoint_info()
            status.update("Checkpoint list refreshed.")
        elif event.button.id == "btn-load":
            arch = self.query_one("#architecture-select", Select).value
            ckpt = self.query_one("#checkpoint-path", Input).value
            self.app.shared_config["architecture"] = arch
            self.app.shared_config["checkpoint_path"] = ckpt

            if not ckpt:
                status.update("Please enter a checkpoint path.")
                return
            path = Path(ckpt)
            if not path.is_file():
                status.update(f"Checkpoint not found: {ckpt}")
                return
            try:
                import torch

                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict):
                    status.update("Checkpoint must be a state_dict mapping.")
                    return
                num_tensors = len(checkpoint)
                status.update(
                    f"Loaded {arch}: {num_tensors} tensors, valid state_dict."
                )
            except Exception as exc:
                status.update(f"Error loading checkpoint: {exc}")
        elif event.button.id == "btn-back":
            self.app._show_screen("tab-dashboard")

    def action_back(self) -> None:
        """Navigate back to dashboard."""
        self.app._show_screen("tab-dashboard")
