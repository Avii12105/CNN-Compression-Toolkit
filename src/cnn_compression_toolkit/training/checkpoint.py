"""Validation-driven training and checkpoint selection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import evaluate_classifier


@dataclass(frozen=True)
class TrainingEpoch:
    """Metrics recorded for one training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float


class BestCheckpointSelector:
    """Keep the first model state with the highest validation accuracy."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_validation_accuracy = float("-inf")
        self.best_validation_loss: float | None = None
        self._best_state: dict[str, torch.Tensor] | None = None

    @property
    def has_checkpoint(self) -> bool:
        return self._best_state is not None

    def consider(
        self,
        model: nn.Module,
        epoch: int,
        validation_accuracy: float,
        validation_loss: float,
    ) -> bool:
        """Snapshot the model if validation accuracy strictly improves."""

        if validation_accuracy <= self.best_validation_accuracy:
            return False
        self.best_epoch = epoch
        self.best_validation_accuracy = validation_accuracy
        self.best_validation_loss = validation_loss
        self._best_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        """Restore the selected CPU snapshot into a model."""

        if self._best_state is None:
            raise RuntimeError("No checkpoint has been selected")
        model.load_state_dict(copy.deepcopy(self._best_state), strict=True)

    def save(self, path: str | Path) -> None:
        """Save the selected raw state_dict to disk."""

        if self._best_state is None:
            raise RuntimeError("No checkpoint has been selected")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._best_state, destination)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    num_classes: int = 10,
) -> list[TrainingEpoch]:
    """Train a classifier and restore its best validation-accuracy state."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    criterion = nn.CrossEntropyLoss()
    selector = BestCheckpointSelector()
    history: list[TrainingEpoch] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            sample_count += labels.size(0)

        validation = evaluate_classifier(model, validation_loader, num_classes, device)
        record = TrainingEpoch(
            epoch=epoch,
            train_loss=total_loss / max(sample_count, 1),
            validation_loss=float(validation["loss"]),
            validation_accuracy=float(validation["accuracy"]),
        )
        history.append(record)
        selector.consider(
            model,
            epoch=epoch,
            validation_accuracy=record.validation_accuracy,
            validation_loss=record.validation_loss,
        )

    selector.restore(model)
    return history