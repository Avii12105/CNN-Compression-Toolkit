"""Conditional knowledge distillation for pruned students."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

from ...evaluation.metrics import evaluate_classifier
from ...training.checkpoint import BestCheckpointSelector, TrainingEpoch


@dataclass(frozen=True)
class DistillationResult:
    skipped: bool
    reason: str
    best_epoch: int | None
    best_validation_accuracy: float | None
    history: tuple[TrainingEpoch, ...]


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Combine hard-label CE and temperature-scaled teacher KL losses."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    cross_entropy = functional.cross_entropy(student_logits, labels)
    teacher_distribution = functional.softmax(teacher_logits / temperature, dim=1)
    student_log_distribution = functional.log_softmax(student_logits / temperature, dim=1)
    kl = functional.kl_div(student_log_distribution, teacher_distribution, reduction="batchmean")
    return (1 - alpha) * cross_entropy + alpha * temperature**2 * kl


def distill_student(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    alpha: float = 0.3,
    temperature: float = 4.0,
    num_classes: int = 10,
    target_accuracy: float | None = None,
    current_accuracy: float | None = None,
) -> DistillationResult:
    """Train a student conditionally and restore its best validation state."""

    target_met = (
        target_accuracy is not None
        and current_accuracy is not None
        and current_accuracy >= target_accuracy
    )
    if target_met:
        return DistillationResult(True, "accuracy_target_met", None, current_accuracy, ())
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    selector = BestCheckpointSelector()
    history: list[TrainingEpoch] = []
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, alpha, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            sample_count += labels.size(0)
        validation = evaluate_classifier(student, validation_loader, num_classes, device)
        record = TrainingEpoch(
            epoch=epoch,
            train_loss=total_loss / max(sample_count, 1),
            validation_loss=float(validation["loss"]),
            validation_accuracy=float(validation["accuracy"]),
        )
        history.append(record)
        selector.consider(student, epoch, record.validation_accuracy, record.validation_loss)

    selector.restore(student)
    return DistillationResult(
        skipped=False,
        reason="trained",
        best_epoch=selector.best_epoch,
        best_validation_accuracy=selector.best_validation_accuracy,
        history=tuple(history),
    )