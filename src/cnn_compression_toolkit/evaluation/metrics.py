"""PyTorch classification metrics for baseline evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable and non-trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def evaluate_classifier(
    model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device
) -> dict[str, Any]:
    """Evaluate logits in one pass and return accuracy, macro metrics, and confusion matrix."""

    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    sample_count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            sample_count += labels.size(0)
            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[int(actual), int(predicted)] += 1

    true_positive = confusion.diag().float()
    predicted_total = confusion.sum(dim=0).float()
    actual_total = confusion.sum(dim=1).float()
    precision = torch.where(
        predicted_total > 0, true_positive / predicted_total, torch.zeros_like(true_positive)
    )
    recall = torch.where(
        actual_total > 0, true_positive / actual_total, torch.zeros_like(true_positive)
    )
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2 * precision * recall / f1_denominator,
        torch.zeros_like(precision),
    )
    return {
        "loss": total_loss / max(sample_count, 1),
        "accuracy": float(true_positive.sum() / max(sample_count, 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "sample_count": sample_count,
    }