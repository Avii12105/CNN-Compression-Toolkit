import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_compression_toolkit.compression.distillation.distiller import (
    distill_student,
    distillation_loss,
)


def test_distillation_loss_is_finite() -> None:
    student = torch.randn(4, 3, requires_grad=True)
    teacher = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])

    loss = distillation_loss(student, teacher, labels)

    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None


def test_distillation_skip_when_accuracy_target_is_met() -> None:
    model = nn.Linear(2, 2)
    result = distill_student(
        model,
        copy.deepcopy(model),
        DataLoader(TensorDataset(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))),
        DataLoader(TensorDataset(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))),
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.device("cpu"),
        epochs=1,
        target_accuracy=0.8,
        current_accuracy=0.9,
    )

    assert result.skipped is True
    assert result.reason == "accuracy_target_met"