import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_compression_toolkit.evaluation.metrics import evaluate_classifier, parameter_count


class FixedClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "logits",
            torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits[: inputs.shape[0]]


def test_metrics_and_confusion_matrix() -> None:
    inputs = torch.zeros(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 1, 1])
    metrics = evaluate_classifier(
        FixedClassifier(),
        DataLoader(TensorDataset(inputs, labels), batch_size=4),
        2,
        torch.device("cpu"),
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["sample_count"] == 4
    assert metrics["confusion_matrix"] == [[1, 0], [1, 2]]
    assert sum(map(sum, metrics["confusion_matrix"])) == 4


def test_parameter_count_includes_all_parameters() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))

    assert parameter_count(model) == sum(parameter.numel() for parameter in model.parameters())