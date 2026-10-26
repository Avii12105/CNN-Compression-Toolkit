from pathlib import Path

import onnx
import pytest
import torch
from torch import nn

from cnn_compression_toolkit.evaluation.onnx_export import compare_pytorch_onnx, export_to_onnx


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(3, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(inputs).flatten(1))


def test_export_and_equivalence(tmp_path: Path) -> None:
    model = TinyClassifier().train()
    sample_input = torch.ones(1, 3, 32, 32)
    output_path = tmp_path / "tiny.onnx"

    result = export_to_onnx(model, sample_input, output_path)
    comparison = compare_pytorch_onnx(model, result, sample_input)

    assert output_path.is_file()
    assert model.training is True
    assert result.input_shape == (1, 3, 32, 32)
    assert result.output_shape == (1, 10)
    assert comparison["passed"] is True
    onnx.checker.check_model(onnx.load(output_path))


def test_export_rejects_wrong_input_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="representative input shape"):
        export_to_onnx(TinyClassifier(), torch.zeros(1, 3, 28, 28), tmp_path / "bad.onnx")