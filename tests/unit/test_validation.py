from pathlib import Path

import torch
from torch import nn

from cnn_compression_toolkit.evaluation.onnx_export import export_to_onnx
from cnn_compression_toolkit.evaluation.validation import validate_fresh_process


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.mean(dim=(2, 3)))


def test_fresh_process_validation(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(TinyClassifier(), torch.zeros(1, 3, 32, 32), path)

    result = validate_fresh_process(path)

    assert result["passed"] is True
    assert result["output_shape"] == [1, 10]


def test_fresh_process_validation_reports_missing_file(tmp_path: Path) -> None:
    result = validate_fresh_process(tmp_path / "missing.onnx")

    assert result["passed"] is False
    assert result["returncode"] != 0