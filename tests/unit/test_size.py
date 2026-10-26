from pathlib import Path

from torch import nn

from cnn_compression_toolkit.evaluation.size import model_size_metrics, serialized_file_metrics


def test_serialized_file_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"phase2")

    result = serialized_file_metrics(artifact)

    assert result["bytes"] == 6
    assert len(result["sha256"]) == 64


def test_model_size_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"onnx")
    model = nn.Linear(3, 2)

    result = model_size_metrics(model, artifact)

    assert result["parameter_count"] == 8
    assert result["onnx"]["bytes"] == 4