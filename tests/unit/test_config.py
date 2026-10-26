from pathlib import Path

import pytest

from cnn_compression_toolkit.config import ConfigurationError, load_config


def test_baseline_config_loads_without_external_assets() -> None:
    config = load_config(Path("configs/baseline.yaml"))

    assert config.data["model"]["architecture"] == "resnet56"
    assert config.data["dataset"]["calibration_size"] == 512
    assert config.artifact_root.name == "artifacts"


def test_invalid_architecture_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "project: {name: test}\n"
        "model: {architecture: mobilenetv2, checkpoint: x}\n"
        "dataset: {root: data, calibration_size: 512}\n"
        "runtime: {device: cpu}\n"
        "benchmark: {provider: CPUExecutionProvider, batch_size: 1, intra_op_threads: 1, "
        "inter_op_threads: 1, warmup_runs: 1, timed_runs: 1}\n"
        "output: {artifact_root: artifacts}\n"
        "logging: {level: INFO}\n"
        "compression: {default_pruning_ratio: 0, default_precision: fp32}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="architecture"):
        load_config(path)


def test_invalid_ratio_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/baseline.yaml").read_text(encoding="utf-8")
    source = source.replace("0.0\n  default_precision", "0.95\n  default_precision")
    path = tmp_path / "invalid.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="pruning_ratio"):
        load_config(path)