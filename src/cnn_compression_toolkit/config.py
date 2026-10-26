"""Configuration loading and validation for Phase 0 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_ARCHITECTURES = {"resnet56", "vgg16_bn"}
SUPPORTED_PRECISIONS = {"fp32", "fp16", "int8"}
SUPPORTED_PROVIDERS = {"CPUExecutionProvider"}


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration with its source path and normalized data."""

    data: dict[str, Any]
    source: Path

    @property
    def artifact_root(self) -> Path:
        return Path(self.data["output"]["artifact_root"])

    @property
    def experiment_name(self) -> str:
        return str(self.data["project"]["name"])


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML configuration without checking external assets."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    data = _normalize(raw, source.parent)
    _validate(data)
    return ExperimentConfig(data=data, source=source)


def _normalize(data: dict[str, Any], config_directory: Path) -> dict[str, Any]:
    normalized = dict(data)
    sections = (
        "project",
        "model",
        "dataset",
        "runtime",
        "benchmark",
        "output",
        "logging",
        "compression",
        "quantization",
        "search",
    )
    for section in sections:
        if not isinstance(normalized.get(section), dict):
            continue
        normalized[section] = dict(normalized[section])

    for section, key in (("model", "checkpoint"), ("dataset", "root"), ("output", "artifact_root")):
        value = normalized.get(section, {}).get(key)
        if isinstance(value, str) and value and not Path(value).expanduser().is_absolute():
            normalized[section][key] = str((config_directory / value).resolve())
    return normalized


def _validate(data: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "model",
        "dataset",
        "runtime",
        "benchmark",
        "output",
        "logging",
        "compression",
    }
    missing = sorted(required_sections - data.keys())
    if missing:
        raise ConfigurationError(f"Missing required sections: {', '.join(missing)}")

    project = data["project"]
    model = data["model"]
    dataset = data["dataset"]
    runtime = data["runtime"]
    benchmark = data["benchmark"]
    onnx = data.get("onnx", {})
    evaluation = data.get("evaluation", {})
    output = data["output"]
    logging = data["logging"]
    compression = data["compression"]

    if not isinstance(project.get("name"), str) or not project["name"].strip():
        raise ConfigurationError("project.name must be a non-empty string")
    if model.get("architecture") not in SUPPORTED_ARCHITECTURES:
        supported = sorted(SUPPORTED_ARCHITECTURES)
        raise ConfigurationError(f"model.architecture must be one of {supported}")
    if not isinstance(model.get("checkpoint"), str) or not model["checkpoint"]:
        raise ConfigurationError("model.checkpoint must be a non-empty path")
    if model.get("checkpoint_format", "state_dict") != "state_dict":
        raise ConfigurationError("model.checkpoint_format must be state_dict")
    if model.get("num_classes", 10) != 10:
        raise ConfigurationError("model.num_classes must be 10 for CIFAR-10")
    if model.get("input_shape", [1, 3, 32, 32]) != [1, 3, 32, 32]:
        raise ConfigurationError("model.input_shape must be [1, 3, 32, 32]")
    if not isinstance(dataset.get("root"), str) or not dataset["root"]:
        raise ConfigurationError("dataset.root must be a non-empty path")
    if dataset.get("calibration_size") != 512:
        raise ConfigurationError("dataset.calibration_size must be exactly 512 in Phase 0")
    if not isinstance(dataset.get("calibration_seed", data["project"].get("seed")), int):
        raise ConfigurationError("dataset.calibration_seed must be an integer")
    validation_size = dataset.get("validation_size", 5000)
    if not isinstance(validation_size, int) or not 0 < validation_size < 50000 - 512:
        raise ConfigurationError("dataset.validation_size must be between 1 and 49487")
    normalization = dataset.get("normalization", {})
    if normalization.get("mean", [0.4914, 0.4822, 0.4465]) != [0.4914, 0.4822, 0.4465]:
        raise ConfigurationError("dataset.normalization.mean must be the CIFAR-10 mean")
    if normalization.get("std", [0.2470, 0.2435, 0.2616]) != [0.2470, 0.2435, 0.2616]:
        raise ConfigurationError(
            "dataset.normalization.std must be the CIFAR-10 standard deviation"
        )
    if runtime.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("runtime.device must be auto, cpu, or cuda")
    if benchmark.get("provider") not in SUPPORTED_PROVIDERS:
        raise ConfigurationError("benchmark.provider must be CPUExecutionProvider")
    for key in ("batch_size", "intra_op_threads", "inter_op_threads", "warmup_runs", "timed_runs"):
        if not isinstance(benchmark.get(key), int) or benchmark[key] <= 0:
            raise ConfigurationError(f"benchmark.{key} must be a positive integer")
    for key in ("batch_size", "num_workers"):
        value = evaluation.get(key, 0)
        if not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"evaluation.{key} must be a non-negative integer")
    if evaluation.get("batch_size", 0) <= 0:
        raise ConfigurationError("evaluation.batch_size must be a positive integer")
    if not isinstance(onnx.get("opset", 18), int) or not 13 <= onnx.get("opset", 18) <= 20:
        raise ConfigurationError("onnx.opset must be an integer between 13 and 20")
    if onnx.get("export_mode", "torch_onnx") != "torch_onnx":
        raise ConfigurationError("onnx.export_mode must be torch_onnx")
    for key in ("input_name", "output_name"):
        if not isinstance(onnx.get(key), str) or not onnx[key].strip():
            raise ConfigurationError(f"onnx.{key} must be a non-empty string")
    if onnx.get("input_shape", [1, 3, 32, 32]) != [1, 3, 32, 32]:
        raise ConfigurationError("onnx.input_shape must be [1, 3, 32, 32]")
    for key in ("rtol", "atol"):
        value = onnx.get(key, 0.0)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigurationError(f"onnx.{key} must be non-negative")
    if not isinstance(output.get("artifact_root"), str) or not output["artifact_root"]:
        raise ConfigurationError("output.artifact_root must be a non-empty path")
    if logging.get("level") not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError("logging.level must be DEBUG, INFO, WARNING, or ERROR")
    ratio = compression.get("default_pruning_ratio")
    if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 0.9:
        raise ConfigurationError("compression.default_pruning_ratio must be between 0 and 0.9")
    precision = compression.get("default_precision")
    if precision not in SUPPORTED_PRECISIONS:
        supported = sorted(SUPPORTED_PRECISIONS)
        raise ConfigurationError(f"compression.default_precision must be one of {supported}")