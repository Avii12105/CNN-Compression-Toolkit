"""Experiment matrix execution for structured compression experiments.

Runs a configurable matrix of (pruning ratio, precision, distillation)
combinations through the compression pipeline, recording all results
including failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..compression.orchestration import CompressionPipeline, StageRecord
from ..compression.pipeline import PipelineState
from ..compression.quantization.precision_gate import probe_host_precision
from ..data import build_cifar10_bundle
from ..experiments.ids import create_experiment_id
from ..experiments.manifest import RunManifest
from ..models.loader import load_model_checkpoint
from ..runtime import collect_environment

# Default experiment matrix from the implementation plan.
DEFAULT_MATRIX: list[dict[str, Any]] = [
    {"pruning_ratio": 0.0, "precision": "fp32", "distillation": False},
    {"pruning_ratio": 0.0, "precision": "int8", "distillation": False},
    {"pruning_ratio": 0.3, "precision": "int8", "distillation": False},
    {"pruning_ratio": 0.5, "precision": "int8", "distillation": False},
    {"pruning_ratio": 0.5, "precision": "int8", "distillation": True},
    {"pruning_ratio": 0.7, "precision": "int8", "distillation": True},
]


@dataclass
class MatrixResult:
    """Result for one matrix cell."""

    config: dict[str, Any]
    experiment_id: str
    succeeded: bool
    stage_records: list[StageRecord] = field(default_factory=list)
    failure: str | None = None
    final_accuracy: float | None = None
    final_latency_ms: float | None = None
    final_size_bytes: int | None = None


class ExperimentMatrix:
    """Execute a matrix of compression experiments."""

    def __init__(
        self,
        config: dict[str, Any],
        artifact_root: str | Path,
        logger,
        matrix: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.artifact_root = Path(artifact_root)
        self.logger = logger
        self.matrix = matrix or DEFAULT_MATRIX
        self.model_config = config["model"]
        self.dataset_config = config["dataset"]
        self.evaluation_config = config["evaluation"]
        self.runtime_config = config["runtime"]
        self.onnx_config = config.get("onnx", {})
        self.benchmark_config = config["benchmark"]
        self.compression_config = config.get("compression", {})
        self.quantization_config = config.get("quantization", {})

    def _device(self):
        import torch

        device_name = self.runtime_config["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return device_name

    def _load(self, device: str):
        bundle = build_cifar10_bundle(
            root=self.dataset_config["root"],
            seed=self.dataset_config.get(
                "calibration_seed", self.config["project"]["seed"]
            ),
            calibration_size=self.dataset_config["calibration_size"],
            validation_size=self.dataset_config.get("validation_size", 5000),
            batch_size=self.evaluation_config["batch_size"],
            num_workers=self.evaluation_config["num_workers"],
            download=self.dataset_config.get("download", False),
        )
        loaded = load_model_checkpoint(
            architecture=self.model_config["architecture"],
            checkpoint_path=self.model_config["checkpoint"],
            input_shape=tuple(self.model_config.get("input_shape", [1, 3, 32, 32])),
            num_classes=self.model_config.get("num_classes", 10),
            device=device,
        )
        return loaded, bundle

    def run_cell(
        self,
        cell_index: int,
        cell_config: dict[str, Any],
        device: str,
    ) -> MatrixResult:
        """Run one matrix cell through the pipeline."""

        pruning_ratio = float(cell_config["pruning_ratio"])
        precision = str(cell_config["precision"])
        use_distillation = bool(cell_config.get("distillation", False))

        label = f"r{pruning_ratio}-{precision}-{'distill' if use_distillation else 'nodistill'}"
        experiment_id = create_experiment_id(
            {**self.config, "matrix_cell": cell_index, "matrix_label": label}
        )
        manifest = RunManifest(
            self.artifact_root,
            experiment_id,
            {**self.config, "matrix_cell": cell_index, "matrix_config": cell_config},
            f"matrix cell {cell_index}",
        )

        try:
            manifest.add_environment(collect_environment())
            loaded, bundle = self._load(device)

            pipeline = CompressionPipeline(
                loaded=loaded,
                bundle=bundle,
                device=__import__("torch").device(device),
                config={**self.config, "matrix_cell": cell_index, "matrix_config": cell_config},
                manifest=manifest,
                logger=self.logger,
            )

            # M0 baseline.
            pipeline.measure_state(
                PipelineState.M0, loaded.reference, "M0 baseline"
            )

            # Precision gate.
            gate_result = probe_host_precision("auto")
            selected_precision = gate_result.selected_precision
            if precision != "auto":
                selected_precision = precision

            # Distillation target.
            distillation_target = None
            if use_distillation:
                distillation_target = self.compression_config.get(
                    "distillation_target_accuracy"
                )

            stages = pipeline.run_full_pipeline(
                pruning_ratio=pruning_ratio,
                requested_alignment=int(
                    self.compression_config.get("simd_alignment", 8)
                ),
                precision=selected_precision,
                sensitivity_bit_width=8,
                protected_count=int(
                    self.compression_config.get("protected_layers_count", 0)
                ),
                protection_factor=float(self.compression_config.get("protection_gamma", 0.5)),
                distillation_target_accuracy=distillation_target,
                distillation_epochs=5 if use_distillation else 0,
                distillation_learning_rate=float(
                    self.compression_config.get("distillation_learning_rate", 1.0e-3)
                ),
                distillation_alpha=float(
                    self.compression_config.get("distillation_alpha", 0.3)
                ),
                distillation_temperature=float(
                    self.compression_config.get("distillation_temperature", 4.0)
                ),
                precision_gate_result={
                    "selected_precision": gate_result.selected_precision,
                    "reason": gate_result.reason,
                    "int8_supported": gate_result.int8_supported,
                    "fp16_efficient": gate_result.fp16_efficient,
                },
            )

            manifest.complete()

            final = stages[-1] if stages else None
            return MatrixResult(
                config=cell_config,
                experiment_id=experiment_id,
                succeeded=True,
                stage_records=stages,
                final_accuracy=final.accuracy if final else None,
                final_latency_ms=final.median_latency_ms if final else None,
                final_size_bytes=final.onnx_size_bytes if final else None,
            )

        except Exception as exc:  # record failure and continue
            self.logger.warning(
                f"Matrix cell {cell_index} ({label}) failed: {exc}"
            )
            manifest.fail(exc)
            return MatrixResult(
                config=cell_config,
                experiment_id=experiment_id,
                succeeded=False,
                stage_records=[],
                failure=str(exc),
            )

    def run_all(self, device: str | None = None) -> list[MatrixResult]:
        """Run the full matrix and return all cell results."""

        if device is None:
            device = self._device()

        results: list[MatrixResult] = []
        for index, cell_config in enumerate(self.matrix):
            self.logger.info(
                f"Matrix cell {index + 1}/{len(self.matrix)}: "
                f"ratio={cell_config['pruning_ratio']}, "
                f"precision={cell_config['precision']}, "
                f"distillation={cell_config.get('distillation', False)}"
            )
            result = self.run_cell(index, cell_config, device)
            results.append(result)
            status = "OK" if result.succeeded else f"FAILED: {result.failure}"
            self.logger.info(f"Cell {index + 1}: {status}")

        return results

    def results_to_dict(self, results: list[MatrixResult]) -> list[dict[str, Any]]:
        """Serialize matrix results for reports."""

        output: list[dict[str, Any]] = []
        for result in results:
            entry: dict[str, Any] = {
                "config": result.config,
                "experiment_id": result.experiment_id,
                "succeeded": result.succeeded,
            }
            if result.succeeded and result.stage_records:
                final = result.stage_records[-1]
                entry.update(
                    {
                        "final_state": final.state.value,
                        "final_accuracy": result.final_accuracy,
                        "final_latency_ms": result.final_latency_ms,
                        "final_size_bytes": result.final_size_bytes,
                        "final_parameter_count": final.parameter_count,
                        "stages": [
                            {
                                "state": r.state.value,
                                "accuracy": r.accuracy,
                                "median_latency_ms": r.median_latency_ms,
                                "p95_latency_ms": r.p95_latency_ms,
                                "parameter_count": r.parameter_count,
                                "onnx_size_bytes": r.onnx_size_bytes,
                            }
                            for r in result.stage_records
                        ],
                    }
                )
            if result.failure:
                entry["failure"] = result.failure
            output.append(entry)
        return output
