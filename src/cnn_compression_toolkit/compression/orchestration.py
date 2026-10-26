"""End-to-end compression pipeline orchestration (M0 → M1 → M2 → M3 → M4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..data.cifar10 import CIFAR10Bundle
from ..evaluation.benchmark import benchmark_onnx
from ..evaluation.metrics import evaluate_classifier, parameter_count
from ..evaluation.onnx_export import compare_pytorch_onnx, export_to_onnx
from ..evaluation.size import model_size_metrics
from ..evaluation.validation import validate_fresh_process
from ..experiments.manifest import RunManifest
from ..models.loader import LoadedModel
from ..sensitivity.analyzer import (
    analyze_sensitivity,
    protected_pruning_ratios,
    sensitivity_results_to_dict,
)
from .distillation.distiller import DistillationResult, distill_student
from .pipeline import PipelineState, validate_transition
from .pruning.pruner import prune_model, pruning_records_to_dict
from .recovery.bn_recovery import recover_batch_norm


@dataclass(frozen=True)
class StageRecord:
    """Measured metrics for one compressed model state."""

    state: PipelineState
    accuracy: float
    median_latency_ms: float
    p95_latency_ms: float
    parameter_count: int
    onnx_size_bytes: int
    onnx_sha256: str
    onnx_path: str
    extra: dict[str, Any] = field(default_factory=dict)


class CompressionPipeline:
    """Execute ordered compression stages with artifact lineage and state guards.

    The pipeline starts at M0 and advances only through legal transitions
    (M0 → M1 → M2 → M3 → M4, with M0 → M2 and M0 → M4 terminal shortcuts).
    Each stage exports ONNX, benchmarks, measures size, and records artifacts.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        bundle: CIFAR10Bundle,
        device: torch.device,
        config: dict[str, Any],
        manifest: RunManifest,
        logger,
    ) -> None:
        self.loaded = loaded
        self.bundle = bundle
        self.device = device
        self.config = config
        self.manifest = manifest
        self.logger = logger
        self.current_state = PipelineState.M0
        self.stage_records: list[StageRecord] = []
        self.transformation_history: list[dict[str, Any]] = []
        self.compression_config = config.get("compression", {})
        self.quantization_config = config.get("quantization", {})
        self.onnx_config = config.get("onnx", {})
        self.benchmark_config = config.get("benchmark", {})
        self.evaluation_config = config.get("evaluation", {})
        self.dataset_config = config.get("dataset", {})

    # ------------------------------------------------------------------ #
    # Measurement helper shared by every stage
    # ------------------------------------------------------------------ #
    def measure_state(
        self,
        state: PipelineState,
        model: nn.Module,
        stage_label: str,
        extra: dict[str, Any] | None = None,
    ) -> StageRecord:
        """Export, benchmark, and record metrics for a model state."""

        self.logger.info(f"Measuring {stage_label} ({state.value})")
        sample_input = next(iter(self.bundle.calibration_loader))[0][:1].to(self.device)
        onnx_path = self.manifest.root / "models" / f"model_{state.value.lower()}.onnx"
        export_result = export_to_onnx(
            model,
            sample_input,
            onnx_path,
            input_name=self.onnx_config.get("input_name", "input"),
            output_name=self.onnx_config.get("output_name", "logits"),
            opset=self.onnx_config.get("opset", 18),
        )
        equivalence = compare_pytorch_onnx(
            model,
            export_result,
            sample_input,
            rtol=self.onnx_config.get("rtol", 1.0e-3),
            atol=self.onnx_config.get("atol", 1.0e-4),
        )
        if not equivalence["passed"]:
            raise RuntimeError(
                f"PyTorch/ONNX equivalence failed for {stage_label}: {equivalence}"
            )

        sample_numpy = sample_input.detach().cpu().numpy().astype("float32")
        benchmark = benchmark_onnx(
            onnx_path,
            export_result.input_name,
            sample_numpy,
            intra_op_threads=self.benchmark_config["intra_op_threads"],
            inter_op_threads=self.benchmark_config["inter_op_threads"],
            warmup_runs=self.benchmark_config["warmup_runs"],
            timed_runs=self.benchmark_config["timed_runs"],
        )
        size_metrics = model_size_metrics(model, onnx_path)
        fresh_process = validate_fresh_process(
            onnx_path,
            input_name=export_result.input_name,
            output_name=export_result.output_name,
            input_shape=export_result.input_shape,
        )
        if not fresh_process["passed"]:
            raise RuntimeError(
                f"Fresh-process validation failed for {stage_label}: {fresh_process}"
            )

        # Accuracy on the reserved test set only.
        test_metrics = evaluate_classifier(
            model, self.bundle.test_loader, self.loaded.num_classes, self.device
        )

        record = StageRecord(
            state=state,
            accuracy=float(test_metrics["accuracy"]),
            median_latency_ms=float(benchmark["median"]),
            p95_latency_ms=float(benchmark["p95"]),
            parameter_count=int(size_metrics["parameter_count"]),
            onnx_size_bytes=int(size_metrics["onnx"]["bytes"]),
            onnx_sha256=str(size_metrics["onnx"]["sha256"]),
            onnx_path=str(onnx_path),
            extra={
                **(extra or {}),
                "equivalence": equivalence,
                "benchmark": benchmark,
                "size": size_metrics,
                "fresh_process_validation": fresh_process,
                "test_metrics": {
                    "loss": test_metrics["loss"],
                    "accuracy": test_metrics["accuracy"],
                    "macro_precision": test_metrics["macro_precision"],
                    "macro_recall": test_metrics["macro_recall"],
                    "macro_f1": test_metrics["macro_f1"],
                    "sample_count": test_metrics["sample_count"],
                },
            },
        )
        self.stage_records.append(record)

        # Persist stage metrics and artifacts.
        metrics_path = (
            self.manifest.root / "metrics" / f"stage_{state.value.lower()}.json"
        )
        self._write_json(metrics_path, record.extra)
        self.manifest.record_artifact(f"{state.value.lower()}_metrics", metrics_path)
        self.manifest.record_artifact(f"{state.value.lower()}_onnx", onnx_path)
        self.logger.info(
            f"{stage_label} complete: accuracy={record.accuracy:.4f}, "
            f"median_latency={record.median_latency_ms:.3f} ms, "
            f"size={record.onnx_size_bytes} bytes, params={record.parameter_count}"
        )
        return record

    # ------------------------------------------------------------------ #
    # Stage: M1 — structured pruning
    # ------------------------------------------------------------------ #
    def run_m1(
        self,
        pruning_ratio: float,
        requested_alignment: int,
        sensitivity_results: list[Any] | None = None,
        protected_count: int = 0,
        protection_factor: float = 0.5,
    ) -> StageRecord:
        """Prune channels (optionally sensitivity-guided) and measure M1."""

        validate_transition(self.current_state, PipelineState.M1)
        self.logger.info("Starting M1 structured pruning")

        compression = self.compression_config
        maximum_ratio = float(compression.get("ratio_clamp_max", 0.9))
        working = self.loaded.working
        example_inputs = next(iter(self.bundle.calibration_loader))[0][:1].to(self.device)

        layer_ratios: dict[str, float] | None = None
        marked_results: list[Any] | None = None
        if sensitivity_results is not None and protected_count > 0:
            layer_ratios, marked_results = protected_pruning_ratios(
                sensitivity_results,
                pruning_ratio,
                protected_count,
                protection_factor,
            )
            self.logger.info(
                f"Sensitivity-guided pruning: protecting {protected_count} layers "
                f"with factor {protection_factor}"
            )

        records = prune_model(
            working,
            example_inputs,
            pruning_ratio,
            requested_alignment=requested_alignment,
            maximum_ratio=maximum_ratio,
            layer_ratios=layer_ratios,
        )

        recovery = recover_batch_norm(
            working,
            self.bundle.calibration_loader,
            self.device,
            expected_samples=self.dataset_config["calibration_size"],
        )

        self.transformation_history.append(
            {
                "stage": "M1",
                "transition": f"{self.current_state.value} -> {PipelineState.M1.value}",
                "pruning_ratio": pruning_ratio,
                "requested_alignment": requested_alignment,
                "maximum_ratio": maximum_ratio,
                "sensitivity_guided": sensitivity_results is not None,
                "protected_count": protected_count,
                "protection_factor": protection_factor,
                "pruning_records": pruning_records_to_dict(records),
                "bn_recovery": {
                    "batch_norm_layers": recovery.batch_norm_layers,
                    "sample_count": recovery.sample_count,
                },
            }
        )
        if marked_results is not None:
            self.transformation_history[-1]["sensitivity_results"] = sensitivity_results_to_dict(
                marked_results
            )

        self.current_state = PipelineState.M1
        return self.measure_state(
            PipelineState.M1,
            working,
            "M1 pruned",
            extra={"pruning_records": pruning_records_to_dict(records)},
        )

    # ------------------------------------------------------------------ #
    # Stage: M2 — BN recovery (standalone, after M1)
    # ------------------------------------------------------------------ #
    def run_m2(self) -> StageRecord:
        """Re-estimate BN statistics on the current working model and measure M2."""

        validate_transition(self.current_state, PipelineState.M2)
        self.logger.info("Starting M2 BN recovery")

        # Measure pre-recovery accuracy on the test set.
        pre_recovery = evaluate_classifier(
            self.loaded.working, self.bundle.test_loader, self.loaded.num_classes, self.device
        )

        recovery = recover_batch_norm(
            self.loaded.working,
            self.bundle.calibration_loader,
            self.device,
            expected_samples=self.dataset_config["calibration_size"],
        )

        # Measure post-recovery accuracy on the test set.
        post_recovery = evaluate_classifier(
            self.loaded.working, self.bundle.test_loader, self.loaded.num_classes, self.device
        )

        self.transformation_history.append(
            {
                "stage": "M2",
                "transition": f"{self.current_state.value} -> {PipelineState.M2.value}",
                "bn_recovery": {
                    "batch_norm_layers": recovery.batch_norm_layers,
                    "sample_count": recovery.sample_count,
                },
                "pre_recovery_accuracy": float(pre_recovery["accuracy"]),
                "post_recovery_accuracy": float(post_recovery["accuracy"]),
            }
        )
        self.current_state = PipelineState.M2
        return self.measure_state(
            PipelineState.M2,
            self.loaded.working,
            "M2 BN-recovered",
            extra={
                "pre_recovery_accuracy": float(pre_recovery["accuracy"]),
                "post_recovery_accuracy": float(post_recovery["accuracy"]),
            },
        )

    # ------------------------------------------------------------------ #
    # Stage: M3 — conditional knowledge distillation
    # ------------------------------------------------------------------ #
    def run_m3(
        self,
        target_accuracy: float,
        epochs: int,
        learning_rate: float,
        alpha: float = 0.3,
        temperature: float = 4.0,
    ) -> StageRecord | None:
        """Distill a student only if M2 misses the accuracy target.

        Returns None when distillation is skipped (target already met).
        """

        validate_transition(self.current_state, PipelineState.M3)
        self.logger.info(
            f"Checking M3 distillation condition: target_accuracy={target_accuracy}"
        )

        current_accuracy = (
            self.stage_records[-1].accuracy if self.stage_records else None
        )
        if current_accuracy is not None and current_accuracy >= target_accuracy:
            self.logger.info(
                f"Skipping distillation: current accuracy {current_accuracy:.4f} "
                f"meets target {target_accuracy:.4f}"
            )
            self.transformation_history.append(
                {
                    "stage": "M3",
                    "transition": f"{self.current_state.value} -> {PipelineState.M3.value}",
                    "skipped": True,
                    "reason": "accuracy_target_met",
                    "current_accuracy": float(current_accuracy),
                    "target_accuracy": float(target_accuracy),
                }
            )
            self.current_state = PipelineState.M3
            return None

        if epochs <= 0:
            raise ValueError("epochs must be positive when distillation is required")

        teacher = self.loaded.reference
        student = self.loaded.working
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        result: DistillationResult = distill_student(
            teacher=teacher,
            student=student,
            train_loader=self.bundle.train_loader,
            validation_loader=self.bundle.validation_loader,
            optimizer=optimizer,
            device=self.device,
            epochs=epochs,
            alpha=alpha,
            temperature=temperature,
            num_classes=self.loaded.num_classes,
            target_accuracy=target_accuracy,
            current_accuracy=current_accuracy,
        )

        self.transformation_history.append(
            {
                "stage": "M3",
                "transition": f"{self.current_state.value} -> {PipelineState.M3.value}",
                "skipped": result.skipped,
                "reason": result.reason,
                "best_epoch": result.best_epoch,
                "best_validation_accuracy": result.best_validation_accuracy,
                "alpha": alpha,
                "temperature": temperature,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "history": [
                    {
                        "epoch": record.epoch,
                        "train_loss": record.train_loss,
                        "validation_loss": record.validation_loss,
                        "validation_accuracy": record.validation_accuracy,
                    }
                    for record in result.history
                ],
            }
        )
        self.current_state = PipelineState.M3
        return self.measure_state(
            PipelineState.M3,
            student,
            "M3 distilled",
            extra={
                "distillation": {
                    "skipped": result.skipped,
                    "reason": result.reason,
                    "best_epoch": result.best_epoch,
                    "best_validation_accuracy": result.best_validation_accuracy,
                    "alpha": alpha,
                    "temperature": temperature,
                }
            },
        )

    # ------------------------------------------------------------------ #
    # Stage: M4 — terminal quantization
    # ------------------------------------------------------------------ #
    def run_m4(
        self,
        precision: str,
        precision_gate_result: dict[str, Any] | None = None,
    ) -> StageRecord:
        """Apply terminal quantization (INT8 or FP16) and measure M4."""

        validate_transition(self.current_state, PipelineState.M4)
        self.logger.info(f"Starting M4 quantization: precision={precision}")

        source_model = self.loaded.working
        onnx_path = self.manifest.root / "models" / "model_m4.onnx"
        sample_input = next(iter(self.bundle.calibration_loader))[0][:1].to(self.device)

        if precision == "int8":
            from ..compression.quantization.quantizer import quantize_onnx_static

            quantize_onnx_static(
                source_model,
                sample_input,
                onnx_path,
                calibration_loader=self.bundle.calibration_loader,
                device=self.device,
                input_name=self.onnx_config.get("input_name", "input"),
                output_name=self.onnx_config.get("output_name", "logits"),
                opset=self.onnx_config.get("opset", 18),
            )
        elif precision == "fp16":
            from ..compression.quantization.quantizer import convert_to_fp16_onnx

            convert_to_fp16_onnx(
                source_model,
                sample_input,
                onnx_path,
                input_name=self.onnx_config.get("input_name", "input"),
                output_name=self.onnx_config.get("output_name", "logits"),
                opset=self.onnx_config.get("opset", 18),
            )
        else:
            raise ValueError(f"Unsupported M4 precision: {precision}")

        # Reload the quantized ONNX for measurement (it may differ from the in-memory model).
        import onnx
        import onnxruntime as ort

        graph = onnx.load(onnx_path)
        onnx.checker.check_model(graph)

        session = ort.InferenceSession(
            onnx_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        sample_numpy = sample_input.detach().cpu().numpy().astype("float32")
        benchmark = benchmark_onnx(
            onnx_path,
            input_name,
            sample_numpy,
            intra_op_threads=self.benchmark_config["intra_op_threads"],
            inter_op_threads=self.benchmark_config["inter_op_threads"],
            warmup_runs=self.benchmark_config["warmup_runs"],
            timed_runs=self.benchmark_config["timed_runs"],
        )
        size_metrics = {
            "parameter_count": parameter_count(source_model),
            "onnx": {
                "path": str(onnx_path),
                "bytes": onnx_path.stat().st_size,
                "sha256": __import__("hashlib").sha256(onnx_path.read_bytes()).hexdigest(),
            },
        }
        fresh_process = validate_fresh_process(
            onnx_path,
            input_name=input_name,
            output_name=output_name,
            input_shape=tuple(sample_input.shape),
        )
        if not fresh_process["passed"]:
            raise RuntimeError(f"Fresh-process validation failed for M4: {fresh_process}")

        # Test-set accuracy via ONNX Runtime (batch size 1 due to fixed input shape).
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.bundle.test_loader:
                for i in range(images.shape[0]):
                    single_image = images[i : i + 1].numpy().astype("float32")
                    outputs = session.run([output_name], {input_name: single_image})[0]
                    prediction = int(outputs.argmax(axis=1)[0])
                    correct += int(prediction == int(labels[i]))
                    total += 1
        accuracy = correct / max(total, 1)

        self.transformation_history.append(
            {
                "stage": "M4",
                "transition": f"{self.current_state.value} -> {PipelineState.M4.value}",
                "precision": precision,
                "precision_gate": precision_gate_result or {},
                "onnx_path": str(onnx_path),
            }
        )
        self.current_state = PipelineState.M4
        return self.measure_state(
            PipelineState.M4,
            source_model,
            "M4 quantized",
            extra={
                "precision": precision,
                "precision_gate": precision_gate_result or {},
                "benchmark": benchmark,
                "size": size_metrics,
                "fresh_process_validation": fresh_process,
                "test_metrics": {
                    "accuracy": float(accuracy),
                    "sample_count": total,
                },
            },
        )

    # ------------------------------------------------------------------ #
    # Sensitivity analysis helper
    # ------------------------------------------------------------------ #
    def compute_sensitivity(self, bit_width: int = 8) -> list[Any]:
        """Run weights-only sensitivity analysis on the current working model."""

        self.logger.info("Computing layer sensitivity analysis")
        results = analyze_sensitivity(
            self.loaded.working,
            self.bundle.calibration_loader,
            self.device,
            bit_width=bit_width,
        )
        self.transformation_history.append(
            {
                "stage": "sensitivity",
                "sensitivity_results": [
                    {
                        "layer_name": r.layer_name,
                        "layer_type": r.layer_type,
                        "mse": r.mse,
                        "sqnr": r.sqnr,
                        "bit_width": r.bit_width,
                        "protected": r.protected,
                    }
                    for r in results
                ],
            }
        )
        return results

    # ------------------------------------------------------------------ #
    # Full pipeline driver
    # ------------------------------------------------------------------ #
    def run_full_pipeline(
        self,
        pruning_ratio: float,
        requested_alignment: int,
        precision: str,
        sensitivity_bit_width: int = 8,
        protected_count: int = 0,
        protection_factor: float = 0.5,
        distillation_target_accuracy: float | None = None,
        distillation_epochs: int = 0,
        distillation_learning_rate: float = 1.0e-3,
        distillation_alpha: float = 0.3,
        distillation_temperature: float = 4.0,
        precision_gate_result: dict[str, Any] | None = None,
    ) -> list[StageRecord]:
        """Execute the configured compression sequence end to end.

        Sequence: sensitivity → M1 (prune) → M2 (BN recover) →
        M3 (conditional distill) → M4 (quantize).
        """

        # Sensitivity analysis always runs first to inform pruning protection.
        sensitivity_results = self.compute_sensitivity(bit_width=sensitivity_bit_width)

        # M1: prune (with sensitivity-guided protection if configured).
        self.run_m1(
            pruning_ratio=pruning_ratio,
            requested_alignment=requested_alignment,
            sensitivity_results=sensitivity_results,
            protected_count=protected_count,
            protection_factor=protection_factor,
        )

        # M2: BN recovery.
        self.run_m2()

        # M3: conditional distillation.
        if distillation_target_accuracy is not None:
            self.run_m3(
                target_accuracy=distillation_target_accuracy,
                epochs=distillation_epochs,
                learning_rate=distillation_learning_rate,
                alpha=distillation_alpha,
                temperature=distillation_temperature,
            )

        # M4: terminal quantization.
        self.run_m4(
            precision=precision,
            precision_gate_result=precision_gate_result,
        )

        # Persist transformation history and stage summary.
        history_path = self.manifest.root / "metrics" / "transformation_history.json"
        self._write_json(history_path, self.transformation_history)
        self.manifest.record_artifact("transformation_history", history_path)

        summary_path = self.manifest.root / "metrics" / "stage_summary.json"
        self._write_json(
            summary_path,
            {
                "states": [
                    {
                        "state": record.state.value,
                        "accuracy": record.accuracy,
                        "median_latency_ms": record.median_latency_ms,
                        "p95_latency_ms": record.p95_latency_ms,
                        "parameter_count": record.parameter_count,
                        "onnx_size_bytes": record.onnx_size_bytes,
                        "onnx_sha256": record.onnx_sha256,
                    }
                    for record in self.stage_records
                ],
                "final_state": self.current_state.value,
            },
        )
        self.manifest.record_artifact("stage_summary", summary_path)

        self.manifest.complete()
        self.logger.info(
            f"Compression pipeline complete: final state {self.current_state.value}, "
            f"{len(self.stage_records)} stages measured"
        )
        return self.stage_records

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
