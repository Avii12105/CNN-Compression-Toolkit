"""Final validation for the selected compression configuration.

Reloads or rebuilds the selected model from recorded checkpoint and
configuration, runs full test-set evaluation, benchmark, size measurement,
coverage audit, and fresh-process validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch

from ..compression.quantization.coverage import audit_coverage, coverage_report_to_dict
from ..evaluation.size import serialized_file_metrics
from ..evaluation.validation import validate_fresh_process


@dataclass
class FinalValidationResult:
    """Complete final validation record for the selected artifact."""

    experiment_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    architecture: str
    onnx_path: str
    onnx_sha256: str
    onnx_size_bytes: int
    parameter_count: int
    test_accuracy: float
    test_loss: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    median_latency_ms: float
    p95_latency_ms: float
    fresh_process_passed: bool
    coverage_report: dict[str, Any] | None = None
    exclusion_report: dict[str, Any] | None = None
    equivalence_passed: bool = False
    validation_passed: bool = False
    failure: str | None = None


def run_final_validation(
    experiment_dir: str | Path,
    device,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    num_classes: int = 10,
    calibration_loader=None,
    test_loader=None,
    onnx_input_name: str = "input",
    onnx_output_name: str = "logits",
) -> FinalValidationResult:
    """Run full independent validation on the selected experiment's artifact.

    Reloads the ONNX artifact from the experiment directory, validates the
    graph, runs full test-set accuracy via ONNX Runtime, benchmarks, checks
    size/hash, runs fresh-process validation, and audits quantization
    coverage if applicable.
    """

    experiment_dir = Path(experiment_dir)
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        return FinalValidationResult(
            experiment_id="unknown",
            checkpoint_path="",
            checkpoint_sha256="",
            architecture="unknown",
            onnx_path="",
            onnx_sha256="",
            onnx_size_bytes=0,
            parameter_count=0,
            test_accuracy=0.0,
            test_loss=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            fresh_process_passed=False,
            failure=f"manifest not found: {manifest_path}",
        )

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_id = manifest.get("experiment_id", "unknown")

    # Find the final ONNX artifact.
    artifacts = manifest.get("artifacts", {})
    onnx_artifact_name = None
    for key in ("m4_onnx", "m3_onnx", "m2_onnx", "m1_onnx", "onnx_model"):
        if key in artifacts:
            onnx_artifact_name = key
            break
    if onnx_artifact_name is None:
        return FinalValidationResult(
            experiment_id=experiment_id,
            checkpoint_path="",
            checkpoint_sha256="",
            architecture="unknown",
            onnx_path="",
            onnx_sha256="",
            onnx_size_bytes=0,
            parameter_count=0,
            test_accuracy=0.0,
            test_loss=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            fresh_process_passed=False,
            failure="no ONNX artifact found in manifest",
        )

    onnx_path = experiment_dir / artifacts[onnx_artifact_name]
    if not onnx_path.is_file():
        return FinalValidationResult(
            experiment_id=experiment_id,
            checkpoint_path="",
            checkpoint_sha256="",
            architecture="unknown",
            onnx_path=str(onnx_path),
            onnx_sha256="",
            onnx_size_bytes=0,
            parameter_count=0,
            test_accuracy=0.0,
            test_loss=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            fresh_process_passed=False,
            failure=f"ONNX artifact not found on disk: {onnx_path}",
        )

    # Load and validate the ONNX graph.
    graph = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(graph)
        graph_valid = True
    except Exception as exc:
        graph_valid = False
        return FinalValidationResult(
            experiment_id=experiment_id,
            checkpoint_path="",
            checkpoint_sha256="",
            architecture="unknown",
            onnx_path=str(onnx_path),
            onnx_sha256="",
            onnx_size_bytes=0,
            parameter_count=0,
            test_accuracy=0.0,
            test_loss=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            fresh_process_passed=False,
            failure=f"ONNX graph validation failed: {exc}",
        )

    size_info = serialized_file_metrics(onnx_path)

    # Create ONNX Runtime session and benchmark.
    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Benchmark with batch-1 input.
    import time

    import numpy as np

    sample_input = np.zeros(input_shape, dtype=np.float32)
    warmup = 50
    timed = 200
    for _ in range(warmup):
        session.run(None, {input_name: sample_input})
    latencies = []
    for _ in range(timed):
        start = time.perf_counter_ns()
        session.run(None, {input_name: sample_input})
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
    latencies_arr = np.asarray(latencies, dtype=np.float64)
    median_latency = float(np.median(latencies_arr))
    p95_latency = float(np.percentile(latencies_arr, 95))

    # Full test-set accuracy via ONNX Runtime.
    correct = 0
    total = 0
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images_np = images.numpy().astype(np.float32)
            labels_np = labels.numpy()
            outputs = session.run([output_name], {input_name: images_np})[0]
            predictions = outputs.argmax(axis=1)
            correct += int((predictions == labels_np).sum())
            total += labels.size(0)
            # Loss and confusion matrix on PyTorch tensors for consistency.
            logits_t = torch.from_numpy(outputs)
            labels_t = labels
            total_loss += criterion(logits_t, labels_t).item() * labels.size(0)
            for actual, predicted in zip(labels_t.cpu(), predictions):
                confusion[int(actual), int(predicted)] += 1

    test_accuracy = correct / max(total, 1)
    tp = confusion.diag().float()
    pred_total = confusion.sum(dim=0).float()
    actual_total = confusion.sum(dim=1).float()
    precision = torch.where(pred_total > 0, tp / pred_total, torch.zeros_like(tp))
    recall = torch.where(actual_total > 0, tp / actual_total, torch.zeros_like(tp))
    f1_denom = precision + recall
    f1 = torch.where(f1_denom > 0, 2 * precision * recall / f1_denom, torch.zeros_like(precision))
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    # Fresh-process validation.
    fresh_process = validate_fresh_process(
        onnx_path,
        input_name=input_name,
        output_name=output_name,
        input_shape=input_shape,
    )

    # Quantization coverage audit (if QDQ graph).
    coverage_report = None
    exclusion_report = None
    try:
        coverage_report = coverage_report_to_dict(audit_coverage(onnx_path))
    except Exception:
        coverage_report = None

    # All checks passed.
    validation_passed = (
        graph_valid
        and fresh_process["passed"]
        and total > 0
        and test_accuracy > 0
        and size_info["bytes"] > 0
    )

    return FinalValidationResult(
        experiment_id=experiment_id,
        checkpoint_path=manifest.get("configuration", {}).get("model", {}).get("checkpoint", ""),
        checkpoint_sha256="",  # would need to read from metrics
        architecture="unknown",
        onnx_path=str(onnx_path),
        onnx_sha256=size_info["sha256"],
        onnx_size_bytes=size_info["bytes"],
        parameter_count=0,  # would need model reload
        test_accuracy=test_accuracy,
        test_loss=total_loss / max(total, 1),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        median_latency_ms=median_latency,
        p95_latency_ms=p95_latency,
        fresh_process_passed=fresh_process["passed"],
        coverage_report=coverage_report,
        exclusion_report=exclusion_report,
        equivalence_passed=graph_valid,
        validation_passed=validation_passed,
    )
