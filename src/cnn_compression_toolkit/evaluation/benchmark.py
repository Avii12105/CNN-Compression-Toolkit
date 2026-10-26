"""ONNX Runtime CPU benchmark harness."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def create_cpu_session(
    model_path: str | Path,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
) -> ort.InferenceSession:
    """Create an ONNX Runtime CPU session with deterministic thread settings."""

    if intra_op_threads <= 0 or inter_op_threads <= 0:
        raise ValueError("thread counts must be positive")
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise RuntimeError("CPUExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    return ort.InferenceSession(
        Path(model_path).as_posix(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_session(
    session: Any,
    input_name: str,
    sample_input: np.ndarray,
    warmup_runs: int = 50,
    timed_runs: int = 200,
) -> dict[str, Any]:
    """Benchmark only session.run and return latency statistics in milliseconds."""

    if warmup_runs < 0 or timed_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and timed_runs must be positive")
    inputs = {input_name: sample_input}
    for _ in range(warmup_runs):
        session.run(None, inputs)
    latencies_ms: list[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter_ns()
        session.run(None, inputs)
        latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "provider": "CPUExecutionProvider",
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "input_shape": list(sample_input.shape),
        "unit": "milliseconds",
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "stddev": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def benchmark_onnx(
    model_path: str | Path,
    input_name: str,
    sample_input: np.ndarray,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
    warmup_runs: int = 50,
    timed_runs: int = 200,
) -> dict[str, Any]:
    """Create a CPU session and benchmark the exported model."""

    session = create_cpu_session(model_path, intra_op_threads, inter_op_threads)
    result = benchmark_session(session, input_name, sample_input, warmup_runs, timed_runs)
    result.update(
        {
            "intra_op_threads": intra_op_threads,
            "inter_op_threads": inter_op_threads,
        }
    )
    return result