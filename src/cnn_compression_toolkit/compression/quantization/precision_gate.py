"""Host precision gating for terminal quantization (INT8 vs FP16)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort


@dataclass(frozen=True)
class PrecisionGateResult:
    """Result of probing the host for a supported quantization path."""

    selected_precision: str
    reason: str
    cpu_flags: list[str]
    onnxruntime_providers: list[str]
    int8_supported: bool
    fp16_efficient: bool


def probe_host_precision(preferred: str = "auto") -> PrecisionGateResult:
    """Probe CPU ISA and ONNX Runtime execution paths to select a quantization precision.

    INT8 is selected only when the fused integer inference path is supported.
    Otherwise FP16 is selected (with a note that the CPU provider may cast back
    to FP32, in which case size reduction is real but latency may not improve).
    """

    providers = ort.get_available_providers()
    cpu_flags: list[str] = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        flags = next(
            (
                line.split(":", 1)[1].strip().split()
                for line in cpuinfo.splitlines()
                if line.startswith("flags")
            ),
            [],
        )
        cpu_flags = flags
    except OSError:
        cpu_flags = []

    has_avx2 = "avx2" in cpu_flags
    has_avx512f = "avx512f" in cpu_flags

    # ONNX Runtime's CPUExecutionProvider uses MKL-DNN (oneDNN) for INT8.
    # INT8 fused-integer execution generally requires AVX2 at minimum and
    # QDQ operator support in the provider.
    int8_supported = bool(has_avx2 and "CPUExecutionProvider" in providers)

    # FP16 on CPU: ORT can run FP16 models but often casts weights back to FP32.
    # We conservatively mark FP16 as "efficient" only when AVX512F is present;
    # otherwise report it as potentially cast-back.
    fp16_efficient = bool(has_avx512f and "CPUExecutionProvider" in providers)

    if preferred == "int8":
        if int8_supported:
            return PrecisionGateResult(
                selected_precision="int8",
                reason="INT8 requested and host supports fused integer path (AVX2 + CPU EP)",
                cpu_flags=cpu_flags,
                onnxruntime_providers=providers,
                int8_supported=True,
                fp16_efficient=fp16_efficient,
            )
        return PrecisionGateResult(
            selected_precision="fp16",
            reason=(
                "INT8 requested but host lacks fused integer support; "
                "falling back to FP16 (CPU EP may cast FP16 weights back to FP32)"
            ),
            cpu_flags=cpu_flags,
            onnxruntime_providers=providers,
            int8_supported=False,
            fp16_efficient=fp16_efficient,
        )

    if preferred == "fp16":
        return PrecisionGateResult(
            selected_precision="fp16",
            reason="FP16 requested explicitly",
            cpu_flags=cpu_flags,
            onnxruntime_providers=providers,
            int8_supported=int8_supported,
            fp16_efficient=fp16_efficient,
        )

    # preferred == "auto"
    if int8_supported:
        return PrecisionGateResult(
            selected_precision="int8",
            reason="Auto-selected INT8: host supports fused integer path (AVX2 + CPU EP)",
            cpu_flags=cpu_flags,
            onnxruntime_providers=providers,
            int8_supported=True,
            fp16_efficient=fp16_efficient,
        )
    return PrecisionGateResult(
        selected_precision="fp16",
        reason=(
            "Auto-selected FP16: host lacks AVX2 fused-integer support; "
            "CPU EP may cast FP16 weights back to FP32"
        ),
        cpu_flags=cpu_flags,
        onnxruntime_providers=providers,
        int8_supported=False,
        fp16_efficient=fp16_efficient,
    )


def precision_gate_result_to_dict(result: PrecisionGateResult) -> dict[str, Any]:
    """Serialize a precision gate result for experiment artifacts."""

    return {
        "selected_precision": result.selected_precision,
        "reason": result.reason,
        "cpu_flags": result.cpu_flags,
        "onnxruntime_providers": result.onnxruntime_providers,
        "int8_supported": result.int8_supported,
        "fp16_efficient": result.fp16_efficient,
    }
