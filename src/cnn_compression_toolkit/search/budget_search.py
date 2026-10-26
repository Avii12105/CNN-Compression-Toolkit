"""Budget-driven candidate search for compression configurations.

Searches pruning ratios and precisions, runs each candidate through the
compression pipeline, and returns the best configuration that satisfies
hard accuracy, latency, and size budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..compression.orchestration import CompressionPipeline, StageRecord
from ..compression.pipeline import PipelineState
from ..compression.quantization.precision_gate import probe_host_precision
from ..data import build_cifar10_bundle
from ..experiments.manifest import RunManifest
from ..models.loader import load_model_checkpoint
from ..runtime import collect_environment


@dataclass(frozen=True)
class Budget:
    """Hard constraints for candidate search.

    tau_lat: maximum median latency in milliseconds.
    delta_acc: maximum allowed accuracy drop from baseline (M0).
    alpha_min: minimum acceptable absolute accuracy.
    tau_size: maximum serialized ONNX size in bytes.
    Any field set to None means unconstrained.
    """

    tau_lat_ms: float | None = None
    delta_acc: float | None = None
    alpha_min: float | None = None
    tau_size_bytes: int | None = None


@dataclass(frozen=True)
class CandidateResult:
    """Complete record for one search candidate."""

    pruning_ratio: float
    precision: str
    distillation_used: bool
    satisfies_budget: bool
    constraint_failures: list[str]
    efficiency_score: float
    final_state: PipelineState
    stage_records: list[StageRecord]
    transformation_history: list[dict[str, Any]]
    failure_details: str | None = None


def efficiency_score(
    baseline_size: int,
    baseline_latency_ms: float,
    candidate_size: int,
    candidate_latency_ms: float,
    candidate_accuracy: float,
) -> float:
    """Compute the harmonic-mean efficiency score.

    C = 1/2 * [max(0, 1 - size/size0) + max(0, 1 - lat/lat0)]
    E = 2 * A * C / (A + C), with zero-denominator protection.
    """

    size_reduction = max(0.0, 1.0 - candidate_size / max(baseline_size, 1))
    latency_reduction = max(0.0, 1.0 - candidate_latency_ms / max(baseline_latency_ms, 1e-9))
    compression = 0.5 * (size_reduction + latency_reduction)
    accuracy = max(candidate_accuracy, 0.0)
    denominator = accuracy + compression
    if denominator <= 0:
        return 0.0
    return 2.0 * accuracy * compression / denominator


def check_budget(
    budget: Budget,
    baseline_accuracy: float,
    candidate_accuracy: float,
    candidate_latency_ms: float,
    candidate_size_bytes: int,
) -> list[str]:
    """Return a list of constraint failure messages (empty if all pass)."""

    failures: list[str] = []

    if budget.tau_lat_ms is not None and candidate_latency_ms > budget.tau_lat_ms:
        failures.append(
            f"latency {candidate_latency_ms:.3f} ms exceeds budget {budget.tau_lat_ms:.3f} ms"
        )
    if budget.delta_acc is not None:
        drop = baseline_accuracy - candidate_accuracy
        if drop > budget.delta_acc:
            failures.append(
                f"accuracy drop {drop:.4f} exceeds budget {budget.delta_acc:.4f}"
            )
    if budget.alpha_min is not None and candidate_accuracy < budget.alpha_min:
        failures.append(
            f"accuracy {candidate_accuracy:.4f} below minimum {budget.alpha_min:.4f}"
        )
    if budget.tau_size_bytes is not None and candidate_size_bytes > budget.tau_size_bytes:
        failures.append(
            f"size {candidate_size_bytes} bytes exceeds budget {budget.tau_size_bytes} bytes"
        )
    return failures


class BudgetSearch:
    """Search pruning ratios and precisions for budget-satisfying configurations."""

    def __init__(
        self,
        config: dict[str, Any],
        artifact_root: str | Path,
        logger,
        pruning_ratios: tuple[float, ...] = (0.3, 0.5, 0.7),
        precisions: tuple[str, ...] = ("int8", "fp16"),
    ) -> None:
        self.config = config
        self.artifact_root = Path(artifact_root)
        self.logger = logger
        self.pruning_ratios = pruning_ratios
        self.precisions = precisions
        self.model_config = config["model"]
        self.dataset_config = config["dataset"]
        self.evaluation_config = config["evaluation"]
        self.runtime_config = config["runtime"]
        self.onnx_config = config.get("onnx", {})
        self.benchmark_config = config["benchmark"]
        self.compression_config = config.get("compression", {})
        self.quantization_config = config.get("quantization", {})

    def _device(self) -> torch.device:
        device_name = self.runtime_config["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_name)

    def _load(self, device: torch.device):
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

    def _run_candidate(
        self,
        pruning_ratio: float,
        precision: str,
        device: torch.device,
        baseline_accuracy: float,
        baseline_latency_ms: float,
        baseline_size_bytes: int,
        budget: Budget,
        experiment_prefix: str,
    ) -> CandidateResult:
        """Run one candidate through the full pipeline and evaluate against budget."""

        loaded, bundle = self._load(device)
        manifest = RunManifest(
            self.artifact_root,
            f"{experiment_prefix}-r{pruning_ratio}-{precision}",
            self.config,
            f"search candidate ratio={pruning_ratio} precision={precision}",
        )
        try:
            manifest.add_environment(collect_environment())

            pipeline = CompressionPipeline(
                loaded=loaded,
                bundle=bundle,
                device=device,
                config=self.config,
                manifest=manifest,
                logger=self.logger,
            )

            # M0 baseline for this candidate (fresh model load).
            pipeline.measure_state(
                PipelineState.M0, loaded.reference, "M0 baseline"
            )

            # Precision gate.
            gate_result = probe_host_precision("auto")
            selected_precision = gate_result.selected_precision
            if precision != "auto":
                selected_precision = precision

            # Distillation target from config.
            distillation_target = self.compression_config.get("distillation_target_accuracy")

            # Run full pipeline.
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
                distillation_epochs=int(
                    self.compression_config.get("distillation_epochs", 0)
                ),
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

        except Exception as exc:  # candidate failed — record but continue searching
            self.logger.warning(
                f"Candidate ratio={pruning_ratio} precision={precision} failed: {exc}"
            )
            return CandidateResult(
                pruning_ratio=pruning_ratio,
                precision=precision,
                distillation_used=False,
                satisfies_budget=False,
                constraint_failures=["candidate execution failed"],
                efficiency_score=0.0,
                final_state=PipelineState.M0,
                stage_records=[],
                transformation_history=[],
                failure_details=str(exc),
            )

        # Evaluate final stage against budget.
        final = stages[-1] if stages else None
        if final is None:
            return CandidateResult(
                pruning_ratio=pruning_ratio,
                precision=precision,
                distillation_used=False,
                satisfies_budget=False,
                constraint_failures=["no stages produced"],
                efficiency_score=0.0,
                final_state=PipelineState.M0,
                stage_records=[],
                transformation_history=[],
                failure_details="pipeline produced no stages",
            )

        failures = check_budget(
            budget,
            baseline_accuracy,
            final.accuracy,
            final.median_latency_ms,
            final.onnx_size_bytes,
        )
        satisfies = len(failures) == 0
        distillation_used = any(
            s.state == PipelineState.M3 for s in stages
        )
        score = efficiency_score(
            baseline_size_bytes,
            baseline_latency_ms,
            final.onnx_size_bytes,
            final.median_latency_ms,
            final.accuracy,
        )

        return CandidateResult(
            pruning_ratio=pruning_ratio,
            precision=precision,
            distillation_used=distillation_used,
            satisfies_budget=satisfies,
            constraint_failures=failures,
            efficiency_score=score,
            final_state=final.state,
            stage_records=stages,
            transformation_history=pipeline.transformation_history,
            failure_details=(
                "; ".join(failures) if failures else None
            ),
        )

    def search(
        self,
        budget: Budget,
    ) -> dict[str, Any]:
        """Run the budget search and return the best result.

        Returns a dict with:
          - best: CandidateResult | None (best budget-satisfying candidate)
          - best_by_score: CandidateResult | None (highest efficiency score overall)
          - all_candidates: list[CandidateResult]
          - baseline: dict with baseline metrics
          - search_summary: counts and failure info
        """

        device = self._device()
        loaded, bundle = self._load(device)

        # Measure the M0 baseline once for reference.
        baseline_manifest = RunManifest(
            self.artifact_root,
            "search-baseline",
            self.config,
            "search baseline M0",
        )
        baseline_manifest.add_environment(collect_environment())
        pipeline = CompressionPipeline(
            loaded=loaded,
            bundle=bundle,
            device=device,
            config=self.config,
            manifest=baseline_manifest,
            logger=self.logger,
        )
        baseline_record = pipeline.measure_state(
            PipelineState.M0, loaded.reference, "M0 baseline"
        )
        baseline_manifest.complete()

        baseline_accuracy = baseline_record.accuracy
        baseline_latency_ms = baseline_record.median_latency_ms
        baseline_size_bytes = baseline_record.onnx_size_bytes

        self.logger.info(
            f"Baseline M0: accuracy={baseline_accuracy:.4f}, "
            f"latency={baseline_latency_ms:.3f} ms, size={baseline_size_bytes} bytes"
        )

        candidates: list[CandidateResult] = []
        for ratio in self.pruning_ratios:
            for precision in self.precisions:
                candidate = self._run_candidate(
                    pruning_ratio=ratio,
                    precision=precision,
                    device=device,
                    baseline_accuracy=baseline_accuracy,
                    baseline_latency_ms=baseline_latency_ms,
                    baseline_size_bytes=baseline_size_bytes,
                    budget=budget,
                    experiment_prefix="search",
                )
                candidates.append(candidate)
                self.logger.info(
                    f"Candidate ratio={ratio} precision={precision}: "
                    f"satisfies={candidate.satisfies_budget}, "
                    f"score={candidate.efficiency_score:.4f}, "
                    f"failures={candidate.constraint_failures}"
                )

        # Select best budget-satisfying candidate (prefer hard constraints over score).
        satisfying = [c for c in candidates if c.satisfies_budget]
        if satisfying:
            best = max(satisfying, key=lambda c: c.efficiency_score)
        else:
            best = None

        # Best overall by score (even if it violates constraints — for reporting).
        best_by_score = max(candidates, key=lambda c: c.efficiency_score) if candidates else None

        summary = {
            "baseline_accuracy": baseline_accuracy,
            "baseline_latency_ms": baseline_latency_ms,
            "baseline_size_bytes": baseline_size_bytes,
            "budget": {
                "tau_lat_ms": budget.tau_lat_ms,
                "delta_acc": budget.delta_acc,
                "alpha_min": budget.alpha_min,
                "tau_size_bytes": budget.tau_size_bytes,
            },
            "candidates_tested": len(candidates),
            "satisfying_count": len(satisfying),
            "best_satisfying": (
                {
                    "pruning_ratio": best.pruning_ratio,
                    "precision": best.precision,
                    "efficiency_score": best.efficiency_score,
                }
                if best
                else None
            ),
            "best_overall_score": (
                {
                    "pruning_ratio": best_by_score.pruning_ratio,
                    "precision": best_by_score.precision,
                    "efficiency_score": best_by_score.efficiency_score,
                    "satisfies_budget": best_by_score.satisfies_budget,
                }
                if best_by_score
                else None
            ),
        }

        return {
            "best": best,
            "best_by_score": best_by_score,
            "all_candidates": candidates,
            "baseline": {
                "accuracy": baseline_accuracy,
                "latency_ms": baseline_latency_ms,
                "size_bytes": baseline_size_bytes,
            },
            "search_summary": summary,
        }
