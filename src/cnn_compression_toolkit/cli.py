"""Command-line interface for the Phase 0 project foundation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compression.pipeline import PipelineState
from .compression.quantization.precision_gate import (
    precision_gate_result_to_dict,
    probe_host_precision,
)
from .config import ConfigurationError, load_config
from .experiments.ids import create_experiment_id
from .experiments.manifest import RunManifest
from .logging import configure_logging
from .runtime import collect_environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cct", description="CNN Compression Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="show package and runtime information")
    validate = subparsers.add_parser("validate-config", help="validate an experiment YAML file")
    validate.add_argument("config", type=Path)
    run = subparsers.add_parser("run", help="create a configuration-only experiment run")
    run.add_argument("--config", required=True, type=Path)
    baseline = subparsers.add_parser("baseline", help="run the Phase 1 PyTorch M0 baseline")
    baseline.add_argument("--config", required=True, type=Path)
    measure = subparsers.add_parser(
        "measure", help="export and measure the Phase 2 FP32 ONNX baseline"
    )
    measure.add_argument("--config", required=True, type=Path)
    compress = subparsers.add_parser(
        "compress", help="run the full M0 -> M4 compression pipeline"
    )
    compress.add_argument("--config", required=True, type=Path)
    compress.add_argument("--pruning-ratio", type=float, default=None)
    compress.add_argument("--alignment", type=int, default=None)
    compress.add_argument("--precision", type=str, default=None)
    compress.add_argument(
        "--distillation-target-accuracy", type=float, default=None
    )
    compress.add_argument("--distillation-epochs", type=int, default=0)
    compress.add_argument(
        "--protected-count", type=int, default=0, help="number of sensitivity-protected layers"
    )
    search = subparsers.add_parser(
        "search", help="run budget-driven candidate search"
    )
    search.add_argument("--config", required=True, type=Path)
    search.add_argument("--budget-latency-ms", type=float, default=None)
    search.add_argument("--budget-max-accuracy-drop", type=float, default=None)
    search.add_argument("--budget-min-accuracy", type=float, default=None)
    search.add_argument("--budget-max-size-bytes", type=int, default=None)
    matrix = subparsers.add_parser(
        "matrix", help="run the structured experiment matrix"
    )
    matrix.add_argument("--config", required=True, type=Path)
    pareto = subparsers.add_parser(
        "pareto", help="compute Pareto frontier from experiment results"
    )
    pareto.add_argument("--config", required=True, type=Path)
    pareto.add_argument("--experiment-dir", type=Path, default=None)
    validate_final = subparsers.add_parser(
        "validate-final", help="run final validation on a selected experiment"
    )
    validate_final.add_argument("--config", required=True, type=Path)
    validate_final.add_argument("--experiment-dir", type=Path, required=True)
    report = subparsers.add_parser(
        "report", help="generate a final report from experiment results"
    )
    report.add_argument("--config", required=True, type=Path)
    report.add_argument("--experiment-dir", type=Path, required=True)
    tui_cmd = subparsers.add_parser("tui", help="launch the Textual terminal UI")
    tui_cmd.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "info":
            environment = collect_environment()
            payload = {"package_version": __version__, **environment}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "tui":
            from .tui.app import run

            run(config_path=args.config)
            return 0
        config = load_config(args.config)
        if args.command == "validate-config":
            print(f"Configuration is valid: {config.source}")
            return 0

        if args.command == "baseline":
            return _run_baseline(config.data)
        if args.command == "measure":
            return _run_measurement(config.data)
        if args.command == "compress":
            return _run_compress(config.data, args)
        if args.command == "search":
            return _run_search(config.data, args)
        if args.command == "matrix":
            return _run_matrix(config.data, args)
        if args.command == "pareto":
            return _run_pareto(config.data, args)
        if args.command == "validate-final":
            return _run_validate_final(config.data, args)
        if args.command == "report":
            return _run_report(config.data, args)

        experiment_id = create_experiment_id(config.data)
        manifest = RunManifest(config.artifact_root, experiment_id, config.data, " ".join(sys.argv))
        manifest.add_environment(collect_environment())
        logger = configure_logging(
            manifest.root / "logs" / "run.jsonl", config.data["logging"]["level"], experiment_id
        )
        logger.info("Phase 0 configuration validated; model and data execution is deferred")
        manifest.complete()
        print(f"Created configuration-only run: {manifest.root}")
        return 0
    except (ConfigurationError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_baseline(configuration: dict[str, object]) -> int:
    import torch

    from .data import build_cifar10_bundle
    from .evaluation.metrics import evaluate_classifier, parameter_count
    from .models.loader import load_model_checkpoint

    model_config = configuration["model"]
    dataset_config = configuration["dataset"]
    evaluation_config = configuration["evaluation"]
    runtime_config = configuration["runtime"]
    device_name = runtime_config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        if runtime_config.get("allow_cpu_fallback", False):
            device_name = "cpu"
        else:
            raise RuntimeError("CUDA was requested but is unavailable")

    experiment_id = create_experiment_id(configuration)
    manifest = RunManifest(
        Path(configuration["output"]["artifact_root"]), experiment_id, configuration, "baseline"
    )
    try:
        manifest.add_environment(collect_environment())
        bundle = build_cifar10_bundle(
            root=dataset_config["root"],
            seed=dataset_config.get("calibration_seed", configuration["project"]["seed"]),
            calibration_size=dataset_config["calibration_size"],
            validation_size=dataset_config.get("validation_size", 5000),
            batch_size=evaluation_config["batch_size"],
            num_workers=evaluation_config["num_workers"],
            download=dataset_config.get("download", False),
        )
        loaded = load_model_checkpoint(
            architecture=model_config["architecture"],
            checkpoint_path=model_config["checkpoint"],
            input_shape=tuple(model_config.get("input_shape", [1, 3, 32, 32])),
            num_classes=model_config.get("num_classes", 10),
            device=device_name,
        )
        metrics = evaluate_classifier(
            loaded.reference, bundle.test_loader, loaded.num_classes, torch.device(device_name)
        )
        metrics.update(
            {
                "architecture": loaded.architecture,
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "checkpoint_format": "state_dict",
                "input_shape": list(loaded.input_shape),
                "num_classes": loaded.num_classes,
                "device": device_name,
                "parameter_count": parameter_count(loaded.reference),
                "normalization_mean": list(bundle.mean),
                "normalization_std": list(bundle.std),
            }
        )
        baseline_path = manifest.root / "metrics" / "baseline.json"
        _write_json(baseline_path, metrics)
        manifest.record_artifact("baseline_metrics", baseline_path)
        split_path = manifest.root / "metrics" / "split_indices.json"
        _write_json(
            split_path,
            {
                "seed": dataset_config.get("calibration_seed", configuration["project"]["seed"]),
                "train_indices": bundle.train_indices,
                "validation_indices": bundle.validation_indices,
                "calibration_indices": bundle.calibration_indices,
                "test_size": metrics["sample_count"],
            },
        )
        manifest.record_artifact("split_indices", split_path)
        checkpoint_metadata_path = manifest.root / "metrics" / "checkpoint.json"
        _write_json(
            checkpoint_metadata_path,
            {
                "path": str(loaded.checkpoint_path),
                "format": "state_dict",
                "sha256": loaded.checkpoint_sha256,
            },
        )
        manifest.record_artifact("checkpoint_metadata", checkpoint_metadata_path)
        manifest.complete()
        print(f"Created Phase 1 baseline: {manifest.root}")
        return 0
    except Exception as exc:
        manifest.fail(exc)
        raise


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_compress(
    configuration: dict[str, object], args: argparse.Namespace
) -> int:
    """Run the full M0 -> M4 compression pipeline (Phase A)."""

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from .compression.orchestration import CompressionPipeline
    from .data import build_cifar10_bundle
    from .experiments.ids import create_experiment_id
    from .experiments.manifest import RunManifest
    from .logging import configure_logging
    from .models.loader import load_model_checkpoint
    from .runtime import collect_environment

    console = Console()

    model_config = configuration["model"]
    dataset_config = configuration["dataset"]
    evaluation_config = configuration["evaluation"]
    runtime_config = configuration["runtime"]
    compression_config = configuration.get("compression", {})
    quantization_config = configuration.get("quantization", {})

    device_name = runtime_config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        if runtime_config.get("allow_cpu_fallback", False):
            device_name = "cpu"
        else:
            raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    pruning_ratio = (
        args.pruning_ratio
        if args.pruning_ratio is not None
        else compression_config.get("default_pruning_ratio", 0.0)
    )
    requested_alignment = (
        args.alignment
        if args.alignment is not None
        else compression_config.get("simd_alignment", 8)
    )
    precision = (
        args.precision
        if args.precision is not None
        else quantization_config.get("precision", "auto")
    )

    console.print()
    console.print(
        Panel.fit(
            "[bold blue]CNN Compression Toolkit — Compression Pipeline[/bold blue]",
            border_style="blue",
        )
    )
    console.print(f"[bold]Architecture:[/bold] {model_config['architecture']}")
    console.print(f"[bold]Device:[/bold] {device_name}")
    console.print(f"[bold]Pruning ratio:[/bold] {pruning_ratio}")
    console.print(f"[bold]SIMD alignment:[/bold] {requested_alignment}")
    console.print(f"[bold]Precision:[/bold] {precision}")
    console.print(f"[bold]Protected layers:[/bold] {args.protected_count}")
    console.print(f"[bold]Distillation epochs:[/bold] {args.distillation_epochs}")
    console.print()

    experiment_id = create_experiment_id(configuration)
    manifest = RunManifest(
        Path(configuration["output"]["artifact_root"]),
        experiment_id,
        configuration,
        "compress",
    )
    try:
        manifest.add_environment(collect_environment())
        logger = configure_logging(
            manifest.root / "logs" / "run.jsonl",
            configuration["logging"]["level"],
            experiment_id,
        )

        bundle = build_cifar10_bundle(
            root=dataset_config["root"],
            seed=dataset_config.get("calibration_seed", configuration["project"]["seed"]),
            calibration_size=dataset_config["calibration_size"],
            validation_size=dataset_config.get("validation_size", 5000),
            batch_size=evaluation_config["batch_size"],
            num_workers=evaluation_config["num_workers"],
            download=dataset_config.get("download", False),
        )
        loaded = load_model_checkpoint(
            architecture=model_config["architecture"],
            checkpoint_path=model_config["checkpoint"],
            input_shape=tuple(model_config.get("input_shape", [1, 3, 32, 32])),
            num_classes=model_config.get("num_classes", 10),
            device=device_name,
        )

        # M0 baseline measurement (reference model).
        pipeline = CompressionPipeline(
            loaded=loaded,
            bundle=bundle,
            device=device,
            config=configuration,
            manifest=manifest,
            logger=logger,
        )
        pipeline.measure_state(
            PipelineState.M0,
            loaded.reference,
            "M0 baseline",
        )

        # Precision gate for M4.
        preferred = precision if precision != "auto" else "auto"
        gate_result = probe_host_precision(preferred)
        logger.info(f"Precision gate: {gate_result.selected_precision} - {gate_result.reason}")

        # Run the full pipeline.
        pipeline.run_full_pipeline(
            pruning_ratio=pruning_ratio,
            requested_alignment=requested_alignment,
            precision=gate_result.selected_precision,
            sensitivity_bit_width=8,
            protected_count=args.protected_count,
            protection_factor=float(compression_config.get("protection_gamma", 0.5)),
            distillation_target_accuracy=(
                args.distillation_target_accuracy
                if args.distillation_target_accuracy is not None
                else compression_config.get("distillation_target_accuracy")
            ),
            distillation_epochs=args.distillation_epochs,
            distillation_learning_rate=float(
                compression_config.get("distillation_learning_rate", 1.0e-3)
            ),
            distillation_alpha=float(compression_config.get("distillation_alpha", 0.3)),
            distillation_temperature=float(compression_config.get("distillation_temperature", 4.0)),
            precision_gate_result=precision_gate_result_to_dict(gate_result),
        )

        manifest.complete()

        # Print rich summary to console.
        console.print()
        console.print(
            Panel.fit(
                "[bold green]Compression pipeline complete[/bold green]",
                title="CNN Compression Toolkit",
                border_style="green",
            )
        )
        console.print(f"[bold]Experiment ID:[/bold] {experiment_id}")
        console.print(f"[bold]Artifact root:[/bold] {manifest.root}")
        console.print()

        # Summary table of all stages.
        table = Table(title="Stage Summary (M0 → M4)", border_style="cyan")
        table.add_column("State", style="bold magenta")
        table.add_column("Accuracy", justify="right")
        table.add_column("Median Latency", justify="right")
        table.add_column("P95 Latency", justify="right")
        table.add_column("Size (bytes)", justify="right")
        table.add_column("Parameters", justify="right")

        state_labels = {
            "M0": "M0 — Original",
            "M1": "M1 — Pruned",
            "M2": "M2 — BN-recovered",
            "M3": "M3 — Distilled",
            "M4": "M4 — Quantized",
        }
        for record in pipeline.stage_records:
            label = state_labels.get(record.state.value, record.state.value)
            table.add_row(
                label,
                f"{record.accuracy:.4f}",
                f"{record.median_latency_ms:.3f} ms",
                f"{record.p95_latency_ms:.3f} ms",
                f"{record.onnx_size_bytes:,}",
                f"{record.parameter_count:,}",
            )
        console.print(table)
        console.print()

        # Precision gate info.
        gate_precision = gate_result.selected_precision
        gate_style = "green" if gate_result.int8_supported else "yellow"
        console.print(
            f"[bold]Precision gate:[/bold] [{gate_style}]{gate_precision}[/{gate_style}] — "
            f"{gate_result.reason}"
        )
        console.print()
        print(f"Created compression pipeline: {manifest.root}")
        return 0
    except Exception as exc:
        manifest.fail(exc)
        raise


def _run_search(configuration: dict[str, object], args: argparse.Namespace) -> int:
    """Run budget-driven candidate search (Phase E)."""

    import torch

    from .search.budget_search import Budget, BudgetSearch

    runtime_config = configuration["runtime"]

    device_name = runtime_config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    budget = Budget(
        tau_lat_ms=args.budget_latency_ms,
        delta_acc=args.budget_max_accuracy_drop,
        alpha_min=args.budget_min_accuracy,
        tau_size_bytes=args.budget_max_size_bytes,
    )

    experiment_id = create_experiment_id(configuration)
    manifest = RunManifest(
        Path(configuration["output"]["artifact_root"]),
        experiment_id,
        configuration,
        "search",
    )
    try:
        manifest.add_environment(collect_environment())

        from .logging import configure_logging

        logger = configure_logging(
            manifest.root / "logs" / "run.jsonl",
            configuration["logging"]["level"],
            experiment_id,
        )

        searcher = BudgetSearch(
            config=configuration,
            artifact_root=configuration["output"]["artifact_root"],
            logger=logger,
        )
        result = searcher.search(budget=budget)

        manifest.complete()
        summary = result["search_summary"]
        print(
            f"Search complete: {summary['candidates_tested']} candidates tested, "
            f"{summary['satisfying_count']} satisfy budget"
        )
        best = result["best"]
        if best:
            print(
                f"Best satisfying: ratio={best.pruning_ratio}, "
                f"precision={best.precision}, score={best.efficiency_score:.4f}"
            )
        else:
            print("No candidate satisfies all budget constraints.")
        best_by_score = result["best_by_score"]
        if best_by_score:
            print(
                f"Best overall score: ratio={best_by_score.pruning_ratio}, "
                f"precision={best_by_score.precision}, "
                f"score={best_by_score.efficiency_score:.4f}, "
                f"satisfies={best_by_score.satisfies_budget}"
            )
        return 0
    except Exception as exc:
        manifest.fail(exc)
        raise


def _run_matrix(configuration: dict[str, object], args: argparse.Namespace) -> int:
    """Run the structured experiment matrix (Phase F)."""

    import torch

    from .experiments.matrix import ExperimentMatrix
    from .logging import configure_logging

    device_name = configuration["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_id = create_experiment_id(configuration)
    manifest = RunManifest(
        Path(configuration["output"]["artifact_root"]),
        experiment_id,
        configuration,
        "matrix",
    )
    try:
        manifest.add_environment(collect_environment())
        logger = configure_logging(
            manifest.root / "logs" / "run.jsonl",
            configuration["logging"]["level"],
            experiment_id,
        )

        matrix_runner = ExperimentMatrix(
            config=configuration,
            artifact_root=configuration["output"]["artifact_root"],
            logger=logger,
        )
        results = matrix_runner.run_all(device=device_name)

        manifest.complete()

        succeeded = sum(1 for r in results if r.succeeded)
        failed = len(results) - succeeded
        print(
            f"Matrix complete: {len(results)} cells, {succeeded} succeeded, {failed} failed"
        )
        for result in results:
            status = "OK" if result.succeeded else f"FAILED: {result.failure}"
            print(
                f"  {result.config['pruning_ratio']} {result.config['precision']} "
                f"{'distill' if result.config.get('distillation') else 'nodistill'}: {status}"
            )
        return 0
    except Exception as exc:
        manifest.fail(exc)
        raise


def _run_pareto(configuration: dict[str, object], args: argparse.Namespace) -> int:
    """Compute Pareto frontier from experiment results (Phase F)."""

    from pathlib import Path

    from .reports.pareto import compute_pareto_frontier

    experiment_dir = (
        args.experiment_dir
        or Path(configuration["output"]["artifact_root"]) / "runs"
    )

    # Collect results from experiment directories.
    results = []
    if experiment_dir.is_dir():
        for run_dir in sorted(experiment_dir.iterdir()):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                continue
            # Look for stage summary.
            summary_path = run_dir / "metrics" / "stage_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                for stage in summary.get("states", []):
                    results.append(
                        {
                            "succeeded": True,
                            "config": manifest.get("configuration", {}).get(
                                "matrix_config", {}
                            ),
                            "experiment_id": manifest.get("experiment_id"),
                            "stages": [stage],
                        }
                    )

    pareto = compute_pareto_frontier(results)
    print(
        f"Pareto analysis: {pareto['total_points']} points, "
        f"{pareto['frontier_size']} on frontier"
    )
    for point in pareto["frontier"]:
        print(
            f"  {point['label']}: acc={point['accuracy']:.4f}, "
            f"lat={point['latency_ms']:.3f} ms, size={point['size_bytes']} bytes"
        )
    return 0


def _run_validate_final(configuration: dict[str, object], args: argparse.Namespace) -> int:
    """Run final validation on a selected experiment (Phase G)."""

    import torch

    from .data import build_cifar10_bundle
    from .reports.validation import run_final_validation

    device_name = configuration["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    bundle = build_cifar10_bundle(
        root=configuration["dataset"]["root"],
        seed=configuration["dataset"].get(
            "calibration_seed", configuration["project"]["seed"]
        ),
        calibration_size=configuration["dataset"]["calibration_size"],
        validation_size=configuration["dataset"].get("validation_size", 5000),
        batch_size=configuration["evaluation"]["batch_size"],
        num_workers=configuration["evaluation"]["num_workers"],
        download=configuration["dataset"].get("download", False),
    )

    result = run_final_validation(
        experiment_dir=args.experiment_dir,
        device=device,
        test_loader=bundle.test_loader,
    )

    print(f"Final validation for experiment {result.experiment_id}:")
    print(f"  Validation passed: {result.validation_passed}")
    print(f"  Test accuracy: {result.test_accuracy:.4f}")
    print(f"  Median latency: {result.median_latency_ms:.3f} ms")
    print(f"  ONNX size: {result.onnx_size_bytes} bytes")
    print(f"  Fresh process: {'PASS' if result.fresh_process_passed else 'FAIL'}")
    if result.coverage_report:
        print(f"  Quantization coverage: {result.coverage_report['coverage']:.2%}")
    return 0 if result.validation_passed else 1


def _run_report(configuration: dict[str, object], args: argparse.Namespace) -> int:
    """Generate a final report from experiment results (Phase G)."""

    import json

    from .reports.generator import generate_report

    experiment_dir = args.experiment_dir
    output_dir = experiment_dir / "reports"

    # Load manifest and stage summary.
    manifest_path = experiment_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    summary_path = experiment_dir / "metrics" / "stage_summary.json"
    stage_summary = None
    if summary_path.is_file():
        stage_summary = json.loads(summary_path.read_text(encoding="utf-8")).get("states", [])

    # Load validation results if available.
    validation_path = experiment_dir / "reports" / "validation.json"
    validation_result = None
    if validation_path.is_file():
        validation_result = json.loads(validation_path.read_text(encoding="utf-8"))

    # Load Pareto data if available.
    pareto_path = experiment_dir / "reports" / "pareto.json"
    pareto_data = None
    if pareto_path.is_file():
        pareto_data = json.loads(pareto_path.read_text(encoding="utf-8"))

    paths = generate_report(
        output_dir=output_dir,
        stage_summary=stage_summary,
        baseline_metrics=None,
        pareto_data=pareto_data,
        validation_result=validation_result,
        configuration=manifest.get("configuration"),
        environment=None,
    )
    print(f"Report generated: {paths['markdown']}")
    print(f"JSON report: {paths['json']}")
    return 0


def _run_measurement(configuration: dict[str, object]) -> int:
    import numpy as np
    import torch

    from .data import build_cifar10_bundle
    from .evaluation.benchmark import benchmark_onnx
    from .evaluation.onnx_export import compare_pytorch_onnx, export_to_onnx
    from .evaluation.size import model_size_metrics
    from .evaluation.validation import validate_fresh_process
    from .models.loader import load_model_checkpoint

    model_config = configuration["model"]
    dataset_config = configuration["dataset"]
    evaluation_config = configuration["evaluation"]
    runtime_config = configuration["runtime"]
    onnx_config = configuration.get("onnx", {})
    benchmark_config = configuration["benchmark"]
    device_name = runtime_config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        if runtime_config.get("allow_cpu_fallback", False):
            device_name = "cpu"
        else:
            raise RuntimeError("CUDA was requested but is unavailable")

    experiment_id = create_experiment_id(configuration)
    manifest = RunManifest(
        Path(configuration["output"]["artifact_root"]), experiment_id, configuration, "measure"
    )
    try:
        manifest.add_environment(collect_environment())
        bundle = build_cifar10_bundle(
            root=dataset_config["root"],
            seed=dataset_config.get("calibration_seed", configuration["project"]["seed"]),
            calibration_size=dataset_config["calibration_size"],
            validation_size=dataset_config.get("validation_size", 5000),
            batch_size=evaluation_config["batch_size"],
            num_workers=evaluation_config["num_workers"],
            download=dataset_config.get("download", False),
        )
        loaded = load_model_checkpoint(
            architecture=model_config["architecture"],
            checkpoint_path=model_config["checkpoint"],
            input_shape=tuple(model_config.get("input_shape", [1, 3, 32, 32])),
            num_classes=model_config.get("num_classes", 10),
            device=device_name,
        )
        sample_input = next(iter(bundle.calibration_loader))[0][:1].to(device_name)
        onnx_path = manifest.root / "models" / "model.onnx"
        export_result = export_to_onnx(
            loaded.reference,
            sample_input,
            onnx_path,
            input_name=onnx_config.get("input_name", "input"),
            output_name=onnx_config.get("output_name", "logits"),
            opset=onnx_config.get("opset", 18),
        )
        export_metrics = {
            "path": str(onnx_path),
            "input_name": export_result.input_name,
            "output_name": export_result.output_name,
            "opset": export_result.opset,
            "input_shape": list(export_result.input_shape),
            "output_shape": list(export_result.output_shape),
            "passed": True,
        }
        equivalence = compare_pytorch_onnx(
            loaded.reference,
            export_result,
            sample_input,
            rtol=onnx_config.get("rtol", 1.0e-3),
            atol=onnx_config.get("atol", 1.0e-4),
        )
        if not equivalence["passed"]:
            raise RuntimeError(f"PyTorch/ONNX equivalence failed: {equivalence}")
        sample_numpy = sample_input.detach().cpu().numpy().astype(np.float32)
        benchmark = benchmark_onnx(
            onnx_path,
            export_result.input_name,
            sample_numpy,
            intra_op_threads=benchmark_config["intra_op_threads"],
            inter_op_threads=benchmark_config["inter_op_threads"],
            warmup_runs=benchmark_config["warmup_runs"],
            timed_runs=benchmark_config["timed_runs"],
        )
        size_metrics = model_size_metrics(loaded.reference, onnx_path)
        fresh_process = validate_fresh_process(
            onnx_path,
            input_name=export_result.input_name,
            output_name=export_result.output_name,
            input_shape=export_result.input_shape,
        )
        if not fresh_process["passed"]:
            raise RuntimeError(f"Fresh-process validation failed: {fresh_process}")
        artifacts = {
            "onnx_export": (manifest.root / "metrics" / "onnx_export.json", export_metrics),
            "equivalence": (manifest.root / "metrics" / "equivalence.json", equivalence),
            "benchmark": (manifest.root / "metrics" / "benchmark.json", benchmark),
            "size": (manifest.root / "metrics" / "size.json", size_metrics),
            "fresh_process_validation": (
                manifest.root / "metrics" / "fresh_process_validation.json",
                fresh_process,
            ),
        }
        manifest.record_artifact("onnx_model", onnx_path)
        for name, (path, payload) in artifacts.items():
            _write_json(path, payload)
            manifest.record_artifact(name, path)
        manifest.complete()
        print(f"Created Phase 2 measurement: {manifest.root}")
        return 0
    except Exception as exc:
        manifest.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())