# CNN Compression Toolkit

A reproducible experimentation engine for structured CNN compression. It loads trained FP32 convolutional neural networks, applies controlled transformations (structured channel pruning, BatchNorm recovery, conditional knowledge distillation, INT8/FP16 quantization), evaluates each model state under a fixed protocol, and identifies configurations satisfying accuracy, latency, and model-size constraints.

## Features

- **Model Support**: ResNet-56 and VGG-16-BN on CIFAR-10
- **Structured Channel Pruning**: L1/Lp importance scoring with Torch-Pruning dependency groups, SIMD-aware channel alignment (1/2/4/8/16/32), and sensitivity-guided protection
- **BatchNorm Recovery**: Calibration-only running-statistics re-estimation without updating trainable weights
- **Conditional Knowledge Distillation**: Temperature-scaled KL divergence training, invoked only when M2 misses the accuracy target
- **Quantization**: ONNX Runtime static INT8 quantization (QLinearOps/QDQ) and FP16 conversion, with host precision gating (INT8 only when fused-integer path is supported)
- **Operator Coverage Audit**: QDQ coverage per operator type with residual exclusion propagation
- **Budget-Driven Search**: Search pruning ratios and precisions against hard accuracy, latency, and size budgets with efficiency scoring
- **Measurement Pipeline**: Fixed ONNX Runtime CPU benchmarking (50 warmups / 200 timed runs), Top-1 accuracy, macro precision/recall/F1, confusion matrix, serialized size, SHA-256 checksums, fresh-process validation
- **Experiment Tracking**: Experiment IDs, run manifests, config snapshots, environment metadata, artifact lineage (M0→M1→M2→M3→M4)
- **Pareto Analysis**: Multi-objective frontier computation (accuracy, latency, size)
- **Report Generation**: Markdown/JSON reports with stage comparisons, Pareto charts, and validation status
- **Terminal UI**: Full Textual TUI for configuring, running, monitoring, and inspecting experiments (no web UI required)

## Repository Layout

```
cnn-compression-toolkit/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI pipeline (lint + test)
├── configs/
│   ├── baseline.yaml              # ResNet-56 experiment configuration
│   └── vgg16_bn.yaml              # VGG-16-BN experiment configuration
├── docs/
│   └── phase-0.md                 # Phase 0 setup contract
├── src/
│   └── cnn_compression_toolkit/
│       ├── __init__.py
│       ├── cli.py                 # Command-line interface (cct)
│       ├── config.py              # YAML config loading & validation
│       ├── logging.py             # JSONL + console logging
│       ├── runtime.py             # Environment metadata collection
│       ├── compression/
│       │   ├── pipeline.py        # Model state transitions (M0→M4)
│       │   ├── orchestration.py   # End-to-end pipeline orchestration
│       │   ├── pruning/
│       │   │   ├── importance.py  # Lp channel importance
│       │   │   ├── alignment.py   # SIMD-aware retained width
│       │   │   ├── dependency.py  # (in pruner) Torch-Pruning groups
│       │   │   └── pruner.py      # Dependency-aware structured pruning
│       │   ├── recovery/
│       │   │   └── bn_recovery.py # Calibration-only BN statistics recovery
│       │   ├── quantization/
│       │   │   ├── quantizer.py   # INT8 static / FP16 quantization
│       │   │   ├── precision_gate.py # Host precision probing
│       │   │   ├── coverage.py    # QDQ operator coverage audit
│       │   │   └── exclusion.py   # Residual exclusion propagation
│       │   ├── distillation/
│       │   │   └── distiller.py   # Conditional knowledge distillation
│       │   └── __init__.py
│       ├── data/
│       │   ├── cifar10.py         # Deterministic CIFAR-10 loaders
│       │   └── __init__.py
│       ├── evaluation/
│       │   ├── benchmark.py       # ONNX Runtime CPU benchmarking
│       │   ├── metrics.py         # Top-1, macro precision/recall/F1, confusion
│       │   ├── onnx_export.py     # ONNX export & PyTorch equivalence
│       │   ├── size.py            # Parameter count, ONNX bytes, SHA-256
│       │   ├── validation.py      # Fresh-process ONNX validation
│       │   └── __init__.py
│       ├── experiments/
│       │   ├── ids.py             # Experiment ID generation
│       │   ├── manifest.py        # Run manifest & artifact tracking
│       │   ├── matrix.py          # Experiment matrix execution
│       │   └── __init__.py
│       ├── models/
│       │   ├── loader.py          # Strict state-dict loading
│       │   ├── resnet.py          # CIFAR-10 ResNet-56
│       │   ├── vgg.py             # CIFAR-10 VGG-16-BN
│       │   └── __init__.py
│       ├── reports/
│       │   ├── generator.py       # Markdown/JSON report generation
│       │   ├── pareto.py          # Pareto frontier computation
│       │   ├── validation.py      # Final validation
│       │   └── __init__.py
│       ├── search/
│       │   ├── budget_search.py   # Budget-driven candidate search
│       │   └── __init__.py
│       ├── sensitivity/
│       │   ├── analyzer.py        # Weights-only sensitivity (MSE/SQNR)
│       │   └── __init__.py
│       ├── training/
│       │   ├── checkpoint.py      # Best-checkpoint selection
│       │   └── __init__.py
│       └── tui/
│           ├── app.py             # Textual TUI application
│           └── screens/
│               ├── artifact.py
│               ├── charts.py
│               ├── config.py
│               ├── dashboard.py
│               ├── model_select.py
│               ├── results.py
│               ├── run_status.py
│               └── shortcuts.py
├── tests/
│   ├── integration/
│   └── unit/
├── pyproject.toml                 # Project metadata & dependencies
├── uv.lock                        # Locked dependency versions
├── IMPLEMENTATION_STATUS.md       # Implementation progress tracking
├── CNN_Compression_Toolkit_Detailed_Implementation_Plan.md
├── LICENSE                        # MIT license
├── README.md
└── CONTRIBUTING.md
```

## Installation

The project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Clone the repository
git clone https://github.com/Avii12105/CNN-Compression-Toolkit.git
cd CNN-Compression-Toolkit

# Create the virtual environment and install dependencies
uv venv --python 3.11
uv sync --extra dev

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows PowerShell
```

## Quick Start

### 1. Validate configuration

```bash
uv run cct validate-config configs/baseline.yaml
```

### 2. Run the Phase 1 baseline (M0)

Loads the trained FP32 checkpoint, evaluates on CIFAR-10 test set, and records metrics.

```bash
uv run cct baseline --config configs/baseline.yaml
```

### 3. Export and measure ONNX (Phase 2)

```bash
uv run cct measure --config configs/baseline.yaml
```

### 4. Run the full compression pipeline (M0 → M4)

```bash
uv run cct compress \
  --config configs/baseline.yaml \
  --pruning-ratio 0.3 \
  --alignment 8 \
  --precision fp32 \
  --protected-count 4 \
  --distillation-epochs 5 \
  --distillation-target-accuracy 0.90
```

### 5. Budget-driven search

Find configurations satisfying accuracy, latency, and size constraints:

```bash
uv run cct search \
  --config configs/baseline.yaml \
  --budget-latency-ms 2.0 \
  --budget-max-accuracy-drop 0.05 \
  --budget-min-accuracy 0.85 \
  --budget-max-size-bytes 2000000
```

### 6. Launch the Terminal UI

A full terminal UI is available for configuring, running, monitoring, and inspecting experiments:

```bash
uv run cct tui
```

Keyboard shortcuts:
- `?` — Toggle shortcuts help
- `q` / `Ctrl+Q` — Quit
- `Tab` — Switch between screens

## Model States

| State | Description |
|-------|-------------|
| **M0** | Original FP32 model (baseline) |
| **M1** | Structured-pruned model |
| **M2** | BatchNorm-recovered model |
| **M3** | Distilled student (conditional) |
| **M4** | Final quantized model (terminal) |

Pipeline: Prune → BN-Recover → Distill (conditional) → Quantize.

## Configuration

Configuration is YAML-driven. Key sections:

- `project` — name, seed
- `model` — architecture, checkpoint path, input shape, num classes
- `dataset` — root, calibration size (512), validation size (5000), normalization
- `evaluation` — batch size, num workers
- `runtime` — device (auto/cpu/cuda), CPU fallback
- `benchmark` — provider, batch size, threads, warmup/timed runs
- `onnx` — opset, input/output names, tolerances
- `output` — artifact root
- `logging` — level, environment capture
- `compression` — pruning ratio, precision, distillation settings, protection, SIMD alignment
- `quantization` — precision, calibration size, QDQ, per-channel
- `search` — budget constraints (latency, accuracy drop, min accuracy, max size)

Example `configs/baseline.yaml`:

```yaml
project:
  name: cnn-compression-toolkit
  seed: 20261024

model:
  architecture: resnet56
  checkpoint: ../checkpoints/resnet56_cifar10.pt
  checkpoint_format: state_dict
  num_classes: 10
  input_shape: [1, 3, 32, 32]

dataset:
  root: ../data/cifar10
  calibration_size: 512
  calibration_seed: 20261024
  validation_size: 5000
  download: false
  normalization:
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2470, 0.2435, 0.2616]

benchmark:
  provider: CPUExecutionProvider
  batch_size: 1
  intra_op_threads: 1
  inter_op_threads: 1
  warmup_runs: 50
  timed_runs: 200

compression:
  default_pruning_ratio: 0.0
  default_precision: fp32
  distillation_target_accuracy: null
  protection_gamma: 0.5
  protected_layers_count: 4
  simd_alignment: 8
  ratio_clamp_max: 0.9

quantization:
  precision: auto
  calibration_size: 512
  qdq: true
  per_channel: true

search:
  budget_latency_ms: null
  budget_max_accuracy_drop: null
  budget_min_accuracy: null
  budget_max_size_bytes: null
```

## Artifacts

Each experiment is stored under `artifacts/runs/<experiment_id>/` and contains:

```
<experiment_id>/
├── config.yaml              # Snapshot of experiment configuration
├── environment.json         # Python, PyTorch, ONNX, etc. versions
├── manifest.json            # Run lifecycle metadata
├── logs/run.jsonl           # Structured run log
├── metrics/
│   ├── baseline.json        # M0 metrics
│   ├── split_indices.json   # Train/val/cal/test split indices
│   ├── checkpoint.json      # Checkpoint metadata & SHA-256
│   ├── stage_m0.json        # M0 stage metrics
│   ├── stage_m1.json        # M1 stage metrics
│   ├── stage_m2.json        # M2 stage metrics
│   ├── stage_m3.json        # M3 stage metrics
│   ├── stage_m4.json        # M4 stage metrics
│   ├── stage_summary.json   # All stage summary
│   ├── stage_*.json         # Per-stage detailed metrics
│   └── transformation_history.json  # Full transformation lineage
├── models/
│   ├── model_m0.onnx        # M0 ONNX artifact
│   ├── model_m1.onnx        # M1 ONNX artifact
│   ├── model_m2.onnx        # M2 ONNX artifact
│   ├── model_m3.onnx        # M3 ONNX artifact
│   └── model_m4.onnx        # M4 ONNX artifact
└── reports/
    ├── report.md            # Human-readable final report
    ├── report.json          # Machine-readable final report
    └── plot_data.json       # Data for chart regeneration
```

## Testing

```bash
# Run all unit tests
uv run pytest

# Run all tests including integration (requires data/checkpoints)
uv run pytest

# Run only unit tests (fast, no external assets)
uv run pytest -m "not integration"
```
