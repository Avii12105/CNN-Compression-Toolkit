# Phase 0: Project Setup

Phase 0 established the reproducibility boundary for the compression engine. Phase 1 provides the PyTorch dataset/model preparation and M0 baseline path. Phase 2 now provides fixed-shape FP32 ONNX export, output equivalence, CPU latency measurement, serialized size/hash metrics, and fresh-process validation. Pruning, Batch Normalization recovery, distillation, and quantization remain later phases.

## Environment

- Python: 3.11
- Package manager: `uv`
- Development device: CUDA when available, with CPU fallback
- Benchmark provider: ONNX Runtime `CPUExecutionProvider`
- Configuration mode: local paths are validated syntactically; external assets are not required

The project accepts Python 3.11 through 3.13 for the initial package scaffold, but the reproducibility target is Python 3.11. Select the PyTorch wheel that matches the host CUDA runtime before creating the lockfile.

## Setup

```bash
uv venv --python 3.11
uv sync --extra dev
```

For a CUDA host, install the appropriate PyTorch CUDA wheel according to the official PyTorch selector, then run `uv lock` and `uv sync`. The benchmark protocol remains CPU-only so latency comparisons are host-consistent.

## Phase 0 commands

```bash
uv run cct info
uv run cct validate-config configs/baseline.yaml
uv run cct run --config configs/baseline.yaml
uv run cct baseline --config configs/baseline.yaml
uv run cct measure --config configs/baseline.yaml
uv run pytest -m "not integration"
```

The run command creates `artifacts/runs/<experiment_id>/` with the resolved configuration, environment metadata, JSONL logs, and a completed manifest. Failed setup retains a failed manifest for diagnosis.

## Phase 1 status

The repository has provisioned CIFAR-10 and a ResNet-56 raw `state_dict` checkpoint. The baseline command records deterministic split indices, checkpoint metadata, M0 metrics, and manifest artifact references. Best-checkpoint selection uses validation accuracy with first-best tie behavior.

VGG-16-BN architecture support is present, but a separately trained compatible CIFAR-10 VGG-16-BN checkpoint is still required.

## Phase 3 boundary

Phase 3 begins with structured channel importance and pruning. Phase 2 does not implement pruning, Batch Normalization recovery, distillation, quantization, or budget search.