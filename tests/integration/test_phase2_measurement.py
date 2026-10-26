import json
import os
from pathlib import Path

import pytest

from cnn_compression_toolkit.cli import main

pytestmark = [pytest.mark.integration, pytest.mark.requires_onnx, pytest.mark.requires_data]

if os.environ.get("CCT_RUN_ONNX_TESTS") != "1":
    pytest.skip("Set CCT_RUN_ONNX_TESTS=1 to run real ONNX tests", allow_module_level=True)


def test_real_data_measurement_writes_phase2_artifacts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["measure", "--config", "configs/baseline.yaml"]) == 0
    output = capsys.readouterr().out
    run_root = Path(output.strip().split(": ", 1)[1])

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run_root / "metrics" / "benchmark.json").read_text(encoding="utf-8"))
    equivalence = json.loads(
        (run_root / "metrics" / "equivalence.json").read_text(encoding="utf-8")
    )
    fresh_process = json.loads(
        (run_root / "metrics" / "fresh_process_validation.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "completed"
    assert set(manifest["artifacts"]) == {
        "onnx_model",
        "onnx_export",
        "equivalence",
        "benchmark",
        "size",
        "fresh_process_validation",
    }
    assert benchmark["warmup_runs"] == 50
    assert benchmark["timed_runs"] == 200
    assert benchmark["provider"] == "CPUExecutionProvider"
    assert equivalence["passed"] is True
    assert fresh_process["passed"] is True