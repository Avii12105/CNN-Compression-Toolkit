import json
import os
from pathlib import Path

import pytest

from cnn_compression_toolkit.cli import main

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]

if os.environ.get("CCT_RUN_REAL_DATA_TESTS") != "1":
    pytest.skip(
        "Set CCT_RUN_REAL_DATA_TESTS=1 to run real-data Phase 1 tests",
        allow_module_level=True,
    )


def test_real_data_baseline_writes_complete_artifacts(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["baseline", "--config", "configs/baseline.yaml"]) == 0
    output = capsys.readouterr().out
    run_root = Path(output.strip().split(": ", 1)[1])

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_root / "metrics" / "baseline.json").read_text(encoding="utf-8"))
    split_indices = json.loads(
        (run_root / "metrics" / "split_indices.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "completed"
    assert manifest["schema_version"] == 2
    assert set(manifest["artifacts"]) == {
        "baseline_metrics",
        "split_indices",
        "checkpoint_metadata",
    }
    assert metrics["sample_count"] == 10000
    assert len(metrics["confusion_matrix"]) == 10
    assert split_indices["test_size"] == 10000
    assert len(split_indices["calibration_indices"]) == 512
    assert 0.0 <= metrics["accuracy"] <= 1.0