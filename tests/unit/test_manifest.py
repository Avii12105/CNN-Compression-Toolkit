import json
from pathlib import Path

from cnn_compression_toolkit.experiments.manifest import RunManifest


def test_manifest_creates_expected_run_layout(tmp_path: Path) -> None:
    manifest = RunManifest(tmp_path, "run-id", {"seed": 1}, "test")
    manifest.add_environment({"python_version": "3.11"})
    artifact = manifest.root / "metrics" / "baseline.json"
    artifact.write_text("{}")
    manifest.record_artifact("baseline_metrics", artifact)
    manifest.complete()

    assert (manifest.root / "config.yaml").is_file()
    assert (manifest.root / "environment.json").is_file()
    assert (manifest.root / "logs").is_dir()
    manifest_data = json.loads((manifest.root / "manifest.json").read_text())
    assert manifest_data["status"] == "completed"
    assert manifest_data["artifacts"]["baseline_metrics"] == "metrics/baseline.json"


def test_manifest_retains_failure(tmp_path: Path) -> None:
    manifest = RunManifest(tmp_path, "failed-run", {}, "test")
    manifest.fail(RuntimeError("boom"))

    data = json.loads((manifest.root / "manifest.json").read_text())
    assert data["status"] == "failed"
    assert "boom" in data["error"]