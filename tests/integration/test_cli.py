import json
from pathlib import Path

import pytest

from cnn_compression_toolkit.cli import main


@pytest.mark.integration
def test_configuration_only_run_creates_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path("configs/baseline.yaml").read_text(encoding="utf-8")
    source = source.replace("artifacts", str(tmp_path / "artifacts"), 1)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(source, encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    run_root = Path(output.strip().split(": ", 1)[1])

    assert json.loads((run_root / "manifest.json").read_text())["status"] == "completed"
    assert (run_root / "config.yaml").is_file()
    assert (run_root / "environment.json").is_file()
    assert (run_root / "logs" / "run.jsonl").is_file()