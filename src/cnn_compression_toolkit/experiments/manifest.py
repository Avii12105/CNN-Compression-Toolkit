"""Run artifact directories and manifest lifecycle."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MANIFEST_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class RunManifest:
    """Create and update the files belonging to one experiment run."""

    def __init__(
        self, root: Path, experiment_id: str, configuration: dict[str, Any], command: str
    ) -> None:
        self.root = root / "runs" / experiment_id
        self.experiment_id = experiment_id
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("logs", "metrics", "models", "reports"):
            (self.root / directory).mkdir(exist_ok=True)
        _atomic_write(self.root / "config.yaml", yaml.safe_dump(configuration, sort_keys=False))
        self.data: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "command": command,
            "status": "started",
            "started_at": _utc_now(),
            "completed_at": None,
            "error": None,
            "artifacts": {},
        }
        self._write()

    def _write(self) -> None:
        content = json.dumps(self.data, indent=2, sort_keys=True) + "\n"
        _atomic_write(self.root / "manifest.json", content)

    def add_environment(self, environment: dict[str, Any]) -> None:
        content = json.dumps(environment, indent=2, sort_keys=True) + "\n"
        _atomic_write(self.root / "environment.json", content)

    def record_artifact(self, name: str, path: Path) -> None:
        """Record a run-relative artifact path in the manifest."""

        relative_path = path.relative_to(self.root)
        self.data["artifacts"][name] = str(relative_path)
        self._write()

    def complete(self) -> None:
        self.data.update({"status": "completed", "completed_at": _utc_now()})
        self._write()

    def fail(self, error: BaseException) -> None:
        self.data.update({"status": "failed", "completed_at": _utc_now(), "error": repr(error)})
        self._write()