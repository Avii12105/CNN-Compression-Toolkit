"""Console and JSONL logging helpers."""

from __future__ import annotations

import json
import logging as std_logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("experiment_id", "stage"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(
    log_path: str | Path, level: str = "INFO", experiment_id: str | None = None
) -> std_logging.Logger:
    """Configure the toolkit logger with console and JSONL handlers."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = std_logging.getLogger("cnn_compression_toolkit")
    logger.handlers.clear()
    logger.setLevel(getattr(std_logging, level.upper()))
    logger.propagate = False

    formatter = JsonlFormatter()
    file_handler = std_logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = std_logging.StreamHandler()
    console_handler.setFormatter(std_logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(console_handler)
    logger = std_logging.LoggerAdapter(logger, {"experiment_id": experiment_id, "stage": "phase0"})
    return logger