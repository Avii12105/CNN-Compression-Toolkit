"""Stable experiment identifier generation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def configuration_hash(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def create_experiment_id(configuration: dict[str, Any], now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{configuration_hash(configuration)}"