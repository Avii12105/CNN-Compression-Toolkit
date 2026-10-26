"""Legal model-state transitions for the compression pipeline."""

from __future__ import annotations

from enum import StrEnum


class PipelineState(StrEnum):
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"


class InvalidTransitionError(ValueError):
    """Raised when a transformation is requested in an invalid order."""


_ALLOWED_TRANSITIONS = {
    PipelineState.M0: {PipelineState.M1, PipelineState.M4},
    PipelineState.M1: {PipelineState.M2},
    PipelineState.M2: {PipelineState.M3, PipelineState.M4},
    PipelineState.M3: {PipelineState.M4},
    PipelineState.M4: set(),
}


def validate_transition(current: PipelineState, target: PipelineState) -> None:
    """Validate one forward pipeline transition."""

    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(f"Cannot transition from {current} to {target}")