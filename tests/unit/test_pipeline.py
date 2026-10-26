import pytest

from cnn_compression_toolkit.compression.pipeline import (
    InvalidTransitionError,
    PipelineState,
    validate_transition,
)


def test_pipeline_allows_terminal_quantization_after_m0() -> None:
    validate_transition(PipelineState.M0, PipelineState.M4)


def test_pipeline_rejects_transformations_after_quantization() -> None:
    with pytest.raises(InvalidTransitionError):
        validate_transition(PipelineState.M4, PipelineState.M1)