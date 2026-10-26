import numpy as np
import pytest

from cnn_compression_toolkit.evaluation.benchmark import benchmark_session


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, output_names: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        return [inputs["input"]]


def test_benchmark_counts_warmups_and_timed_runs() -> None:
    session = FakeSession()
    result = benchmark_session(session, "input", np.zeros((1, 3, 32, 32), dtype=np.float32), 2, 5)

    assert session.calls == 7
    assert result["warmup_runs"] == 2
    assert result["timed_runs"] == 5
    assert result["input_shape"] == [1, 3, 32, 32]
    assert result["median"] >= 0


def test_benchmark_rejects_invalid_timing_counts() -> None:
    with pytest.raises(ValueError):
        benchmark_session(FakeSession(), "input", np.zeros(1), -1, 5)
    with pytest.raises(ValueError):
        benchmark_session(FakeSession(), "input", np.zeros(1), 0, 0)