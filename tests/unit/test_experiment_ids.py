from datetime import UTC, datetime

from cnn_compression_toolkit.experiments.ids import configuration_hash, create_experiment_id


def test_configuration_hash_is_order_independent() -> None:
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})


def test_experiment_id_contains_utc_timestamp_and_hash() -> None:
    identifier = create_experiment_id({"seed": 1}, datetime(2026, 10, 24, tzinfo=UTC))

    assert identifier == "20261024T000000Z-" + configuration_hash({"seed": 1})