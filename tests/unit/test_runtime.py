from cnn_compression_toolkit.runtime import collect_environment


def test_environment_capture_has_reproducibility_fields() -> None:
    environment = collect_environment()

    assert environment["python_version"]
    assert "packages" in environment
    assert "cuda" in environment
    assert isinstance(environment["cuda"]["available"], bool)