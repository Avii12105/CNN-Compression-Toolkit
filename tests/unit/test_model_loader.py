from pathlib import Path

import pytest
import torch

from cnn_compression_toolkit.models.loader import load_model_checkpoint
from cnn_compression_toolkit.models.resnet import resnet56


def _write_checkpoint(path: Path) -> None:
    torch.save(resnet56().state_dict(), path)


def test_state_dict_loads_and_models_are_independent(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    _write_checkpoint(checkpoint_path)

    loaded = load_model_checkpoint("resnet56", checkpoint_path)
    reference_before = {
        name: value.detach().clone() for name, value in loaded.reference.state_dict().items()
    }
    with torch.no_grad():
        next(loaded.working.parameters()).add_(1.0)

    assert loaded.checkpoint_sha256
    assert all(
        torch.equal(loaded.reference.state_dict()[name], value)
        for name, value in reference_before.items()
    )


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint("resnet56", tmp_path / "missing.pt")


def test_unsupported_checkpoint_format_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "full-model.pt"
    torch.save(resnet56(), checkpoint_path)

    with pytest.raises(TypeError, match="raw state_dict"):
        load_model_checkpoint("resnet56", checkpoint_path)