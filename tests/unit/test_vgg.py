from pathlib import Path

import pytest
import torch

from cnn_compression_toolkit.models.loader import load_model_checkpoint
from cnn_compression_toolkit.models.vgg import vgg16_bn


def test_vgg16_bn_forward_contract() -> None:
    model = vgg16_bn()
    output = model(torch.zeros(2, 3, 32, 32))

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()


def test_vgg16_bn_state_dict_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "vgg16_bn.pt"
    torch.save(vgg16_bn().state_dict(), checkpoint_path)

    loaded = load_model_checkpoint("vgg16_bn", checkpoint_path)
    assert loaded.architecture == "vgg16_bn"
    assert loaded.reference(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


def test_vgg_rejects_resnet_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "wrong.pt"
    torch.save(vgg16_bn(num_classes=9).state_dict(), checkpoint_path)

    with pytest.raises(RuntimeError):
        load_model_checkpoint("vgg16_bn", checkpoint_path)