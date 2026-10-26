import torch

from cnn_compression_toolkit.models.resnet import resnet56


def test_resnet56_forward_contract() -> None:
    model = resnet56()
    output = model(torch.zeros(2, 3, 32, 32))

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 855770