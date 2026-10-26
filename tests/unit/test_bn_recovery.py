import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_compression_toolkit.compression.recovery.bn_recovery import recover_batch_norm


def test_bn_recovery_updates_stats_without_updating_parameters() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
    model.train()
    inputs = torch.randn(8, 3, 8, 8)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(8, dtype=torch.long)), batch_size=4)
    parameters_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model[1].running_mean.fill_(5.0)

    result = recover_batch_norm(model, loader, torch.device("cpu"), expected_samples=8)

    assert result.batch_norm_layers == ("1",)
    assert result.sample_count == 8
    assert model.training is True
    assert torch.allclose(
        model[1].running_mean, torch.zeros_like(model[1].running_mean), atol=1.0
    )
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in parameters_before.items()
    )


def test_bn_recovery_rejects_wrong_calibration_size() -> None:
    model = nn.Sequential(nn.BatchNorm1d(2))
    loader = DataLoader(torch.randn(3, 2), batch_size=3)

    try:
        recover_batch_norm(model, loader, torch.device("cpu"), expected_samples=4)
    except ValueError as error:
        assert "expected 4" in str(error)
    else:
        raise AssertionError("Expected a calibration-size validation error")