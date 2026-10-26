import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_compression_toolkit.sensitivity.analyzer import analyze_sensitivity


def test_sensitivity_is_deterministic_and_preserves_reference() -> None:
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    inputs = torch.randn(6, 4)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(6, dtype=torch.long)), batch_size=3)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    first = analyze_sensitivity(model, loader, torch.device("cpu"))
    second = analyze_sensitivity(model, loader, torch.device("cpu"))

    assert [result.layer_name for result in first] == [result.layer_name for result in second]
    assert all(result.weights_only_proxy for result in first)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_sensitivity_rejects_invalid_bit_width() -> None:
    model = nn.Linear(2, 2)
    loader = DataLoader(torch.zeros(1, 2), batch_size=1)

    try:
        analyze_sensitivity(model, loader, torch.device("cpu"), bit_width=1)
    except ValueError as error:
        assert "bit_width" in str(error)
    else:
        raise AssertionError("Expected invalid bit width to be rejected")