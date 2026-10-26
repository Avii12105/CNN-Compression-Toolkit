import copy
import os
from pathlib import Path

import pytest
import torch

from cnn_compression_toolkit.compression.pruning.pruner import prune_model
from cnn_compression_toolkit.compression.recovery.bn_recovery import recover_batch_norm
from cnn_compression_toolkit.data import build_cifar10_bundle
from cnn_compression_toolkit.models.loader import load_model_checkpoint

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]

if os.environ.get("CCT_RUN_PRUNING_TESTS") != "1":
    pytest.skip("Set CCT_RUN_PRUNING_TESTS=1 to run real pruning tests", allow_module_level=True)


def test_resnet_dependency_pruning_preserves_forward() -> None:
    checkpoint = Path("checkpoints/resnet56_cifar10.pt")
    loaded = load_model_checkpoint("resnet56", checkpoint)
    working = copy.deepcopy(loaded.working)
    example = torch.zeros(1, 3, 32, 32)

    records = prune_model(working, example, pruning_ratio=0.1, requested_alignment=1)
    output = working(example)

    assert records
    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()
    assert working.fc.out_features == 10


def test_pruned_resnet_bn_recovery_preserves_parameters() -> None:
    checkpoint = Path("checkpoints/resnet56_cifar10.pt")
    loaded = load_model_checkpoint("resnet56", checkpoint)
    working = copy.deepcopy(loaded.working)
    example = torch.zeros(1, 3, 32, 32)
    prune_model(working, example, pruning_ratio=0.1, requested_alignment=1)
    bundle = build_cifar10_bundle(
        "data/cifar10",
        seed=20261024,
        batch_size=128,
        num_workers=0,
        download=False,
    )
    parameters_before = {
        name: value.detach().clone() for name, value in working.named_parameters()
    }

    result = recover_batch_norm(
        working,
        bundle.calibration_loader,
        torch.device("cpu"),
        expected_samples=512,
    )

    assert result.sample_count == 512
    assert result.batch_norm_layers
    assert all(
        torch.equal(working.state_dict()[name], value)
        for name, value in parameters_before.items()
    )
    assert torch.isfinite(working(example)).all()