from pathlib import Path

import torch
from torch import nn

from cnn_compression_toolkit.training.checkpoint import BestCheckpointSelector


def test_selector_restores_best_validation_accuracy_and_first_tie(tmp_path: Path) -> None:
    model = nn.Linear(1, 1, bias=False)
    selector = BestCheckpointSelector()

    with torch.no_grad():
        model.weight.fill_(1.0)
    assert selector.consider(model, epoch=1, validation_accuracy=0.8, validation_loss=0.5)

    with torch.no_grad():
        model.weight.fill_(2.0)
    assert not selector.consider(model, epoch=2, validation_accuracy=0.8, validation_loss=0.4)

    with torch.no_grad():
        model.weight.fill_(3.0)
    selector.restore(model)

    assert selector.best_epoch == 1
    assert selector.best_validation_accuracy == 0.8
    assert model.weight.item() == 1.0

    checkpoint_path = tmp_path / "best.pt"
    selector.save(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["weight"].item() == 1.0