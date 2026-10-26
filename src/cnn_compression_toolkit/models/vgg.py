"""CIFAR-10 VGG-16-BN architecture."""

from __future__ import annotations

import torch
from torch import nn


class CIFARVGG16BN(nn.Module):
    """VGG-16 with BatchNorm and a CIFAR-10-compatible classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = self._make_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_features() -> nn.Sequential:
        specification: list[int | str] = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        layers: list[nn.Module] = []
        input_channels = 3
        for value in specification:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            layers.extend(
                [
                    nn.Conv2d(input_channels, value, kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(value),
                    nn.ReLU(inplace=True),
                ]
            )
            input_channels = value
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.avgpool(outputs)
        return self.classifier(torch.flatten(outputs, 1))


def vgg16_bn(num_classes: int = 10) -> CIFARVGG16BN:
    """Construct the supported CIFAR-10 VGG-16-BN model."""

    return CIFARVGG16BN(num_classes=num_classes)