"""CIFAR-10 ResNet architectures."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """The two-convolution residual block used by CIFAR ResNet."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + identity)


class CIFARResNet(nn.Module):
    """CIFAR-compatible ResNet with three stages and a 32x32 input."""

    def __init__(self, depth: int = 56, num_classes: int = 10) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must satisfy (depth - 2) % 6 == 0")
        blocks_per_stage = (depth - 2) // 6
        self._in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_stage(16, blocks_per_stage, 1)
        self.layer2 = self._make_stage(32, blocks_per_stage, 2)
        self.layer3 = self._make_stage(64, blocks_per_stage, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: list[nn.Module] = []
        for current_stride in strides:
            layers.append(BasicBlock(self._in_channels, channels, current_stride))
            self._in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer3(self.layer2(self.layer1(outputs)))
        outputs = self.avgpool(outputs)
        return self.fc(torch.flatten(outputs, 1))


def resnet56(num_classes: int = 10) -> CIFARResNet:
    """Construct the supported CIFAR-10 ResNet-56."""

    return CIFARResNet(depth=56, num_classes=num_classes)