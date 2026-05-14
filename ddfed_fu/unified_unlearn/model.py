"""ConvNet model adapted from ddfed_fl/utils/networks.py (net_width=64, net_depth=3)."""

import torch.nn as nn


def conv_block(in_channels: int, out_channels: int, **kwargs) -> list:
    """Single Conv2d → BatchNorm → ReLU → MaxPool block."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    ]


class ConvNet(nn.Module):
    """ConvNet for Fashion‑MNIST / CIFAR‑10 (net_width=64, net_depth=3)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 net_width: int = 64, net_depth: int = 3,
                 im_size: int = 28):
        super().__init__()
        layers = []
        prev = in_channels
        h = w = im_size
        for i in range(net_depth):
            cur = net_width << i  # 64, 128, 256  for depth=3
            layers.extend(conv_block(prev, cur, bias=False))
            prev = cur
            h //= 2
            w //= 2
        self.features = nn.Sequential(*layers)
        num_feat = prev * h * w
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def create_model(dataset: str, num_classes: int = 10,
                 net_width: int = 64, net_depth: int = 3) -> ConvNet:
    in_ch = 3 if dataset.lower().startswith("cifar") else 1
    im_size = 32 if dataset.lower().startswith("cifar") else 28
    return ConvNet(in_channels=in_ch, num_classes=num_classes,
                   net_width=net_width, net_depth=net_depth,
                   im_size=im_size)
