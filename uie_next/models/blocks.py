import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=True)

    def forward(self, x):
        return x + 0.1 * self.conv2(F.silu(self.conv1(x)))


def trunk(in_channels, width=32, depth=4):
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, padding=1, bias=True),
        nn.SiLU(),
        *[ResidualBlock(width) for _ in range(depth)],
    )
