import torch
import torch.nn as nn
from .blocks import trunk


class _CandidateNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        layers = list(trunk(in_channels, 32, 6))
        self.body = nn.Sequential(*layers)
        self.output = nn.Conv2d(32, 3, 3, padding=1, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.body(x))


class Candidate(nn.Module):
    def __init__(self, in_channels=14):
        super().__init__()
        self.net = _CandidateNet(in_channels)

    def forward(self, image, base, P, V):
        raw = self.net(torch.cat([image, base, P, V], dim=1))
        # The residual must be formed after the image-domain clipping.
        return torch.clamp(base + 0.25 * torch.tanh(raw), 0.0, 1.0)
