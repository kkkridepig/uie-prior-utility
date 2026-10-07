import torch
import torch.nn as nn
from .blocks import trunk


class _H(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        self.body = nn.Sequential(*trunk(in_channels, 32, 4))
        self.output = nn.Conv2d(32, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, x):
        return self.output(self.body(x))


class UtilityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = _H(9)

    def forward(self, image, base, r):
        g_a = r.square().mean(1, keepdim=True)
        c = g_a.sqrt()
        active = c > 1e-6
        safe = torch.where(active, c, torch.ones_like(c))
        u = torch.where(active, r / safe, torch.zeros_like(r))
        hp = self.h(torch.cat([image, base, u], dim=1))
        hn = self.h(torch.cat([image, base, -u], dim=1))
        v_hat = 0.5 * (hp - hn)
        v_hat = torch.where(active, v_hat, torch.zeros_like(v_hat))
        return {"v_hat": v_hat, "b_hat": c * v_hat, "u": u, "a": g_a,
                "c": c, "active": active}
