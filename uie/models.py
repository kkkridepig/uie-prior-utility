import math

import torch
from torch import nn
import torch.nn.functional as F

from .physics import PhysicalPrior, corrupt_prior
from .vendor.wwe import myModel


def pad_image(x, multiple=8):
    h, w = x.shape[-2:]
    target_h = max(16, math.ceil(h / multiple) * multiple)
    target_w = max(16, math.ceil(w / multiple) * multiple)
    return F.pad(x, (0, target_w - w, 0, target_h - h), mode="replicate")


class Coarse(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.backbone = myModel(feature_channels=width, use_white_balance=True)

    def forward(self, image):
        h, w = image.shape[-2:]
        return self.backbone(pad_image(image))[..., :h, :w]


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(4, width),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 1),
            nn.SiLU(),
            nn.Conv2d(width * 2, width, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class DenseCondition(nn.Module):
    """Small trainable patch-grid baseline, not a pretrained DINO representation."""
    def __init__(self, width, mode):
        super().__init__()
        if mode not in ("dense", "global", "none"):
            raise ValueError("patch_mode must be dense/global/none")
        self.mode = mode
        self.encoder = nn.Sequential(nn.Conv2d(3, width, 4, stride=4), nn.SiLU(), Block(width))

    def forward(self, x):
        if self.mode == "none":
            return x.new_zeros((x.shape[0], self.encoder[0].out_channels, *x.shape[-2:]))
        z = self.encoder(pad_image(x))
        if self.mode == "global":
            z = z.mean((-2, -1), keepdim=True)
        return F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)


class FlowField(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stem = nn.Conv2d(17 + width, width, 3, padding=1)
        self.enc = Block(width)
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.mid = nn.Sequential(Block(width * 2), Block(width * 2))
        self.up = nn.Conv2d(width * 2, width, 1)
        self.dec = Block(width)
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image, base, state, time, prior, gate, features):
        t = time.reshape(-1, 1, 1, 1).expand(image.shape[0], 1, *image.shape[-2:])
        x = self.stem(torch.cat((image, base, state, t, prior * gate, features), 1))
        skip = self.enc(x)
        mid = self.mid(self.down(skip))
        x = self.up(F.interpolate(mid, size=skip.shape[-2:], mode="bilinear", align_corners=False)) + skip
        return self.out(self.dec(x))


class UtilityHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, width, 3, padding=1), nn.SiLU(), Block(width),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 1, 1),
        )

    def forward(self, image, base, initial_state, prior):
        return self.net(torch.cat((image, base, initial_state, prior), 1))


class RestorationSystem(nn.Module):
    def __init__(self, width=32, patch_mode="dense", sigma=0.05, sampler="flow"):
        super().__init__()
        if width < 8 or width % 4:
            raise ValueError("width must be a multiple of 4 and >=8")
        if sigma < 0:
            raise ValueError("sigma must be nonnegative")
        if sampler not in ("flow", "ddim"):
            raise ValueError("sampler must be flow/ddim")
        self.spec = dict(width=width, patch_mode=patch_mode, sigma=sigma, sampler=sampler)
        self.sigma = sigma
        self.sampler = sampler
        self.coarse = Coarse(width)
        self.prior = PhysicalPrior()
        self.condition = DenseCondition(width, patch_mode)
        self.flow = FlowField(width)
        self.utility = UtilityHead(width)
        self.register_buffer("temperature", torch.tensor(1.0))

    def noise(self, image, seed=None):
        scale = self.sigma if self.sampler == "flow" else 1.0
        if seed is None:
            return torch.randn_like(image) * scale
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(image.shape, generator=generator, dtype=torch.float32).to(image.device) * scale

    @staticmethod
    def alpha_bar(time):
        return torch.cos((time + 0.008) / 1.008 * (math.pi / 2)).square().clamp(1e-4, 0.9999)

    def velocity(self, image, base, state, time, prior, gate=1.0, features=None):
        if features is None:
            features = self.condition(image)
        return self.flow(image, base, state, time, prior, gate, features)

    def integrate(self, image, base, prior, initial, gate, steps, features=None):
        if steps < 1:
            raise ValueError("steps must be >=1")
        if features is None:
            features = self.condition(image)
        state = initial
        for k in range(steps):
            if self.sampler == "flow":
                time = image.new_full((image.shape[0],), k / steps)
                state = state + self.velocity(image, base, state, time, prior, gate, features) / steps
            else:
                # Deterministic DDIM (eta=0), predicting clean residual r0.
                time = image.new_full((image.shape[0],), 1 - k / steps)
                residual = self.velocity(image, base, state, time, prior, gate, features)
                alpha = self.alpha_bar(time)[:, None, None, None]
                epsilon = (state - alpha.sqrt() * residual) / (1-alpha).sqrt()
                if k == steps-1:
                    state = residual
                else:
                    nxt = self.alpha_bar(time - 1/steps)[:, None, None, None]
                    state = nxt.sqrt() * residual + (1-nxt).sqrt() * epsilon
        return base + state

    def restore(self, image, steps=4, gate_mode="utility", seed=42, corruption="clean",
                fixed_gate=0.5, depth=None, initial=None):
        base = self.coarse(image).clamp(0, 1)
        prior = corrupt_prior(self.prior(image, depth), corruption)
        initial = self.noise(image, seed) if initial is None else initial
        if gate_mode in ("utility", "learned"):
            gate = torch.sigmoid(self.utility(image, base, initial, prior) / self.temperature.clamp_min(0.05))
        elif gate_mode in ("none", "all", "fixed"):
            value = {"none": 0.0, "all": 1.0, "fixed": fixed_gate}[gate_mode]
            gate = image.new_full((image.shape[0], 1, *image.shape[-2:]), value)
        else:
            raise ValueError(f"Unknown gate mode: {gate_mode}")
        return self.integrate(image, base, prior, initial, gate, steps), gate
