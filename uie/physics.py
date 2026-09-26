"""Conservative image-formation candidate, not calibrated physical recovery."""
import torch
from torch import nn
import torch.nn.functional as F


class PhysicalPrior(nn.Module):
    """RGB candidate + RGB transmission + scalar clipping diagnostic.

    The default range is an image-derived heuristic, NOT metric depth.
    Supplying depth requires a map whose larger values mean farther away.
    This branch is fixed so cycle consistency cannot collapse via free parameters.
    """
    def forward(self, image, depth=None):
        x = image.float().clamp(0, 1)
        blur = F.avg_pool2d(F.pad(x, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        if depth is None:
            # A weak low-contrast/red-deficit proxy deliberately exposed to tests.
            contrast = (x - blur).abs().mean(1, keepdim=True)
            distance = (0.5 + blur[:, 1:2] - blur[:, :1] - contrast).clamp(0, 1)
        else:
            if depth.shape != x[:, :1].shape:
                raise ValueError("depth must be Bx1xHxW and spatially aligned")
            lo = depth.amin((-2, -1), keepdim=True)
            hi = depth.amax((-2, -1), keepdim=True)
            distance = ((depth.float() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
        # Effective optical thickness; no sigmoid(beta)*unit-depth ceiling.
        rates = x.new_tensor([2.4, 1.2, 0.8]).view(1, 3, 1, 1)
        tau = rates * distance
        transmission = torch.exp(-tau).clamp_min(0.05)
        ambient = blur.amax((-2, -1), keepdim=True).clamp(0.05, 0.95)
        raw = (x - ambient * (1 - transmission)) / transmission
        clipping = ((raw < 0) | (raw > 1)).float().mean(1, keepdim=True)
        return torch.cat((raw.clamp(0, 1), transmission, clipping), 1)


def corrupt_prior(prior, mode, strength=0.15):
    if mode == "clean":
        return prior
    p = prior.clone()
    if mode == "shift":
        p = torch.roll(p, shifts=(max(1, p.shape[-2] // 8), max(1, p.shape[-1] // 8)), dims=(-2, -1))
    elif mode == "color":
        p[:, :3] = (p[:, :3] + p.new_tensor([strength, -strength, 0.0]).view(1, 3, 1, 1)).clamp(0, 1)
    elif mode == "invert":
        p[:, :6] = 1 - p[:, :6]
    elif mode == "missing":
        p[..., p.shape[-2] // 4:p.shape[-2] * 3 // 4, p.shape[-1] // 4:p.shape[-1] * 3 // 4] = 0
    else:
        raise ValueError(f"Unknown prior corruption: {mode}")
    return p
