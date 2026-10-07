import torch
import torch.nn.functional as F


def geometry(r, eps=1e-6):
    """Return per-pixel shared-RGB quadratic geometry for residual r."""
    if r.ndim!=4 or r.shape[1]!=3 or not r.is_floating_point():raise ValueError('expected floating NCHW RGB residual')
    a = r.square().mean(dim=1, keepdim=True)
    c = a.sqrt()
    active = c > eps
    safe = torch.where(active, c, torch.ones_like(c))
    u = torch.where(active, r / safe, torch.zeros_like(r))
    return {"r": r, "a": a, "c": c, "u": u, "active": active}


def labels(base, candidate, target, eps=1e-6):
    g = geometry(candidate - base, eps=eps)
    e = target - base
    b = (e * g["r"]).mean(dim=1, keepdim=True)
    v = (e * g["u"]).mean(dim=1, keepdim=True)
    l0 = (base - target).square().mean(dim=1, keepdim=True)
    l1 = (candidate - target).square().mean(dim=1, keepdim=True)
    U = 2.0 * b - g["a"]
    return {**g, "e": e, "b": b, "v": v, "U": U, "l0": l0, "l1": l1}


def decision(base, g, b_hat, tau=0.0, lamb=1e-4):
    if tau<0 or lamb<=0:raise ValueError('tau >=0 and lambda >0 required')
    a = g["a"]
    active = g["active"]
    denom = a + lamb
    alpha = ((b_hat - tau) / denom).clamp(0.0, 1.0)
    alpha = torch.where(active, alpha, torch.zeros_like(alpha))
    out = (base + alpha * g["r"]).clamp(0.0, 1.0)
    return out, alpha


def oracle(base, candidate, target, block=1, eps=1e-6):
    """Reference-only block oracle; block aggregates a and b before solving."""
    if block < 1 or base.shape[-1] % block or base.shape[-2] % block:
        raise ValueError("block must divide spatial dimensions")
    r = candidate - base
    e = target - base
    n, _, h, w = r.shape
    if block > 1:
        hh, ww = h // block, w // block
        # Block objective uses aggregate RGB pixel energies over each block.
        a = r.square().view(n, 3, hh, block, ww, block).mean((1, 3, 5), keepdim=False).unsqueeze(1)
        b = (e * r).view(n, 3, hh, block, ww, block).mean((1, 3, 5), keepdim=False).unsqueeze(1)
        active = a > 0
        alpha = (b / torch.where(active, a, torch.ones_like(a))).clamp(0, 1)
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
        alpha = F.interpolate(alpha, size=(h, w), mode="nearest")
    else:
        a = r.square().mean(1, keepdim=True)
        b = (e * r).mean(1, keepdim=True)
        active = a.sqrt() > eps
        alpha = (b / torch.where(active, a, torch.ones_like(a))).clamp(0, 1)
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
    return (base + alpha * r).clamp(0, 1), alpha
