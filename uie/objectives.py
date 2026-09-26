import torch
import torch.nn.functional as F


def reconstruction_loss(output, target, edge_weight=0.1):
    loss = torch.sqrt((output.float() - target.float()).square() + 1e-6).mean()
    dx = output[..., :, 1:] - output[..., :, :-1]
    dy = output[..., 1:, :] - output[..., :-1, :]
    tx = target[..., :, 1:] - target[..., :, :-1]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return loss + edge_weight * (F.l1_loss(dx, tx) + F.l1_loss(dy, ty))


@torch.no_grad()
def utility_target(off, on, target, delta=0.001, window=9):
    if window < 1 or window % 2 == 0:
        raise ValueError("utility window must be positive and odd")
    # Evaluate the actual clipped image delivered by the inference pipeline.
    off_error = (off.float().clamp(0, 1) - target.float()).abs().mean(1, keepdim=True)
    on_error = (on.float().clamp(0, 1) - target.float()).abs().mean(1, keepdim=True)
    benefit = F.avg_pool2d(off_error - on_error, window, stride=1, padding=window // 2, count_include_pad=False)
    return benefit.detach(), (benefit > delta).float().detach()
