"""Stateless, sample-keyed DDPM/DDIM sampling for the exploration protocol."""
import hashlib
import torch
from mpa_diff.diffusion.core import time_grid


def stable_seed(*parts):
    data = '\0'.join(str(x) for x in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') & ((1 << 63) - 1)


def keyed_noise(shape, device, sample_id, role, timestep=0):
    generator = torch.Generator(device=device).manual_seed(stable_seed('eval_protocol_v2', sample_id, role, timestep))
    return torch.randn(shape, device=device, generator=generator)


@torch.no_grad()
def sample_fixed(denoiser, condition, shape, schedule, sample_id, name, steps, clip=False):
    if name not in ('ddpm', 'ddim') or (name == 'ddpm' and steps != schedule.steps):
        raise ValueError('Invalid solver or DDPM step count')
    grid = time_grid(schedule.steps, steps)
    x = keyed_noise(shape, schedule.beta.device, sample_id, 'initial')
    for t, s in zip(grid[:-1], grid[1:]):
        index = torch.full((shape[0],), t - 1, device=x.device, dtype=torch.long)
        x0 = denoiser(x, index, condition)
        if clip:
            x0 = x0.clamp(0, 1)
        if s == 0:
            x = x0
            continue
        at, ass = schedule.alpha_bar_math[t], schedule.alpha_bar_math[s]
        if name == 'ddim':
            eps = schedule.epsilon(x, x0, index)
            x = ass.sqrt() * x0 + (1 - ass).sqrt() * eps
        else:
            beta = schedule.beta[t - 1]
            alpha = 1 - beta
            mean = ass.sqrt() * beta / (1 - at) * x0 + alpha.sqrt() * (1 - ass) / (1 - at) * x
            variance = beta * (1 - ass) / (1 - at)
            x = mean + variance.sqrt() * keyed_noise(shape, x.device, sample_id, 'ddpm', t)
    return x, {'solver': name, 'nfe': steps, 'math_timesteps': grid, 'eta': 0.0,
               'noise_protocol': 'sha256(eval_protocol_v2,sample_id,role,timestep)'}
