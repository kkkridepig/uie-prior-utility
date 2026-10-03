import torch

from mpa_diff.diffusion.core import Schedule, time_grid
from scripts.explore_ag.sampling import keyed_noise, sample_fixed, stable_seed


def test_sample_keyed_noise_is_order_independent():
    a = keyed_noise((1, 3, 7, 9), 'cpu', 'UIEB/a', 'ddpm', 9)
    _ = keyed_noise((1, 3, 7, 9), 'cpu', 'UIEB/b', 'ddpm', 9)
    b = keyed_noise((1, 3, 7, 9), 'cpu', 'UIEB/a', 'ddpm', 9)
    assert torch.equal(a, b)
    assert stable_seed('a', 'b') != stable_seed('b', 'a')
    assert not torch.equal(a, keyed_noise((1, 3, 7, 9), 'cpu', 'UIEB/a', 'ddpm', 8))


def test_ddim_and_ddpm_end_at_clean_prediction():
    schedule = Schedule(steps=20)
    truth = torch.full((1, 3, 9, 11), .37)

    def denoiser(x, index, condition):
        assert index.min() >= 0 and index.max() < 20
        return truth

    for name, steps in (('ddpm', 20), ('ddim', 5), ('ddim', 1)):
        output, diag = sample_fixed(denoiser, {}, truth.shape, schedule, 'sample', name, steps)
        assert torch.equal(output, truth)
        assert diag['nfe'] == steps
        assert diag['math_timesteps'][-1] == 0
    assert time_grid(20, 5)[0] == 20
