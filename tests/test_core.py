import torch
import pytest

from uie.models import RestorationSystem
from uie.physics import PhysicalPrior, corrupt_prior
from uie.objectives import utility_target


def test_physics_constant_images_and_tau_range():
    prior = PhysicalPrior()
    for value in (0.0, 0.3, 1.0):
        result = prior(torch.full((2, 3, 17, 19), value))
        assert result.shape == (2, 7, 17, 19)
        assert torch.isfinite(result).all()
    x = torch.rand(2, 3, 20, 24)
    clean = prior(x)
    noisy = corrupt_prior(clean, "shift")
    assert clean.shape == noisy.shape
    assert not torch.equal(clean, noisy)


@pytest.mark.parametrize("spatial", ["dense", "global", "none"])
@pytest.mark.parametrize("sampler", ["flow", "ddim"])
def test_full_pipeline_odd_size_backward(spatial, sampler):
    model = RestorationSystem(width=8, patch_mode=spatial, sampler=sampler)
    x = torch.rand(1, 3, 19, 23)
    y = torch.rand_like(x)
    out = model.coarse(x)
    assert out.shape == x.shape
    (out - y).abs().mean().backward()
    model.zero_grad(set_to_none=True)
    base = model.coarse(x).detach()
    prior = model.prior(x)
    state = torch.randn_like(x) * 0.05
    velocity = model.velocity(x, base, state, torch.tensor([0.5]), prior, gate=1.0)
    assert velocity.shape == x.shape
    velocity.square().mean().backward()
    assert any(p.grad is not None for p in model.flow.parameters())
    with torch.no_grad():
        out, gate = model.restore(x, steps=2, gate_mode="utility", seed=7)
    assert out.shape == x.shape
    assert gate.shape == (1, 1, 19, 23)
    assert torch.isfinite(out).all()


def test_counterfactual_label_sign_and_no_grad():
    target = torch.zeros(1, 3, 16, 16)
    off = torch.full_like(target, 0.4, requires_grad=True)
    on = torch.full_like(target, 0.1, requires_grad=True)
    benefit, label = utility_target(off, on, target, delta=0.001, window=5)
    assert (benefit > 0).all()
    assert label.min() == 1
    assert not benefit.requires_grad
    _, reverse = utility_target(on, off, target, delta=0.001, window=5)
    assert reverse.max() == 0


def test_same_noise_and_null_prior_are_reproducible():
    model = RestorationSystem(width=8).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        a, _ = model.restore(x, steps=2, gate_mode="none", seed=3)
        b, _ = model.restore(x, steps=2, gate_mode="none", seed=3)
        c, _ = model.restore(x, steps=2, gate_mode="none", seed=3, corruption="shift")
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, c)
