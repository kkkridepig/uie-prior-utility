import copy

import torch

from mpa_diff.config import DEFAULT
from mpa_diff.priors.provider import MPADiff
from scripts.explore_ag.branches import ExplorationModel, phase_spatial


def parent_fixture():
    config = copy.deepcopy(DEFAULT)
    config['depth']['provider'] = 'synthetic_fixture'
    config['runtime']['test_fixture'] = True
    config['experiment']['evidence_profile'] = 'synthetic_test_only'
    base = MPADiff(config)
    return {'config': config, 'model': base.state_dict()}


def test_phase_zero_and_odd_shapes_are_finite():
    for shape in ((1, 3, 15, 17), (2, 3, 18, 21)):
        result = phase_spatial(torch.zeros(shape))
        assert result.shape == shape
        assert torch.isfinite(result).all()
        assert torch.count_nonzero(result) == 0


def test_a_split_starts_identically_to_coupled_renderer():
    parent = parent_fixture()
    coupled = ExplorationModel(parent, 'A_COUPLED').eval()
    split = ExplorationModel(parent, 'A_SPLIT').eval()
    image = torch.rand(1, 3, 24, 24)
    with torch.no_grad():
        ca, pa = coupled.condition(image)
        cb, pb = split.condition(image)
    assert torch.allclose(ca['physical'], cb['physical'], atol=1e-6)
    assert torch.allclose(pa.fields['t_D'], pb.fields['t_D'], atol=1e-6)
    assert torch.allclose(pa.fields['b'], pb.fields['b'], atol=1e-6)


def test_controls_keep_parent_path_and_have_gradients():
    parent = parent_fixture()
    image = torch.rand(1, 3, 24, 24)
    state = torch.rand_like(image)
    index = torch.tensor([10])
    for branch in ('B_PROXY', 'C_CONV', 'C_FIXED', 'C_ROUTED',
                   'D_PHASE', 'D_SOBEL_CONTROL', 'F_UNIFORM', 'F_ROUTED'):
        model = ExplorationModel(parent, branch)
        condition, prior = model.condition(image)
        prediction = model.predict(state, index, condition, prior)
        assert prediction.shape == image.shape
        assert torch.isfinite(prediction).all()
        prediction.mean().backward()
        assert any(p.grad is not None for p in model.base.denoiser.parameters())


def test_all_invalid_additive_route_recovers_parent_feature():
    parent = parent_fixture()
    model = ExplorationModel(parent, 'C_ROUTED').eval()
    image = torch.rand(1, 3, 24, 24)
    state = torch.rand_like(image)
    index = torch.tensor([10])
    condition, prior = model.condition(image)
    prior.depth.valid_mask.zero_()
    prior.metadata.update(histogram_missing=True, highfreq_missing=True)
    with torch.no_grad():
        routed = model.predict(state, index, condition, prior)
        baseline = model.base.denoiser(state, index, condition)
    assert torch.equal(routed, baseline)


def test_new_branch_initialization_preserves_public_rng_stream():
    parent = parent_fixture()
    states = []
    for branch in ('BASE_CONT', 'D_PHASE', 'D_SOBEL_CONTROL', 'C_ROUTED', 'F_ROUTED'):
        torch.manual_seed(1234)
        ExplorationModel(parent, branch)
        states.append(torch.get_rng_state())
    assert all(torch.equal(states[0], state) for state in states[1:])
