import torch

from scripts.cde_v3.inference import inference_model
from scripts.cde_v3.update_cost_prediction import costs
from scripts.cde_v3.dispatch import command_identity
from scripts.cde_v3.common import ROOT


def test_layout_preserves_parameter_values_and_freezing():
    model = torch.nn.Conv2d(3, 8, 3).requires_grad_(False)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    inference_model(model)
    assert model.weight.is_contiguous(memory_format=torch.channels_last)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_full_matrix_budget_gate_blocks_slow_inference():
    branches = ('BASE_CONT_V3', 'C_BANK', 'C_RGB_CONTROL', 'C_ALL_ONLY')
    old = {'training_seconds_5000': {k: 3500 for k in branches}}
    slow = {k: 1.3 for k in (*branches, 'PARENT')}
    fast = {k: .3 for k in slow}
    blocked = costs(old, slow, 91, 141, 71, 915)
    approved = costs(old, fast, 91, 141, 71, 915)
    assert not blocked['dispatch_allowed']
    assert approved['dispatch_allowed']
    assert approved['final_named_models_per_seed'] == 9
    assert approved['final_image_count'] == 915
    assert blocked['stage_seconds_per_seed']['training_all_five_enhancers'] == approved['stage_seconds_per_seed']['training_all_five_enhancers']


def test_resume_command_identity_preserves_args():
    relative = ['.venv/bin/python', 'scripts/cde_v3/profile_selector.py', '--run']
    absolute = [str(ROOT/'.venv/bin/python'), *relative[1:]]
    assert command_identity(relative) == command_identity(absolute)
    assert command_identity(relative) != command_identity(absolute + ['--different'])
