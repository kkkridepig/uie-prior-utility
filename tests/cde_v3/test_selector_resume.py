import torch

from scripts.cde_v3.model import Selector
from scripts.cde_v3.train_selector import selector_step, save_selector


def test_nonperiodic_checkpoint_retains_optimizer_and_exact_next_update(tmp_path):
    torch.manual_seed(17)
    image = torch.rand(2, 3, 64, 64)
    summaries = [torch.rand(2, 128) for _ in range(3)]
    stats = torch.ones(2, 4)
    targets = torch.rand(2, 5)
    def initialize():
        torch.manual_seed(29)
        m = Selector()
        return m, torch.optim.Adam(m.parameters(), lr=.001)
    contiguous, co = initialize()
    split, so = initialize()
    for _ in range(20): selector_step(contiguous, co, image, summaries, stats, targets, 'UTILITY')
    for _ in range(7): selector_step(split, so, image, summaries, stats, targets, 'UTILITY')
    save_selector(tmp_path, {'profiling': True}, split, so, 7)
    checkpoint = torch.load(tmp_path/'last.pt', map_location='cpu')
    restored, ro = initialize()
    restored.load_state_dict(checkpoint['model']); ro.load_state_dict(checkpoint['optimizer'])
    assert checkpoint['step'] == 7
    assert all(state['step'] == 7 for state in ro.state_dict()['state'].values())
    for _ in range(checkpoint['step'], 20): selector_step(restored, ro, image, summaries, stats, targets, 'UTILITY')
    assert all(torch.equal(a, b) for a, b in zip(contiguous.parameters(), restored.parameters()))
    for i, state in co.state_dict()['state'].items():
        assert all(torch.equal(v, ro.state_dict()['state'][i][k]) for k, v in state.items())
