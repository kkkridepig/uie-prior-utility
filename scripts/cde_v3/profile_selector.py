"""PPU selector throughput/resume with real training features and zero profiling targets."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.eval_development import load_branch
from scripts.cde_v3.model import Selector, token_summary
from scripts.cde_v3.train_selector import selector_step, save_selector
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', required=True)
    p.parse_args()
    setup()
    model, _ = load_branch(RUN/'profiles/C_BANK/last.pt')
    row = roles('route_fit', 'labels')[0]
    cfg = model.config
    with torch.no_grad():
        x = read_image(ROOT/cfg['data']['data_root']/row['image_path'], cfg['data']['resize_hw'])[None].cuda()
        cond, extra = model.prepare(x, static_cached(model.base.priors, x, cfg))
        enc = model.encode(extra)[0]
        images = torch.nn.functional.interpolate(x, (64, 64), mode='bilinear', align_corners=False).repeat(32, 1, 1, 1)
        summaries = [z.repeat(32, 1) for z in token_summary(enc)]
        stats = extra[2].repeat(32, 1)
        targets = torch.zeros(32, 5, device='cuda')
    del model
    results = {}
    for kind in ('UTILITY', 'WINNER_CE', 'SHUFFLED'):
        torch.manual_seed(key(20261004, 'gate_init'))
        gate = Selector().cuda()
        opt = optimizer(gate)
        times = []
        for step in range(100):
            if deadline(): sys.exit(75)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = selector_step(gate, opt, images, summaries, stats, targets, kind)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        results[kind] = {'updates': 100, 'p95_seconds': float(np.quantile(times[10:], .95)),
                         'median_seconds': float(np.median(times[10:])), 'last_loss': loss}
    # Save on a non-periodic boundary and compare restored optimizer/parameters
    # with the same contiguous updates, using the actual shared update routine.
    def new_gate():
        torch.manual_seed(key(20261004, 'gate_init'))
        g = Selector().cuda()
        return g, optimizer(g)
    contiguous, co = new_gate()
    split, so = new_gate()
    for _ in range(20): selector_step(contiguous, co, images, summaries, stats, targets, 'UTILITY')
    for _ in range(7): selector_step(split, so, images, summaries, stats, targets, 'UTILITY')
    out = RUN/'profiles/selector_resume'
    out.mkdir(parents=True, exist_ok=True)
    ident = {'profiling_only': True, 'zero_targets_not_scientific_labels': True}
    save_selector(out, ident, split, so, 7)
    cp = torch.load(out/'last.pt', map_location='cpu')
    restored, ro = new_gate()
    restored.load_state_dict(cp['model']); ro.load_state_dict(cp['optimizer'])
    assert cp['step'] == 7
    for _ in range(7, 20): selector_step(restored, ro, images, summaries, stats, targets, 'UTILITY')
    pd = max((a-b).abs().max().item() for a,b in zip(contiguous.parameters(), restored.parameters()))
    od = max((v-ro.state_dict()['state'][i][k]).abs().max().item()
             for i,state in co.state_dict()['state'].items() for k,v in state.items() if torch.is_tensor(v))
    assert pd <= 1e-5 and od <= 1e-5
    bound = read(RUN/'cost_prediction.json')['selector_step_seconds_bound']
    assert max(r['p95_seconds'] for r in results.values()) <= bound, 'Selector exceeds predispatch cost bound'
    write(RUN/'selector_profile_resume.json', {'profiles': results, 'resume_7_plus_13_vs_20': {
        'max_parameter_delta': pd, 'max_optimizer_delta': od, 'tolerance': 1e-5, 'passed': True},
        'profiling_only': True, 'targets': 'zero; no measured utility labels before5000',
        'features_role': 'route_fit', 'scientific_training_updates': 0, 'budget_bound_seconds': bound,
        'code_sha256': sha(Path(__file__)), 'training_step_code_sha256': sha(ROOT/'scripts/cde_v3/train_selector.py')})
    print(results, pd, od, flush=True)


if __name__ == '__main__':
    main()
