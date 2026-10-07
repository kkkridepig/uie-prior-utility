"""Bounded V3 plumbing check, without optimizer updates or model checkpoints."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mpa_diff.config import DEFAULT
from mpa_diff.priors.provider import MPADiff
from scripts.cde_v3.common import noise, write
from scripts.cde_v3.model import V3Model, Selector, token_summary, phase_feature
from scripts.cde_v3.sampling import sample


def state_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a recovery receipt: ' + str(output))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('PPU unavailable; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.monotonic()
    cfg = copy.deepcopy(DEFAULT)
    cfg['depth']['provider'] = 'synthetic_fixture'
    cfg['model']['dropout'] = 0.0
    # This random parent exists only in memory and is never a historical weight.
    parent = MPADiff(cfg)
    model = V3Model({'config': cfg, 'model': parent.state_dict()}, 'C_BANK', 20261004)
    del parent
    model = model.to(args.device).eval()
    original = state_digest(model.base)
    image = torch.rand(1, 3, 32, 32, device=args.device)
    state = noise(image.shape, 101, 'recovery_fixture', device=args.device)
    index = torch.tensor([450], device=args.device)
    cond, extras = model.prepare(image)
    with torch.no_grad():
        ref = model.base.denoiser(state, index, cond)
        encoded = model.encode(extras)
        for mode in ('null', 'physical', 'histogram', 'highfreq', 'all'):
            assert torch.equal(ref, model.predict(state, index, cond, encoded, mode)), mode
    # Backward only: no optimizer is constructed and no parameter update occurs.
    pred = model.predict(state, index, cond, model.encode(extras), 'all')
    pred.square().mean().backward()
    grads = {name: float(p.grad.detach().abs().sum().cpu())
             for name, p in model.adapters.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in model.adapters.parameters() if p.grad is not None)
    assert any(value > 0 for name, value in grads.items() if 'outputs.' in name)
    assert all(p.grad is None for p in model.base.parameters())
    assert state_digest(model.base) == original
    measured = {}
    for nfe in (1, 2, 4, 20):
        null = lambda xt, idx, c: model.predict(xt, idx, c, None, 'null')
        a, diag = sample(null, cond, image.shape, model.schedule, 'recovery_fixture', 101, nfe)
        b, _ = sample(model.base.denoiser, cond, image.shape, model.schedule, 'recovery_fixture', 101, nfe)
        assert torch.equal(a, b) and torch.isfinite(a).all()
        assert diag['nfe_measured'] == nfe
        measured[str(nfe)] = diag
    assert torch.equal(state, noise(image.shape, 101, 'recovery_fixture', device=args.device))
    assert not torch.equal(state, noise(image.shape, 102, 'recovery_fixture', device=args.device))
    selector = Selector().to(args.device).eval()
    with torch.no_grad():
        scores = selector(image, token_summary(encoded[0]), extras[2])
        assert scores.shape == (1, 5) and torch.isfinite(scores).all()
        for kind in ('sobel', 'phase', 'softphase'):
            value, _ = phase_feature(image, kind)
            assert torch.isfinite(value).all()
            value, _ = phase_feature(torch.ones(1, 3, 17, 19, device=args.device), kind)
            assert value.abs().max() < 1e-6
    if args.device == 'cuda':
        torch.cuda.synchronize()
    report = {
        'status': 'passed', 'device': args.device,
        'device_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU',
        'torch': torch.__version__, 'torch_path': torch.__file__,
        'shape': list(image.shape), 'precision': 'float32',
        'parent_source': 'random_in_memory_test_fixture_only',
        'depth_source': 'synthetic_fixture_only',
        'historical_parent_loaded': False, 'scientific_result': False,
        'optimizer_updates': 0, 'checkpoints_written': 0,
        'frozen_parent_unchanged': True, 'zero_adapter_exact': True,
        'null_trajectory_exact': True, 'noise_seeds_distinct': True,
        'selector_shape': list(scores.shape), 'samplers': measured,
        'adapter_gradient_sums': grads,
        'elapsed_seconds': time.monotonic() - started,
    }
    write(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
