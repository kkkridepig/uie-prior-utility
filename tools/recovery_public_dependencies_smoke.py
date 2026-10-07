"""Check public V3 dependencies on a training input; no historical model claim."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mpa_diff.data.manifest import read_manifest
from mpa_diff.engine.losses import VGG19Loss
from mpa_diff.priors.provider import MPADiff
from mpa_diff.utils.io import read_image
from scripts.cde_v3.common import sha, read, write, noise
from scripts.cde_v3.sampling import sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists() or (output.parent / 'metric_identity.json').exists():
        raise FileExistsError('Use a fresh public-dependency receipt directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('PPU unavailable')
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    started = time.monotonic()
    cfg = read(ROOT / 'runs/prior_utility_cde_v3_20261004/code_and_environment.json')['config']
    model = MPADiff(cfg).to(args.device).eval()
    if args.device == 'cpu':
        for name, module in list(sys.modules.items()):
            if name.startswith('depth_anything_v2.') and name.endswith('.attention'):
                module.XFORMERS_AVAILABLE = False
    row = next(r for r in read_manifest(ROOT / cfg['data']['manifest']) if r['split'] == 'train')
    image_path = ROOT / cfg['data']['data_root'] / row['image_path']
    image = read_image(image_path, cfg['data']['resize_hw'])[None].to(args.device)
    cond, priors = model.condition(image)
    assert torch.isfinite(priors.depth.raw).all() and priors.depth.valid_mask.any()
    for value in cond.values():
        assert torch.isfinite(value).all()
    state = noise(image.shape, 101, 'public_dependency_fixture', device=args.device)
    pred = model.denoiser(state, torch.tensor([450], device=args.device), cond)
    assert pred.shape == image.shape and torch.isfinite(pred).all()
    vgg = VGG19Loss(cfg['loss']['vgg_checkpoint'], cfg['loss']['vgg_sha256']).to(args.device)
    loss = (pred - image).square().mean() + .1 * vgg(pred, image)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in model.denoiser.parameters())
    assert all(v.grad is None for v in model.priors.depth.parameters())
    result, diag = sample(model.denoiser, cond, image.shape, model.schedule,
                          'public_dependency_fixture', 101, 2)
    assert result.shape == image.shape and torch.isfinite(result).all() and diag['nfe_measured'] == 2
    # The original constructor writes a receipt. Redirect only that receipt;
    # its frozen V3 source and historical metric identity stay unchanged.
    import scripts.cde_v3.metrics as metric_module
    old_run = metric_module.RUN
    old_identity = old_run / 'metric_identity.json'
    old_hash = sha(old_identity)
    try:
        metric_module.RUN = output.parent
        metrics = metric_module.Metrics(args.device)
        equal = metrics(image, image)
        perturbed = metrics(image * .95, image)
    finally:
        metric_module.RUN = old_run
    assert sha(old_identity) == old_hash
    assert abs(equal['lpips']) < 1e-6 and perturbed['lpips'] > 0
    if args.device == 'cuda':
        torch.cuda.synchronize()
    report = {
        'status': 'passed', 'device': args.device,
        'device_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU',
        'torch': torch.__version__, 'shape': list(image.shape),
        'input_sample_id': row['sample_id'], 'input_role': 'historical_train_plumbing_only',
        'input_sha256': sha(image_path), 'reference_image_read': False,
        'historical_parent_loaded': False, 'restoration_parameters': 'random_in_memory_only',
        'public_depth_and_vgg_weights_verified': True,
        'optimizer_updates': 0, 'checkpoints_written': 0, 'scientific_result': False,
        'vgg19_backward_finite': True, 'frozen_depth_gradient_absent': True,
        'ddim2': diag, 'lpips_self_check': equal, 'lpips_perturbation_check': perturbed,
        'historical_metric_identity_unchanged': True,
        'elapsed_seconds': time.monotonic() - started,
    }
    write(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
