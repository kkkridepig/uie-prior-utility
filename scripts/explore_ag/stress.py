"""Source-validation prior interventions; never used to revise frozen test selection."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.priors.kernels import pad_image, sobel, haar
from mpa_diff.metrics.image import psnr, ssim
from mpa_diff.utils.io import read_image, sha256
from scripts.explore_ag.dispatch import atomic_json
from scripts.explore_ag.evaluate import load_experiment, fixed_rows
from scripts.explore_ag.sampling import sample_fixed, keyed_noise


CASES = ('clean', 'depth_shift_002', 'depth_shift_005', 'depth_noise_005',
         'depth_noise_010', 'depth_missing', 'histogram_mismatch',
         'histogram_missing', 'highfreq_noise_001', 'highfreq_noise_003')


def depth_intervention(value, valid, case, sample_id):
    if case.startswith('depth_shift_'):
        fraction = .02 if case.endswith('002') else .05
        offset = max(1, round(value.shape[-1]*fraction))
        # Replicate border, no circular wraparound across unrelated edges.
        return F.pad(value, (offset, 0, 0, 0), mode='replicate')[..., :-offset]
    if case.startswith('depth_noise_'):
        sigma = .05 if case.endswith('005') else .10
        noise = keyed_noise((value.shape[0], 1, 16, 16), value.device,
                            sample_id, case)
        noise = F.interpolate(noise, size=value.shape[-2:], mode='bilinear', align_corners=False)
        return torch.where(valid.bool(), (value+sigma*noise).clamp(0, 1), value)
    if case == 'depth_missing':
        return torch.zeros_like(value)
    return value


def corrupt_static(static, image, provider, case, sample_id, mismatch):
    altered = copy.deepcopy(static)
    depth = altered['depth']
    if case.startswith('depth_'):
        distance = depth_intervention(depth.distance_proxy, depth.valid_mask, case, sample_id)
        depth.distance_proxy = distance
        legacy = depth.metadata['coordinate_kind'] == 'legacy_inverse_normalized'
        depth.physical_coordinate = 1-distance if legacy else distance
        depth.physical_coordinate = torch.where(depth.valid_mask.bool(), depth.physical_coordinate,
                                                torch.zeros_like(distance))
        if case == 'depth_missing':
            depth.valid_mask.zero_()
            depth.confidence.zero_()
            depth.physical_coordinate.zero_()
    elif case == 'histogram_mismatch':
        altered['histogram'] = mismatch
    elif case == 'histogram_missing':
        altered['histogram'].zero_()
    elif case.startswith('highfreq_noise_'):
        sigma = .01 if case.endswith('001') else .03
        noisy = (image + sigma*keyed_noise(image.shape, image.device, sample_id, case)).clamp(0, 1)
        altered['edges'], altered['wavelet'] = sobel(noisy), haar(noisy)[1]
    return altered


@torch.no_grad()
def run(args):
    model, state, branch = load_experiment(args.checkpoint, args.device)
    torch.set_num_threads(state['config']['runtime']['threads'])
    cfg = model.config
    base = model if branch == 'PARENT' else model.base
    rows = fixed_rows(cfg['data']['manifest'], 'val', 24)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'checkpoint_sha256': sha256(args.checkpoint), 'split': 'source_val',
                'sample_ids': [r['sample_id'] for r in rows], 'cases': list(CASES),
                'scopes': ['all_priors'] + (['added_route_only'] if branch.startswith('C_') else []),
                'solver': args.solver, 'nfe': args.steps, 'selection_use': False,
                'implementation_sha256': sha256(__file__),
                'timing': 'Supplemental protocol fixed after initial clean validation; not preregistered before training'}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Stress protocol changed')
    else:
        atomic_json(out/'protocol.json', protocol)
    path = out/'per_image.jsonl'
    records = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done = {(r['sample_id'], r['scope'], r['case']) for r in records}
    root = Path(cfg['data']['data_root'])
    for i, row in enumerate(rows):
        image = read_image(root/row['image_path'], cfg['data']['resize_hw']).to(args.device)[None]
        target = read_image(root/row['reference_path'], cfg['data']['resize_hw']).to(args.device)
        image, info = pad_image(image)
        static = base.priors.static(image)
        next_image = read_image(root/rows[(i+1)%len(rows)]['image_path'], cfg['data']['resize_hw']).to(args.device)[None]
        next_image, _ = pad_image(next_image)
        mismatch = base.priors.static(next_image)['histogram']
        clean_condition, clean_prior = model.condition(image, static)
        for scope in protocol['scopes']:
            for case in CASES:
                key = (row['sample_id'], scope, case)
                if key in done:
                    continue
                if time.time() >= args.deadline_unix-900:
                    atomic_json(out/'progress.json', {'status': 'budget_stopped', 'records': len(records)})
                    return
                altered = corrupt_static(static, image, base.priors, case, row['sample_id'], mismatch)
                condition, prior = model.condition(image, altered)
                handles = []
                if scope == 'added_route_only':
                    def change_extras(module, inputs):
                        h, extras, t = inputs
                        extras = dict(extras)
                        extras.update(depth=prior.depth.distance_proxy, depth_valid=prior.depth.valid_mask,
                                      histogram=condition['histogram'], highfreq=condition['highfreq'],
                                      histogram_missing=case == 'histogram_missing')
                        return h, extras, t
                    handles = [net.register_forward_pre_hook(change_extras) for net in model.injected.controls.values()]
                    used_condition, used_prior = clean_condition, clean_prior
                else:
                    used_condition, used_prior = condition, prior
                    used_prior.metadata['histogram_missing'] = case == 'histogram_missing'
                denoiser = model.denoiser if branch == 'PARENT' else lambda x,t,c: model.predict(x,t,c,used_prior)
                try:
                    prediction, _ = sample_fixed(denoiser, used_condition, image.shape, model.schedule,
                        row['sample_id'], args.solver, args.steps, cfg['sampler']['clip_intermediate_x0'])
                finally:
                    for handle in handles:
                        handle.remove()
                prediction = prediction[0, :, :info['hw'][0], :info['hw'][1]].clamp(0,1)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite stress output')
                record = {'sample_id': row['sample_id'], 'scene_id': row['scene_id'],
                          'scope': scope, 'case': case, 'psnr': psnr(prediction,target),
                          'ssim': ssim(prediction,target)}
                with path.open('a') as f:
                    f.write(json.dumps(record)+'\n')
                records.append(record)
    summary = {}
    for scope in protocol['scopes']:
        clean = {r['sample_id']: r['psnr'] for r in records if r['scope']==scope and r['case']=='clean'}
        summary[scope] = {case: {'count': len([r for r in records if r['scope']==scope and r['case']==case]),
            'mean_psnr': float(np.mean([r['psnr'] for r in records if r['scope']==scope and r['case']==case])),
            'mean_drop_db': float(np.mean([clean[r['sample_id']]-r['psnr'] for r in records if r['scope']==scope and r['case']==case]))}
            for case in CASES}
    atomic_json(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--solver', default='ddim', choices=['ddim','ddpm'])
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--device', default='cuda')
    p.add_argument('--deadline-unix', type=float, required=True)
    run(p.parse_args())
