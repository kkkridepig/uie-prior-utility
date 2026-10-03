"""Evaluate sampler quality and cost on a frozen source-validation subset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.data.manifest import read_manifest
from mpa_diff.engine.checkpoint import load_model
from mpa_diff.metrics.image import psnr, ssim, finite_json
from mpa_diff.priors.kernels import pad_image
from mpa_diff.utils.io import read_image, sha256, write_json
from scripts.explore_ag.sampling import sample_fixed


def fixed_rows(manifest, count=24):
    rows = [r for r in read_manifest(manifest) if r['split'] == 'val']
    version = sha256(manifest)
    return sorted(rows, key=lambda r: hashlib.sha256((version + r['sample_id']).encode()).digest())[:count]


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--full', action='store_true', help='Confirm an accepted sampler on the full source validation set')
    p.add_argument('--solvers', nargs='+', default=['ddpm:1000', 'ddim:100', 'ddim:50', 'ddim:20'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and (out/'summary.json').exists():
        print('Completed E0 result already exists: {}'.format(out), flush=True)
        return
    model, state = load_model(args.checkpoint, args.device)
    cfg = state['config']
    rows = fixed_rows(cfg['data']['manifest'], 100000 if args.full else 24)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'checkpoint': str(Path(args.checkpoint).resolve()),
                                    'checkpoint_sha256': sha256(args.checkpoint),
                                    'checkpoint_step': state['step'],
                                    'manifest_sha256': sha256(cfg['data']['manifest']),
                                    'samples': [r['sample_id'] for r in rows],
                                    'solvers': args.solvers, 'noise': 'sample-keyed SHA256; DDPM keyed by math timestep',
                                    'split': 'val', 'size': cfg['data']['resize_hw']}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Existing E0 protocol mismatch; use a new output')
    else:
        write_json(out/'protocol.json', protocol)
    records = [json.loads(line) for line in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
    completed = {(r['sample_id'], r['solver'], r['nfe']) for r in records}
    device = next(model.parameters()).device
    for i, row in enumerate(rows):
        if args.deadline_unix and time.time() >= args.deadline_unix - 900:
            write_json(out/'progress.json', {'status': 'budget_stopped',
                       'completed_pairs': len(records), 'total_pairs': len(rows)*len(args.solvers)})
            return
        x = read_image(Path(cfg['data']['data_root'])/row['image_path'], cfg['data']['resize_hw']).to(device)[None]
        y = read_image(Path(cfg['data']['data_root'])/row['reference_path'], cfg['data']['resize_hw']).to(device)
        sync(device)
        start = time.perf_counter()
        padded, info = pad_image(x)
        cond, _ = model.condition(padded)
        sync(device)
        prior_seconds = time.perf_counter() - start
        for label in args.solvers:
            if args.deadline_unix and time.time() >= args.deadline_unix - 900:
                write_json(out/'progress.json', {'status': 'budget_stopped',
                           'completed_pairs': len(records), 'total_pairs': len(rows)*len(args.solvers)})
                return
            name, steps_str = label.split(':')
            steps = int(steps_str)
            if (row['sample_id'], name, steps) in completed:
                continue
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            begin = time.perf_counter()
            pred, diag = sample_fixed(model.denoiser, cond, padded.shape, model.schedule,
                                      row['sample_id'], name, steps,
                                      cfg['sampler']['clip_intermediate_x0'])
            pred = pred[..., :info['hw'][0], :info['hw'][1]].clamp(0, 1)[0]
            sync(device)
            sample_seconds = time.perf_counter() - begin
            record = {'sample_id': row['sample_id'], 'scene_id': row['scene_id'], 'split': 'val',
                      'solver': name, 'nfe': diag['nfe'], 'psnr': finite_json(psnr(pred, y)),
                      'ssim': ssim(pred, y), 'prior_seconds': prior_seconds,
                      'sample_seconds': sample_seconds, 'end_to_end_seconds': prior_seconds + sample_seconds,
                      'peak_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
            records.append(record)
            with (out/'per_image.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        print(json.dumps({'completed_images': i+1, 'total_images': len(rows), 'sample_id': row['sample_id']}), flush=True)
    summary = {}
    for label in args.solvers:
        name, nfe = label.split(':')
        rs = [r for r in records if r['solver'] == name and r['nfe'] == int(nfe)]
        summary[label] = {'count': len(rs), 'mean_psnr': float(np.mean([r['psnr'] for r in rs])),
                          'mean_ssim': float(np.mean([r['ssim'] for r in rs])),
                          'median_end_to_end_seconds': float(np.median([r['end_to_end_seconds'] for r in rs])),
                          'median_sampling_seconds': float(np.median([r['sample_seconds'] for r in rs]))}
    write_json(out/'summary.json', summary)
    baseline = summary.get('ddpm:1000')
    if baseline:
        accepted = [label for label, score in summary.items() if label.startswith('ddim:')
                    and score['mean_psnr'] >= baseline['mean_psnr']-0.10
                    and score['mean_ssim'] >= baseline['mean_ssim']-0.002]
        recommendation = min(accepted, key=lambda label: int(label.split(':')[1])) if accepted else 'ddpm:1000'
        write_json(out/'recommendation.json', {'subset_recommendation': recommendation,
                                               'requires_full_validation_confirmation': not args.full,
                                               'thresholds': {'max_psnr_drop_db': 0.10, 'max_ssim_drop': 0.002}})


if __name__ == '__main__':
    main()
