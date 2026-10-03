"""Resumable source-validation evaluation for parent and screening branches."""
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
from scripts.explore_ag.branches import ExplorationModel
from scripts.explore_ag.sampling import sample_fixed


def fixed_rows(manifest, split, limit):
    if split != 'val':
        raise ValueError('Development evaluator accepts source validation only')
    rows = [r for r in read_manifest(manifest) if r['split'] == split]
    version = sha256(manifest)
    ordered = sorted(rows, key=lambda r: hashlib.sha256((version+r['sample_id']).encode()).digest())
    return ordered[:limit] if limit else ordered


def load_experiment(path, device):
    state = torch.load(path, map_location='cpu')
    if 'branch' not in state:
        model, state = load_model(path, device)
        return model, state, 'PARENT'
    parent = torch.load(state['parent_checkpoint'], map_location='cpu')
    if sha256(state['parent_checkpoint']) != state['parent_checkpoint_sha256']:
        raise ValueError('Parent weight changed')
    model = ExplorationModel(parent, state['branch'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), state, state['branch']


@torch.no_grad()
def evaluate(checkpoint, output, solver, steps, limit=None, device='cuda', deadline_unix=None):
    model, state, branch = load_experiment(checkpoint, device)
    cfg = model.config
    rows = fixed_rows(cfg['data']['manifest'], 'val', limit)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': sha256(checkpoint),
                'parent_checkpoint_sha256': state.get('parent_checkpoint_sha256'),
                'branch': branch, 'step': state['step'], 'solver': solver, 'nfe': steps,
                'manifest_sha256': sha256(cfg['data']['manifest']),
                'sample_ids': [r['sample_id'] for r in rows], 'split': 'val'}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Existing evaluation uses another checkpoint/protocol')
    else:
        write_json(out/'protocol.json', protocol)
    records = [json.loads(line) for line in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
    done = {r['sample_id'] for r in records}
    dev = next(model.parameters()).device
    for row in rows:
        if row['sample_id'] in done:
            continue
        if deadline_unix and time.time() >= deadline_unix - 900:
            write_json(out/'progress.json', {'status': 'budget_stopped', 'done': len(records), 'total': len(rows)})
            return None
        x = read_image(Path(cfg['data']['data_root'])/row['image_path'], cfg['data']['resize_hw']).to(dev)[None]
        y = read_image(Path(cfg['data']['data_root'])/row['reference_path'], cfg['data']['resize_hw']).to(dev)
        if dev.type == 'cuda': torch.cuda.synchronize()
        begin = time.perf_counter()
        padded, info = pad_image(x)
        condition, prior = model.condition(padded)
        if branch == 'PARENT':
            denoiser = model.denoiser
        else:
            denoiser = lambda xt, index, cond: model.predict(xt, index, cond, prior)
        prediction, diag = sample_fixed(denoiser, condition, padded.shape, model.schedule,
                                        row['sample_id'], solver, steps,
                                        cfg['sampler']['clip_intermediate_x0'])
        prediction = prediction[..., :info['hw'][0], :info['hw'][1]].clamp(0, 1)[0]
        if dev.type == 'cuda': torch.cuda.synchronize()
        record = {'sample_id': row['sample_id'], 'scene_id': row['scene_id'],
                  'branch': branch, 'checkpoint_step': state['step'], 'split': 'val',
                  'psnr': finite_json(psnr(prediction, y)), 'ssim': ssim(prediction, y),
                  'nfe': diag['nfe'], 'seconds': time.perf_counter()-begin}
        with (out/'per_image.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
        records.append(record)
        write_json(out/'progress.json', {'status': 'validating', 'done': len(records), 'total': len(rows),
                                         'last_sample_id': row['sample_id']})
    scores = [r['psnr'] for r in records if isinstance(r['psnr'], (float, int))]
    summary = {'count': len(records), 'mean_psnr': float(np.mean(scores)),
               'mean_ssim': float(np.mean([r['ssim'] for r in records])),
               'median_seconds': float(np.median([r['seconds'] for r in records])),
               'branch': branch, 'step': state['step'], 'solver': solver, 'nfe': steps}
    write_json(out/'summary.json', summary)
    write_json(out/'worst_cases.json', sorted(records, key=lambda r: r['psnr'])[:10])
    write_json(out/'progress.json', {'status': 'completed', 'done': len(records), 'total': len(rows)})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--solver', choices=['ddpm','ddim'], required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.output, args.solver, args.steps,
                              args.limit, args.device, args.deadline_unix)), flush=True)


if __name__ == '__main__':
    main()
