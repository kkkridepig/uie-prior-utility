"""One-time fixed-model paired and no-reference benchmark after selection freeze."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.data.manifest import read_manifest
from mpa_diff.metrics.image import psnr, ssim, finite_json
from mpa_diff.priors.kernels import pad_image
from mpa_diff.utils.io import read_image, save_image, sha256, write_json
from scripts.explore_ag.evaluate import load_experiment
from scripts.explore_ag.sampling import sample_fixed


ROOT = Path(__file__).resolve().parents[2]


def benchmark_rows(dataset, cfg):
    if dataset == 'uieb_test':
        rows = [r for r in read_manifest(cfg['data']['manifest']) if r['split'] == 'test']
        return rows, Path(cfg['data']['data_root']), sha256(cfg['data']['manifest'])
    if dataset == 'lsui_test':
        manifest = ROOT/'manifests/lsui_recon_grouped_v1.jsonl'
        rows = [r for r in read_manifest(manifest) if r['split'] == 'test']
        return rows, ROOT/'data/lsui', sha256(manifest)
    if dataset == 'u45':
        root = ROOT/'data/benchmarks/U45/underwater-test-dataset-U45--master/upload/U45/U45'
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
        rows = [{'sample_id': 'U45/'+p.stem, 'scene_id': None, 'image_path': p.name,
                 'reference_path': None, 'split': 'test'} for p in files]
        return rows, root, None
    raise ValueError('Unknown benchmark dataset')


@torch.no_grad()
def run(args):
    freeze = json.loads(Path(args.freeze).read_text())
    if not freeze.get('frozen_before_test'):
        raise ValueError('Model/protocol must be frozen before test access')
    if sha256(args.checkpoint) not in freeze['allowed_checkpoint_sha256']:
        raise ValueError('Checkpoint not on predeclared fixed Benchmark list')
    model, state, branch = load_experiment(args.checkpoint, args.device)
    rows, root, manifest_hash = benchmark_rows(args.dataset, model.config)
    if not rows:
        raise ValueError('Benchmark has no images')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'dataset': args.dataset, 'checkpoint': str(Path(args.checkpoint).resolve()),
                'checkpoint_sha256': sha256(args.checkpoint), 'branch': branch,
                'solver': freeze['solver'], 'nfe': freeze['nfe'],
                'manifest_sha256': manifest_hash, 'sample_ids': [r['sample_id'] for r in rows],
                'freeze_sha256': sha256(args.freeze)}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Existing Benchmark protocol mismatch')
    else:
        write_json(out/'protocol.json', protocol)
    records = [json.loads(line) for line in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
    done = {r['sample_id'] for r in records}
    dev = next(model.parameters()).device
    for row in rows:
        if row['sample_id'] in done:
            continue
        if args.deadline_unix and time.time() >= args.deadline_unix - 900:
            write_json(out/'progress.json', {'status':'budget_stopped', 'done':len(records), 'total':len(rows)})
            return None
        x = read_image(root/row['image_path'], model.config['data']['resize_hw']).to(dev)[None]
        if dev.type == 'cuda': torch.cuda.synchronize()
        begin = time.perf_counter()
        padded, info = pad_image(x)
        condition, prior = model.condition(padded)
        denoiser = model.denoiser if branch == 'PARENT' else (lambda xt, index, cond: model.predict(xt, index, cond, prior))
        pred, diag = sample_fixed(denoiser, condition, padded.shape, model.schedule,
                                  row['sample_id'], freeze['solver'], freeze['nfe'],
                                  model.config['sampler']['clip_intermediate_x0'])
        pred = pred[..., :info['hw'][0], :info['hw'][1]].clamp(0,1)[0]
        if dev.type == 'cuda': torch.cuda.synchronize()
        target = read_image(root/row['reference_path'], model.config['data']['resize_hw']).to(dev) if row['reference_path'] else None
        record = {'sample_id': row['sample_id'], 'scene_id': row['scene_id'],
                  'dataset': args.dataset, 'branch': branch, 'split': 'test',
                  'raw_psnr': finite_json(psnr(x[0], target)) if target is not None else None,
                  'raw_ssim': ssim(x[0], target) if target is not None else None,
                  'psnr': finite_json(psnr(pred,target)) if target is not None else None,
                  'ssim': ssim(pred,target) if target is not None else None,
                  'nfe': diag['nfe'], 'seconds': time.perf_counter()-begin,
                  'prediction_path': str((out/'images'/(row['sample_id'].replace('/','__')+'.png')).resolve())}
        save_image(record['prediction_path'], pred)
        with (out/'per_image.jsonl').open('a') as file:
            file.write(json.dumps(record)+'\n')
            file.flush()
        records.append(record)
        write_json(out/'progress.json', {'status':'testing', 'done':len(records), 'total':len(rows)})
    scores = [r['psnr'] for r in records if isinstance(r['psnr'], (float,int))]
    summary = {'count':len(records), 'dataset':args.dataset, 'branch':branch,
               'raw_mean_psnr':float(np.mean([r['raw_psnr'] for r in records])) if scores else None,
               'raw_mean_ssim':float(np.mean([r['raw_ssim'] for r in records])) if scores else None,
               'mean_psnr':float(np.mean(scores)) if scores else None,
               'mean_ssim':float(np.mean([r['ssim'] for r in records])) if scores else None,
               'median_seconds':float(np.median([r['seconds'] for r in records])),
               'p95_seconds':float(np.percentile([r['seconds'] for r in records], 95)),
               'missing_metrics':{'lpips':'weight not provisioned and audited',
                                  'uciqe':'formula implementation not independently audited',
                                  'uiqm':'formula implementation not independently audited'}
                                  if args.dataset == 'u45' else {}}
    write_json(out/'summary.json', summary)
    if scores:
        write_json(out/'worst_cases.json', sorted(records, key=lambda r: r['psnr'])[:10])
    write_json(out/'progress.json', {'status':'completed', 'done':len(records), 'total':len(rows)})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', choices=['uieb_test','lsui_test','u45'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    print(json.dumps(run(args)), flush=True)


if __name__ == '__main__':
    main()
