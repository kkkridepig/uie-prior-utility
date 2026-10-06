"""Validate inference layout on train-only images and measure the complete cost matrix."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.eval_development import load_branch
from scripts.cde_v3.inference import inference_model, INFERENCE_RUNTIME
from scripts.cde_v3.model import V3Model
from scripts.cde_v3.metrics import Metrics
from scripts.cde_v3.sampling import sample
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image


@torch.no_grad()
def measure(model, x, y, row, metric, mode):
    torch.cuda.synchronize()
    start = time.perf_counter()
    static = static_cached(model.base.priors, x, model.config)
    cond, extra = model.prepare(x, static)
    encoded = model.encode(extra)
    pred, diagnostic = sample(lambda xt, t, c: model.predict(xt, t, c, encoded, mode),
                              cond, x.shape, model.schedule, row['sample_id'], 101, 20)
    pred = pred[0].clamp(0, 1)
    torch.cuda.synchronize()
    model_seconds = time.perf_counter() - start
    scores = metric(pred, y)
    torch.cuda.synchronize()
    return pred, scores, model_seconds, time.perf_counter() - start


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', required=True)
    p.parse_args()
    setup()
    metric = Metrics('cuda')
    rows = sorted(roles('adapter_fit'), key=lambda r: key(20261004, 'layout_preflight', r['sample_id']))[:8]
    thresholds = {'max_abs': .002, 'rms': .0002, 'psnr': .01, 'ssim': .0002, 'lpips': .001}
    evidence = []
    timings = {}
    null_exact = True
    for branch in ['BASE_CONT_V3', 'C_BANK', 'C_RGB_CONTROL', 'C_ALL_ONLY', 'PARENT']:
        if deadline():
            sys.exit(75)
        if branch == 'PARENT':
            model = V3Model(torch.load(PARENT, map_location='cpu'), 'ANCHOR', 20261004).cuda().eval()
        else:
            model, _ = load_branch(RUN / 'profiles' / branch / 'last.pt', optimized=False)
        cfg = model.config
        timings[branch] = []
        for i, row in enumerate(rows):
            x = read_image(ROOT / cfg['data']['data_root'] / row['image_path'], cfg['data']['resize_hw'])[None].cuda()
            y = read_image(ROOT / cfg['data']['data_root'] / row['reference_path'], cfg['data']['resize_hw']).cuda()
            mode = 'all' if model.adapters else 'null'
            model.to(memory_format=torch.contiguous_format)
            before, bs, _, _ = measure(model, x, y, row, metric, mode)
            inference_model(model)
            after, aps, seconds, total = measure(model, x, y, row, metric, mode)
            difference = after - before
            r = {'branch': branch, 'sample_id': row['sample_id'],
                 'max_abs': difference.abs().max().item(),
                 'rms': difference.square().mean().sqrt().item(),
                 'metric_abs_deltas': {k: abs(aps[k] - bs[k]) for k in ('psnr', 'ssim', 'lpips')},
                 'model_seconds': seconds, 'model_plus_metrics_seconds': total}
            r['pass'] = r['max_abs'] <= thresholds['max_abs'] and r['rms'] <= thresholds['rms'] and all(
                r['metric_abs_deltas'][k] <= thresholds[k] for k in ('psnr', 'ssim', 'lpips'))
            evidence.append(r)
            if i >= 2:
                timings[branch].append(total)
            if branch == 'C_BANK':
                static = static_cached(model.base.priors, x, cfg)
                cond, extra = model.prepare(x, static)
                encoded = model.encode(extra)
                a, _ = sample(lambda xt, t, c: model.predict(xt, t, c, encoded, 'null'),
                              cond, x.shape, model.schedule, row['sample_id'], 101, 20)
                b, _ = sample(model.base.denoiser, cond, x.shape, model.schedule,
                              row['sample_id'], 101, 20)
                null_exact = null_exact and torch.equal(a, b)
            print(r, flush=True)
        del model
        torch.cuda.empty_cache()
    measured = {k: float(np.quantile(v, .95)) for k, v in timings.items()}
    result = {'runtime': INFERENCE_RUNTIME, 'runtime_sha256': sha(Path(__file__).with_name('inference.py')),
              'role': 'adapter_fit', 'evidence': evidence, 'tolerances_frozen_before_measurement': thresholds,
              'hard_null_full_DDIM20_exact': null_exact, 'passed': null_exact and all(r['pass'] for r in evidence),
              'model_plus_metrics_p95_seconds': measured, 'raw_timings': timings,
              'no_holdout_scores_accessed': True, 'training_unchanged': True}
    write(RUN / 'inference_layout_validation.json', result)
    if not result['passed']:
        raise RuntimeError('Layout failed numerical contract; do not dispatch')
    print(result['model_plus_metrics_p95_seconds'], flush=True)


if __name__ == '__main__':
    main()
