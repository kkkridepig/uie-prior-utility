"""Charged train-only inference diagnosis; never changes the frozen protocol."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import ROOT, RUN, deadline, roles, setup, write
from scripts.cde_v3.eval_development import load_branch
from scripts.cde_v3.metrics import Metrics
from scripts.cde_v3.sampling import sample
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    setup()
    checkpoint = RUN / 'profiles/C_BANK/last.pt'
    row = roles('adapter_fit')[0]
    metric = Metrics('cuda')
    cases = []
    baseline = None
    options = [
        ('frozen_backend', False, True, False, 1),
        ('benchmark_deterministic', True, True, False, 1),
        ('benchmark_nondeterministic', True, False, False, 1),
        ('channels_last_deterministic', True, True, True, 1),
        ('batch4_deterministic', False, True, False, 4),
    ]
    for name, benchmark, deterministic, channels_last, batch in options:
        if deadline():
            sys.exit(75)
        model, _ = load_branch(checkpoint)
        cfg = model.config
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        x = read_image(ROOT / cfg['data']['data_root'] / row['image_path'],
                       cfg['data']['resize_hw'])[None].cuda().repeat(batch, 1, 1, 1)
        if channels_last:
            model.to(memory_format=torch.channels_last)
            x = x.contiguous(memory_format=torch.channels_last)
        result = dict(name=name, batch=batch, benchmark=benchmark,
                      deterministic=deterministic, channels_last=channels_last,
                      precision='float32_no_AMP', role='adapter_fit')
        try:
            times = []
            model_times = []
            # Batching here is only a throughput probe. The first noise tensor is
            # identical to batch1; repeated images are not research observations.
            for iteration in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                static = static_cached(model.base.priors, x, cfg)
                cond, extra = model.prepare(x, static)
                enc = model.encode(extra)
                pred, diag = sample(lambda xt, t, c: model.predict(xt, t, c, enc, 'all'),
                                    cond, x.shape, model.schedule, row['sample_id'], 101, 20)
                pred = pred.clamp(0, 1)
                torch.cuda.synchronize()
                model_seconds = time.perf_counter() - start
                metric(pred[0], x[0])
                torch.cuda.synchronize()
                if iteration >= 2:
                    times.append(time.perf_counter() - start)
                    model_times.append(model_seconds)
            assert diag['nfe_measured'] == 20
            assert torch.isfinite(pred).all()
            if baseline is None:
                baseline = pred[:1].detach().clone()
            difference = pred[:1] - baseline
            result.update(model_seconds=model_times,
                          one_metric_plus_model_seconds=times,
                          model_p95_seconds_per_image=float(np.quantile(model_times, .95)) / batch,
                          first_image_max_abs_difference=difference.abs().max().item(),
                          first_image_rms_difference=difference.square().mean().sqrt().item(),
                          status='measured')
        except (RuntimeError, NotImplementedError) as exc:
            result.update(status='unsupported', error=str(exc))
        cases.append(result)
        write(RUN / 'inference_options_diagnostic.json', dict(
            checkpoint=str(checkpoint), sample_id=row['sample_id'], cases=cases,
            protocol_changed=False, no_holdout_scores_accessed=True,
            metric_timing_note='batch4 has one metric; per-image model throughput only'))
        print(result, flush=True)
        del model, x
        torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
