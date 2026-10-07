"""Supplement selected-policy timing with the complete O path, after closeout.

If the deployed policy returns J0, its shortcut time is not the full mechanism
time. This isolated process executes backbone/prior/producer/controller anyway,
while keeping the final frozen policy and output. No quality/reference is read.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_next.records import ROOT, read, write, sha
from uie_next.backbones.ssuie import load_official
from uie_next.data.manifest import image_tensor
from uie_next.priors.heuristic import make_prior
from uie_next.models.controls import Controller
from uie_next.v2.context import State
from uie_next.v2.diagnostics import checkpoint_model
from uie_next.v2.evaluation import apply_policy


def worker(s):
    torch.set_num_threads(2)
    registry = read(s.run / 'method_registry.json')
    selected = read(s.run / 'selection/calibration_selection.json')
    policy = selected['methods']['O']['policy']; entry = registry['O']
    backbone, backbone_identity = load_official(ROOT, s.config['backbone']['checkpoint'],
                                               s.config['backbone']['checkpoint_sha256'])
    backbone.to('cuda:0'); candidate = checkpoint_model(entry['candidate'])
    controller = Controller('O').to('cuda:0')
    controller.load_state_dict(torch.load(entry['controller']['path'], map_location='cpu')['model_state'], strict=True)
    controller.eval().requires_grad_(False)
    scales = read(s.run / 'normalization_stats.json')
    import json
    rows = sorted([json.loads(x) for x in (s.run / 'roles.jsonl').read_text().splitlines()
                   if json.loads(x)['role'] == 'model_fit'], key=lambda r: r['sample_id'])[:20]

    def infer(image):
        base = backbone(image)
        field = make_prior(image)
        corrected = candidate(image, base, field['P'], field['V'])
        prediction = controller(image, base, corrected, se2=scales['s_e2'])
        return apply_policy('O', policy, base, corrected, prediction)[0]

    model_times = []; service_times = []
    image = image_tensor(rows[0]['input_path'])[None].to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(20):
            infer(image)
        for _ in range(100):
            torch.cuda.synchronize(); start = time.perf_counter()
            infer(image); torch.cuda.synchronize(); model_times.append(time.perf_counter() - start)
        for i, row in enumerate(rows):
            torch.cuda.synchronize(); start = time.perf_counter()
            x = image_tensor(row['input_path'])[None].to('cuda:0'); out = infer(x)
            torch.cuda.synchronize()
            arr = (out[0].cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
            path = s.run / 'timing/full_O_service_png' / ('%02d.png' % i)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)):
                raise IOError('PNG write failed')
            service_times.append(time.perf_counter() - start)
    summarize = lambda values: {'mean': float(np.mean(values)), 'p50': float(np.median(values)),
                                 'p95': float(np.quantile(values, .95))}
    write(s.run / 'timing/O_full_pipeline.json', {
        'method': 'O', 'measurement_kind': 'complete_pipeline_without_return_base_shortcut',
        'frozen_policy': policy, 'frozen_policy_output_preserved': True,
        'model': summarize(model_times), 'service': summarize(service_times),
        'model_raw_seconds': model_times, 'service_raw_seconds': service_times,
        'peak_single_process_allocated_bytes': torch.cuda.max_memory_allocated(),
        'backbone_parameters': sum(p.numel() for p in backbone.parameters()),
        'candidate_parameters': sum(p.numel() for p in candidate.parameters()),
        'controller_parameters': sum(p.numel() for p in controller.parameters()),
        'scales_loaded_once_outside_timing': True, 'reference_images_read': False,
        'metric_network_loaded': False, 'candidate_or_base_cache_used': False,
        'device': torch.cuda.get_device_name(0), 'precision': 'float32', 'batch': 1,
        'size': [256,256], 'warmup': 20, 'timed': 100, 'service_inputs': 20,
        'file_cache_state': 'warm OS filesystem; page cache not explicitly evicted',
        'identity': {'script': sha(__file__), 'registry': sha(s.run/'method_registry.json'),
            'selection': sha(s.run/'selection/calibration_selection.json'), 'roles': sha(s.run/'roles.jsonl'),
            'backbone': backbone_identity, 'candidate': entry['candidate']['checkpoint_sha256'],
            'controller': entry['controller']['checkpoint_sha256'], 'scales': sha(s.run/'normalization_stats.json')},
        'service_sample_ids': [r['sample_id'] for r in rows],
        'quality_selection_unchanged': True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); s = State()
    if args.worker:
        worker(s); return
    if not s.state.get('closeout_complete') or s.state['scientific_status'] in ['RUNNING', 'INTERRUPTED_RECOVERABLE']:
        raise RuntimeError('Wait for scientific closeout')
    if not (s.run / 'selection/calibration_selection.json').exists():
        raise RuntimeError('No frozen O network/policy available')
    output = s.run / 'timing/O_full_pipeline.json'
    if output.exists():
        identity = read(output)['identity']
        if (identity['script'] != sha(__file__) or identity['selection'] != sha(s.run/'selection/calibration_selection.json')
                or identity['registry'] != sha(s.run/'method_registry.json')):
            raise RuntimeError('Supplemental timing identity changed')
        print('Reusing identity-checked full O timing'); return
    from uie_next.records import append
    cmd = [sys.executable, '-m', 'scripts.ssuie_v2_full_pipeline_timing', '--worker']
    append(s.run / 'commands.jsonl', {'argv': cmd, 'purpose': 'full O timing only; no selection changes'})
    with s.device_job('CLOSEOUT_FULL_O_TIMING', 300, final=True):
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (s.run / 'timing/O_full_pipeline.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError('Full O timing worker failed; see log')
    s.live(current_job='none: scientific closeout; full O timing recorded')
    print('Full O path measured; frozen quality selection unchanged')


if __name__ == '__main__':
    main()
