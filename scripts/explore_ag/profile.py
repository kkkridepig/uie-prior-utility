"""Measure effective-batch-four update throughput with the real training path."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.utils.io import sha256, write_json
from scripts.explore_ag.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for micro in (1, 2, 4):
        if args.deadline_unix and time.time() >= args.deadline_unix - 1800:
            break
        directory = out/('micro_{}'.format(micro))
        try:
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            train(argparse.Namespace(parent=args.parent, branch='BASE_CONT',
                  output=str(directory), target_steps=120, micro_batch=micro,
                  deadline_unix=args.deadline_unix, device=args.device))
            records = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
            by_step = {row['added_step']: row for row in records}
            stable = [by_step[step]['seconds'] for step in range(21, 121) if step in by_step]
            results[str(micro)] = {
                'status': 'measured' if len(stable) == 100 else 'incomplete',
                'warmup_steps': 20, 'stable_steps': len(stable),
                'mean_update_seconds': float(np.mean(stable)) if stable else None,
                'median_update_seconds': float(np.median(stable)) if stable else None,
                'peak_memory_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                'checkpoint': str((directory/'last.pt').resolve()),
            }
        except torch.cuda.OutOfMemoryError as error:
            results[str(micro)] = {'status': 'oom', 'error': str(error)}
            torch.cuda.empty_cache()
        write_json(out/'summary.json', {'parent_sha256': sha256(args.parent),
                   'effective_batch': 4, 'results': results,
                   'cache_note': 'Shared frozen depth cache can warm across variants; compare full update time with this limitation.'})
    valid = [(int(k), value['mean_update_seconds']) for k, value in results.items()
             if value['status'] == 'measured']
    if not valid:
        raise RuntimeError('No completed 20+100 update throughput profile')
    selected = min(valid, key=lambda item: item[1])[0]
    write_json(out/'selection.json', {'micro_batch': selected, 'gradient_accumulation': 4//selected,
               'selection_rule': 'lowest measured mean seconds/update over steps 21-120',
               'parent_sha256': sha256(args.parent)})
    print(json.dumps({'selected_micro_batch': selected, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
