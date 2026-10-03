"""Summarize frozen Benchmark records without selecting models on test data."""
import argparse
import json
from pathlib import Path

import numpy as np


LEAKAGE_SENSITIVITY_ID = 'UIEB/356_img_'


def read_records(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = {row['sample_id']: row for row in rows}
    if len(records) != len(rows):
        raise ValueError('Duplicate sample records: {}'.format(path))
    return records


def paired_delta(candidate, reference, metric, excluded=frozenset(), draws=1000):
    ids = sorted((candidate.keys() & reference.keys()) - set(excluded))
    ids = [key for key in ids if isinstance(candidate[key].get(metric), (float, int))
           and isinstance(reference[key].get(metric), (float, int))]
    if not ids:
        return {'count': 0, 'mean_delta': None, 'ci95': None}
    groups = {}
    for key in ids:
        scene = candidate[key].get('scene_id') or key
        groups.setdefault(scene, []).append(float(candidate[key][metric]) - float(reference[key][metric]))
    values = np.array([delta for deltas in groups.values() for delta in deltas])
    names = sorted(groups)
    generator = np.random.default_rng(20261003)
    boot = []
    for _ in range(draws):
        chosen = generator.choice(len(names), len(names), replace=True)
        boot.append(float(np.mean([delta for index in chosen for delta in groups[names[index]]])))
    return {'count': len(ids), 'scene_groups': len(names), 'mean_delta': float(np.mean(values)),
            'improved_fraction': float(np.mean(values > 0)),
            'ci95': [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            'resampling': 'scene_group_1000_draws' if len(names) < len(ids) else 'image_1000_draws_scene_unknown'}


def summarize(root):
    root = Path(root)
    freeze = json.loads((root/'selection_freeze_before_test.json').read_text())
    branches = [r['branch'] for r in freeze['representatives']]
    comparisons = {}
    coverage = {}
    for dataset in ('uieb_test', 'lsui_test', 'u45'):
        records = {branch: read_records(root/'benchmark'/branch/dataset/'per_image.jsonl')
                   for branch in branches}
        coverage[dataset] = {branch: len(rows) for branch, rows in records.items()}
        for branch in branches:
            if branch == 'PARENT' or not records[branch]:
                continue
            references = ['PARENT'] + (['BASE_CONT'] if branch != 'BASE_CONT' else [])
            for baseline in references:
                if not records.get(baseline):
                    continue
                key = '{}/{}/vs_{}'.format(dataset, branch, baseline)
                metrics = ('psnr', 'ssim') if dataset != 'u45' else ()
                comparisons[key] = {metric: paired_delta(records[branch], records[baseline], metric)
                                    for metric in metrics}
                if dataset == 'uieb_test':
                    comparisons[key]['without_confirmed_scene_overlap'] = {
                        metric: paired_delta(records[branch], records[baseline], metric,
                                             excluded={LEAKAGE_SENSITIVITY_ID}) for metric in metrics}
    return {'freeze_sha256_required': True, 'selection_rule': freeze['selection_rule'],
            'coverage': coverage, 'paired_comparisons': comparisons,
            'ci_scope': 'Fixed model and paired test samples only; not training-seed variability',
            'sensitivity_excluded_sample': LEAKAGE_SENSITIVITY_ID}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
