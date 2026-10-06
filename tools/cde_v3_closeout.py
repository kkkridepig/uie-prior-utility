"""CPU-only, post-freeze audit/export. Never trains, selects models, or edits freezes.

Run: .venv/bin/python tools/cde_v3_closeout.py
The frozen experiment source remains untouched; these are posthoc descriptive tables.
"""
import collections
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cde_v3.common import RUN, DOC, PARENT, MODES, sha, read, write, code_hash
from scripts.cde_v3.summarize import paired_summary

OUT = RUN / 'closeout'


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records):
    grouped = collections.defaultdict(list)
    for row in records:
        grouped[(row['sample_id'], row['scene_id'])].append(row)
    output = []
    for (sid, scene), rows in sorted(grouped.items()):
        assert sorted(r['eval_noise_seed'] for r in rows) == [101, 102, 103]
        for model in rows[0]['scores']:
            output.append(dict(sample_id=sid, scene_id=scene, model=model,
                               **{m: float(np.mean([r['scores'][model][m] for r in rows]))
                                  for m in ('psnr', 'ssim', 'lpips')}))
    return output


def scene_table(rows):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[(r['scene_id'], r['model'])].append(r)
    return [dict(scene_id=scene, model=model, images=len(rs),
                 **{m: float(np.mean([r[m] for r in rs])) for m in ('psnr', 'ssim', 'lpips')})
            for (scene, model), rs in sorted(grouped.items())]


def compare(records, method, control, metric):
    result = paired_summary(records, method, control, metric)
    per = collections.defaultdict(list)
    for r in records:
        per[r['sample_id']].append(r['scores'][method][metric] - r['scores'][control][metric])
    raw = np.array([np.mean(v) for v in per.values()])
    oriented = -raw if metric == 'lpips' else raw
    result['improved_fraction'] = float(np.mean(oriented > 0))
    result.pop('remove_largest5_mean', None)
    keep = np.argsort(oriented)[:-5]
    result['remove_best5_raw_delta_mean'] = float(raw[keep].mean())
    result['delta_definition'] = 'method_minus_control; LPIPS lower is better'
    return result


def main():
    begin = time.monotonic()
    OUT.mkdir(exist_ok=True)
    budget = read(RUN / 'budget.json')
    assert budget['active'] is None
    freeze = read(RUN / 'final_freeze.json')
    assert code_hash() == freeze['code_sha256']
    assert read(RUN / 'development_decision.json')['status'] == 'pilot_no_supported_gain'
    assert read(RUN / 'development_decision.json')['seeds'] == [20261004]
    assert all(sha(Path(p)) == h for p, h in freeze['checkpoints'].items())
    assert all(sha(Path(p)) == h for p, h in freeze['auxiliary_hashes'].items())
    root = RUN / 'pilot/20261004'

    # Verify the existing frozen artifact inventory before adding this audit.
    manifest = read(RUN / 'artifact_manifest.json')
    mismatches = [p for p, h in manifest.items() if not (RUN / p).exists() or sha(RUN / p) != h]
    assert not mismatches, mismatches
    weights = []
    for path in [PARENT] + sorted(root.glob('*/step_05000.pt')) + sorted(root.glob('selectors/*/last.pt')):
        cp = torch.load(path, map_location='cpu')
        if path != PARENT:
            assert cp['step'] == (2000 if 'selectors' in path.parts else 5000)
            assert cp['identity']['finetune_seed'] == 20261004
            assert cp['optimizer']['state']
            state = cp.get('delta', cp.get('model'))
            assert all(torch.isfinite(v).all().item() for v in state.values() if torch.is_tensor(v))
            assert all(torch.isfinite(v).all().item() for s in cp['optimizer']['state'].values()
                       for v in s.values() if torch.is_tensor(v))
        weights.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                            sha256=sha(path), step=cp.get('step'), cpu_load_verified=True,
                            parent_required=path != PARENT))
        del cp
    table('weights.csv', weights)

    # Check all 5000 common data/t/noise/dropout keys across seven trained branches.
    pairing = {}
    shared_fields = ('step', 'sample_ids', 't', 'noise_sha256', 'dropout_seed')
    base_rng = [{k: r[k] for k in shared_fields} for r in lines(root / 'BASE_CONT_V3/rng_pairing.jsonl')]
    assert len(base_rng) == 5000
    for p in sorted(root.glob('*/rng_pairing.jsonl')):
        rows = [{k: r[k] for k in shared_fields} for r in lines(p)]
        pairing[p.parent.name] = len(rows) == 5000 and rows == base_rng
        assert pairing[p.parent.name]

    # Complete, unique key grids and finite recorded metrics.
    record_audit = {}
    for p in sorted(RUN.rglob('per_image.jsonl')):
        rows = lines(p)
        if '/E/' in str(p):
            continue  # E has sampler-specific keys, already validated by its own diagnostics.
        keys = [(r['sample_id'], r['eval_noise_seed'], r.get('mode'), r.get('corruption'),
                 r.get('branch'), r.get('sigma')) for r in rows]
        assert len(keys) == len(set(keys)), str(p)
        for r in rows:
            scores = [r] if 'psnr' in r else r.get('scores', r.get('candidate_scores', {})).values()
            assert all(math.isfinite(float(s[m])) for s in scores for m in ('psnr', 'ssim', 'lpips'))
        record_audit[str(p.relative_to(RUN))] = dict(rows=len(rows), unique=True, finite=True, sha256=sha(p))
    expected = {'C_BANK/eval': 1365, 'selector_evaluation': 2457, 'D_noise': 2457,
                'labels/route_fit/scores': 1410, 'labels/route_cal/scores': 710}
    for part, n in expected.items():
        assert len(lines(root / part / 'per_image.jsonl')) == n

    # Merge existing source_dev results without new inference.
    dev = {}
    for branch in ('BASE_CONT_V3', 'C_BANK', 'C_RGB_CONTROL', 'C_ALL_ONLY',
                   'D_SOBEL_STD', 'D_PHASE_STD', 'D_SOFTPHASE_STD'):
        for r in lines(root / branch / 'eval/per_image.jsonl'):
            key = (r['sample_id'], r['eval_noise_seed'])
            item = dev.setdefault(key, dict(sample_id=r['sample_id'], scene_id=r['scene_id'],
                                            eval_noise_seed=r['eval_noise_seed'], scores={}))
            name = 'BANK_' + r['mode'] if branch == 'C_BANK' else branch
            item['scores'][name] = {m: r[m] for m in ('psnr', 'ssim', 'lpips')}
    routing = lines(root / 'selector_evaluation/per_image.jsonl')
    for r in routing:
        if r['corruption'] == 'clean':
            dev[(r['sample_id'], r['eval_noise_seed'])]['scores'].update(r['deployed_results'])
    dev = list(dev.values())
    assert len(dev) == 273
    per = aggregate(dev)
    table('development_per_image.csv', per)
    table('development_per_scene.csv', scene_table(per))
    summaries = {}
    for method, control in [('UTILITY', c) for c in ('BEST_FIXED', 'WINNER_CE', 'SHUFFLED',
                            'BASE_CONT_V3', 'C_RGB_CONTROL', 'C_ALL_ONLY', 'BANK_null')] + [
                            ('D_SOFTPHASE_STD', c) for c in ('D_PHASE_STD', 'D_SOBEL_STD', 'BASE_CONT_V3')]:
        summaries[method + '_vs_' + control] = {m: compare(dev, method, control, m) for m in ('psnr', 'ssim', 'lpips')}
    write(OUT / 'development_paired_statistics.json', summaries)

    diagnostics = []
    risk_curves = []
    for corruption in sorted({r['corruption'] for r in routing}):
        grouped = collections.defaultdict(list)
        for r in routing:
            if r['corruption'] == corruption:
                grouped[r['sample_id']].append(r)
        details = []
        for sid, rs in sorted(grouped.items()):
            assert len(rs) == 3
            assert len({r['selected_modes']['UTILITY'] for r in rs}) == 1
            truth = np.array([np.mean([r['candidate_scores'][m]['psnr'] for r in rs]) for m in MODES])
            utility = truth - truth[0]
            predicted = np.mean([r['predicted_utility'] for r in rs], axis=0)
            mode = rs[0]['selected_modes']['UTILITY']
            actual = utility[MODES.index(mode)]
            details.append(dict(sample_id=sid, scene_id=rs[0]['scene_id'], corruption=corruption,
                                mode=mode, utility=float(actual), predicted_selected=float(predicted[MODES.index(mode)]),
                                nonnull_mse=float(np.mean((predicted[1:] - utility[1:]) ** 2)),
                                oracle_regret=float(truth.max() - truth[MODES.index(mode)])))
        accepted = [r for r in details if r['mode'] != 'null']
        diagnostics.append(dict(corruption=corruption, images=len(details), nonnull_accepted=len(accepted),
                                coverage=len(accepted)/len(details), null_rate=1-len(accepted)/len(details),
                                negative_acceptance_conditional=sum(r['utility'] < -.1 for r in accepted)/len(accepted) if accepted else None,
                                negative_acceptance_all_images=sum(r['utility'] < -.1 for r in accepted)/len(details),
                                utility_mse=float(np.mean([r['nonnull_mse'] for r in details])),
                                regret=float(np.mean([r['oracle_regret'] for r in details])),
                                **{'count_'+m: sum(r['mode'] == m for r in details) for m in MODES}))
        table('selector_' + corruption + '_per_image.csv', details)
        for tau in (0, .05, .1, .2):
            kept = [r for r in accepted if r['predicted_selected'] >= tau]
            risk_curves.append(dict(corruption=corruption, threshold=tau, images=len(details),
                                    accepted=len(kept), coverage=len(kept)/len(details),
                                    risk_loss_over_0_1_conditional=sum(r['utility'] < -.1 for r in kept)/len(kept) if kept else None,
                                    realized_mean_gain_vs_null=sum(r['utility'] for r in kept)/len(details),
                                    use='posthoc_descriptive_not_threshold_selection'))
    table('selector_diagnostics.csv', diagnostics)
    table('risk_coverage.csv', risk_curves)

    d_groups = collections.defaultdict(list)
    for row in lines(root / 'D_noise/per_image.jsonl'):
        if row['eval_noise_seed'] == 101:
            d_groups[(row['branch'], row['sigma'])].append(row)
    d_table = []
    for (branch, sigma), rows in sorted(d_groups.items()):
        taus = [v for r in rows for v in (r['tau'] or [])]
        imaginary = [r['imaginary_max'] for r in rows if r['imaginary_max'] is not None]
        d_table.append(dict(branch=branch, sigma=sigma, images=len(rows),
                            tau_min=min(taus) if taus else None,
                            tau_median=float(np.median(taus)) if taus else None,
                            tau_max=max(taus) if taus else None,
                            fft_imaginary_max=max(imaginary) if imaginary else None,
                            mean_channel_rms_before=float(np.mean([r['before_rms'] for r in rows])),
                            mean_channel_rms_after=float(np.mean([r['after_rms'] for r in rows]))))
    table('D_fft_tau_rms_diagnostics.csv', d_table)

    final_stats = {}
    panels = {}
    for out in sorted((RUN / 'final/20261004').iterdir()):
        rows = lines(out / 'per_image.jsonl')
        assert len(rows) == {'confirm_holdout': 1173, 'UIEB_legacy_exposed_regression': 291,
                             'LSUI_legacy_exposed_regression': 1281}[out.name]
        assert all(set(r['scores']) == {'PARENT', 'BASE_CONT_V3'} for r in rows)
        per = aggregate(rows)
        table(out.name + '_per_image.csv', per)
        table(out.name + '_per_scene.csv', scene_table(per))
        final_stats[out.name] = {m: compare(rows, 'BASE_CONT_V3', 'PARENT', m) for m in ('psnr', 'ssim', 'lpips')}
        visual = read(out / 'visual_manifest.json')
        selected = set(visual['fixed'] + visual['worst10'] + visual['largest_regression10'])
        assert len(visual['fixed']) == 12 and len(visual['worst10']) == len(visual['largest_regression10']) == 10
        for sid in selected:
            for suffix in ('', '_detail', '_abs_difference'):
                assert (out / 'panels' / (sid.replace('/', '__') + suffix + '.png')).exists()
        panels[out.name] = dict(fixed=12, worst=10, largest_regression=10, unique_images=len(selected),
                               columns=visual['columns'], files_verified=True)
    write(OUT / 'final_statistics_corrected.json', final_stats)

    packages = collections.defaultdict(float)
    for event in budget['events']:
        packages[event['package']] += event['elapsed_seconds']/3600
    assert all(hours <= budget['package_caps'][p] for p, hours in packages.items())
    audit = dict(timestamp_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 status='pilot_no_supported_gain', actual_finetune_seeds=[20261004], repeats=0,
                 source_and_freeze_unchanged=True, old_manifest_entries_verified=len(manifest),
                 weight_files_cpu_verified=len(weights), rng_pairing_5000_updates=pairing,
                 records=record_audit, visual_panels=panels, device_hours=budget['charged_seconds']/3600,
                 package_hours=dict(packages), extra_device_hours_this_audit=0,
                 cpu_audit_seconds=time.monotonic()-begin,
                 errata=['Frozen seed_claim and final_summary evidence_scope are generic planned-matrix text; actual pilot count is 1 and repeat count 0.',
                         'Frozen LPIPS improved_fraction used positive raw deltas. Corrected fractions and remove-best5 statistics are in final_statistics_corrected.json.',
                         'No method passed: final inference contains only PARENT and BASE_CONT_V3; no C/D holdout confirmation.',
                         '391 formerly unscored LSUI images now have baseline evaluation exposure and cannot be claimed untouched for future development.'],
                 unexecuted=['BASE_CONT_DATA_MATCHED: C did not pass the prerequisite gate',
                             '20261005/20261006: neither C nor D advanced',
                             'C/D holdout benchmarks and their final panels: no selected method',
                             'all-condition contamination: optional diagnostic not executed',
                             'URanker: no audited usable weights; not reported'])
    write(OUT / 'verification.json', audit)
    inventory = {str(p.relative_to(RUN)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in RUN.rglob('*') if p.is_file() and p.suffix in ('.jsonl', '.log', '.xml', '.csv')}
    write(OUT / 'evidence_inventory.json', inventory)
    print(json.dumps({k: audit[k] for k in ('status', 'weight_files_cpu_verified', 'old_manifest_entries_verified',
                                          'device_hours', 'package_hours', 'cpu_audit_seconds')}, indent=2))


if __name__ == '__main__':
    main()
