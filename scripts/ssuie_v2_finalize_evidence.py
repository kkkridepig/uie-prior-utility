"""CPU-only closeout supplement; never selects candidates or reads dataset images.

Run after the V2 dispatcher stops. This script adds descriptive tables and repairs
delivery scope (including checkpoint sidecars needed by checksum-aware resume).
It does not modify the frozen scientific implementation or any V1 files.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from uie_next.records import ROOT, read, sha, write
from uie_next.reporting import archive
from uie_next.v2.context import RUN_ID, OLD, METHOD_ORDER
from uie_next.v2.diagnostics import csv_write
from uie_next.v2.statistics import paired_stats


def load_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    numbers = ['mse', 'log_mse', 'log_mse_eps', 'psnr', 'ssim', 'lpips',
               'delta_mse', 'delta_psnr_db', 'delta_ssim', 'alpha_mean',
               'alpha_zero_fraction', 'alpha_one_fraction', 'boundary_fraction']
    for r in rows:
        for k in numbers:
            if k in r:
                r[k] = float(r[k]) if r[k] not in ['', 'None', 'null'] else None
    return rows


def describe(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['checkpoint_id'], r['role'], r['prediction_kind']), []).append(r)
    result = []
    for (cid, role, strategy), part in sorted(groups.items()):
        row = {'checkpoint_id': cid, 'role': role, 'strategy': strategy,
               'seed': 20261007, 'n_images': len(part),
               'n_groups': len({r['group_id'] for r in part}), 'aggregation': 'image_weighted'}
        for key in ['psnr', 'mse', 'log_mse', 'log_mse_eps', 'ssim', 'lpips',
                    'delta_psnr_db', 'delta_mse', 'delta_ssim', 'alpha_mean',
                    'alpha_zero_fraction', 'alpha_one_fraction', 'boundary_fraction']:
            vals = [r[key] for r in part if r.get(key) is not None]
            if not vals:
                row[key + '_mean'] = None
                row[key + '_status'] = 'not_scheduled_in_D0'
                continue
            if not np.isfinite(vals).all():
                raise ValueError('Nonfinite diagnostic: ' + str((cid, role, strategy, key)))
            row[key + '_mean'] = float(np.mean(vals))
            row[key + '_median'] = float(np.median(vals))
            for name, val in zip(['05', '10', '90', '95'], np.quantile(vals, [.05, .1, .9, .95])):
                row[key + '_image_q' + name] = float(val)
        ds = np.array([r['delta_psnr_db'] for r in part])
        row.update(worst10_mean=float(np.sort(ds)[:math.ceil(.1 * len(ds))].mean()),
                   improvement_over_010_fraction=float((ds > .1).mean()),
                   harm_below_minus010_fraction=float((ds < -.1).mean()))
        result.append(row)
    return result


def loss_intervals(rows):
    """Additional paired MSE/log-MSE estimates; same 5000 group draws as PSNR."""
    output = {}
    keys = sorted({(r['checkpoint_id'], r['role']) for r in rows})
    for cid, role in keys:
        a = {r['sample_id']: r for r in rows if r['checkpoint_id'] == cid and
             r['role'] == role and r['prediction_kind'] == 'endpoint'}
        b = {r['sample_id']: r for r in rows if r['checkpoint_id'] == cid and
             r['role'] == role and r['prediction_kind'] == 'B0'}
        if set(a) != set(b):
            raise ValueError('Unpaired endpoint/base')
        ids = sorted(a)
        record = {}
        for metric in ['mse', 'log_mse', 'log_mse_eps', 'psnr']:
            st = paired_stats([a[i][metric] - b[i][metric] for i in ids],
                              [a[i]['group_id'] for i in ids])
            record[metric] = {k: st[k] for k in ['mean_image', 'ci95', 'n_images',
                                              'n_groups', 'draws', 'seed', 'aggregation', 'scope']}
        record['log_mse_is_not_independent_of_psnr'] = True
        record['mean_mse_relative_change'] = (np.mean([a[i]['mse'] for i in ids]) /
                                             np.mean([b[i]['mse'] for i in ids]) - 1)
        record['same_image_mse_lower_fraction'] = float(np.mean([a[i]['mse'] < b[i]['mse'] for i in ids]))
        output[cid + '|' + role] = record
    return output


def repair_deliveries(run, doc):
    """All official/actual scientific checkpoints AND their verified sidecars."""
    original = run / 'delivery/archive_receipts.json'
    if original.exists():
        previous = read(original)
        write(run / 'delivery/archive_receipts_dispatcher.json', previous)
    weights = read(run / 'delivery/server_weights.json')
    for entry in weights:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Weight changed after closeout')
    weight_files = [Path(r['path']) for r in weights]
    for p in list(weight_files):
        sidecar = Path(str(p) + '.json')
        if p.suffix == '.pt':
            if not sidecar.exists() or read(sidecar)['sha256'] != sha(p):
                raise ValueError('Missing or invalid recovery sidecar: ' + str(p))
            weight_files.append(sidecar)
    weight_files += [run / 'delivery/server_weights.json', run / 'delivery/recovery_instructions.md',
                     run / 'roles.jsonl', run / 'protocol_resolved.yaml', run / 'source_snapshot.json']
    weight_files += [p for p in (run / 'selection').rglob('*.json')]
    # Resume lineage needs the old metadata as well as its untouched weights.
    weight_files += [p for p in OLD.glob('*.json')]
    weight_files += [p for p in (OLD / 'checkpoints').rglob('selection.json')]
    bundle = ROOT.parent / (RUN_ID + '_local_history.bundle')
    subprocess.run(['git', 'bundle', 'create', str(bundle), '--all'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    write(run / 'delivery/git_bundle.json', {'path': str(bundle), 'sha256': sha(bundle),
                                           'bytes': bundle.stat().st_size, 'independent_backup_verified': False})
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    image_or_weight = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.pth', '.pt', '.zip', '.pdf'}
    source = [ROOT / p for p in tracked if p and (ROOT / p).is_file() and
              Path(p).suffix.lower() not in image_or_weight]
    source += [ROOT / p for p in read(run / 'source_snapshot.json')]
    source += [run / n for n in ['protocol_source.md', 'protocol_resolved.yaml', 'source_snapshot.json',
                                'source_recovery_receipt.json', 'roles.jsonl', 'exposure_ledger.json',
                                'environment.json']]
    source += [p for p in (run / 'delivery').rglob('*.md')]
    upstream = ROOT / 'third_party/ss_uie'
    source += [p for p in upstream.rglob('*') if p.is_file() and '.git' not in p.parts and
               '__pycache__' not in p.parts and p.suffix.lower() not in image_or_weight]
    source += [p for p in OLD.glob('*.json')]
    review = [p for p in list(run.rglob('*')) + list(doc.rglob('*')) if p.is_file() and
              p.suffix.lower() not in image_or_weight and '/cache/' not in str(p)]
    visuals = [p for p in (run / 'figures').rglob('*') if p.is_file() and p.suffix in {'.png', '.json', '.md'}]
    packages = {'review': review, 'source_protocol': source, 'weights_recovery': weight_files, 'visuals': visuals}
    receipts = {}
    for name, files in packages.items():
        receipts[name] = archive(ROOT.parent / (RUN_ID + '_' + name + '.zip'), sorted(set(files)), ROOT)
    six = ['__init__.py', 'audit.py', 'cache.py', 'manifest.py', 'roles.py', 'runtime.py']
    snapshot = read(OLD / 'source_snapshot.json')
    six_receipt = {}
    with zipfile.ZipFile(receipts['source_protocol']['path']) as z:
        for name in six:
            member = 'uie_next/data/' + name
            actual = hashlib.sha256(z.read(member)).hexdigest()
            if actual != snapshot[member]:
                raise ValueError('Original six-module payload mismatch')
            six_receipt[member] = {'sha256': actual, 'matches_original_v1_snapshot': True}
    write(run / 'delivery/source_six_modules_package_receipt.json', {'passed': True, 'members': six_receipt})
    write(original, {'packages': receipts, 'independent_backup_verified': False, 'same_server_disk_only': True,
                     'checkpoint_sidecars_included': True, 'six_original_data_modules_verified': True,
                     'git_bundle': read(run / 'delivery/git_bundle.json'),
                     'package_scope': 'review excludes dataset images/weights/cache; source includes original six data modules, protocol and official public source; recovery includes official and all actual checkpoints plus optimizer/RNG/sampling states and checksum sidecars; visuals separate',
                     'hash_scope': 'ZIP CRC and every included member SHA256; V1 preservation separately verified; no client-side independent-copy claim'})
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    run = ROOT / 'runs' / RUN_ID
    doc = ROOT / 'docs/experiments' / RUN_ID
    state = read(run / 'state.json')
    if state['scientific_status'] in ['RUNNING', 'INTERRUPTED_RECOVERABLE'] or not state.get('closeout_complete'):
        raise RuntimeError('Closeout only after scientific dispatcher stops')
    start = time.monotonic()
    rows = load_csv(run / 'diagnostics/checkpoint_image_metrics.csv')
    detailed = describe(rows)
    csv_write(run / 'diagnostics/checkpoint_distribution_summary.csv', detailed)
    intervals = loss_intervals(rows)
    write(run / 'diagnostics/paired_loss_intervals.json', intervals)
    write(run / 'delivery/analysis_identity.json', {'script': str(Path(__file__).resolve()),
          'script_sha256': sha(__file__), 'input_long_table_sha256': sha(run / 'diagnostics/checkpoint_image_metrics.csv'),
          'selection_unchanged': True, 'CPU_wall_seconds': time.monotonic() - start,
          'dataset_images_read': False, 'device_work': False, 'same_single_seed': 20261007})
    print(json.dumps({'analysis_tables': len(detailed), 'paired_endpoint_groups': len(intervals)}, indent=2))
    if not args.analysis_only:
        print(json.dumps(repair_deliveries(run, doc), indent=2))


if __name__ == '__main__':
    main()
