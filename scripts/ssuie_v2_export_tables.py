"""CPU-only faithful exports after closeout, without changing any selection.

JSON remains the scientific record. These CSVs expose every registered method,
single-seed and content-group results, stress conditions, and actual stage status.
No dataset image is opened and no missing score is replaced with a number.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from uie_next.records import ROOT, read, sha, write, append
from uie_next.v2.context import RUN_ID, METHOD_ORDER
from uie_next.v2.diagnostics import csv_write
from uie_next.v2.statistics import paired_stats


def grouped(rows, key_fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, part in sorted(groups.items()):
        record = dict(zip(key_fields, key))
        record.update(seed=20261007, n_images=len(part),
                      n_groups=len({r['group_id'] for r in part}),
                      aggregation='image_weighted', group_type='content_group_proxy')
        for metric in ['mse', 'psnr', 'ssim', 'lpips', 'delta_psnr_db', 'alpha_mean',
                       'coverage_over_0_05', 'oracle_regret_mse', 'U_hat_mse',
                       'U_hat_true_correlation']:
            values = [r[metric] for r in part if r.get(metric) is not None]
            record[metric] = float(np.mean(values)) if values else None
            record[metric + '_n_provided'] = len(values)
            if not values:
                record[metric + '_status'] = 'not_provided_or_not_applicable_see_per_image'
        delta = np.asarray([r['delta_psnr_db'] for r in part])
        record.update(harm_rate=float((delta < -.1).mean()),
                      improvement_rate=float((delta > .1).mean()),
                      worst10_delta_psnr=float(np.sort(delta)[:math.ceil(.1*len(delta))].mean()))
        output.append(record)
    return output


def summarize_accesses(records):
    """Keep grouped file-identity audits distinct from per-image model access."""
    accesses = {}
    identity_audits = []
    for row in records:
        if {'role', 'operation', 'sample_id'} <= set(row):
            key = (row['role'], row['operation'])
            entry = accesses.setdefault(key, {'events': 0, 'ids': set()})
            entry['events'] += 1
            entry['ids'].add(row['sample_id'])
        elif (row.get('event') == 'file_identity_only' and
              row.get('does_not_release_sealed') is True and
              row.get('operation') ==
              'SHA256_only_no_decode_for_quality_no_model_forward_no_J0_J1_cache_no_score' and
              {'roles', 'counts', 'audit_result_sha256', 'per_file_expected_identity_in'} <= set(row)):
            identity_audits.append(row)
        else:
            raise ValueError('Unrecognized access audit event; refusing incomplete export')
    return accesses, identity_audits


def self_check():
    # Unequal content-group sizes: the main average must retain equal image weight.
    rows = [{'role': 'synthetic', 'condition': 'nominal', 'method': 'G1',
             'sample_id': str(i), 'group_id': ('a' if i < 2 else 'b'),
             'psnr': v, 'delta_psnr_db': v, 'U_hat_mse': None}
            for i, v in enumerate([0., 0., 9.])]
    out = grouped(rows, ['role', 'condition', 'method'])[0]
    assert out['psnr'] == 3. and out['n_images'] == 3 and out['n_groups'] == 2
    assert out['U_hat_mse'] is None and out['U_hat_mse_n_provided'] == 0
    assert sum(r['n_images'] for r in grouped(rows, ['group_id'])) == 3
    guard = {'role': 'utility_val', 'operation': 'develop', 'sample_id': 'fixture'}
    audit = {'event': 'file_identity_only', 'does_not_release_sealed': True,
             'operation': 'SHA256_only_no_decode_for_quality_no_model_forward_no_J0_J1_cache_no_score',
             'roles': ['sealed_eval'], 'counts': {'sealed_eval': 177},
             'audit_result_sha256': 'fixture', 'per_file_expected_identity_in': 'roles.jsonl'}
    accesses, audits = summarize_accesses([audit, guard, guard])
    assert accesses[('utility_val', 'develop')]['events'] == 2
    assert len(accesses[('utility_val', 'develop')]['ids']) == 1
    assert audits == [audit] and not any(k[0] == 'sealed_eval' for k in accesses)
    try:
        summarize_accesses([{'operation': 'unrecognized'}])
        raise AssertionError('Malformed access audit accepted')
    except ValueError:
        pass
    return {'passed': True, 'fixture_only': True,
            'unequal_group_sizes_image_weighted': True, 'missing_prediction_remains_null': True,
            'identity_only_event_kept_separate_from_model_access': True,
            'malformed_audit_rejected': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    run = ROOT / 'runs' / RUN_ID
    if args.self_check:
        receipt = self_check()
        write(run / 'tests/table_export_self_check.json', receipt)
        print(json.dumps(receipt))
        return
    state = read(run / 'state.json')
    if not state.get('closeout_complete') or state['scientific_status'] in ['RUNNING', 'INTERRUPTED_RECOVERABLE']:
        raise RuntimeError('Scientific dispatcher must stop first')
    start = time.monotonic()
    inputs = {}; formal = []; stress = []
    # The intake audit predates calibration. Keep it immutable and separately
    # expose actual guarded runtime accesses instead of relabeling old facts.
    access_path = run / 'access_events.jsonl'
    if access_path.exists():
        with access_path.open(encoding='utf-8') as stream:
            accesses, identity_audits = summarize_accesses(json.loads(line) for line in stream)
        for audit in identity_audits:
            if audit['audit_result_sha256'] != sha(run / 'data_audit.json'):
                raise ValueError('File-identity audit source changed')
        write(run / 'delivery/data_access_closeout.json', {
            'source_sha256': sha(access_path),
            'counts_are_guard_events_not_unique_forwards': True,
            'intake_data_audit_sha256': sha(run / 'data_audit.json'),
            'intake_audit_not_rewritten_after_scoring': True,
            'sealed_eval_released': state['sealed_eval_released'],
            'identity_only_audit_events': identity_audits,
            'guarded_accesses': [{'role': role, 'operation': operation,
                                 'events': entry['events'], 'unique_images': len(entry['ids'])}
                                for (role, operation), entry in sorted(accesses.items())]})
    for path in sorted((run / 'metrics/parts').glob('*.json')):
        record = read(path)
        if path.name.endswith('_cal_grid.json'):
            continue
        rows = record.get('rows', [])
        if len(record.get('done', [])) != len({r['sample_id'] for r in rows}):
            raise ValueError('Incomplete per-image export: ' + str(path))
        inputs[str(path.relative_to(run))] = sha(path)
        if rows and rows[0]['condition'] == 'nominal':
            formal.extend(rows)
        else:
            stress.extend(rows)
    if formal:
        csv_write(run / 'metrics/method_summary.csv', grouped(formal, ['role', 'condition', 'method']))
        csv_write(run / 'metrics/per_seed.csv', grouped(formal, ['role', 'condition', 'method']))
        csv_write(run / 'metrics/per_content_group.csv', grouped(formal, ['role', 'condition', 'method', 'group_id']))
        gate_path = run / 'development_gate.json'
        if gate_path.exists():
            primary = read(gate_path)['primary_control']
            dev = [r for r in formal if r['role'] == 'utility_val']
            control = {r['sample_id']: r for r in dev if r['method'] == primary}
            ours = sorted([r for r in dev if r['method'] == 'O'],
                          key=lambda r: (r['delta_psnr_db'], r['sample_id']))
            text = '# 固定策略失败案例与解释边界\n\n'
            text += '以下按完整utility_val的O−J0逐图PSNR排序列出最差8例，使用已冻结CAL策略；没有因案例修改检查点、先验或参数。图像面板ID沿规约哈希/最好/最差/中位规则，不以视觉结果筛选。\n\n'
            text += '| 样本 | 内容代理组 | O−J0(dB) | O−主对照(dB) | alpha均值 | 同候选oracle regret(MSE) |\n|---|---|---:|---:|---:|---:|\n'
            for row in ours[:8]:
                ref = control[row['sample_id']]
                text += '| %s | %s | %+.6f | %+.6f | %.6f | %.9g |\n' % (
                    row['sample_id'], row['group_id'], row['delta_psnr_db'],
                    row['psnr']-ref['psnr'], row['alpha_mean'], row['oracle_regret_mse'])
            text += '\n主对照为 `%s`。regret只比较该方法所属候选融合空间的精确oracle，不能当作可部署收益。真实U和误差差图只用于诊断，不进入推理。小幅alpha或回退不保证平均质量、尾部或机制闸门通过。压力族单独判定，不用压力改善替代名义失败；结论仅限本轮固定配方、单新增模块种子和历史已暴露数据。\n' % primary
            (run / 'FAILURE_CASES.md').write_text(text, encoding='utf-8')
            (ROOT / 'docs/experiments' / RUN_ID / 'FAILURE_CASES.md').write_text(text, encoding='utf-8')
    if stress:
        csv_write(run / 'metrics/stress_per_image.csv', stress)
        csv_write(run / 'metrics/stress_summary.csv', grouped(stress, ['role', 'condition', 'method']))
        csv_write(run / 'metrics/stress_per_content_group.csv', grouped(stress, ['role', 'condition', 'method', 'group_id']))
        stress_pairs = []
        for role, condition in sorted({(r['role'], r['condition']) for r in stress}):
            part = [r for r in stress if r['role'] == role and r['condition'] == condition]
            by = {m: {r['sample_id']: r for r in part if r['method'] == m}
                  for m in sorted({r['method'] for r in part})}
            ids = sorted(by['O'])
            for method, scores in by.items():
                if set(scores) != set(ids):
                    raise ValueError('Incomplete stress comparison: ' + method)
                stats = paired_stats([by['O'][i]['psnr'] - scores[i]['psnr'] for i in ids],
                                     [by['O'][i]['group_id'] for i in ids])
                stress_pairs.append({'role': role, 'condition': condition,
                                     'comparison': 'O-minus-' + method, **stats})
        csv_write(run / 'metrics/stress_O_paired_comparisons.csv', stress_pairs)
    pairs = []
    for name in ['development_gate.json', 'confirmation_results.json']:
        path = run / name
        if not path.exists():
            continue
        gate = read(path); inputs[name] = sha(path)
        role = 'utility_val' if name.startswith('development') else 'sealed_eval'
        for method, stats in sorted(gate['paired'].items()):
            pairs.append({'role': role, 'comparison': 'O-minus-' + method, **stats})
        # A failed gate was executed; it is not an unrun stage. No frozen policy changes.
        stage = 'DEV_GATE' if role == 'utility_val' else 'SEALED_ONCE'
        if not gate['passed'] and state['stages'][stage]['status'] == 'not_run':
            state['stages'][stage] = {'status': 'executed_failed', 'receipt_sha256': sha(path),
                                     'reason': '完整判据已执行但未通过；失败不等于not_run。'}
    if pairs:
        csv_write(run / 'metrics/O_paired_comparisons.csv', pairs)
    if (run / 'selection/producer_freeze.json').exists() and not (run / 'selection/rescue_choice.json').exists():
        for stage in ['ONE_RESCUE', 'RESCUE_MODEL_FREEZE', 'RESCUE_UTILITY_GATE']:
            if stage not in state['completed']:
                state['stages'][stage] = {'status': 'not_run',
                    'reason': '旧非零池已提供合格producer，迁移检查通过；唯一救援条件未触发。'}
    diagnostic_path = run / 'diagnostics/checkpoint_image_metrics.csv'
    if diagnostic_path.exists():
        with diagnostic_path.open(newline='', encoding='utf-8') as f:
            diagnostic = list(csv.DictReader(f))
        diag_groups = {}
        for r in diagnostic:
            key = tuple(r[k] for k in ['role', 'checkpoint_id', 'prediction_kind', 'group_id'])
            diag_groups.setdefault(key, []).append(r)
        diag_out = []
        for key, part in sorted(diag_groups.items()):
            rec = dict(zip(['role', 'checkpoint_id', 'prediction_kind', 'group_id'], key))
            rec.update(seed=20261007, n_images=len(part), aggregation='within_group_image_weighted',
                       group_type='content_group_proxy')
            for metric in ['psnr', 'mse', 'ssim', 'delta_psnr_db']:
                rec[metric] = float(np.mean([float(r[metric]) for r in part]))
            diag_out.append(rec)
        csv_write(run / 'diagnostics/checkpoint_per_content_group.csv', diag_out)
        inputs[str(diagnostic_path.relative_to(run))] = sha(diagnostic_path)
    # Keep a receipt that distinguishes completed jobs from temporary fixtures.
    jobs = []
    defaults = []
    for method in METHOD_ORDER:
        selection = run / 'checkpoints' / method / 'selection.json'
        records = []
        log = selection.parent / 'training.jsonl'
        if log.exists():
            records = [json.loads(x) for x in log.read_text().splitlines()]
        jobs.append({'method': method, 'status': 'complete' if selection.exists() else 'not_run_or_incomplete',
                     'actual_final_logged_update': records[-1]['step'] if records else None,
                     'source_presentations': records[-1]['source_presentations'] if records else None,
                     'view_presentations': records[-1]['view_presentations'] if records else None,
                     'unique_sources': records[-1]['distinct_sources_cumulative'] if records else None,
                     'loss_last50_mean': records[-1]['last50_mean'] if records else None,
                     'clip_fraction_last50': records[-1]['last50_gradient_clip_fraction'] if records else None})
        if selection.exists():
            chosen = read(selection)
            policy = ({'alpha': 1.0} if method == 'B4' else
                      {'temperature': 1.0, 'margin': 0.0} if method.startswith('G') else
                      {'tau': 0.0, 'lamb': 1e-4})
            matches = [row for row in chosen['all_grid_scores']
                       if row['step'] == chosen['selected_step'] and row['policy'] == policy]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous frozen-network default-policy score: ' + method)
            defaults.append({'method': method, 'role': 'utility_val',
                             'selected_step': chosen['selected_step'],
                             'selected_sha256': chosen['selected_sha256'],
                             'record_kind': 'uncalibrated_default_diagnostic_not_final_deployment',
                             **matches[0]})
    csv_write(run / 'metrics/training_job_receipts.csv', jobs)
    if defaults:
        csv_write(run / 'metrics/default_development_policy_scores.csv', defaults)
    write(run / 'state.json', state)
    append(run / 'events.jsonl', {'event': 'CPU_closeout_exports', 'selection_unchanged': True,
                                 'sealed_access_added': False, 'stage_status_only_correction': True})
    write(run / 'delivery/table_export_identity.json', {'script_sha256': sha(__file__),
          'inputs': inputs, 'CPU_wall_seconds': time.monotonic() - start,
          'dataset_images_read': False, 'device_work': False, 'selection_unchanged': True,
          'n_formal_per_image_rows': len(formal), 'n_stress_per_image_rows': len(stress)})
    print(json.dumps({'formal_rows': len(formal), 'stress_rows': len(stress), 'jobs': len(jobs)}))


if __name__ == '__main__':
    main()
