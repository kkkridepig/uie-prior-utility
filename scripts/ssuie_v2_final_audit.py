"""CPU-only final consistency audit of the completed V2 evidence.

Reads records and file hashes, never decodes data images or invokes a model.
It does not change network/policy selections, the budget, or scientific state.
"""
import csv
import json
import math
import subprocess
import time
from collections import Counter, defaultdict

from uie_next.records import ROOT, read, sha, write
from uie_next.v2.context import RUN_ID, OLD, PROTOCOL_SHA, METHOD_ORDER


def main():
    start = time.monotonic()
    run = ROOT / 'runs' / RUN_ID
    state = read(run / 'state.json')
    budget = read(run / 'budget.json')
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise RuntimeError('Final consistency check failed: ' + name)

    check('terminal_scientific_stop', state['scientific_status'] == 'STOP_CURRENT_RECIPE_NOT_SUPPORTED')
    check('closeout_complete', state['closeout_complete'])
    check('sealed_not_released', state['sealed_eval_released'] is False)
    check('budget_not_active_and_under_limit', budget['active'] is None and
          budget['inherited_device_seconds'] <= budget['used_device_seconds'] <= 57600)
    check('budget_inheritance_preserved', budget['inherited_device_seconds'] == 1503.1731119155884 and
          sha(OLD / 'budget.json') == budget['inherited_budget_sha256'])
    check('protocol_identity', sha(run / 'protocol_source.md') == PROTOCOL_SHA)
    snapshot = read(run / 'source_snapshot.json')
    check('all_74_frozen_scientific_sources_unchanged', len(snapshot) == 74 and
          all(sha(ROOT / p) == h for p, h in snapshot.items()))
    old_snapshot = read(OLD / 'source_snapshot.json')
    six = ['__init__.py', 'audit.py', 'cache.py', 'manifest.py', 'roles.py', 'runtime.py']
    check('six_original_data_sources_match_v1', all(
        sha(ROOT / 'uie_next/data' / name) == old_snapshot['uie_next/data/' + name] for name in six))
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    check('six_data_sources_tracked', all('uie_next/data/' + name in tracked for name in six))
    preservation = read(run / 'legacy_preservation.json')
    check('all_legacy_files_unchanged', all((ROOT / p).is_file() and sha(ROOT / p) == h
                                         for p, h in preservation.items()))
    upstream = ROOT / 'third_party/ss_uie'
    check('official_commit_and_clean_tree',
          subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() ==
          '88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df' and not
          subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip())
    roles = [json.loads(x) for x in (run / 'roles.jsonl').read_text().splitlines()]
    check('roles_unchanged_from_v1', sha(run / 'roles.jsonl') == sha(OLD / 'roles.jsonl'))
    check('complete_role_counts', Counter(r['role'] for r in roles) == {
        'model_fit': 3608, 'model_val': 671, 'utility_fit': 443, 'utility_val': 136,
        'calibration': 133, 'sealed_eval': 177, 'excluded_overlap': 1})
    check('sealed_no_result_or_final_release_freeze', not any((run / p).exists() for p in [
        'metrics/sealed_per_image.csv', 'confirmation_results.json',
        'selection/selection_freeze_before_eval.json', 'metrics/parts/sealed_eval_nominal.json']))
    access = read(run / 'delivery/data_access_closeout.json')
    check('access_export_current', access['source_sha256'] == sha(run / 'access_events.jsonl'))
    check('no_sealed_model_access', not any(r['role'] == 'sealed_eval' for r in access['guarded_accesses']))
    check('identity_audit_retained_separately', len(access['identity_only_audit_events']) == 1)
    summaries = read(run / 'diagnostics/checkpoint_summary.json')
    expected = {h + '_%06d' % step for h in ['B1', 'B3'] for step in [1000, 2000, 3000, 4000]}
    check('eight_checkpoints_all_three_roles', all(
        {r['checkpoint_id'] for r in summaries if r['role'] == role and r['training_step'] > 0} == expected
        for role in ['model_fit_probe', 'model_val', 'utility_val']))
    check('producer_frozen_and_not_changed_after_transfer',
          read(run / 'selection/producer_freeze.json')['producer']['checkpoint_id'] == 'B1_003000' and
          read(run / 'selection/producer_transfer_gate.json')['passed'])
    check('conditional_rescue_not_run', all(state['stages'][p]['status'] == 'not_run' for p in [
        'ONE_RESCUE', 'RESCUE_MODEL_FREEZE', 'RESCUE_UTILITY_GATE']) and
        not (run / 'selection/rescue_choice.json').exists())
    check('all_11_formal_jobs_3000_updates', all(
        read(run / 'checkpoints' / m / 'selection.json')['updates'] == 3000 for m in METHOD_ORDER) and
        state['formal_training_updates_v2'] == 33000)
    registry = read(run / 'method_registry.json')
    check('registered_weights_unchanged', all(sha(entry[field]['path']) == entry[field]['checkpoint_sha256']
        for entry in registry.values() for field in ['candidate', 'controller'] if entry[field]))
    calibration = read(run / 'selection/calibration_selection.json')
    check('calibration_identity_and_full_count', calibration['n_images'] == 133 and
          calibration['registry'] == sha(run / 'method_registry.json') and
          calibration['network_freeze'] == sha(run / 'selection/network_freeze_before_calibration.json'))
    gate = read(run / 'development_gate.json')
    check('failed_gate_recorded_as_executed', not gate['passed'] and
          state['stages']['DEV_GATE']['status'] == 'executed_failed')
    with (run / 'metrics/development_per_image.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    check('complete_20_method_development_table', Counter(r['method'] for r in rows) ==
          {m: 136 for m in registry} and len(registry) == 20)
    ids = {r['sample_id'] for r in roles if r['role'] == 'utility_val'}
    check('all_methods_paired_on_same_136_images', all(
        {r['sample_id'] for r in rows if r['method'] == m} == ids for m in registry))
    check('inference_never_uses_reference', all(r['uses_reference_at_inference'] == 'False' for r in rows))
    check('psnr_exact_per_image_mse', all(math.isfinite(float(r['psnr'])) and
          abs(float(r['psnr']) + 10 * math.log10(float(r['mse']))) < 1e-9 for r in rows))
    check('gates_missing_prediction_remains_null', all(
        r['U_hat_mse'] == '' and r['U_hat_true_correlation'] == '' and
        r['U_hat_status'] == 'not_provided_by_method'
        for r in rows if r['method'] in ['G0', 'G1', 'G2']))
    check('gate_means_match_per_image_csv', all(abs(
        sum(float(r['psnr']) for r in rows if r['method'] == m) / 136 -
        gate['summary'][m]['psnr']) < 1e-10 for m in registry))
    with (run / 'metrics/stress_per_image.csv').open(newline='') as f:
        stress = list(csv.DictReader(f))
    check('all_four_stress_families_complete', Counter((r['condition'], r['method']) for r in stress) == {
        (condition, m): 136 for condition in ['tau_050', 'tau_150', 'ambient_green', 'shift_right_4']
        for m in registry})
    figures = read(run / 'figures/comparison_utility_val_manifest.json')
    check('fixed_candidate_visual_ids_and_hashes', len(figures['cases']) == 20 and
          figures['identity']['fixed_candidate_case_ids'] == [r['sample_id'] for r in figures['cases']] and
          all(sha(r['path']) == r['sha256'] for r in figures['cases']))
    check('full_O_timing_present_and_same_policy',
          read(run / 'timing/O_full_pipeline.json')['frozen_policy'] == calibration['methods']['O']['policy'])
    result = {'passed': True, 'checks': checks, 'n_checks': len(checks),
              'script_sha256': sha(__file__), 'CPU_wall_seconds': time.monotonic() - start,
              'device_work': False, 'dataset_images_decoded': False,
              'new_quality_scores_generated': False, 'selection_changed': False,
              'legacy_files_rehashed': len(preservation),
              'independent_backup_verified': False, 'budget_sha256': sha(run / 'budget.json')}
    write(run / 'tests/final_evidence_consistency.json', result)
    print(json.dumps({'passed': True, 'checks': len(checks), 'CPU_wall_seconds': result['CPU_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
