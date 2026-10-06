"""Versioned cost correction following a validated inference-only optimization."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.corruption import CORRUPTIONS
from scripts.cde_v3.inference import INFERENCE_RUNTIME


def costs(old, measured, ndev, nfit, ncal, final_images, selector_seconds=.05):
    # Include I/O/cold-prior headroom on every output, although the actual
    # evaluators amortize prior extraction across modes and noise keys.
    infer = {k: v + .10 for k, v in measured.items()}
    training = dict(old['training_seconds_5000'])
    training['BASE_CONT_DATA_MATCHED'] = training['BASE_CONT_V3']
    components = {
        'training_all_five_enhancers': sum(training.values()),
        'development_all_five_enhancers': ndev * 3 * (
            2 * infer['BASE_CONT_V3'] + 5 * infer['C_BANK']
            + infer['C_RGB_CONTROL'] + infer['C_ALL_ONLY']),
        'route_fit_and_cal_labels': (nfit + ncal) * 5 * 2 * infer['C_BANK'],
        'selector_all_clean_and_corruption_groups': ndev * 3 * 5 * len(CORRUPTIONS) * infer['C_BANK'],
        'three_selector_trainings': 3 * 2000 * selector_seconds,
        'startup_and_export_headroom': 300,
    }
    perseed = sum(components.values())
    # Parent plus the eight named deployable controls/methods. No controls or
    # images are deleted and no bank-output caching is assumed here.
    final_per_image = (infer['PARENT'] + 2 * infer['BASE_CONT_V3']
                       + infer['C_RGB_CONTROL'] + infer['C_ALL_ONLY']
                       + 4 * infer['C_BANK'])
    final_hours = 3 * final_images * 3 * final_per_image * 1.25 / 3600 + .5
    result = dict(old)
    result.update(prediction_version=2, stage_seconds_per_seed=components,
                  inference_seconds_by_branch_with_IO_headroom=infer,
                  IO_cold_prior_headroom_seconds_per_output=.10,
                  evaluation_seconds_per_image_conservative=max(infer.values()),
                  initial_C_package_hours=(sum(training[b] for b in ('BASE_CONT_V3', 'C_BANK', 'C_RGB_CONTROL'))
                                           + ndev * 3 * (infer['BASE_CONT_V3'] + 5 * infer['C_BANK'] + infer['C_RGB_CONTROL'])) / 3600,
                  full_C_pilot_hours_with_controls=perseed / 3600,
                  two_repeats_hours=2 * perseed / 3600,
                  C_pilot_predicted_hours_with_safety=perseed * 1.25 / 3600,
                  repeats_predicted_hours_with_safety=perseed * 2.5 / 3600,
                  final_3seed_full_controls_hours_with_safety=final_hours,
                  final_image_count=final_images,
                  final_named_models_per_seed=9,
                  dispatch_allowed=perseed * 1.25 / 3600 <= 18 and perseed * 2.5 / 3600 <= 22 and final_hours <= 12,
                  prediction_note='NCHW training unchanged; verified channels-last inference; per-branch PPU p95 plus 0.10s/output I/O/cold-prior headroom; 1.25x margin; full controls',
                  selector_step_seconds_bound=selector_seconds)
    return result


def main():
    validation = read(RUN / 'inference_layout_validation.json')
    assert validation['passed'] and validation['hard_null_full_DDIM20_exact']
    assert validation['runtime'] == INFERENCE_RUNTIME
    assert validation['runtime_sha256'] == sha(ROOT / 'scripts/cde_v3/inference.py')
    previous = RUN / 'cost_prediction_v1_nchw.json'
    if not previous.exists():
        shutil.copy2(RUN / 'cost_prediction.json', previous)
    old = read(previous)
    count = read(RUN / 'holdout_frozen_candidates.json')['count'] + 97 + 427
    result = costs(old, validation['model_plus_metrics_p95_seconds'],
                   len(roles('source_dev')), len(roles('route_fit')), len(roles('route_cal')), count)
    result['previous_prediction_sha256'] = sha(previous)
    result['layout_validation_sha256'] = sha(RUN / 'inference_layout_validation.json')
    result['current_prediction_code_sha256'] = sha(Path(__file__))
    assert result['dispatch_allowed'], 'Complete matrix still exceeds frozen caps'
    write(RUN / 'cost_prediction_v2_channels_last.json', result)
    write(RUN / 'cost_prediction.json', result)
    freeze = {'runtime': INFERENCE_RUNTIME, 'runtime_sha256': validation['runtime_sha256'],
              'validation_sha256': result['layout_validation_sha256'],
              'previous_prediction_sha256': sha(previous),
              'prediction_sha256': sha(RUN / 'cost_prediction_v2_channels_last.json'),
              'training_implementation': train_identity(), 'no_training_recipe_change': True,
              'effective_before_first_pilot': True, 'holdout_scores_accessed': False}
    target = RUN / 'inference_runtime_freeze.json'
    if target.exists():
        assert read(target) == freeze, 'Inference runtime freeze immutable'
    else:
        write(target, freeze)
    (DOC / 'BUDGET_AND_DATA_PLAN.md').write_text(
        '# 预算与数据计划\n\n新72设备小时上限，最终保留12小时，原阶段上限不变。'
        'NCHW旧预测保留在cost_prediction_v1_nchw.json，因预测超限停过一次；'
        '正式pilot尚未启动时完成训练集上的channels_last数值/计时核验。'
        '训练配方、DDIM20、图数、噪声键、seed与所有对照未变。\n\n'
        '新预测按各分支真实PPU p95、每次输出额外0.10秒I/O/冷先验余量、'
        '全矩阵1.25倍安全系数估算；最终额外预留0.5小时导出。'
        '不假定删除或缓存任何命名对照。所有实际设备时间继续累计到同一账本。\n\n'
        '```json\n' + json.dumps(result, indent=2, ensure_ascii=False) + '\n```\n')
    print({k: result[k] for k in ('C_pilot_predicted_hours_with_safety', 'repeats_predicted_hours_with_safety',
                                'final_3seed_full_controls_hours_with_safety', 'dispatch_allowed')})


if __name__ == '__main__':
    main()
