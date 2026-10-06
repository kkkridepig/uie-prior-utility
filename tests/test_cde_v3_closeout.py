import pytest
from tools.cde_v3_closeout import compare, aggregate


def test_lpips_improvement_direction_and_drop_best_five():
    deltas = [-0.6, -0.5, -0.4, -0.3, -0.2, 0.1]
    rows = [dict(sample_id=str(i), scene_id=str(i), eval_noise_seed=101,
                 scores={'method': {'lpips': 1 + d}, 'control': {'lpips': 1.0}})
            for i, d in enumerate(deltas)]
    result = compare(rows, 'method', 'control', 'lpips')
    assert result['improved_fraction'] == pytest.approx(5 / 6)
    assert result['mean'] == pytest.approx(sum(deltas) / 6)
    assert result['remove_best5_raw_delta_mean'] == pytest.approx(0.1)


def test_aggregation_requires_three_distinct_frozen_noise_keys():
    rows = [dict(sample_id='x', scene_id='s', eval_noise_seed=seed,
                 scores={'method': {'psnr': 20., 'ssim': .9, 'lpips': .1}})
            for seed in (101, 102, 103)]
    assert aggregate(rows)[0]['psnr'] == 20.
    rows[-1]['eval_noise_seed'] = 102
    with pytest.raises(AssertionError):
        aggregate(rows)
