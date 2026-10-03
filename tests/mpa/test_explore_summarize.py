import pytest

from scripts.explore_ag.summarize import paired_delta


def test_paired_delta_and_overlap_sensitivity():
    parent = {key: {'psnr': 20., 'scene_id': key} for key in ('a', 'b', 'c')}
    method = {'a': {'psnr': 21., 'scene_id': 'a'},
              'b': {'psnr': 21., 'scene_id': 'b'},
              'c': {'psnr': 18., 'scene_id': 'c'}}
    all_rows = paired_delta(method, parent, 'psnr')
    clean = paired_delta(method, parent, 'psnr', excluded={'c'})
    assert all_rows['count'] == 3
    assert all_rows['mean_delta'] == pytest.approx(0.)
    assert clean['count'] == 2
    assert clean['mean_delta'] == pytest.approx(1.)
    assert paired_delta(method, {}, 'psnr')['ci95'] is None
