import torch

from scripts.explore_ag.stress import depth_intervention, corrupt_static
from mpa_diff.priors.depth import normalize_depth


def test_shift_uses_border_not_wrap():
    image = torch.arange(10.).view(1, 1, 1, 10)
    shifted = depth_intervention(image, torch.ones_like(image), 'depth_shift_005', 'a')
    torch.testing.assert_close(shifted, torch.tensor([0.,0.,1.,2.,3.,4.,5.,6.,7.,8.]).view_as(image))


def test_noise_is_sample_keyed_and_preserves_invalid_pixels():
    image = torch.full((1,1,32,32), .5)
    valid = torch.ones_like(image).bool()
    valid[..., :8] = False
    altered = depth_intervention(image, valid, 'depth_noise_010', 'a')
    torch.testing.assert_close(altered[~valid], image[~valid])
    assert not torch.equal(altered[valid], image[valid])
    torch.testing.assert_close(altered, depth_intervention(image,valid,'depth_noise_010','a'))


def test_global_distance_change_rebuilds_legacy_coordinate_without_mutation():
    image = torch.arange(10.).view(1,1,1,10)/9
    depth = normalize_depth(image)
    static = {'depth': depth}
    changed = corrupt_static(static, image, None, 'depth_shift_005', 'a', None)
    torch.testing.assert_close(changed['depth'].physical_coordinate,
                               1-changed['depth'].distance_proxy)
    torch.testing.assert_close(depth.physical_coordinate, image)
    missing = corrupt_static(static, image, None, 'depth_missing', 'a', None)['depth']
    assert not missing.valid_mask.any()
    assert missing.physical_coordinate.count_nonzero() == 0
