import torch
import torch.nn.functional as F

from uie_next.priors.heuristic import make_prior, interventions
from uie_next.losses import utility_loss


def test_T24_T25_global_ambient_rgb_and_nonwrapping_fields():
    torch.manual_seed(7)
    image = torch.rand(2, 3, 16, 20)
    n = make_prior(image)
    blur = F.avg_pool2d(F.pad(image, (4,4,4,4), mode='replicate'), 9, 1)
    assert n['A'].shape == (2,3,1,1)
    assert torch.equal(n['A'], blur.amax((2,3),keepdim=True).clamp(.05,.95))
    views = interventions(image)
    assert len(views) == 7
    assert torch.allclose(views['tau_075']['tau'], n['tau']*.75)
    assert torch.allclose(views['ambient_rb_plus']['A'], (n['A']+image.new_tensor([.04,0,-.04]).view(1,3,1,1)).clamp(.05,.95))
    right = views['shift_right_2']
    assert torch.equal(right['d'][...,2:], n['d'][...,:-2])
    assert not right['V'][...,:2].any() and not right['P'][...,:2].any()
    assert torch.allclose(right['tau'][...,2:], right['d'][...,2:]*image.new_tensor([2.4,1.2,.8]).view(1,3,1,1))
    for field in views.values():
        assert field['P'].shape == (2,7,16,20)
        assert torch.isfinite(field['P']).all()
    assert not torch.equal(views['tau_075']['P'], n['P'])


def test_T15_equal_source_view_weights_and_true_pair_difference():
    shape=(2,2,1,2,2)
    target={'v':torch.zeros(shape), 'a':torch.zeros(shape), 'c':torch.ones(shape),
            'active':torch.ones(shape,dtype=torch.bool),'U':torch.zeros(shape),
            'target':torch.zeros(2,2,3,2,2)}
    # Source 0 only has one active pixel per view; source 1 has four.
    target['active'][0]=False;target['active'][0,:,:,0,0]=True
    target['active'][0,1]=False
    p=torch.ones(shape);p[1]=3
    pred={'v_hat':p,'b_hat':p,'output':torch.zeros_like(target['target'])}
    loss,parts=utility_loss(pred,target,{'s_v':1.,'s_U':1.,'s_e2':1.},eta_pair=0,eta_dec=0)
    assert torch.allclose(loss,torch.tensor(1.5))
    assert parts['pair']==0  # Both copies have identical predicted and true effects.
    pred['output']=torch.ones_like(target['target'])*.2
    loss,parts=utility_loss(pred,target,{'s_v':1.,'s_U':1.,'s_e2':.5},eta_pair=0,eta_dec=.1)
    assert torch.allclose(parts['dec'],torch.tensor(.08))
    target['active'].zero_()
    loss,parts=utility_loss(pred,target,{'s_v':1.,'s_U':1.,'s_e2':.5},eta_pair=0,eta_dec=0)
    assert loss==0 and torch.isfinite(loss)


def test_T15_inactive_views_do_not_dilute_projection():
    shape=(2,2,1,2,2)
    target={'v':torch.zeros(shape),'a':torch.zeros(shape),'active':torch.zeros(shape,dtype=torch.bool),
            'U':torch.zeros(shape),'target':torch.zeros(2,2,3,2,2)}
    target['active'][0]=True
    pred={'v_hat':torch.ones(shape),'b_hat':torch.zeros(shape),'output':torch.zeros_like(target['target'])}
    loss,_=utility_loss(pred,target,{'s_v':1.,'s_U':1.,'s_e2':1.},eta_pair=0,eta_dec=0)
    assert loss==.5
