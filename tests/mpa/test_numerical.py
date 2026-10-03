import copy
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from mpa_diff.priors.kernels import haar,inverse_haar,sobel,histogram,pad_image
from mpa_diff.priors.depth import normalize_depth
from mpa_diff.physics.renderer import render,invert
from mpa_diff.diffusion.core import Schedule,sample
from mpa_diff.models.unet import UNet,BetaUNet
from mpa_diff.config import load_config,validate,merge_strict,DEFAULT
from mpa_diff.priors.provider import MPADiff
from mpa_diff.engine.losses import masked_mse
from mpa_diff.metrics.image import psnr,ssim

torch.set_num_threads(2)

def test_haar_hand_computed():
    x=torch.tensor([[[[1.,2.],[3.,4.]]]])
    low,high=haar(x)
    torch.testing.assert_close(low,torch.tensor([[[[5.]]]]))
    torch.testing.assert_close(high,torch.tensor([[[[-1.]],[[-2.]],[[0.]]]]))
    torch.testing.assert_close(inverse_haar(low,high),x)

@pytest.mark.parametrize('batch',[1,4])
def test_haar_roundtrip(batch):
    x=torch.randn(batch,3,18,22,requires_grad=True);lo,hi=haar(x);y=inverse_haar(lo,hi)
    torch.testing.assert_close(x,y,atol=5e-7,rtol=1e-6);y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))

def test_physics_boundaries_and_gradients():
    y=torch.rand(2,3,8,8);d=torch.rand(2,1,8,8);kd=torch.rand(2,3,1,1,requires_grad=True);kb=torch.rand_like(kd);a=torch.rand_like(kd)
    x,f=render(y,d,kd,kb,a);reconstructed,diag=invert(x,f)
    torch.testing.assert_close(y,reconstructed,atol=1e-6,rtol=1e-6)
    zero,_=render(y,torch.zeros_like(d),kd,kb,a);torch.testing.assert_close(zero,y)
    x.sum().backward();assert torch.isfinite(kd.grad).all() and kd.grad.abs().sum()>0
    _,diag=invert(x,dict(f,t_D=torch.zeros_like(f['t_D'])))
    assert not diag['valid_mask'].any() and torch.isfinite(diag['unclipped']).all()

@pytest.mark.parametrize('value',[0.,.5,1.])
def test_constant_priors(value):
    x=torch.full((4,3,16,16),value);h=histogram(x,bins=8)
    assert torch.isfinite(h).all() and torch.isfinite(sobel(x)).all()
    depth=normalize_depth(x[:,:1]);assert not depth.valid_mask.any();assert depth.distance_proxy.count_nonzero()==0

def test_histogram_independent_scalar():
    x=torch.tensor([[[[.2]],[[.4]],[[.7]]]])
    h=histogram(x,bins=3,bandwidth=.3)[0].numpy()
    rgb=x.numpy().ravel();log=np.log(rgb+1e-6);q=np.linspace(-3,3,3);planes=[]
    for c,others in enumerate(((1,2),(0,2),(0,1))):
        planes.append(np.array([[np.sqrt(np.sum(rgb**2)+1e-6)/(1+((log[c]-log[others[0]]-u)/.3)**2)/(1+((log[c]-log[others[1]]-v)/.3)**2) for v in q] for u in q]))
    expected=np.array(planes);expected/=expected.sum()+1e-6
    np.testing.assert_allclose(h,expected,rtol=1e-5,atol=1e-7)

def test_diffusion_oracle_and_parameterizations():
    schedule=Schedule(20,.0001,.02);x0=torch.rand(4,3,8,8);noise=torch.randn_like(x0);idx=torch.tensor([0,1,10,19]);xt=schedule.q_sample(x0,idx,noise);a,s=schedule.coefficients(idx)
    torch.testing.assert_close(schedule.to_x0(noise,xt,idx,'epsilon'),x0)
    torch.testing.assert_close(schedule.to_x0(a*noise-s*x0,xt,idx,'v'),x0)
    torch.testing.assert_close(schedule.epsilon(xt,x0,idx),noise,atol=5e-6,rtol=1e-5)
    for kind in ('ddpm','ddim'):
        calls=[]
        def oracle(x,t,c):calls.append(t);return x0
        result,diag=sample(oracle,None,x0.shape,schedule,torch.Generator().manual_seed(3),kind,20 if kind=='ddpm' else 7)
        torch.testing.assert_close(result,x0);assert diag['nfe']==len(calls);assert min(t.min() for t in calls)>=0
    with pytest.raises(ValueError):schedule.coefficients(torch.tensor([-1]))

@pytest.mark.parametrize('blocks',[1,2])
@pytest.mark.parametrize('batch',[1,4])
def test_unet_shape_gradient(blocks,batch):
    x=torch.randn(batch,3,16,24);model=UNet(base=4,blocks=blocks,dropout=0);beta=BetaUNet(base=4,blocks=blocks,dropout=0)
    condition={'image':x,'physical':x,'histogram':x,'highfreq':torch.zeros(batch,4,16,24)}
    out=model(x,torch.ones(batch,dtype=torch.long),condition)
    assert out.shape==x.shape and beta(x).shape==(batch,3,1,1)
    out.square().mean().backward();assert model.encoder.input.weight.grad.abs().sum()>0

@pytest.mark.parametrize('hw',[(1,1),(1,31),(17,19)])
def test_padding(hw):
    x=torch.randn(1,3,*hw);out,info=pad_image(x)
    assert min(out.shape[-2:])>=16 and out.shape[-1]%8==0
    torch.testing.assert_close(out[...,:hw[0],:hw[1]],x)

def test_prior_no_reference_and_beta_gradient():
    c=load_config('configs/base/synthetic_smoke.yaml');m=MPADiff(c).eval();x=torch.rand(1,3,16,16)
    a=m.priors.build(x);b=m.priors.build(x)
    torch.testing.assert_close(a.physical_image,b.physical_image)
    with pytest.raises(TypeError):m.priors.build(x,reference=torch.rand_like(x))
    condition,_=m.condition(x);out=m.denoiser(torch.randn_like(x),torch.tensor([10]),condition);out.mean().backward()
    assert sum(p.grad.abs().sum() for p in m.priors.beta.parameters() if p.grad is not None)>0
    assert sum(p.grad.abs().sum() for p in m.highfreq.parameters() if p.grad is not None)>0

def test_metrics_independent_scipy():
    rng=np.random.RandomState(3);a=rng.rand(3,24,27);b=np.clip(a+.05*rng.randn(3,24,27),0,1)
    f=lambda x:gaussian_filter(x,sigma=(0,1.5,1.5),truncate=3.5)[...,5:-5,5:-5]
    u,v=f(a),f(b);score=((2*u*v+.0001)*(2*(f(a*b)-u*v)+.0009))/((u*u+v*v+.0001)*(f(a*a)-u*u+f(b*b)-v*v+.0009))
    assert abs(ssim(torch.tensor(a),torch.tensor(b))-score.mean())<1e-12
    assert psnr(torch.zeros(3,12,12),torch.ones(3,12,12))==0
    assert psnr(torch.ones(3,12,12),torch.ones(3,12,12))==float('inf')

def test_config_fails_closed():
    with pytest.raises(ValueError):merge_strict(copy.deepcopy(DEFAULT),{'typo':1})
    c=copy.deepcopy(DEFAULT);c['extensions']['A']='enabled'
    with pytest.raises(ValueError):validate(c)
    c=copy.deepcopy(DEFAULT);c['depth']['provider']='synthetic_fixture'
    with pytest.raises(ValueError):validate(c)
