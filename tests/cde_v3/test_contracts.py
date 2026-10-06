import copy
import pytest
import torch
from mpa_diff.diffusion.core import Schedule,time_grid
from mpa_diff.config import DEFAULT
from mpa_diff.priors.provider import MPADiff
from scripts.cde_v3.model import CrossInjection,pool_tokens,phase_feature,V3Model,fields,token_summary,Selector
from scripts.cde_v3.sampling import sample
from scripts.cde_v3.common import noise,key,roles,MODES

torch.set_num_threads(2)

def encoded(device='cpu'):
    return [(torch.randn(1,12,64,device=device),torch.ones(1,12,dtype=torch.bool,device=device)) for _ in range(3)]

def test_sampler_conversion_endpoint_and_calls():
    s=Schedule(); x=torch.rand(1,3,8,8); e=torch.randn_like(x); t=torch.tensor([999]); xt=s.q_sample(x,t,e)
    assert torch.allclose(s.epsilon(xt,x,t),e,atol=1e-6)
    for n,last in [(1,999),(2,0),(4,0),(20,0)]:
        seen=[]
        def f(xt,index,c): seen.append(index.item()); return x
        result,diag=sample(f,{},x.shape,s,'a',101,n)
        assert seen[-1]==last and len(seen)==n and diag['nfe_measured']==n
        assert torch.equal(result,x)
    assert time_grid(1000,1)==[1000,0] and time_grid(1000,2)==[1000,1,0]

def test_noise_seed_and_stream_isolation():
    a=noise((2,3),101,'x'); torch.manual_seed(45); torch.rand(256)
    assert torch.equal(a,noise((2,3),101,'x'))
    assert not torch.equal(a,noise((2,3),102,'x'))
    assert not torch.equal(a,noise((2,3),101,'x','label'))

def test_histogram_mass_bin_coordinates():
    x=torch.rand(2,3,64,64); t,m=pool_tokens(x,torch.ones(2,1,64,64,dtype=torch.bool),True)
    assert t.shape==(2,256,5)
    assert torch.allclose(t[:,:,:3].sum(1),x.sum((2,3)),atol=.001)
    assert t[0,0,-2:].tolist()==[-1.,-1.] and t[0,-1,-2:].tolist()==[1.,1.]

def test_true_attention_distant_value_and_mask():
    m=CrossInjection(8); h=torch.randn(1,8,4,4); z=encoded(); t=torch.tensor([900])
    assert m.heads==4 and all(p.bias is None for p in m.outputs)
    assert torch.equal(m(h,t,z,'all'),h)
    torch.nn.init.normal_(m.outputs[0].weight,std=.1)
    y=m(h,t,z,'physical'); z2=[(a.clone(),v.clone()) for a,v in z]; z2[0][0][:,-1,0]+=30
    assert not torch.allclose(y[:,:,0,0],m(h,t,z2,'physical')[:,:,0,0])
    z[0][1][:,-1]=False; z2[0][1][:,-1]=False
    assert torch.equal(m(h,t,z,'physical'),m(h,t,z2,'physical'))
    for a,v in z: v[:]=False
    assert torch.equal(m(h,t,z,'all'),h)
    assert torch.equal(m(h,t,z,'null'),h)

def test_mask_nan_sanitized():
    x=torch.randn(1,20,16,16); v=torch.zeros(1,1,16,16,dtype=torch.bool); x[:]=float('nan')
    t,m=pool_tokens(x,v); assert torch.isfinite(t).all() and not m.any()

def test_active_adapter_gradient_after_three_steps():
    m=CrossInjection(8); opt=torch.optim.Adam(m.parameters(),lr=.001); h=torch.rand(1,8,4,4); t=torch.tensor([3])
    inputs=[torch.rand(1,c,16,16) for c in (20,3,32)]; masks=[torch.ones(1,1,16,16,dtype=torch.bool) for _ in inputs]
    for _ in range(3):
        opt.zero_grad(); y=m(h,t,m.encode(inputs,masks),'all'); y.square().mean().backward(); opt.step()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    assert torch.equal(m(h,t,m.encode(inputs,masks),'null'),h)

@pytest.mark.parametrize('shape',[(1,3,17,19),(1,3,16,16)])
@pytest.mark.parametrize('kind',['sobel','phase','softphase'])
def test_fft_constants_odd_and_finite(shape,kind):
    for x in (torch.zeros(shape),torch.ones(shape),torch.randn(shape)*1e-9):
        y,d=phase_feature(x,kind); assert torch.isfinite(y).all()
        if x.std()==0: assert y.abs().max()<1e-6
        if d: assert d['imaginary_max']<1e-5 and d['tau'].min()>=1e-6

def test_roles_guard():
    for role in ['route_fit','route_cal','source_dev','legacy_exposed_regression','confirm_holdout']:
        with pytest.raises(ValueError): roles(role,'enhancer','C_BANK')
    for role in ['route_cal','source_dev','legacy_exposed_regression']:
        with pytest.raises(ValueError): roles(role,'labels')
    assert len(roles('route_fit','enhancer','BASE_CONT_DATA_MATCHED'))>0

def test_oracle_is_not_selector_or_mixture_bound():
    scores=torch.tensor([[0.,2.],[2.,0.]])
    oracle=scores.max(1).values.mean(); uninformed=scores[:,0].mean()
    assert oracle>uninformed  # Identical input features cannot predict alternating winners.
    a=torch.tensor([1.]); b=-a; target=torch.zeros(1)
    assert ((a+b)/2-target).square().mean()<min((a-target).square().mean(),(b-target).square().mean())

def test_selector_only_deployable_inputs():
    m=Selector(); z=encoded(); out=m(torch.rand(1,3,32,32),token_summary(z),torch.ones(1,4))
    assert out.shape==(1,5) and out[0,0]==0
