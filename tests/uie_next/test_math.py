import pytest
import torch

from uie_next.math.utility import geometry, labels, decision, oracle
from uie_next.models.utility import UtilityNetwork
from uie_next.models.controls import Controller, covariance
from uie_next.models.candidate import Candidate


def fixture():
    torch.manual_seed(20261007)
    base = torch.rand(2, 3, 8, 8, dtype=torch.float64)
    target = torch.rand_like(base)
    candidate = torch.rand_like(base)
    return base, candidate, target


def test_T01_T02_quadratic_and_endpoint():
    base, candidate, target = fixture()
    g = geometry(candidate - base)
    lab = labels(base, candidate, target)
    l0 = (base-target).square().mean(1, keepdim=True)
    l1 = (candidate-target).square().mean(1, keepdim=True)
    assert (2*lab['b'] - (l0+g['a']-l1)).abs().max() < 1e-10
    for strength in [0., .17, .5, 1.]:
        direct = (base+strength*g['r']-target).square().mean(1, keepdim=True)-l0
        assert (direct-(g['a']*strength**2-2*lab['b']*strength)).abs().max()<1e-10
    assert torch.allclose(lab['b'], g['c']*lab['v'], atol=1e-10, rtol=0)
    assert torch.allclose(lab['U'], l0-l1, atol=1e-10, rtol=0)


def test_T03_T04_oracles():
    base, candidate, target = fixture()
    lab = labels(base, candidate, target)
    for block in [1, 4]:
        out, alpha = oracle(base, candidate, target, block=block)
        assert ((out-target).square().mean() <= (base-target).square().mean()+1e-10)
        for strength in torch.linspace(0, 1, 1001):
            constant = base+strength*(candidate-base)
            assert (out-target).square().mean() <= (constant-target).square().mean()+1e-10
        assert torch.isfinite(alpha).all()
    zero, a = oracle(base, base, target)
    assert torch.equal(zero, base) and not a.any()


def test_T05_actual_clipped_residual():
    m = Candidate().double()
    with torch.no_grad(): m.net.output.bias.fill_(3.)
    base = torch.ones(1,3,8,8,dtype=torch.float64)*.99
    image = torch.ones_like(base)*.2
    candidate = m(image, base, torch.zeros(1,7,8,8,dtype=torch.float64), torch.ones(1,1,8,8,dtype=torch.float64))
    r = candidate-base
    assert r.max() <= .01000000000001
    assert not torch.allclose(r, torch.ones_like(r)*.25*torch.tanh(torch.tensor(3.)))


def test_T06_T09_T11_T12_boundaries_and_no_safety_theorem():
    base, candidate, target = fixture()
    for magnitude in [0., 1e-7, 1e-6, 1.01e-6]:
        g=geometry(torch.ones_like(base)*magnitude)
        out, alpha = decision(base, g, torch.ones_like(g['a']), 0, 1e-4)
        assert torch.isfinite(g['u']).all() and torch.isfinite(out).all()
        if magnitude <= 1e-6: assert torch.equal(out, base) and not alpha.any()
    g=geometry(candidate-base)
    out, _=decision(base,g,torch.ones_like(g['a']),0,1e-4)
    assert out.min()>=0 and out.max()<=1
    base=torch.zeros(1,3,2,2,dtype=torch.float64);candidate=torch.ones_like(base)
    out,_=decision(base,geometry(candidate-base),torch.ones(1,1,2,2,dtype=torch.float64)*2,0,1e-4)
    assert out.square().mean()>base.square().mean()


def test_T07_T08_T10_shared_odd_signed_homogeneous():
    m=UtilityNetwork().double()
    with torch.no_grad(): torch.nn.init.normal_(m.h.output.weight, std=.02)
    base,candidate,_=fixture();image=base*.5;r=candidate-base
    original=m(image,base,r)
    negative=m(image,base,-r)
    assert torch.allclose(original['b_hat'],-negative['b_hat'],atol=1e-10,rtol=0)
    assert original['v_hat'].min()<0 and original['v_hat'].max()>0
    for factor in [.5,2.]:
        scaled=m(image,base,r*factor)
        assert torch.allclose(scaled['b_hat'],original['b_hat']*factor,atol=1e-10,rtol=0)


def test_T13_covariance_psd_and_oracle():
    base,candidate,target=fixture();e0=base-target;e1=candidate-target
    c00=e0.square().mean(1,keepdim=True);c11=e1.square().mean(1,keepdim=True);c01=(e0*e1).mean(1,keepdim=True)
    g=geometry(candidate-base);lab=labels(base,candidate,target)
    assert torch.allclose(g['a'],c00+c11-2*c01,atol=1e-10,rtol=0)
    assert torch.allclose(lab['b'],c00-c01,atol=1e-10,rtol=0)
    alpha_c=((c00-c01)/(c00+c11-2*c01)).clamp(0,1)
    alpha_b=(lab['b']/g['a']).clamp(0,1)
    assert torch.allclose(alpha_c,alpha_b,atol=1e-10,rtol=0)
    c=covariance(torch.randn(3,3,8,8,dtype=torch.float64),.02)
    assert ((c[:,0]*c[:,1]-c[:,2].square())>=0).all()


def test_T04_independent_block_grid_minimum():
    base,candidate,target=fixture();out,alpha=oracle(base,candidate,target,block=4)
    strengths=torch.linspace(0,1,1001,dtype=torch.float64).view(1001,1,1,1)
    for i in range(2):
        for y in [0,4]:
            for x in [0,4]:
                j0=base[i:i+1,:,y:y+4,x:x+4];j1=candidate[i:i+1,:,y:y+4,x:x+4];gt=target[i:i+1,:,y:y+4,x:x+4]
                grid=(j0+strengths*(j1-j0)-gt).square().mean((1,2,3)).min()
                actual=(out[i:i+1,:,y:y+4,x:x+4]-gt).square().mean()
                assert actual<=grid+1e-12
                assert alpha[i,:,y:y+4,x:x+4].max()==alpha[i,:,y:y+4,x:x+4].min()


@pytest.mark.parametrize('name,expected',[('B1',115907),('B3',115907),('G0',10),('G1',76897),('G2',150256),('F0',77475),('R0',76611),('O',76896),('O-NI',76896),('O-NP',76896),('O-NS',77185),('O-ND',76896)])
def test_parameter_contract(name,expected):
    m=Candidate() if name in ('B1','B3') else Controller(name)
    assert sum(p.numel() for p in m.parameters())==expected
