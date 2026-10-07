"""Local head gradients and recovery, without an official backbone substitute."""
import copy
import torch
import pytest
from uie_next.models.candidate import Candidate
from uie_next.models.controls import Controller
from uie_next.models.utility import UtilityNetwork
from uie_next.math.utility import labels
from uie_next.priors.heuristic import interventions
from uie_next.training import pair_objective,frozen_candidate,optimizer,streams
from uie_next.checkpoint import save,resume


def fixture(n=2,size=16):
    torch.manual_seed(20261007)
    image=torch.rand(n,3,size,size);base=.1+.8*image
    target=(base+.05*torch.randn_like(base)).clamp(0,1)
    c=(base[:,None]+.03*torch.randn(n,2,3,size,size)).clamp(0,1)
    return image,base,c,target


def test_T14_intervention_labels_from_actual_candidate_not_names():
    image,base,_,target=fixture()
    m=Candidate()
    with torch.no_grad():torch.nn.init.normal_(m.net.output.weight,std=.04)
    views=interventions(image);out=[]
    for key in ['nominal','tau_075']:
        f=views[key];candidate=m(image,base,f['P'],f['V'])
        lab=labels(base,candidate,target);out.append(lab)
        assert torch.allclose(lab['U'],lab['l0']-lab['l1'],atol=1e-7,rtol=0)
    assert not torch.equal(out[0]['r'],out[1]['r'])
    assert torch.equal(out[0]['U']-out[1]['U'],-(out[1]['U']-out[0]['U']))


@pytest.mark.parametrize('name',['G0','G1','G2','F0','R0','O','O-NI','O-NP','O-NS','O-ND'])
def test_T28_T29_controller_gradients_frozen_candidate(name):
    image,base,c,target=fixture()
    candidate=Candidate()
    with torch.no_grad():torch.nn.init.normal_(candidate.net.output.weight,std=.03)
    candidate=frozen_candidate(candidate);old=copy.deepcopy(candidate.state_dict())
    views=interventions(image)
    with torch.no_grad():
        c=torch.stack([candidate(image,base,views[k]['P'],views[k]['V']) for k in ['nominal','tau_075']],1)
    if name=='O-NI':c=c[:,0:1].repeat(1,2,1,1,1)
    controller=Controller(name);opt,_=optimizer(controller,1000)
    loss,parts=pair_objective(controller,image,base,c,target,{'s_v':.05,'s_U':.003,'s_e2':.003})
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in controller.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    opt.step()
    assert all(torch.equal(candidate.state_dict()[k],v) for k,v in old.items())
    assert set(map(id,candidate.parameters())).isdisjoint(set(id(p) for g in opt.param_groups for p in g['params']))


def test_T30_shared_h_parameter_objects_and_signed_gradient():
    m=UtilityNetwork();seen=[]
    hook=m.h.register_forward_hook(lambda layer,args,out:seen.append(id(layer.output.weight)))
    image,base,c,target=fixture();r=c[:,0]-base
    lab=labels(base,c[:,0],target)
    out=m(image,base,r)
    ((out['v_hat']-lab['v']).square().mean()).backward()
    hook.remove()
    assert len(seen)==2 and len(set(seen))==1
    assert m.h.output.weight.grad.abs().sum()>0
    assert lab['v'].min()<0<lab['v'].max()


def test_T31_accumulation_by_source():
    image,base,c,target=fixture(n=4)
    a=Controller('O').double();b=copy.deepcopy(a)
    items=[x.double() for x in [image,base,c,target]]
    sc={'s_v':.05,'s_U':.003,'s_e2':.003}
    pair_objective(a,*items,sc)[0].backward()
    for j in range(4):pair_objective(b,*[x[j:j+1] for x in items],sc)[0].div(4).backward()
    for x,y in zip(a.parameters(),b.parameters()):
        assert torch.allclose(x.grad,y.grad,atol=1e-10,rtol=1e-9)


def test_T32_rng_optimizer_scheduler_resume(tmp_path):
    torch.manual_seed(32);initial=Controller('O');s=streams(20261007)
    starts={n:g.get_state() for n,g in s.items()}
    image,base,c,target=fixture();sc={'s_v':.05,'s_U':.003,'s_e2':.003}
    def advance(m,o,sched,g,steps):
        for _ in range(steps):
            idx=torch.randperm(2,generator=g['data'])
            jitter=torch.randn(c.shape,generator=g['view'])*.001
            o.zero_grad();loss,_=pair_objective(m,image[idx],base[idx],(c+jitter)[idx],target[idx],sc)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step();sched.step()
    full=copy.deepcopy(initial);fo,fs=optimizer(full,1000)
    advance(full,fo,fs,s,20)
    for n,g in s.items():g.set_state(starts[n])
    part=copy.deepcopy(initial);po,ps=optimizer(part,1000)
    advance(part,po,ps,s,10)
    path=tmp_path/'resume.pt';identity={'test':'synthetic_CPU_only','seed':20261007}
    save(path,part,po,ps,10,identity,s)
    newer=Controller('O');no,ns=optimizer(newer,1000);ng=streams(20261007)
    state=resume(path,newer,no,ns,identity,ng);assert state['global_step']==10
    advance(newer,no,ns,ng,10)
    assert all(torch.equal(full.state_dict()[k],v) for k,v in newer.state_dict().items())
    assert fs.state_dict()==ns.state_dict()
