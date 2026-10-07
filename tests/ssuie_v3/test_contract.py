import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from uie_next.v3.moments import *
from uie_next.v3.descriptors import descriptors,torch_descriptors
from uie_next.v3.models import MomentController,objective
from uie_next.v3.ridge_probe import fit,predict,decide
from uie_next.v3.context import Data
from uie_next.records import RunContext


def test_quadratic_and_relative_identity():
    rng=np.random.default_rng(7);base=rng.random((2,3,64,64));j=rng.random(base.shape);y=rng.random(base.shape);g=geometry(base,j,y,32);alpha=rng.random(g['A'].shape)
    actual=((base+expand(alpha,32)*g['r']-y)**2).mean((1,2,3))
    assert np.max(np.abs(actual-mse_from(g['mse0'],g['A'],g['B'],alpha)))<1e-10
    ref=.3;C=g['B']-ref*g['A'];t=alpha-ref
    assert np.max(np.abs((2*g['B']*alpha-g['A']*alpha**2)-(2*g['B']*ref-g['A']*ref**2)-(2*C*t-g['A']*t*t)))<1e-10


def test_regret_boundary_and_zero():
    A=np.array([0.,1,1,1]);B=np.array([0.,-1,.3,2]);alpha=np.array([.9,.2,.7,.4]);star=oracle_alpha(A,B)
    total,sq,bd=regret(A,B,alpha)
    assert np.allclose(total,A*alpha**2-2*B*alpha-(A*star**2-2*B*star),atol=1e-10)
    assert bd[1]>0 and bd[3]>0 and star[0]==0


def test_aggregate_before_ratio_and_order():
    rng=np.random.default_rng(4);b=rng.random((1,3,64,64));j=rng.random(b.shape);y=rng.random(b.shape);ms=[]
    for block in [1,32,64]:
        g=geometry(b,j,y,block);ms.append(mse_from(g['mse0'],g['A'],g['B'],oracle_alpha(g['A'],g['B']))[0])
    assert ms[0]<=ms[1]<=ms[2]
    A=np.array([1.,9]);B=np.array([1.,0]);assert not np.isclose(oracle_alpha(A.mean(),B.mean()),oracle_alpha(A,B).mean())


def test_regularization_center_grid_and_negative_replacement():
    A=np.array([1.,2,0]);B=np.array([.8,-.2,0]);ref=.4;lam=.2
    al=relative_alpha(A,B-ref*A,ref,lam);grid=np.linspace(0,1,100001)
    for i in [0,1]:
        best=grid[np.argmin(A[i]*grid**2-2*B[i]*grid+lam*(grid-ref)**2)]
        assert abs(best-al[i])<1e-5
    # A predictor can undo regularization: replacement term is signed.
    astar=oracle_alpha(A[:1],B[:1]);assert (A[:1]*astar**2-2*B[:1]*astar-(A[:1]*al[:1]**2-2*B[:1]*al[:1]))[0]<0


def test_descriptor_and_nearest_contract():
    I=np.zeros((1,3,256,256));I[:,:,:,128:]=1;J=I*.5;r=I*.1
    fr,fg=descriptors(I,J,J+r)
    assert fr.shape==(1,26,8,8) and fg.shape==(1,26,1,1)
    assert fg[0,13,0,0]>.4 and np.mean(fr[0,13])==0
    a=np.arange(64).reshape(1,1,8,8);ex=expand(a,32);assert np.all(ex[:,:,32:64,64:96]==10)
    tr,tg=torch_descriptors(torch.tensor(I).float(),torch.tensor(J).float(),torch.tensor(J+r).float())
    assert np.allclose(tr.numpy(),fr,atol=1e-5) and np.allclose(tg.numpy(),fg,atol=1e-5)


def test_ridge_fit_only_validation_targets_and_intercept():
    rng=np.random.default_rng(2);N=12;A=rng.random((N,64))*.001;B=A*.2
    d={'A':A,'B':B,'mse0':np.ones(N)*.01,'f_region':rng.random((N,64,26)).astype('float32'),'f_global':rng.random((N,26)).astype('float32')}
    train=np.arange(8);m=fit(d,train,'global');d2=copy.deepcopy(d);d2['B'][8:]+=10
    assert fit(d2,train,'global')==m
    assert not m['intercept_penalized'] and m['ridge_penalty']==.01
    p,_=predict(m,d,np.arange(8,12));assert p.shape==(4,64)


def test_step0_shared_initialization_zero_residual():
    c={'alpha_ref':.3,'lambda0':1e-4,'s_C':.01};methods=['MOM_R','MOM_G','MOM_R_NC','MOM_R_NM','DIRECT_R','MOM_G_NM','DIRECT_G'];models=[MomentController(m,c) for m in methods]
    r=torch.randn(2,26,8,8);g=torch.randn(2,26,1,1)
    for m in models:
        assert sum(p.numel() for p in m.parameters())==5505
        A=torch.ones(2,1,1,1) if '_G' in m.method else torch.ones(2,1,8,8)
        p=m(r,g,A);assert torch.allclose(p['alpha'],torch.full_like(A,.3),atol=1e-7)
        assert torch.equal(m.net[0].weight,models[0].net[0].weight)
    z=models[0](r,g,torch.zeros(2,1,8,8));assert torch.allclose(z['alpha'],torch.full_like(z['alpha'],.3))


def test_no_context_does_not_read_global():
    c={'alpha_ref':.3,'lambda0':1e-4,'s_C':.01};m=MomentController('MOM_R_NC',c)
    with torch.no_grad():m.net[-1].weight.fill_(.2)
    r=torch.randn(1,26,8,8);A=torch.ones(1,1,8,8)
    assert torch.equal(m(r,torch.randn(1,26,1,1),A)['alpha'],m(r,torch.randn(1,26,1,1)*100,A)['alpha'])


@pytest.mark.parametrize('sign',[-1,1])
def test_gradients_both_signs_and_early_layers(sign):
    c={'alpha_ref':.3,'lambda0':1e-4,'s_C':.01};m=MomentController('MOM_R',c);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    r=torch.randn(2,26,8,8);g=torch.randn(2,26,1,1);A=torch.full((2,1,8,8),.01);B=A*.3+sign*.001;C=B-.3*A
    for n in range(2):
        opt.zero_grad();p=m(r,g,A);loss,parts=objective(p,A,C,torch.ones(2)*.1,B,c,'MOM_R');loss.backward()
        assert m.net[-1].weight.grad.norm()>0
        if n==1:assert m.net[0].weight.grad.norm()>0
        opt.step()


def test_decision_loss_per_image_and_label_detach():
    c={'alpha_ref':.3,'s_C':.01};A=torch.ones(2,1,8,8)*.01;B=A*.2;alpha=torch.ones_like(A,requires_grad=True)*.4;C=(B-.3*A).requires_grad_();pred={'alpha':alpha,'C_hat':torch.zeros_like(A,requires_grad=True)}
    loss,parts=objective(pred,A,C,torch.tensor([.1,.2]),B,c,'MOM_R');loss.backward()
    assert C.grad is None
    assert torch.allclose(parts['dec'],torch.log((parts['mse']+1e-6)/(torch.tensor([.1,.2])+(.01*.09-2*.002*.3)+1e-6)).mean())


def test_null_amplitude_and_alpha_shuffle():
    rng=np.random.default_rng(5);base=rng.random((3,8,8));r=rng.random(base.shape)*.1;candidate=np.clip(base+r,0,1);r=candidate-base
    rn,rm,eta=matched_null(base,r,'fixture',1)
    assert np.allclose(np.mean(rn**2,0),np.mean(rm**2,0),atol=1e-14)
    assert (base+rn).min()>=-1e-12 and (base+rn).max()<=1+1e-12
    al=rng.random((1,1,8,8));variants=spatial_variants(al,np.ones_like(al),'fixture')
    assert np.array_equal(np.sort(al.ravel()),np.sort(variants['shuffle'].ravel()))
    assert np.array_equal(variants['shuffle'],spatial_variants(al,np.ones_like(al),'fixture')['shuffle'])


def test_role_and_write_guards(tmp_path):
    d=Data.__new__(Data);d.s=SimpleNamespace(run=tmp_path,state={'status':'RECOVERY_AUDIT'})
    for role in ['calibration','sealed_eval','model_fit','excluded_overlap']:
        with pytest.raises(PermissionError):d.check({'sample_id':'fixture','role':role})
    ctx=RunContext(tmp_path,'fixture','a')
    with pytest.raises(PermissionError):ctx.write(tmp_path/'runs/old/state.json',{})


def test_route_priority_and_stop():
    def p(g):return {'vs_fixed':{'mean_image':g},'delta_base':g,'harm_rate':0}
    def s(g,r):return {'RIDGE_GLOBAL':p(g),'RIDGE_REGION':p(r),'FIXED_FIT':{'harm_rate':0},'region_minus_global':{'mean_image':r-g},'oracle_region_minus_global':.1}
    assert decide({'oof':s(.04,.07),'utility_val':s(.04,.07)})['route']=='TRAIN_REGION'
    assert decide({'oof':s(.04,.04),'utility_val':s(.04,.04)})['route']=='TRAIN_GLOBAL'
    assert decide({'oof':s(0,0),'utility_val':s(0,0)})['route']=='STOP_NO_PREDICTABILITY_SIGNAL'
    assert decide({'oof':s(0,.04),'utility_val':s(0,.04)})['route']=='TRAIN_REGION'


def test_ridge_matches_independent_augmented_least_squares():
    rng=np.random.default_rng(19);N=30;d={'A':rng.uniform(.001,.01,(N,64)),'B':rng.uniform(-.002,.006,(N,64)),
        'mse0':np.full(N,.1),'f_region':rng.normal(size=(N,64,26)).astype('float32'),'f_global':rng.normal(size=(N,26)).astype('float32')}
    from uie_next.v3.ridge_probe import features
    from uie_next.v3.descriptors import standardize
    for scale in ['global','region']:
        idx=np.arange(N);m=fit(d,idx,scale);X=features(d,idx,scale).reshape(-1,len(m['coefficients'])-1);X=standardize(X,m['standardization']);X=np.column_stack([X,np.ones(len(X))]);A=d['A'];B=d['B']
        if scale=='global':A=A.mean(1,keepdims=True);B=B.mean(1,keepdims=True)
        y=((B-m['constants']['alpha_ref']*A)/m['constants']['s_C']).ravel();pen=np.eye(X.shape[1]);pen[-1,-1]=0
        aug=np.concatenate([X/np.sqrt(len(X)),np.sqrt(.01)*pen]);yy=np.concatenate([y/np.sqrt(len(X)),np.zeros(X.shape[1])])
        w=np.linalg.lstsq(aug,yy,rcond=None)[0];assert np.allclose(w,m['coefficients'],atol=1e-10,rtol=1e-10)


def test_group_bootstrap_image_weighting_unequal_sizes():
    from uie_next.v2.statistics import paired_stats
    values=np.array([1.,1.,1.,-1.]);groups=['a','a','a','b'];s=paired_stats(values,groups,repeats=5000)
    assert s['mean_image']==.5
    assert np.mean([np.mean(values[:3]),values[-1]])==0
    assert s==paired_stats(values,groups,repeats=5000)


def test_original_loss_component_gradients_live_cpu():
    from uie_next.models.controls import Controller
    from uie_next.training import pair_objective
    torch.manual_seed(6);m=Controller('O').double();I=torch.rand(2,3,12,12,dtype=torch.float64);base=torch.rand_like(I);cand=torch.rand(2,2,3,12,12,dtype=torch.float64);Y=torch.rand_like(I)
    total,parts=pair_objective(m,I,base,cand,Y,{'s_v':.1,'s_U':.01,'s_e2':.1});params=list(m.parameters());components=[total-.25*parts['pair_live']-.1*parts['dec_live'],.25*parts['pair_live'],.1*parts['dec_live']]
    grads=[torch.cat([g.reshape(-1) for g in torch.autograd.grad(t,params,retain_graph=True)]) for t in components];gt=torch.cat([g.reshape(-1) for g in torch.autograd.grad(total,params)])
    assert np.max(np.abs((sum(grads)-gt).detach().numpy()))<1e-10
    assert not parts['proj'].requires_grad and parts['pair_live'].requires_grad


def test_ordinary_gating_missing_predicted_utility_is_null():
    from uie_next.scientific_evaluation import prediction_utility_statistics
    p={'alpha':torch.ones(1,1,3,3)*.5};truth={'U':torch.ones(1,1,3,3)}
    s=prediction_utility_statistics(p,truth)
    assert s['U_hat_mse'] is None and s['U_hat_true_correlation'] is None


def test_stage_budget_cap_and_orphan_reconciliation(tmp_path,monkeypatch):
    import time
    import uie_next.v3.context as ctxmod
    from uie_next.records import write,read
    from uie_next.budget import BudgetStop
    from uie_next.v3.context import State
    monkeypatch.setattr(ctxmod,'ROOT',tmp_path);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    s=State.__new__(State);s.run=tmp_path/'runs/new';s.run.mkdir(parents=True);s.live=lambda **kw:None;s.active=None;s.stage=None
    write(s.run/'budget.json',{'max_device_hours':16,'reserve_device_hours':3.5,'used_device_seconds':100.,'inherited_device_seconds':100.,'active':None})
    write(s.run/'stage_budget.json',{'1':3599.,'2':0.,'3':0.})
    with pytest.raises(BudgetStop):
        with s.device_job('too_large',estimate=2,stage=1):assert False
    write(s.run/'stage_budget.json',{'1':0.,'2':0.,'3':0.,'_active':{'stage':1,'job':'orphan','baseline_seconds':100.}})
    write(s.run/'budget.json',{'max_device_hours':16,'reserve_device_hours':3.5,'used_device_seconds':105.,'inherited_device_seconds':100.,'active':{'job':'orphan','devices':1,'last_accounted_at':time.time()-20,'final':False}})
    with s.device_job('resume',estimate=2,stage=1):pass
    state=read(s.run/'stage_budget.json');assert state['1']>=25 and '_active' not in state
    before=state['1']
    with s.device_job('resume_again',estimate=2,stage=1):pass
    assert read(s.run/'stage_budget.json')['1']-before<2


def test_reference_free_ridge_deploy_signature():
    import inspect
    from uie_next.v3.evaluation import deploy_ridge
    assert list(inspect.signature(deploy_ridge).parameters)==['image','backbone','producer','ridge','kappa']


def test_resume_rejects_missing_or_changed_artifact(tmp_path):
    import hashlib
    from uie_next.v3.recovery import require_hashes
    p=tmp_path/'paired.csv';p.write_text('same identity\n');h=hashlib.sha256(p.read_bytes()).hexdigest();require_hashes(tmp_path,{'paired.csv':h})
    p.write_text('different identity\n')
    with pytest.raises(ValueError):require_hashes(tmp_path,{'paired.csv':h})
    p.unlink()
    with pytest.raises(ValueError):require_hashes(tmp_path,{'paired.csv':h})
