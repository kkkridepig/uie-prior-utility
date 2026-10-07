"""Local utility protocol tests with names independent of the legacy suite."""
import copy
import inspect
import json
import math
import torch
import pytest
from uie_next.budget import DeviceBudget,BudgetStop
from uie_next.checkpoint import atomic_torch
from uie_next.data.cache import put,get,REQUIRED
from uie_next.data.roles import RoleGuard
from uie_next.data.audit import Components,features,exact_edges,clarify_membership
from uie_next.evaluation import psnr,ssim
from uie_next.models.controls import Controller
from uie_next.calibration import grid,select
from uie_next.backbones.ssuie import FrozenBackbone,postprocess


def test_T16_role_and_unknown_exposure_guard():
    rows=[{'sample_id':n,'role':n,'upstream_exposure':'documented_nonoverlap','backbone_provenance_verified':True} for n in ['utility_fit','calibration','sealed_eval']]
    guard=RoleGuard(rows)
    guard.check('utility_fit','utility_train')
    for role in ['calibration','sealed_eval']:
        with pytest.raises(PermissionError):guard.check(role,'utility_train')
    with pytest.raises(PermissionError):guard.check('sealed_eval','final')
    rows[0]['upstream_exposure']='unknown_upstream_exposure'
    with pytest.raises(PermissionError):RoleGuard(rows).check('utility_fit','label_scales')


def test_author_source_pool_does_not_prove_each_image_was_trained():
    rows=[{'dataset':'LSUI','role':'model_fit','upstream_exposure':'known_upstream_train'},
          {'dataset':'UIEB','role':'excluded_overlap','upstream_exposure':'known_upstream_train'},
          {'dataset':'UIEB','role':'utility_fit','upstream_exposure':'documented_nonoverlap'}]
    clarified=clarify_membership(rows)
    assert clarified[0]['upstream_exposure']==clarified[1]['upstream_exposure']=='unknown_upstream_exposure'
    assert clarified[2]['upstream_exposure']=='documented_nonoverlap'
    assert all(not r['backbone_provenance_verified'] for r in rows)
    assert [r['role'] for r in rows]==['model_fit','excluded_overlap','utility_fit']


def test_T18_transitive_group():
    c=Components(4);c.join(0,1);c.join(1,2)
    assert c.find(0)==c.find(2) and c.find(0)!=c.find(3)


def test_T17_input_reference_swapped_duplicates(tmp_path):
    from PIL import Image
    import numpy as np
    rng=np.random.RandomState(17)
    first=rng.randint(0,256,(16,16,3),dtype=np.uint8);second=rng.randint(0,256,(16,16,3),dtype=np.uint8)
    entries=[]
    for owner,kind,array in [(0,'input',first),(0,'reference',second),(1,'input',second),(1,'reference',first)]:
        path=tmp_path/('%d_%s.png'%(owner,kind));Image.fromarray(array).save(path)
        feat,_=features(path);entries.append(dict(feat,owner=owner,kind=kind))
    c=Components(2);edges=exact_edges(entries,c)
    assert c.find(0)==c.find(1) and len(edges)==2
    assert all(entries[e['left']]['kind']!=entries[e['right']]['kind'] for e in edges)


def test_T20_reference_file_is_never_read_by_deployment(tmp_path):
    from PIL import Image
    import numpy as np
    from uie_next.data.manifest import inference_input
    from uie_next.models.system import Deployment
    from uie_next.models.candidate import Candidate
    path=tmp_path/'input.png';ref=tmp_path/'reference.png'
    Image.fromarray(np.full((256,256,3),100,dtype=np.uint8)).save(path)
    Image.fromarray(np.zeros((256,256,3),dtype=np.uint8)).save(ref)
    raw=torch.nn.Conv2d(3,3,1);system=Deployment(FrozenBackbone(raw),Candidate(),Controller('O'))
    before=system(inference_input(path)[None])['output']
    ref.unlink()
    after=system(inference_input(path)[None])['output']
    assert torch.equal(before,after)


def test_T19_all_identity_fields_invalidate(tmp_path):
    identity={k:k for k in REQUIRED};path=tmp_path/'cache.pt'
    put(path,identity,{'base':torch.zeros(1,3,4,4)})
    get(path,identity)
    for key in REQUIRED:
        altered=dict(identity);altered[key]='changed'
        with pytest.raises(ValueError):get(path,altered)


def test_T21_deployment_has_no_metadata():
    assert set(inspect.signature(Controller.forward).parameters).isdisjoint({'Y','target','role','sample_id','intervention_id','oracle'})


def test_T22_T23_local_output_and_actual_candidate():
    m=Controller('O')
    torch.nn.init.normal_(m.net.h.output.weight,std=.03)
    image=torch.rand(1,3,16,16);base=torch.rand_like(image)*.5
    candidate=(base+.05*torch.randn_like(image)).clamp(0,1)
    out=m(image,base,candidate)
    assert out['alpha'].std()>0
    zero=m(image,base,base)
    assert not zero['alpha'].any() and not zero['v_hat'].any()
    assert torch.equal(zero['output'],base)
    assert torch.allclose(out['a'],(candidate-base).square().mean(1,keepdim=True))


def test_T27_frozen_wrapper_contract_synthetic_not_official():
    raw=torch.nn.Sequential(torch.nn.BatchNorm2d(3),torch.nn.Conv2d(3,3,1))
    m=FrozenBackbone(raw);before=copy.deepcopy(m.state_dict());m.train()
    assert not raw.training and not any(p.requires_grad for p in raw.parameters())
    m(torch.rand(2,3,256,256))
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_T33_atomic_failure_preserves_old(tmp_path,monkeypatch):
    path=tmp_path/'model.pt';atomic_torch(path,{'old':torch.tensor(2)})
    import uie_next.checkpoint as cp
    def fail(*args):raise OSError('simulated interrupt before rename')
    monkeypatch.setattr(cp.os,'replace',fail)
    with pytest.raises(OSError):atomic_torch(path,{'new':torch.tensor(3)})
    assert 'old' in torch.load(path)


def test_T34_device_budget_reserve_lease_and_stop(tmp_path):
    now=[0.]
    with DeviceBudget(tmp_path,max_hours=.02,reserve_hours=.01,clock=lambda:now[0]) as b:
        b.start('test',20)
        with pytest.raises(BudgetStop):
            with DeviceBudget(tmp_path,max_hours=.02,reserve_hours=.01,clock=lambda:now[0]):pass
        now[0]=35
        with pytest.raises(BudgetStop):b.guard(save_period_seconds=2)
        b.stop()
        with pytest.raises(BudgetStop):b.start('next',2)
    s=json.loads((tmp_path/'budget.json').read_text())
    assert s['used_device_seconds']==35 and s['active'] is None


def test_T36_psnr_ssim_independent():
    import numpy as np
    from skimage.metrics import structural_similarity
    torch.manual_seed(36);x=torch.rand(2,3,32,32);y=torch.rand_like(x)
    val,mse=psnr(x,y)
    assert val[0]==-10*math.log10(float(mse[0]))
    exact,_=psnr(x,x);assert exact==['+inf','+inf']
    values=ssim(x,y)
    for i in range(2):
        a=x[i].double().permute(1,2,0).numpy();b=y[i].double().permute(1,2,0).numpy()
        independent=structural_similarity(a,b,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)
        assert abs(independent-float(values[i]))<1e-10


def test_T37_batch_postprocess_and_constants():
    a=torch.rand(1,3,4,4)*.6-1;b=torch.rand(1,3,4,4)*4+2
    for policy in ['clip01','official_minmax_float']:
        batch=postprocess(torch.cat([a,b]),policy)
        assert torch.equal(batch[:1],postprocess(a,policy))
        assert torch.equal(batch[1:],postprocess(b,policy))
        assert torch.isfinite(postprocess(torch.ones_like(a),policy)).all()


def test_fixed_calibration_ties_and_constraints():
    assert grid('O')[0]=={'return_base':True} and len(grid('O'))==10
    assert len(grid('G1'))==10 and len(grid('B4'))==5
    rows=[{'psnr':1,'ssim_delta':0,'lpips_delta':0},{'psnr':3,'ssim_delta':-.002,'lpips_delta':0},{'psnr':1+1e-9,'ssim_delta':0,'lpips_delta':0}]
    assert select(rows)==0


def test_T35_pipeline_resume_idempotent(tmp_path):
    from uie_next.pipeline import advance
    config={'runtime':{'run_dir':str(tmp_path/'runs/test'),'run_id':'test'}}
    events=[]
    def ok():events.append('ok');return {'passed':True}
    def blocked():events.append('blocked');return {'passed':False,'status':'BLOCKED_BACKBONE','reasons':['missing']}
    handlers=[('S0',ok),('S1',blocked)]
    advance(config,handlers);advance(config,handlers)
    assert events==['ok','blocked','blocked']
    state=json.loads((tmp_path/'runs/test/state.json').read_text())
    assert state['completed']==['S0'] and not state['sealed_eval_released']
