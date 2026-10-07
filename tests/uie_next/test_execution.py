import math
import torch
from types import SimpleNamespace

from uie_next.training import optimizer


def test_learning_rate_is_set_for_the_update_being_executed():
    model=torch.nn.Linear(1,1)
    total=1000
    opt,schedule=optimizer(model,total)
    warm=math.ceil(.05*total)
    for update in range(1,total+1):
        if update<=warm:
            expected=1e-4*update/warm
        else:
            expected=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(update-warm)/(total-warm)))
        assert abs(opt.param_groups[0]['lr']-expected)<1e-16
        opt.step()
        schedule.step()
    assert abs(expected-1e-5)<1e-16


def test_B4_checkpoint_selection_reads_only_model_val():
    from uie_next.experiment import Experiment
    e=object.__new__(Experiment);access=[]
    rows=[{'sample_id':'model/1'}]
    target=torch.full((1,3,16,16),.4)
    def batch(ids,operation,views):
        access.append((ids,operation,views));return {'image':target,'base':target,'target':target}
    def role(name):
        assert name=='model_val';return rows
    e.data=SimpleNamespace(role=role,batch=batch,real_candidate=lambda image,base,model,rgb:base+.01)
    e.guard=lambda:None
    model=torch.nn.Linear(1,1)
    assert e.validate('B4',model)>30
    assert access==[(['model/1'],'candidate_select',False)]


def test_deployment_has_no_reference_or_cache_dependency(tmp_path):
    from uie_next.timing import Deployment
    from uie_next.records import write
    from uie_next.backbones.ssuie import FrozenBackbone
    from uie_next.models.candidate import Candidate
    from uie_next.models.controls import Controller
    write(tmp_path/'normalization_stats.json',{'s_e2':.01})
    e=SimpleNamespace(run=tmp_path,backbone=FrozenBackbone(torch.nn.Identity()),data=SimpleNamespace(policy='clip01'))
    image=torch.rand(1,3,256,256)
    models={'B1':Candidate(),'O':Controller('O')}
    deploy=Deployment(e,models,{'O':{'policy':{'tau':0.,'lambda':1e-4}}})
    assert torch.equal(deploy(image,'O'),image)
    deploy.selection['O']['policy']={'return_base':True}
    models.clear()
    assert torch.equal(deploy(image,'O'),image)


def test_visual_selection_is_paired_fixed_and_deduplicated():
    from uie_next.visuals import case_ids
    rows=[]
    for n in range(30):
        rows.extend([{'method':'O','sample_id':str(n),'psnr':float(n)},
                     {'method':'B4','sample_id':str(n),'psnr':0.}])
    cases=case_ids(rows,'O','B4')
    assert len({c['sample_id'] for c in cases})==len(cases)
    assert {'0','1','2','3','4'}.issubset({c['sample_id'] for c in cases})
    assert cases==case_ids(rows[::-1],'O','B4')
