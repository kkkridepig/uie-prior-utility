import copy
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from uie_next.records import RunContext,ROOT,digest,write
from uie_next.v2.selection import choose_producer,eligibility,standalone,grid,choose_grid,rescue_choice
from uie_next.v2.diagnostics import exact_strategies,numpy_ssim
from uie_next.v2.statistics import paired_stats
from uie_next.v2.context import V2RoleGuard,config_template,load_config
from uie_next.evaluation import ssim
from uie_next.math.utility import labels
from uie_next.training import optimizer,streams,GroupSampler
from uie_next.checkpoint import save,resume
from uie_next.models.candidate import Candidate


def row(cid='B1_1000',step=1000,**kw):
    return dict(checkpoint_id=cid,training_step=step,S=.001,H32=.2,G32=.1,Hp=.3,Gp=.2,finite_pass=True,
                oracle_block32_psnr=25.,oracle_pixel_psnr=25.1,endpoint_psnr=24.,**kw)


def test_T02_T15_explicit_context_write_guard(tmp_path):
    ctx=RunContext(tmp_path,'v2','abc')
    ctx.write(ctx.run/'state.json',{'test':True})
    with pytest.raises(PermissionError):ctx.write(tmp_path/'runs/v1/selection.json',{})
    for role in ['calibration','sealed_eval']:
        rows=[{'sample_id':'x','role':role,'upstream_exposure':'documented_nonoverlap','backbone_provenance_verified':True}]
        g=V2RoleGuard(rows,ctx)
        for op in ['candidate_diagnostic','model_diagnostic','utility_train','calibrate','final']:
            with pytest.raises(PermissionError):g.check('x',op)


def test_T05_exact_oracle_and_order_active_difference():
    torch.manual_seed(5);b=torch.rand(2,3,256,256);j=(b+.15*torch.randn_like(b)).clamp(0,1);y=torch.rand_like(b)
    variants,err=exact_strategies(b,j,y);assert err <= 1e-10
    bb=torch.full((1,3,256,256),.5);jj=bb+5e-7;yy=bb+.1
    exact,_=exact_strategies(bb,jj,yy)
    assert exact['oracle_pixel'][1].min()==1
    assert exact['oracle_pixel_active'][1].max()==0


def test_T06_standalone_does_not_delete_local_space():
    b=torch.full((1,3,256,256),.5,dtype=torch.float64);y=b+.1;j=b.clone();j[...,:128]=b[...,:128]+.2;j[...,128:]=b[...,128:]-.1
    variants,_=exact_strategies(b.float(),j.float(),y.float())
    mse=lambda z:float((z-y).square().mean())
    assert mse(variants['endpoint'][0])>mse(b)
    assert mse(variants['oracle_pixel'][0])<mse(b)
    zero=row('B1_0',0);zero['endpoint_psnr']=25
    assert standalone([zero,row()])['training_step']==0
    assert choose_producer([zero,row()])['selected']['training_step']==1000


def test_T07_selection_qualification_ties_transfer():
    a=row();b=row('B1_2000',2000);assert choose_producer([b,a])['selected']['checkpoint_id']==a['checkpoint_id']
    fine=row();fine['H32']=.1;assert eligibility(fine)=='FINE_ONLY_HEADROOM'
    bad=row();bad['S']=0;assert choose_producer([bad])['selected'] is None
    assert choose_producer([fine])['status']=='FINE_ONLY_HEADROOM'
    assert choose_producer([a,b])['selected']['checkpoint_id']==a['checkpoint_id']
    transfer=dict(a,S=0);assert eligibility(transfer).startswith('NO_QUALIFIED')


def test_T08_checkpoint_grid_replaces_default_selection():
    rows=[{'psnr':24.,'step':0,'strategy_index':0,'return_base':True,'ssim_delta':0,'lpips_delta':0},
          {'psnr':23.,'step':750,'strategy_index':1,'ssim_delta':0,'lpips_delta':0},
          {'psnr':24.2,'step':750,'strategy_index':2,'ssim_delta':0,'lpips_delta':0}]
    assert choose_grid(rows)['step']==750
    rows[2]['psnr']=24.+1e-9;assert choose_grid(rows)['return_base']
    assert len(grid('O'))==len(grid('G1'))==10 and len(grid('B4'))==5


def test_T13_LOG_per_image_and_negative_loss():
    x=torch.tensor([.001,.01],dtype=torch.float64,requires_grad=True);s=.02
    loss=s*torch.log((x+1e-6)/(s+1e-6)).mean();assert loss<0
    loss.backward();assert torch.allclose(x.grad,s/(2*(x+1e-6)))
    a=row();a.update(endpoint_mse=.009,baseline_mse=.01,endpoint_psnr=23.9,baseline_psnr=24.)
    assert rescue_choice(a)['route']=='LOSS_COMPARISON'
    a['endpoint_mse']=.0101;assert rescue_choice(a)['route']=='LOW_LR_CONTINUATION'


def test_T14_four_vs_two_resume_with_sampler(tmp_path):
    torch.manual_seed(14);initial=Candidate();rng0=streams(20261007);init={k:v.get_state() for k,v in rng0.items()}
    rows=[{'sample_id':str(i),'group_id':str(i//2)} for i in range(8)]
    def advance(m,o,sc,rng,n):
        sampler=GroupSampler(rows,rng['data']);seen=[]
        for _ in range(n):
            ids=sampler.sample(2);noise=torch.rand(2,14,12,12,generator=rng['missing_prior']);seen.append((ids,float(noise.sum())))
            image=noise[:,:3];base=image*.8;target=image*.7
            o.zero_grad();out=m(image,base,noise[:,6:13],noise[:,13:14]);loss=(out-target).square().mean();loss.backward();o.step();sc.step()
        return seen
    full=copy.deepcopy(initial);fo,fs=optimizer(full,80);fr=streams(20261007)
    allseen=advance(full,fo,fs,fr,4)
    part=copy.deepcopy(initial);po,ps=optimizer(part,80);pr=streams(20261007);seen=advance(part,po,ps,pr,2)
    save(tmp_path/'x.pt',part,po,ps,2,{'test':'v2'},pr)
    new=Candidate();no,ns=optimizer(new,80);nr=streams(20261007);resume(tmp_path/'x.pt',new,no,ns,{'test':'v2'},nr)
    seen+=advance(new,no,ns,nr,2)
    assert seen==allseen and all(torch.equal(v,new.state_dict()[k]) for k,v in full.state_dict().items())
    assert fs.state_dict()==ns.state_dict()


def test_T16_image_weighted_group_bootstrap():
    s=paired_stats([0,0,0,4],['a','a','a','b'],repeats=100)
    assert s['mean_image']==1 and s['n_groups']==2
    assert s['aggregation']=='image_weighted'


def test_D0_numpy_metric_matches_locked_original():
    torch.manual_seed(16);x=torch.rand(2,3,24,25);y=torch.rand_like(x)
    a=numpy_ssim(x.numpy(),y.numpy());b=ssim(x,y).numpy()
    assert np.max(np.abs(a-b))<1e-12


def test_T18_terminal_resume_does_not_dispatch(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from uie_next.v2.cli import execute
    import uie_next.v2.delivery as delivery
    state=SimpleNamespace(run=tmp_path,state={'scientific_status':'STOP_PRODUCER_TRANSFER_GATE'})
    calls=[]
    def done(s):calls.append('closeout');return {'terminal':True}
    monkeypatch.setattr(delivery,'deliver',done)
    assert execute(state)=={'terminal':True}
    assert execute(state)=={'terminal':True}
    assert calls==['closeout','closeout']


def test_shared_stop_exception_for_module_execution():
    from uie_next.v2.errors import Stop
    from uie_next.v2 import cli,training,evaluation
    assert cli.Stop is training.Stop is evaluation.Stop is Stop
    try:
        raise training.Stop('STOP_NO_USABLE_PRODUCER','finite scientific stop')
    except cli.Stop as exc:
        assert exc.status=='STOP_NO_USABLE_PRODUCER'
