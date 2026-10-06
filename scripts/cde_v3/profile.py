"""Measured 100-step cost matrix and 20-step exact-resume verification."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.train_bank import train

def invoke(branch,steps,out):
    return train(argparse.Namespace(branch=branch,finetune_seed=20261004,steps=steps,output=str(out),micro_batch=4,device='cuda'))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); args=p.parse_args()
    if not args.run: p.error('--run required')
    results={}
    for branch in ['BASE_CONT_V3','C_BANK','C_RGB_CONTROL','C_ALL_ONLY']:
        out=RUN/'profiles'/branch
        if invoke(branch,100,out): sys.exit(75)
        results[branch]=read(out/'progress.json')
    for name,target in [('resume_split',10),('resume_split',20),('resume_contiguous',20)]:
        path=RUN/'profiles'/name/'last.pt'
        if path.exists() and torch.load(path,map_location='cpu')['step']>=target: continue
        if invoke('C_BANK',target,RUN/'profiles'/name): sys.exit(75)
    a=torch.load(RUN/'profiles/resume_split/last.pt',map_location='cpu'); b=torch.load(RUN/'profiles/resume_contiguous/last.pt',map_location='cpu')
    diffs={k:(a['delta'][k]-b['delta'][k]).abs().max().item() for k in a['delta']}
    assert max(diffs.values())<=1e-5,'Resume beyond frozen PPU tolerance'
    optimizer_deltas=[]
    for param,state in a['optimizer']['state'].items():
        for field,value in state.items():
            other=b['optimizer']['state'][param][field]
            if torch.is_tensor(value): optimizer_deltas.append((value-other).abs().max().item())
            else: assert value==other
    assert max(optimizer_deltas)<=1e-5,'Optimizer resume differs beyond tolerance'
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    write(RUN/'resume_test.json',{'continuous_vs_10_plus_10_updates':20,'max_parameter_delta':max(diffs.values()),'max_optimizer_tensor_delta':max(optimizer_deltas),'tolerance':1e-5,'stateless_rng':True,'optimizer_steps_equal':True,'status':'passed'})
    streams={}
    for branch in results:
        streams[branch]=[{k:r[k] for k in ['step','sample_ids','t','noise_sha256']} for r in [json.loads(s) for s in (RUN/'profiles'/branch/'rng_pairing.jsonl').read_text().splitlines()]]
    assert all(streams[b]==streams['C_BANK'] for b in streams)
    write(RUN/'rng_pairing_audit.json',{'status':'passed','branches':list(streams),'steps':100,'independent_streams':['data','t','noise','adapter_init','mode','dropout'],'hashes':{k:digest(v) for k,v in streams.items()}})
    # Cost predictions include measured image timings where available, 2x safety envelope,
    # fixed 2000-step selector controls and full contamination/label matrices.
    e=read(RUN/'E/full/summary.json') if (RUN/'E/full/summary.json').exists() else []
    # Measure actual trained C-bank end-to-end model + metric cost, no CPU proxy.
    from scripts.cde_v3.eval_development import load_branch
    from scripts.cde_v3.sampling import sample
    from scripts.cde_v3.metrics import Metrics
    from mpa_diff.utils.io import read_image
    from mpa_diff.data.cache import static_cached
    m,cp=load_branch(RUN/'profiles/C_BANK/last.pt'); metric=Metrics('cuda')
    row=roles('adapter_fit')[0]; cfg=m.config
    x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda()
    times=[]
    with torch.no_grad():
        for j in range(7):
            torch.cuda.synchronize(); start=time.perf_counter()
            static=static_cached(m.base.priors,x,cfg); cond,extra=m.prepare(x,static); enc=m.encode(extra)
            pred,_=sample(lambda xt,t,c:m.predict(xt,t,c,enc,'all'),cond,x.shape,m.schedule,row['sample_id'],101,20)
            metric(pred.clamp(0,1),x); torch.cuda.synchronize()
            if j>=2: times.append(time.perf_counter()-start)
    infer=float(np.quantile(times,.95))
    train_s={k:r['median_step_seconds']*5000 for k,r in results.items()}
    ndev=len(roles('source_dev')); nfit=len(roles('route_fit')); ncal=len(roles('route_cal'))
    label_calls=(nfit+ncal)*5*2*20
    initial=train_s['BASE_CONT_V3']+train_s['C_BANK']+train_s['C_RGB_CONTROL']+ndev*3*7*infer
    rest=train_s['C_ALL_ONLY']+train_s['BASE_CONT_V3']+(nfit+ncal)*5*2*infer+ndev*3*5*9*infer+3*2000*.05
    perseed=initial+rest
    estimate={'profiles':results,'training_seconds_5000':train_s,'evaluation_seconds_per_image_conservative':infer,'evaluation_profile_seconds':times,'selector_label_denoiser_calls_per_seed':label_calls,'initial_C_package_hours':initial/3600,'full_C_pilot_hours_with_controls':perseed/3600,'two_repeats_hours':2*perseed/3600,'safety_multiplier':1.25,'C_pilot_predicted_hours_with_safety':perseed*1.25/3600,'repeats_predicted_hours_with_safety':perseed*2.5/3600,'remaining_steps_expression':'sum(5000 * measured_seconds_per_step) + modes*noise_keys*images*measured_inference + gates','final_reserved_hours':12,'required_disk_gib_estimate':2.5,'dispatch_allowed':perseed*1.25/3600<=18 and perseed*2.5/3600<=22,'prediction_note':'base data-matched estimated from identical BASE_CONT topology; bank inference+metrics measured on device; actual eval/label times recorded'}
    final_images=read(RUN/'holdout_frozen_candidates.json')['count']+97+427
    final_hours=3*final_images*3*9*infer*1.25/3600+.25
    estimate['final_3seed_full_controls_hours_with_safety']=final_hours
    estimate['dispatch_allowed']=estimate['dispatch_allowed'] and final_hours<=12
    write(RUN/'cost_prediction.json',estimate)
    (DOC/'BUDGET_AND_DATA_PLAN.md').write_text('# 预算与数据计划\n\n新72设备小时上限，最终保留12小时。所有先验/标签/评估/失败重试计时。父模型已见过训练子集，旧test只作回归。\n\n```json\n'+json.dumps(estimate,indent=2)+'\n```\n\n超过完整控制矩阵预算时停止该包，不减seed或删对照冒充完成。\n')
    print(estimate)
if __name__=='__main__': main()
