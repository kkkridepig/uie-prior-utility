"""No-training, source-dev-only sampling diagnosis with exact NFE/timing boundaries."""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.sampling import sample
from scripts.cde_v3.metrics import Metrics
from scripts.explore_ag.evaluate import load_experiment
from scripts.explore_ag.sampling import sample_fixed
from mpa_diff.utils.io import read_image,save_image
from mpa_diff.priors.kernels import pad_image

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',choices=['subset','full','legacy'],required=True); a=p.parse_args(); setup()
    metric=Metrics('cuda'); cfgrows=roles('source_dev','development'); fixed=read(RUN/'data_roles.json')['fixed24']
    rows=cfgrows if a.stage=='full' else [r for r in cfgrows if r['sample_id'] in fixed]
    checkpoints={'PARENT':PARENT,'V2_BASE_CONT':ROOT/'runs/explore_ag_single_seed_v2_20261003/branches/BASE_CONT/step_05000.pt'}
    out=RUN/'E'/a.stage; out.mkdir(parents=True,exist_ok=True)
    identity={'stage':a.stage,'checkpoint_hashes':{k:sha(v) for k,v in checkpoints.items()},'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'sampler_sha256':sha(ROOT/'scripts/cde_v3/sampling.py'),'metric_identity':read(RUN/'metric_identity.json'),'image_ids':[r['sample_id'] for r in rows]}
    if (out/'identity.json').exists(): assert read(out/'identity.json')==identity
    else: write(out/'identity.json',identity)
    records=[json.loads(s) for s in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
    done={(r['model'],r['sample_id'],r['eval_noise_seed'],r['solver'],r['nfe']) for r in records}
    for name,path in checkpoints.items():
        model,state,branch=load_experiment(path,'cuda'); cfg=model.config
        # Two full warmups, fixed first source image, excluded from latency percentiles, charged to budget.
        warm=read_image(ROOT/cfg['data']['data_root']/rows[0]['image_path'],cfg['data']['resize_hw'])[None].cuda()
        for _ in range(2):
            cond,prior=model.condition(warm); den=model.denoiser if branch=='PARENT' else lambda xt,t,c:model.predict(xt,t,c,prior)
            sample(den,cond,warm.shape,model.schedule,'warmup',101,20)
        del warm
        for row in rows:
            for seed in ([101] if a.stage=='legacy' else [101,102,103]):
                reference20=None
                for solver,nfe in ([('ddim',20)] if a.stage=='legacy' else [('ddim',n) for n in (20,1,2,4,8)]+([('ddpm',1000)] if a.stage=='subset' else [])):
                    task=(name,row['sample_id'],seed,solver,nfe)
                    if task in done and nfe!=20: continue
                    if deadline(): print('budget_stopped'); sys.exit(75)
                    torch.cuda.synchronize(); service_start=time.perf_counter()
                    x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda()
                    y=read_image(ROOT/cfg['data']['data_root']/row['reference_path'],cfg['data']['resize_hw']).cuda()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); model_start=time.perf_counter()
                    padded,info=pad_image(x); t0=time.perf_counter(); cond,prior=model.condition(padded); torch.cuda.synchronize(); prior_s=time.perf_counter()-t0
                    den=model.denoiser if branch=='PARENT' else lambda xt,t,c:model.predict(xt,t,c,prior)
                    calls=[0]
                    def instrument(xt,t,c): calls[0]+=1; return den(xt,t,c)
                    t0=time.perf_counter()
                    if a.stage=='legacy': raw,diag=sample_fixed(instrument,cond,padded.shape,model.schedule,row['sample_id'],solver,nfe,cfg['sampler']['clip_intermediate_x0'])
                    else: raw,diag=sample(instrument,cond,padded.shape,model.schedule,row['sample_id'],seed,nfe,solver,trace=a.stage=='subset')
                    torch.cuda.synchronize(); sampling_s=time.perf_counter()-t0
                    pred=raw[0,:,:info['hw'][0],:info['hw'][1]].clamp(0,1); torch.cuda.synchronize(); model_s=time.perf_counter()-model_start
                    memory=int(torch.cuda.max_memory_allocated()); peak_reserved=int(torch.cuda.max_memory_reserved())
                    dest=out/'images'/name/(row['sample_id'].replace('/','__')+'_%s_%s_%s.png'%(seed,solver,nfe)); save_image(dest,pred); torch.cuda.synchronize(); service_s=time.perf_counter()-service_start
                    if solver=='ddim' and nfe==20: reference20=pred.clone()
                    assert calls[0]==nfe
                    diff=pred-reference20
                    r={'model':name,'sample_id':row['sample_id'],'scene_id':row['scene_id'],'eval_noise_seed':seed,'solver':solver,'nfe':nfe,'actual_calls':calls[0],**metric(pred,y),'output_float32_sha256':__import__('hashlib').sha256(pred.cpu().numpy().tobytes()).hexdigest(),'output_png_sha256':sha(dest),'max_difference_ddim20':diff.abs().max().item(),'rms_difference_ddim20':diff.square().mean().sqrt().item(),'prior_seconds':prior_s,'sampling_seconds':sampling_s,'model_seconds':model_s,'service_seconds':service_s,'peak_allocated_bytes':memory,'peak_reserved_bytes':peak_reserved,'cache':'uncached_prior_two_model_warmups_OS_cache_not_flushed','diag':diag}
                    if task not in done: append(out/'per_image.jsonl',r); records.append(r); done.add(task)
                    write(out/'progress.json',{'model':name,'last':list(task),'records':len(records)})
                if a.stage=='subset' and seed==101:
                    # Fixed state, alternative index (out-of-distribution diagnostic only).
                    statex=noise(padded.shape,seed,row['sample_id'],device='cuda'); t=torch.tensor([999],device='cuda')
                    true=den(statex,t,cond); pert=noise(statex.shape,401,row['sample_id'],'sensitivity',device='cuda')*.01
                    states={'sample_id':row['sample_id'],'model':name,'alternative_index_mse':{},'delta_x_gain':((den(statex+pert,t,cond)-true).square().mean()/pert.square().mean()).sqrt().item(),'zero_state_mse':(den(torch.zeros_like(statex),t,cond)-true).square().mean().item(),'input_state_mse':(den(padded,t,cond)-true).square().mean().item()}
                    for idx in (0,499): states['alternative_index_mse'][str(idx)]=(den(statex,torch.tensor([idx],device='cuda'),cond)-true).square().mean().item()
                    append(out/'sensitivity.jsonl',states)
        del model; gc.collect(); torch.cuda.empty_cache()
    summary=[]
    for name in checkpoints:
        for solver,nfe in sorted({(r['solver'],r['nfe']) for r in records}):
            rr=[r for r in records if r['model']==name and r['solver']==solver and r['nfe']==nfe]
            if not rr: continue
            rec={'model':name,'solver':solver,'nfe':nfe,'count':len(rr)}
            for field in ('psnr','ssim','lpips'): rec[field]=float(np.mean([r[field] for r in rr]))
            for field in ('prior_seconds','sampling_seconds','model_seconds','service_seconds'):
                rec[field+'_p50']=float(np.median([r[field] for r in rr])); rec[field+'_p95']=float(np.quantile([r[field] for r in rr],.95))
            summary.append(rec)
    write(out/'summary.json',summary); print(summary)
    (DOC/'E_DIAGNOSTICS.md').write_text('# E 无训练采样诊断\n\n当前已完成阶段 '+a.stage+'。主算法比较仍固定DDIM20。子集和完整验证、V2兼容噪声分别记录，不混为一条曲线。\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n模型时间排除指标计算；服务时间含读取/解码/H2D/PNG保存，先验不使用缓存，OS缓存未清空。分数不是创新证据。全部原始记录在 runs/prior_utility_cde_v3_20261004/E。\n')
if __name__=='__main__': main()
