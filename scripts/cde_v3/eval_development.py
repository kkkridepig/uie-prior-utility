"""DDIM20 development evaluation, finite bank oracle after fixed5000 only."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import V3Model
from scripts.cde_v3.sampling import sample
from scripts.cde_v3.metrics import Metrics
from scripts.cde_v3.inference import inference_model, INFERENCE_RUNTIME
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image,save_image

def load_branch(path,device='cuda',optimized=True):
    cp=torch.load(path,map_location='cpu'); ident=cp['identity']
    assert ident['parent_sha256']==sha(PARENT)
    assert ident['data_roles_sha256']==sha(RUN/'data_roles.json')
    assert ident['protocol_sha256']==sha(RUN/'protocol_frozen.yaml')
    assert ident['implementation']==train_identity()
    if ident['branch'].startswith('D_'): assert ident['d_rms_sha256']==sha(RUN/'d_rms.json')
    model=V3Model(torch.load(PARENT,map_location='cpu'),ident['branch'],ident['finetune_seed']).to(device)
    model.load_delta(cp['delta']); model.eval()
    if optimized and device=='cuda': inference_model(model)
    return model,cp

@torch.no_grad()
def evaluate(checkpoint,output,role='source_dev',mode_list=None,noise_seeds=(101,102,103),labels=False):
    setup(); rows=roles(role,'labels' if labels else 'calibration' if role=='route_cal' else 'development')
    model,cp=load_branch(checkpoint); assert cp['step']==5000,'No oracle/quality decision before5000'
    cfg=model.config; metric=Metrics('cuda'); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    modes=mode_list or (MODES if cp['identity']['branch']=='C_BANK' else ['all'] if model.adapters else ['null'])
    ident={'checkpoint_sha256':sha(checkpoint),'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'role':role,'row_hash':digest(rows),'seeds':list(noise_seeds),'modes':list(modes),'metric':read(RUN/'metric_identity.json'),'code_sha256':sha(Path(__file__)),'inference_runtime':INFERENCE_RUNTIME,'inference_runtime_sha256':sha(Path(__file__).with_name('inference.py'))}
    if (out/'identity.json').exists(): assert read(out/'identity.json')==ident
    else: write(out/'identity.json',ident)
    records=[json.loads(s) for s in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
    done={(r['sample_id'],r['mode'],r['eval_noise_seed']) for r in records}
    for r in rows:
        x=read_image(ROOT/cfg['data']['data_root']/r['image_path'],cfg['data']['resize_hw'])[None].cuda()
        y=read_image(ROOT/cfg['data']['data_root']/r['reference_path'],cfg['data']['resize_hw']).cuda()
        static=static_cached(model.base.priors,x,cfg); cond,extras=model.prepare(x,static); enc=model.encode(extras)
        for mode in modes:
            for seed in noise_seeds:
                task=(r['sample_id'],mode,seed)
                if task in done: continue
                if deadline(): return 75
                calls=[0]
                def den(xt,t,c): calls[0]+=1; return model.predict(xt,t,c,enc,mode)
                torch.cuda.synchronize(); begin=time.perf_counter()
                raw,diag=sample(den,cond,x.shape,model.schedule,r['sample_id'],seed,20)
                pred=raw[0].clamp(0,1); torch.cuda.synchronize(); seconds=time.perf_counter()-begin
                record={'sample_id':r['sample_id'],'scene_id':r['scene_id'],'mode':mode,'eval_noise_seed':seed,'finetune_seed':cp['identity']['finetune_seed'],'role':role,'sampling_seconds':seconds,'calls':calls[0],**metric(pred,y)}
                if labels: record['label_psnr']=min(float(record['psnr']),100.); record['label_mse_floored']=float(record['psnr'])>100.
                append(out/'per_image.jsonl',record); records.append(record); done.add(task)
                if seed==noise_seeds[0] and r['sample_id'] in read(RUN/'data_roles.json')['fixed_visual12']:
                    save_image(out/'images'/mode/(r['sample_id'].replace('/','__')+'.png'),pred)
        write(out/'progress.json',{'status':'evaluating','records':len(records),'last_sample':r['sample_id']})
    summary={mode:{m:float(np.mean([r[m] for r in records if r['mode']==mode])) for m in ('psnr','ssim','lpips')} for mode in modes}
    write(out/'summary.json',summary); write(out/'progress.json',{'status':'complete','records':len(records)}); return 0

def oracle(seed):
    root=RUN/'pilot'/str(seed); path=root/'C_BANK/eval/per_image.jsonl'
    records=[json.loads(s) for s in path.read_text().splitlines()]
    rows=roles('source_dev','development'); scores=[]; per_image=[]
    for r in rows:
        means=[]
        for mode in MODES:
            rr=[x for x in records if x['sample_id']==r['sample_id'] and x['mode']==mode]
            assert sorted(x['eval_noise_seed'] for x in rr)==[101,102,103]
            means.append(float(np.mean([x['psnr'] for x in rr])))
        scores.append(means); per_image.append({'sample_id':r['sample_id'],'scene_id':r['scene_id'],'noise_averaged_mode_psnr':dict(zip(MODES,means)),'oracle_mode':MODES[int(np.argmax(means))]})
    scores=np.array(scores); fixed=scores.mean(0); omniscient=float(scores.max(1).mean())
    base=read(root/'BASE_CONT_V3/eval/summary.json')['null']['psnr']; rgb=read(root/'C_RGB_CONTROL/eval/summary.json')['all']['psnr']
    space=omniscient-float(fixed.max()); diff=omniscient-base
    result={'finetune_seed':seed,'bank_step':5000,'oracle_psnr':omniscient,'dev_oracle_fixed':dict(zip(MODES,fixed.tolist())),'oracle_minus_dev_fixed':space,'oracle_minus_base':diff,'rgb_psnr':rgb,'mode_psnr_spread':float(np.ptp(scores,axis=1).mean()),'pass':space>=.1 and diff>=-.1,'status':'space_available_not_selector_evidence' if space>=.1 and diff>=-.1 else 'bank_no_selectable_gain','per_image':per_image,'oracle_order':'mean over101/102/103 then max over5modes','metrics_complete':True}
    write(root/'oracle.json',result)
    (DOC/'C_BANK_ORACLE.md').write_text('# C bank oracle\n\n仅限训练后固定5000步bank，先跨三个噪声求平均，再逐图取最优。oracle不可部署，也不支持混合输出上界。\n\n```json\n'+json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2)+'\n```\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint'); p.add_argument('--output'); p.add_argument('--oracle',type=int); a=p.parse_args()
    if a.oracle: print(oracle(a.oracle)); return
    if not a.checkpoint or not a.output: p.error('--checkpoint and --output required')
    sys.exit(evaluate(a.checkpoint,a.output))
if __name__=='__main__': main()
