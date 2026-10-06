"""Freeze hashes first; holdout/regression scores cannot alter the decision."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.eval_development import load_branch
from scripts.cde_v3.model import V3Model,token_summary
from scripts.cde_v3.eval_selectors import load_selectors
from scripts.cde_v3.train_selector import choose
from scripts.cde_v3.metrics import Metrics
from scripts.cde_v3.inference import inference_model, INFERENCE_RUNTIME
from scripts.cde_v3.sampling import sample
from scripts.cde_v3.summarize import paired_summary
from mpa_diff.data.manifest import read_manifest
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image,save_image


def freeze(decision):
    payload={'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'code_sha256':code_hash(),'data_roles_sha256':sha(RUN/'data_roles.json'),'lineage_sha256':sha(RUN/'lineage.json'),'metric_identity':read(RUN/'metric_identity.json'),'holdout_sha256':sha(RUN/'holdout_frozen_candidates.json'),'decision':decision,'checkpoints':{path:sha(path) for path in decision.get('checkpoints',[])},'sampler':{'solver':'ddim','nfe':20,'eval_noise_seed':[101,102,103]},'new_test_access_before_freeze':False}
    aux=[RUN/'d_rms.json',ROOT/'mpa_diff/metrics/image.py',ROOT/'scripts/cde_v3/metrics.py',ROOT/'scripts/cde_v3/inference.py',RUN/'inference_runtime_freeze.json']
    payload['inference_runtime']=INFERENCE_RUNTIME
    aux += [RUN/'pilot'/str(s)/'calibration.json' for s in decision.get('seeds',[]) if decision.get('method')=='C_UTILITY']
    payload['auxiliary_hashes']={str(p):sha(p) for p in aux if p.exists()}
    path=RUN/'final_freeze.json'
    if path.exists(): assert read(path)==payload,'Final freeze immutable'
    else: write(path,payload)
    return payload

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--freeze-only',action='store_true'); a=p.parse_args(); setup()
    decision=read(RUN/'development_decision.json'); f=freeze(decision)
    if a.freeze_only: print('frozen; no test accessed'); return
    assert sha(RUN/'protocol_frozen.yaml')==f['protocol_sha256'] and code_hash()==f['code_sha256']
    for path,expected in f['checkpoints'].items(): assert sha(path)==expected,'Frozen weight mismatch'
    for path,expected in f['auxiliary_hashes'].items(): assert sha(path)==expected,'Frozen auxiliary identity mismatch'
    if decision['status'] in ('resource_blocked','implementation_blocked','metrics_blocked'):
        print('No method claim/test for blocked development'); return
    holdout=read(RUN/'holdout_frozen_candidates.json')
    cfg=torch.load(PARENT,map_location='cpu')['config']; metric=Metrics('cuda')
    datasetsets={'confirm_holdout':(holdout['rows'],Path(holdout['root'])),'UIEB_legacy_exposed_regression':(read(RUN/'data_roles.json')['roles']['legacy_exposed_regression'],ROOT/cfg['data']['data_root'])}
    import yaml
    lcfg=yaml.safe_load((ROOT/'configs/data/lsui_grouped.yaml').read_text())
    datasetsets['LSUI_legacy_exposed_regression']=([r for r in read_manifest(ROOT/lcfg['manifest']) if r['split']=='test'],ROOT/lcfg['data_root'])
    for seed in decision['seeds']:
        root=RUN/'pilot'/str(seed); models={}; selectors=None
        models['PARENT']=inference_model(V3Model(torch.load(PARENT,map_location='cpu'),'ANCHOR',seed).cuda())
        for name,path in decision['models'][str(seed)].items():
            assert sha(path)==f['checkpoints'][path]; models[name]=load_branch(path)[0]
        if decision.get('method')=='C_UTILITY':
            selectors=load_selectors(root,sha(root/'C_BANK/step_05000.pt')); cal=read(root/'calibration.json')
        for dataset,(rows,dataroot) in datasetsets.items():
            out=RUN/'final'/str(seed)/dataset; out.mkdir(parents=True,exist_ok=True)
            recs=[json.loads(s) for s in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
            done={(r['sample_id'],r['eval_noise_seed']) for r in recs}
            fixed={r['sample_id'] for r in sorted(rows,key=lambda r:key(20261004,'fixed_test_visual',r['sample_id']))[:12]}
            for row in rows:
                if deadline(): sys.exit(75)
                x=read_image(dataroot/row['image_path'],cfg['data']['resize_hw'])[None].cuda(); y=read_image(dataroot/row['reference_path'],cfg['data']['resize_hw']).cuda()
                for ns in (101,102,103):
                    if (row['sample_id'],ns) in done: continue
                    scores={}; base_images=[]
                    for name,m in models.items():
                        torch.cuda.synchronize(); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                        static=static_cached(m.base.priors,x,cfg); cond,extra=m.prepare(x,static); enc=m.encode(extra)
                        mode='all' if m.adapters else 'null'
                        if name=='C_UTILITY':
                            logits=selectors['UTILITY'](x,token_summary(enc[0]),extra[2]); mode=MODES[int(choose(logits,extra[2])[0])]
                        elif name=='C_BEST_FIXED': mode=cal['best_fixed_mode']
                        elif name in ('C_WINNER_CE','C_SHUFFLED'):
                            kind=name[2:]; logits=selectors[kind](x,token_summary(enc[0]),extra[2]); mode=MODES[int(choose(logits,extra[2],kind=kind)[0])]
                        pred,diag=sample(lambda xt,t,c:m.predict(xt,t,c,enc,mode),cond,x.shape,m.schedule,row['sample_id'],ns,20)
                        pred=pred[0].clamp(0,1); torch.cuda.synchronize(); seconds=time.perf_counter()-start
                        peak=int(torch.cuda.max_memory_allocated()); scores[name]=dict(metric(pred,y),model_seconds=seconds,peak_bytes=peak,mode=mode,nfe=diag['nfe_measured'])
                        if ns==101 and row['sample_id'] in fixed:
                            dest=out/'images'/name/(row['sample_id'].replace('/','__')+'.png'); save_image(dest,pred)
                    record={'sample_id':row['sample_id'],'scene_id':row['scene_id'],'dataset_role':dataset,'finetune_seed':seed,'eval_noise_seed':ns,'scores':scores}
                    append(out/'per_image.jsonl',record); recs.append(record); done.add((row['sample_id'],ns))
                write(out/'progress.json',{'status':'evaluating','records':len(recs)})
            method=decision.get('method') or 'BASE_CONT_V3'; controls=[n for n in models if n!=method]
            summary={n:{metric:float(np.mean([r['scores'][n][metric] for r in recs])) for metric in ('psnr','ssim','lpips','model_seconds')} for n in models}
            comparisons={c:{metric:paired_summary(recs,method,c,metric,seed) for metric in ('psnr','ssim','lpips')} for c in controls}
            write(out/'summary.json',{'models':summary,'comparisons':comparisons,'dataset_role':dataset,'freeze_sha256':sha(RUN/'final_freeze.json')})
            worst=sorted(rows,key=lambda row:np.mean([r['scores'][method]['psnr'] for r in recs if r['sample_id']==row['sample_id']]))[:10]
            strong=decision.get('strong_control','BASE_CONT_V3'); strong='PARENT' if strong==method else strong
            degraded=sorted(rows,key=lambda row:np.mean([r['scores'][method]['psnr']-r['scores'][strong]['psnr'] for r in recs if r['sample_id']==row['sample_id']]))[:10]
            samples={r['sample_id']:r for r in rows if r['sample_id'] in fixed or r in worst or r in degraded}
            for sid,row in samples.items():
                names=['PARENT',strong,method]; pieces=[]
                # Retain only predefined/failure panels; reproduce worst examples
                # with the same frozen key after ranking, avoiding multi-GB images.
                for name in set(names):
                    dest=out/'images'/name/(sid.replace('/','__')+'.png')
                    if dest.exists(): continue
                    if deadline(): sys.exit(75)
                    m=models[name]
                    x=read_image(dataroot/row['image_path'],cfg['data']['resize_hw'])[None].cuda()
                    static=static_cached(m.base.priors,x,cfg); cond,extra=m.prepare(x,static); enc=m.encode(extra)
                    mode='all' if m.adapters else 'null'
                    if name=='C_UTILITY':
                        logits=selectors['UTILITY'](x,token_summary(enc[0]),extra[2]); mode=MODES[int(choose(logits,extra[2])[0])]
                    elif name=='C_BEST_FIXED': mode=cal['best_fixed_mode']
                    elif name in ('C_WINNER_CE','C_SHUFFLED'):
                        kind=name[2:]; logits=selectors[kind](x,token_summary(enc[0]),extra[2]); mode=MODES[int(choose(logits,extra[2],kind=kind)[0])]
                    pred,_=sample(lambda xt,t,c:m.predict(xt,t,c,enc,mode),cond,x.shape,m.schedule,sid,101,20)
                    save_image(dest,pred[0].clamp(0,1))
                for path in (dataroot/row['image_path'],dataroot/row['reference_path']):
                    with Image.open(path) as im: pieces.append(im.convert('RGB').resize((336,336),Image.BILINEAR))
                for name in names:
                    with Image.open(out/'images'/name/(sid.replace('/','__')+'.png')) as im: pieces.append(im.convert('RGB').copy())
                canvas=Image.new('RGB',(336*len(pieces),336))
                for i,im in enumerate(pieces): canvas.paste(im,(336*i,0))
                panel=out/'panels'/(sid.replace('/','__')+'.png'); panel.parent.mkdir(exist_ok=True); canvas.save(panel)
                # Shared fixed center crop and raw absolute difference, no auto-contrast.
                detail=Image.new('RGB',(168*len(pieces),168))
                for i,im in enumerate(pieces): detail.paste(im.crop((84,84,252,252)),(168*i,0))
                detail.save(panel.with_name(panel.stem+'_detail.png'))
                difference=np.abs(np.asarray(pieces[-1],dtype=np.int16)-np.asarray(pieces[-2],dtype=np.int16)).astype(np.uint8)
                Image.fromarray(difference).save(panel.with_name(panel.stem+'_abs_difference.png'))
            write(out/'visual_manifest.json',{'columns':['input','reference','PARENT',strong,method],'fixed':sorted(fixed),'worst10':[r['sample_id'] for r in worst],'largest_regression10':[r['sample_id'] for r in degraded],'posthoc_failures_marked':True,'uniform_scale':True})
            write(out/'progress.json',{'status':'complete','records':len(recs)})
        del models; torch.cuda.empty_cache()
    print('Frozen test export completed; see summaries, not a claim of innovation')
if __name__=='__main__': main()
