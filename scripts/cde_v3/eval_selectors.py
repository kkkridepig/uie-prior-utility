"""Input-only selector gate with route-cal fixed control and all frozen corruption groups."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import Selector,token_summary
from scripts.cde_v3.train_selector import choose
from scripts.cde_v3.eval_development import load_branch
from scripts.cde_v3.sampling import sample
from scripts.cde_v3.metrics import Metrics
from scripts.cde_v3.inference import INFERENCE_RUNTIME
from scripts.cde_v3.corruption import CORRUPTIONS,corrupt
from mpa_diff.utils.io import read_image,save_image
from mpa_diff.data.cache import static_cached


def load_selectors(root,bank_sha):
    result={}
    for kind in ('UTILITY','WINNER_CE','SHUFFLED'):
        cp=torch.load(root/'selectors'/kind/'last.pt',map_location='cpu')
        assert cp['step']==2000 and cp['identity']['bank_sha256']==bank_sha
        m=Selector().cuda().eval(); m.load_state_dict(cp['model']); result[kind]=m
    return result

@torch.no_grad()
def calibrate(root,selectors):
    cal=torch.load(root/'labels/route_cal/utilities.pt',map_location='cpu'); assert cal['identity']['role']=='route_cal'
    entries=cal['entries']; u=torch.stack([r['utility'] for r in entries]).cuda()
    fixed=int(u.mean(0).argmax())
    x=torch.stack([r['image64'] for r in entries]).cuda(); z=torch.stack([r['summaries'] for r in entries]).cuda(); stats=torch.stack([r['stats'] for r in entries]).cuda()
    scores=selectors['UTILITY'](x,[z[:,i] for i in range(3)],stats); values=[]
    for threshold in (0,.05,.10,.20):
        actions=choose(scores,stats,threshold,'UTILITY'); values.append([float(u[torch.arange(len(u),device='cuda'),actions].mean()),threshold])
    _,tau=max(values)
    record={'calibration_role':'route_cal','best_fixed_mode':MODES[fixed],'utility_deployment_tau':tau,'main_tables':'no additional threshold; CE never compared to dB tau','calibration_sha256':sha(root/'labels/route_cal/utilities.pt'),'threshold_scores':values}
    if (root/'calibration.json').exists(): assert read(root/'calibration.json')==record
    else: write(root/'calibration.json',record)
    return fixed,tau

@torch.no_grad()
def enhance_selected(model,selector,x,static,sid,seed,stats_transform=None):
    # Deployment path never receives a reference or candidate scores. sample ID only
    # enters the sampler noise, never the selector CNN/summary feature computation.
    cond,extra=model.prepare(x,static)
    if stats_transform: extra=stats_transform(extra)
    enc=model.encode(extra); logits=selector(x,token_summary(enc[0]),extra[2]); action=int(choose(logits,extra[2])[0]); mode=MODES[action]
    return sample(lambda xt,t,c:model.predict(xt,t,c,enc,mode),cond,x.shape,model.schedule,sid,seed,20)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,required=True); a=p.parse_args(); setup()
    root=RUN/'pilot'/str(a.seed); bank=root/'C_BANK/step_05000.pt'; model,cp=load_branch(bank); cfg=model.config
    selectors=load_selectors(root,sha(bank)); fixed,tau=calibrate(root,selectors)
    out=root/'selector_evaluation'; out.mkdir(parents=True,exist_ok=True)
    identity={'bank_sha256':sha(bank),'selector_sha256':{k:sha(root/'selectors'/k/'last.pt') for k in selectors},'calibration_sha256':sha(root/'calibration.json'),'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'code_sha256':sha(Path(__file__)),'corruptions':CORRUPTIONS,'role':'source_dev','inference_runtime':INFERENCE_RUNTIME,'inference_runtime_sha256':sha(Path(__file__).with_name('inference.py'))}
    if (out/'identity.json').exists(): assert read(out/'identity.json')==identity
    else: write(out/'identity.json',identity)
    records=[json.loads(s) for s in (out/'per_image.jsonl').read_text().splitlines()] if (out/'per_image.jsonl').exists() else []
    done={(r['sample_id'],r['corruption'],r['eval_noise_seed']) for r in records}
    metric=Metrics('cuda'); rms=torch.tensor(read(RUN/'d_rms.json')['highfreq_rms'],device='cuda')[None,:,None,None]; rr=roles('route_fit','labels')
    for row in roles('source_dev','development'):
        x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda(); y=read_image(ROOT/cfg['data']['data_root']/row['reference_path'],cfg['data']['resize_hw']).cuda()
        static=static_cached(model.base.priors,x,cfg); cond,extra=model.prepare(x,static)
        other=rr[key(401,row['sample_id'],'histogram_replace')%len(rr)]
        ox=read_image(ROOT/cfg['data']['data_root']/other['image_path'],cfg['data']['resize_hw'])[None].cuda()
        from mpa_diff.priors.kernels import histogram
        hc=cfg['histogram']; replacement=histogram(ox,hc['bins'],hc['bandwidth'],hc['max_input_size'],hc['mode'])
        for corruption in CORRUPTIONS:
            altered=corrupt(extra,x,corruption,row['sample_id'],rms,replacement); enc=model.encode(altered)
            z=token_summary(enc[0]); st=altered[2]
            logits={k:m(x,z,st) for k,m in selectors.items()}
            decisions={k:int(choose(v,st,kind=k)[0]) for k,v in logits.items()}
            decisions.update(BEST_FIXED=fixed,UTILITY_CALIBRATED=int(choose(logits['UTILITY'],st,tau)[0]))
            for seed in (101,102,103):
                task=(row['sample_id'],corruption,seed)
                if task in done: continue
                if deadline(): sys.exit(75)
                scores={}
                for mode in MODES:
                    pred,_=sample(lambda xt,t,c:model.predict(xt,t,c,enc,mode),cond,x.shape,model.schedule,row['sample_id'],seed,20)
                    scores[mode]=metric(pred[0].clamp(0,1),y)
                rec={'sample_id':row['sample_id'],'scene_id':row['scene_id'],'corruption':corruption,'scope':'route_only','eval_noise_seed':seed,'finetune_seed':a.seed,'selected_modes':{k:MODES[v] for k,v in decisions.items()},'predicted_utility':logits['UTILITY'][0].cpu().tolist(),'candidate_scores':scores,'deployed_results':{k:scores[MODES[v]] for k,v in decisions.items()},'enumeration_is_diagnostic_not_deployment_cost':True}
                append(out/'per_image.jsonl',rec); records.append(rec); done.add(task)
            write(out/'progress.json',{'records':len(records),'last':list(task),'status':'running'})
    clean=[r for r in records if r['corruption']=='clean']; expected=len(roles('source_dev'))*3
    assert len(clean)==expected
    summary={k:{m:float(np.mean([r['deployed_results'][k][m] for r in clean])) for m in ('psnr','ssim','lpips')} for k in ['UTILITY','BEST_FIXED','WINNER_CE','SHUFFLED','UTILITY_CALIBRATED']}
    strongest=[]
    for branch,mode in [('BASE_CONT_V3','null'),('C_RGB_CONTROL','all'),('C_ALL_ONLY','all')]: strongest.append((branch,read(root/branch/'eval/summary.json')[mode]))
    name,strong=max(strongest,key=lambda p:p[1]['psnr'])
    drops={c:float(np.mean([r['deployed_results']['UTILITY']['psnr']-r['deployed_results']['BEST_FIXED']['psnr'] for r in records if r['corruption']==c])) for c in CORRUPTIONS[1:]}
    u=summary['UTILITY']; checks={'best_fixed':u['psnr']-summary['BEST_FIXED']['psnr']>=.1,'ce':u['psnr']>summary['WINNER_CE']['psnr'],'shuffle':u['psnr']-summary['SHUFFLED']['psnr']>=.05,'strong':u['psnr']-strong['psnr']>=.1,'ssim':u['ssim']>=strong['ssim']-.002,'lpips':u['lpips']<=strong['lpips']+.01,'corruption':all(d>=-.1 for d in drops.values())}
    per_image=[]
    for row in roles('source_dev'):
        rs=[r for r in clean if r['sample_id']==row['sample_id']]; candidate=np.mean([[r['candidate_scores'][m]['psnr'] for m in MODES] for r in rs],0)
        actual=float(np.mean([r['deployed_results']['UTILITY']['psnr'] for r in rs])); diff=actual-float(np.mean([r['deployed_results']['BEST_FIXED']['psnr'] for r in rs]))
        per_image.append({'sample_id':row['sample_id'],'delta_vs_fixed':diff,'oracle_regret':float(candidate.max()-actual),'mode':rs[0]['selected_modes']['UTILITY']})
    ds=np.array([r['delta_vs_fixed'] for r in per_image]); sensitivities={'median':float(np.median(ds)),'improved_fraction':float(np.mean(ds>0)),'remove_largest5_mean':float(np.sort(ds)[:-5].mean()),'mean_oracle_regret':float(np.mean([r['oracle_regret'] for r in per_image]))}
    result={'status':'pilot_gate_pass_data_match_pending' if all(checks.values()) else 'utility_not_learned','pass':all(checks.values()),'metrics_complete':True,'summary':summary,'strongest_control':name,'strongest_scores':strong,'checks':checks,'route_only_group_deltas':drops,'sensitivity':sensitivities,'per_image':per_image,'calibrated_not_used_for_gate':True}
    write(root/'gate_result.json',result); write(out/'progress.json',{'status':'complete','records':len(records)})
    print({k:v for k,v in result.items() if k!='per_image'})
if __name__=='__main__': main()
