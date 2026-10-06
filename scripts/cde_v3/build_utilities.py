"""Detached final-output utility labels; route_fit trains gate, route_cal calibrates only."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import token_summary
from scripts.cde_v3.eval_development import evaluate,load_branch
from mpa_diff.utils.io import read_image
from mpa_diff.data.cache import static_cached

@torch.no_grad()
def build(checkpoint,role,output):
    if role not in ('route_fit','route_cal'): raise ValueError('Label/calibration role required')
    rows=roles(role,'labels' if role=='route_fit' else 'calibration'); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if evaluate(checkpoint,out/'scores',role,MODES,(17,29),labels=role=='route_fit'): return 75
    records=[json.loads(s) for s in (out/'scores/per_image.jsonl').read_text().splitlines()]
    model,cp=load_branch(checkpoint); model.requires_grad_(False); cfg=model.config
    entries=[]
    for row in rows:
        if deadline(): return 75
        x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda()
        static=static_cached(model.base.priors,x,cfg); cond,extra=model.prepare(x,static); enc=model.encode(extra)[0]
        scores=[]
        for mode in MODES:
            rr=[r for r in records if r['sample_id']==row['sample_id'] and r['mode']==mode]; assert len(rr)==2
            scores.append(float(np.mean([min(float(r['psnr']),100.) for r in rr])))
        entries.append({'sample_id':row['sample_id'],'image64':torch.nn.functional.interpolate(x,(64,64),mode='bilinear',align_corners=False)[0].cpu(),'summaries':torch.stack(token_summary(enc),1)[0].cpu(),'stats':extra[2][0].cpu(),'utility':torch.tensor(scores)-scores[0]})
    data={'identity':{'role':role,'bank_sha256':sha(checkpoint),'parent_sha256':PARENT_HASH,'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'role_hash':digest(rows),'noise_keys':[17,29],'labels_stop_gradient':True,'allowed_use':'selector_optimization' if role=='route_fit' else 'fixed_mode_and_deployment_tau_only','finetune_seed':cp['identity']['finetune_seed'],'evaluation_identity_sha256':sha(out/'scores/identity.json'),'sampler_sha256':sha(ROOT/'scripts/cde_v3/sampling.py'),'preprocess_sha256':sha(ROOT/'mpa_diff/utils/io.py'),'inference_runtime_sha256':sha(ROOT/'scripts/cde_v3/inference.py')},'entries':entries}
    tmp=out/'utilities.pt.tmp'; torch.save(data,tmp); tmp.replace(out/'utilities.pt'); write(out/'identity.json',dict(data['identity'],utilities_sha256=sha(out/'utilities.pt')))
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',required=True); p.add_argument('--role',choices=['route_fit','route_cal'],required=True); p.add_argument('--output',required=True); a=p.parse_args(); setup(); sys.exit(build(a.checkpoint,a.role,a.output))
if __name__=='__main__': main()
