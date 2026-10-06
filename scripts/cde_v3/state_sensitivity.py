"""Interior-trajectory sensitivity and exact floating-output noise variance on fixed24."""
import sys,time,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.sampling import sample
from scripts.explore_ag.evaluate import load_experiment
from mpa_diff.utils.io import read_image
from mpa_diff.diffusion.core import time_grid

@torch.no_grad()
def main():
    setup(); fixed=set(read(RUN/'data_roles.json')['fixed24']); rows=[r for r in roles('source_dev','development') if r['sample_id'] in fixed]
    out=RUN/'E/interior_sensitivity.jsonl'; existing=[json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []; done={(r['model'],r['sample_id']) for r in existing}
    for name,path in [('PARENT',PARENT),('V2_BASE_CONT',ROOT/'runs/explore_ag_single_seed_v2_20261003/branches/BASE_CONT/step_05000.pt')]:
        model,cp,branch=load_experiment(path,'cuda'); cfg=model.config
        for row in rows:
            if (name,row['sample_id']) in done: continue
            if deadline(): sys.exit(75)
            image=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda(); cond,prior=model.condition(image)
            den=model.denoiser if branch=='PARENT' else lambda x,t,c:model.predict(x,t,c,prior)
            captured=[]
            def capture(x,t,c):
                if len(captured) in (0,9,19): captured.append((x.clone(),t.clone()))
                else: captured.append(None)
                return den(x,t,c)
            outputs=[]
            for seed in (101,102,103):
                pred,_=sample(capture if seed==101 else den,cond,image.shape,model.schedule,row['sample_id'],seed,20)
                outputs.append(pred.clamp(0,1))
            stack=torch.stack(outputs); result={'model':name,'sample_id':row['sample_id'],'eval_noise_seeds':[101,102,103],'output_pixel_variance_mean':stack.var(0,unbiased=False).mean().item(),'output_pixel_variance_max':stack.var(0,unbiased=False).max().item(),'states':[]}
            for j in (0,9,19):
                x,t=captured[j]; prediction=den(x,t,cond); delta=.01*noise(x.shape,401,row['sample_id'],'interior_delta',j,'cuda'); perturbed=den(x+delta,t,cond)
                state={'trajectory_position':j,'index':int(t.item()),'state_rms':x.square().mean().sqrt().item(),'correct_prediction_rms':prediction.square().mean().sqrt().item(),'gain_norm_delta_f_over_delta_x':((perturbed-prediction).square().sum()/delta.square().sum()).sqrt().item(),'zero_input_delta_rms':(den(torch.zeros_like(x),t,cond)-prediction).square().mean().sqrt().item(),'image_input_delta_rms':(den(image,t,cond)-prediction).square().mean().sqrt().item(),'alternative_time_delta_rms':{}}
                for alt in (0,499,999): state['alternative_time_delta_rms'][str(alt)]=(den(x,torch.tensor([alt],device='cuda'),cond)-prediction).square().mean().sqrt().item()
                result['states'].append(state)
            append(out,result)
        del model; torch.cuda.empty_cache()
    print('48 sample/model sensitivity records complete')
if __name__=='__main__': main()
