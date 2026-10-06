"""Frozen-RMS D route-only noise diagnostic; not an alternative quality objective."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import phase_feature
from scripts.cde_v3.eval_development import load_branch
from scripts.cde_v3.sampling import sample
from scripts.cde_v3.metrics import Metrics
from mpa_diff.utils.io import read_image
from mpa_diff.priors.kernels import sobel
from mpa_diff.data.cache import static_cached

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,required=True); a=p.parse_args(); setup()
    root=RUN/'pilot'/str(a.seed); output=root/'D_noise'; output.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in (output/'per_image.jsonl').read_text().splitlines()] if (output/'per_image.jsonl').exists() else []
    done={(r['branch'],r['sample_id'],r['sigma'],r['eval_noise_seed']) for r in records}; metric=Metrics('cuda')
    for branch,kind in [('D_SOBEL_STD','sobel'),('D_PHASE_STD','phase'),('D_SOFTPHASE_STD','softphase')]:
        m,cp=load_branch(root/branch/'step_05000.pt'); cfg=m.config
        for row in roles('source_dev','development'):
            x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda(); y=read_image(ROOT/cfg['data']['data_root']/row['reference_path'],cfg['data']['resize_hw']).cuda()
            static=static_cached(m.base.priors,x,cfg); clean_cond,_=m.base.condition(x,static)
            for sigma in (.005,.02,.05):
                altered=(x+sigma*noise(x.shape,401,row['sample_id'],'D_feature_noise',device='cuda')).clamp(0,1)
                feature,diag=phase_feature(altered,kind); normalized=feature/m.frozen_rms.clamp_min(.001)
                cond=dict(clean_cond,highfreq=clean_cond['highfreq']+m.phase_adapter(normalized))
                for seed in (101,102,103):
                    task=(branch,row['sample_id'],sigma,seed)
                    if task in done: continue
                    if deadline(): sys.exit(75)
                    pred,_=sample(m.base.denoiser,cond,x.shape,m.schedule,row['sample_id'],seed,20); pred=pred[0].clamp(0,1)
                    r={'branch':branch,'sample_id':row['sample_id'],'scene_id':row['scene_id'],'sigma':sigma,'eval_noise_seed':seed,'scope':'new_D_condition_only_parent_clean','frozen_rms_sha256':sha(RUN/'d_rms.json'),'before_rms':feature.square().mean((0,2,3)).sqrt().cpu().tolist(),'after_rms':normalized.square().mean((0,2,3)).sqrt().cpu().tolist(),'tau':diag['tau'].flatten().cpu().tolist() if diag else None,'imaginary_max':diag['imaginary_max'].item() if diag else None,'edge_mse':(sobel(pred[None])-sobel(y[None])).square().mean().item(),**metric(pred,y)}
                    append(output/'per_image.jsonl',r); records.append(r); done.add(task)
            write(output/'progress.json',{'status':'running','records':len(records)})
        del m; torch.cuda.empty_cache()
    summary={branch:{str(sigma):{k:float(np.mean([r[k] for r in records if r['branch']==branch and r['sigma']==sigma])) for k in ['psnr','ssim','lpips','edge_mse']} for sigma in (.005,.02,.05)} for branch in ['D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']}
    write(output/'summary.json',summary); write(output/'progress.json',{'status':'complete','records':len(records)})
    print(summary)
if __name__=='__main__': main()
