"""Triggered D package: same-topology measured profiles, immutable RMS, gradient checks."""
import argparse,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.train_bank import train
from scripts.cde_v3.model import phase_feature,V3Model

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); a=p.parse_args(); setup()
    trigger=read(RUN/'D_trigger.json'); assert trigger['C_correct_complete'] and trigger['metrics_complete']
    # Actual PPU math: odd/constant/low-energy input, CPU comparison and imaginary error.
    rows=[]
    for kind in ['sobel','phase','softphase']:
        for x in [torch.zeros(1,3,17,19),torch.ones(1,3,17,19),noise((1,3,17,19),17,'d_preflight')*.01]:
            cpu,_=phase_feature(x,kind); ppu,diag=phase_feature(x.cuda(),kind)
            err=(cpu-ppu.cpu()).abs().max().item(); assert err<1e-3 and torch.isfinite(ppu).all()
            rows.append({'kind':kind,'cpu_ppu_max_delta':err,'imaginary_max':diag['imaginary_max'].item() if diag else None})
    results={}
    for branch in ['D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']:
        out=RUN/'profiles'/branch
        code=train(argparse.Namespace(branch=branch,finetune_seed=20261004,steps=100,output=str(out),micro_batch=4,device='cuda'))
        if code: sys.exit(code)
        results[branch]=read(out/'progress.json'); assert results[branch]['probe_output_rms_change']>0
    infer=read(RUN/'cost_prediction.json')['evaluation_seconds_per_image_conservative']
    eval_seconds=len(roles('source_dev'))*3*(3+3*3)*infer
    estimate=(sum(r['median_step_seconds']*5000 for r in results.values())+eval_seconds)/3600
    write(RUN/'d_preflight.json',{'status':'passed','math_checks':rows,'profiles':results,'estimated_training_eval_hours':estimate,'budget_cap_hours':8,'rms_sha256':sha(RUN/'d_rms.json'),'dispatch_allowed':estimate*1.25<8})
    if estimate*1.25>=8: raise RuntimeError('D full controls exceed package budget; do not train')
if __name__=='__main__': main()
