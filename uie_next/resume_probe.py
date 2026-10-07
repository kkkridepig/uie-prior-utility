"""Synthetic head-only interrupted-training probe, never a scientific checkpoint."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from .models.controls import Controller
from .training import streams,initialize,optimizer,pair_objective
from .checkpoint import save,resume
from .records import ROOT,sha,digest


def run_probe(mode,directory,device='cpu'):
    if device=='cpu':torch.set_num_threads(1)
    random.seed(20261007);np.random.seed(20261007);torch.manual_seed(20261007)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    rng=streams(20261007);m=initialize(lambda:Controller('O'),rng['init']).to(device)
    opt,scheduler=optimizer(m,1000)
    source=torch.Generator().manual_seed(9)
    image=torch.rand(2,3,16,16,generator=source).to(device);base=(.1+.8*image).detach()
    target=(base+.05*torch.randn(image.shape,generator=source).to(device)).clamp(0,1)
    candidate=(base[:,None]+.03*torch.randn(2,2,3,16,16,generator=source).to(device)).clamp(0,1)
    source_files=list((ROOT/'uie_next/models').glob('*.py'))+list((ROOT/'uie_next/math').glob('*.py'))
    source_files+=[ROOT/'uie_next/losses.py',ROOT/'uie_next/training.py',ROOT/'uie_next/resume_probe.py']
    source_hash=digest({str(p.relative_to(ROOT)):sha(p) for p in sorted(source_files)})
    identity={'kind':'SYNTHETIC_HEAD_RECOVERY_ONLY','seed':20261007,'device':device,'official_backbone':False,'source_snapshot_sha256':source_hash}
    start=0;end=20 if mode in ('continuous','resume') else 10
    if mode=='resume':
        state=resume(root/'interrupted.pt',m,opt,scheduler,identity,rng);start=state['global_step']
    for step in range(start,end):
        idx=torch.randperm(2,generator=rng['data']).to(device)
        jitter=torch.randn(candidate.shape,generator=rng['view']).to(device)*.001
        opt.zero_grad();loss,_=pair_objective(m,image[idx],base[idx],(candidate+jitter)[idx],target[idx],{'s_v':.05,'s_U':.003,'s_e2':.003})
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();scheduler.step()
    name={'continuous':'continuous.pt','first':'interrupted.pt','resume':'resumed.pt'}[mode]
    return save(root/name,m,opt,scheduler,end,identity,rng,include_cuda=device.startswith('cuda'),extra={'final_loss':float(loss),'not_scientific_training':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['continuous','first','resume'])
    parser.add_argument('--directory',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();print(run_probe(args.mode,args.directory,args.device))
