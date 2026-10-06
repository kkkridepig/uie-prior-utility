"""Explicit x0 DDIM/DDPM math, independently counted calls and keyed eval noise."""
import torch
from mpa_diff.diffusion.core import time_grid
from scripts.cde_v3.common import noise

@torch.no_grad()
def sample(denoiser,cond,shape,schedule,sid,seed,nfe=20,solver='ddim',clip=False,trace=False):
    grid=time_grid(schedule.steps,nfe)
    if solver=='ddpm' and nfe!=schedule.steps: raise ValueError('DDPM must use T')
    if solver not in ('ddpm','ddim'): raise ValueError(solver)
    x=noise(shape,seed,sid,device=schedule.beta.device); calls=0; records=[]; previous=None
    for t,s in zip(grid[:-1],grid[1:]):
        idx=torch.full((shape[0],),t-1,dtype=torch.long,device=x.device)
        raw=denoiser(x,idx,cond); calls+=1
        x0=raw.clamp(0,1) if clip else raw
        if trace: records.append({'math_t':t,'index':t-1,'state_rms':x.square().mean().sqrt().item(),'x0_rms':x0.square().mean().sqrt().item(),'x0_adjacent_rms':None if previous is None else (x0-previous).square().mean().sqrt().item(),'clip_rms':(raw-raw.clamp(0,1)).square().mean().sqrt().item()})
        if trace: previous=x0
        if s==0: x=x0; continue
        at,ass=schedule.alpha_bar_math[t],schedule.alpha_bar_math[s]
        if solver=='ddim': x=ass.sqrt()*x0+(1-ass).sqrt()*schedule.epsilon(x,x0,idx)
        else:
            beta=schedule.beta[t-1]; alpha=1-beta
            x=ass.sqrt()*beta/(1-at)*x0+alpha.sqrt()*(1-ass)/(1-at)*x+(beta*(1-ass)/(1-at)).sqrt()*noise(shape,seed,sid,'ddpm',t,x.device)
    return x,{'nfe_measured':calls,'grid':grid,'trace':records,'noise_key':['noise_protocol_v3',seed,sid,'initial',0]}
