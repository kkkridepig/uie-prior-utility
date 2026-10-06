"""Frozen route-only interventions. Parent conditioning is never mutated."""
import copy,torch
import torch.nn.functional as F
from scripts.cde_v3.common import noise

CORRUPTIONS=['clean','physical_missing','histogram_missing','highfreq_missing','depth_shift_002','depth_shift_005','highfreq_noise_01','highfreq_noise_03','histogram_replace']

def corrupt(extra,image,name,sid,hf_rms,replacement=None):
    inputs,masks,stats=extra
    inputs=[t.clone() for t in inputs]; masks=[m.clone() for m in masks]; stats=stats.clone()
    if name=='clean': return inputs,masks,stats
    if name.endswith('_missing'):
        k=['physical','histogram','highfreq'].index(name.removesuffix('_missing') if hasattr(name,'removesuffix') else name[:-8]); masks[k][:]=False; inputs[k][:]=0; stats[:,k]=0
        if k==0: stats[:,3]=0
    elif name.startswith('depth_shift'):
        phy=inputs[0]; frac=.02 if name.endswith('002') else .05; shift=max(1,round(image.shape[-1]*frac))
        d=F.pad(phy[:,3:4],(shift,0,0,0),mode='replicate')[...,:image.shape[-1]]
        kd,kb,ambient=phy[:,4:7],phy[:,7:10],phy[:,10:13]
        t=torch.exp(-kd*d); back=ambient*(-torch.expm1(-kb*d)); raw=(image-back)/t.clamp_min(1e-6)
        good=torch.isfinite(raw).all(1,keepdim=True)&torch.isfinite(d)&(t>=1e-6).all(1,keepdim=True)
        phy=torch.cat((raw.clamp(0,1),d,kd,kb,ambient,t,back,good.float()),1)
        inputs[0]=phy; masks[0]=good; stats[:,0]=good.float().mean((1,2,3))
        stats[:,3]=(((raw<0)|(raw>1)).any(1,keepdim=True)&good).float().sum((1,2,3))/good.float().sum((1,2,3)).clamp_min(1)
    elif name.startswith('highfreq_noise'):
        scale=.1 if name.endswith('01') else .3
        inputs[2]+=noise(inputs[2].shape,401,sid,name,device=image.device)*hf_rms*scale
    elif name=='histogram_replace':
        if replacement is None: raise ValueError('Missing training histogram replacement')
        inputs[1]=replacement.clone()
        ok=torch.isfinite(replacement).flatten(1).all(1)&(replacement>=0).flatten(1).all(1)&(replacement.sum((1,2,3))>0)
        masks[1]=ok[:,None,None,None].expand(-1,1,*replacement.shape[-2:]); stats[:,1]=ok.float()
    else: raise ValueError(name)
    return inputs,masks,stats
