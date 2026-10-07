"""Reference-free diagnostic deployment from the same frozen ridge objects.

Formal second-stage checkpoint/CAL/DEV/sealed evaluation is intentionally not
implemented or dispatched after STOP_NO_PREDICTABILITY_SIGNAL.
"""
import numpy as np
import torch
from ..priors.heuristic import make_prior
from .descriptors import descriptors
from .ridge_probe import predict
from .moments import aggregate,expand


@torch.no_grad()
def deploy_ridge(image,backbone,producer,ridge,kappa=1):
    """I only. No reference, sample identity, role or oracle enters prediction."""
    base=backbone(image);f=make_prior(image);candidate=producer(image,base,f['P'],f['V'])
    I=image.detach().cpu().double().numpy();J=base.detach().cpu().double().numpy();r=candidate.detach().cpu().double().numpy()-J
    fr,fg=descriptors(I,J,J+r);n=len(I)
    visible={'f_region':fr.reshape(n,26,64).transpose(0,2,1),'f_global':fg.reshape(n,26),'A':aggregate(np.mean(r*r,1,keepdims=True),32).reshape(n,64)}
    alpha,_=predict(ridge,visible,np.arange(n),kappa=kappa);out=J+expand(alpha.reshape(n,1,8,8),32)*r
    return torch.as_tensor(out,dtype=image.dtype,device=image.device)
