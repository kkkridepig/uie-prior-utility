"""Only the registered 5505-parameter pointwise network."""
import math
import torch
from torch import nn
from .moments import torch_relative


class MomentController(nn.Module):
    def __init__(self,method,constants,seed=20261007):
        super().__init__();self.method=method;self.constants=constants
        if method not in {'MOM_R','MOM_G','MOM_R_NC','MOM_R_NM','MOM_G_NM','DIRECT_R','DIRECT_G'}:raise ValueError(method)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net=nn.Sequential(nn.Conv2d(52,64,1),nn.SiLU(),nn.Conv2d(64,32,1),nn.SiLU(),nn.Conv2d(32,1,1))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        if sum(p.numel() for p in self.parameters())!=5505:raise AssertionError('wrong capacity')

    def forward(self,region,global_,A):
        if self.method.endswith('_G') or self.method=='MOM_G_NM':left=right=global_
        elif self.method=='MOM_R_NC':left=right=region
        else:left=region;right=global_.expand_as(region)
        z=self.net(torch.cat([left,right],1));c=self.constants;ref=c['alpha_ref']
        if self.method.startswith('DIRECT'):alpha=torch.sigmoid(z+math.log(ref/(1-ref)));C=None
        else:
            C=z*c['s_C'];alpha=torch_relative(A,C,ref,c['lambda0'])
        return {'alpha':alpha,'C_hat':C,'z':z}


def objective(pred,A,C,mse0,B,constants,method):
    """Per-image whole-image MSE from exact quadratic sufficient statistics."""
    a=pred['alpha'];ref=constants['alpha_ref']
    m=mse0+(A*a*a-2*B*a).flatten(1).mean(1)
    mref=(mse0+(A*ref*ref-2*B*ref).flatten(1).mean(1)).detach()
    dec=torch.log((m+1e-6)/(mref+1e-6)).mean()
    mom=dec*0
    if pred['C_hat'] is not None:
        v=(pred['C_hat']-C.detach())/constants['s_C']
        mom=(torch.nn.functional.huber_loss(v,torch.zeros_like(v),reduction='none')*(A>0)).flatten(1).mean(1).mean()
    loss=dec+(0 if method.endswith('_NM') or method.startswith('DIRECT') else .1)*mom
    return loss,{'dec':dec,'mom':mom,'mse':m}
