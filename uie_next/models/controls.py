import math
import torch
import torch.nn as nn
from .blocks import trunk
from .utility import UtilityNetwork
from ..math.utility import geometry,decision


def covariance(z, se2):
    c00 = se2 * torch.nn.functional.softplus(z[:, 0:1]) + 1e-8
    c11 = se2 * torch.nn.functional.softplus(z[:, 1:2]) + 1e-8
    c01 = 0.999 * torch.tanh(z[:, 2:3]) * torch.sqrt(c00 * c11)
    return torch.cat([c00, c11, c01], dim=1)


class _ControlBody(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, bias=True, depth=4):
        super().__init__()
        self.body = nn.Sequential(*trunk(in_channels, width, depth))
        self.output = nn.Conv2d(width, out_channels, 3, padding=1, bias=bias)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.body(x))


class Controller(nn.Module):
    def __init__(self, name):
        super().__init__(); self.name = name
        if name == 'G0':
            self.net = nn.Conv2d(9, 1, 1, bias=True); nn.init.zeros_(self.net.weight); nn.init.zeros_(self.net.bias)
        elif name == 'G1': self.net = _ControlBody(9, 1, 32, True)
        elif name == 'G2': self.net = _ControlBody(9, 1, 45, True)
        elif name == 'F0':
            self.net = _ControlBody(9, 3, 32, True)
            with torch.no_grad():
                self.net.output.bias.copy_(torch.tensor([math.log(math.expm1(1)),math.log(math.expm1(1)),math.atanh(.9/.999)]))
        elif name == 'R0': self.net = _ControlBody(6, 3, 32, True)
        elif name == 'O-NS': self.net = _ControlBody(10, 1, 32, True)
        elif name in ('O', 'O-NI', 'O-NP', 'O-ND'):
            self.net = UtilityNetwork()
        else: raise ValueError(name)

    def forward(self,image,base,candidate,se2=1e-4,tau=0.,lamb=1e-4,temperature=1.,margin=0.,error_hat=None):
        g=geometry(candidate-base)
        result=dict(g)
        if self.name.startswith('G'):
            logit=self.net(torch.cat([image,base,g['r']],1))
            alpha=((torch.sigmoid(logit/temperature)-margin)/(1-margin)).clamp(0,1)
            result.update(logit=logit,alpha=alpha,output=(base+alpha*g['r']).clamp(0,1))
            return result
        if self.name=='F0':
            z=self.net(torch.cat([image,base,candidate],1))
            C=covariance(z,se2);c00,c11,c01=C.split(1,dim=1)
            alpha=((c00-c01-tau)/(c00+c11-2*c01+lamb)).clamp(0,1)
            result.update(C=C,U_hat=c00-c11,alpha=alpha,output=(base+alpha*g['r']).clamp(0,1))
            return result
        if self.name=='R0':
            eh=self.net(torch.cat([image,base],1)) if error_hat is None else error_hat
            vh=(eh*g['u']).mean(1,keepdim=True);result['error_hat']=eh
        elif self.name=='O-NS':
            vh=self.net(torch.cat([image,base,g['u'],g['c']],1))
        else:
            result.update(self.net(image,base,g['r']));vh=result['v_hat']
        vh=torch.where(g['active'],vh,torch.zeros_like(vh))
        bh=g['c']*vh
        out,alpha=decision(base,g,bh,tau,lamb)
        result.update(v_hat=vh,b_hat=bh,U_hat=2*bh-g['a'],output=out,alpha=alpha)
        return result
