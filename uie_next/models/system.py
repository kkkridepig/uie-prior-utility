import torch
from torch import nn
from ..priors.heuristic import make_prior,rgb_prior


class Deployment(nn.Module):
    def __init__(self,backbone,candidate=None,controller=None,rgb=False,policy=None):
        super().__init__();self.backbone=backbone;self.candidate=candidate;self.controller=controller;self.rgb=rgb
        self.policy=policy or {}
        if candidate is not None:candidate.requires_grad_(False).eval()
        backbone.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(mode);self.backbone.eval()
        if self.candidate is not None:self.candidate.eval()
        return self

    def forward(self,image):
        base=self.backbone(image)
        if self.candidate is None or self.policy.get('return_base'):return {'output':base,'base':base}
        with torch.no_grad():
            if self.rgb:P,V=rgb_prior(image)
            else:
                f=make_prior(image);P,V=f['P'],f['V']
            candidate=self.candidate(image,base,P,V)
        if self.controller is None:
            alpha=self.policy.get('alpha',1.)
            return {'output':base+alpha*(candidate-base),'base':base,'candidate':candidate}
        kwargs={'lamb':self.policy.get('lambda',1e-4),'tau':self.policy.get('tau',0.),
                'se2':self.policy.get('s_e2',1e-4),'temperature':self.policy.get('temperature',1.),'margin':self.policy.get('margin',0.)}
        result=self.controller(image,base,candidate,**kwargs);result.update(base=base,candidate=candidate)
        return result
