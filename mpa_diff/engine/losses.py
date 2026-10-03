import torch
from torch import nn
from mpa_diff.utils.io import verified_weight


def masked_mse(pred,target,mask=None):
    if mask is None: return (pred-target).square().mean()
    mask=mask.expand_as(pred).to(pred.dtype)
    if mask.sum()==0: raise ValueError('No valid supervision elements')
    return ((pred-target).square()*mask).sum()/mask.sum()

class VGG19Loss(nn.Module):
    def __init__(self,path,digest):
        super().__init__()
        verified_weight(path,digest)
        from torchvision.models.vgg import make_layers, cfgs
        self.features=make_layers(cfgs['E'])[:18]
        state=torch.load(path,map_location='cpu')
        filtered={k[len('features.'):]:v for k,v in state.items() if k.startswith('features.') and int(k.split('.')[1])<18}
        self.features.load_state_dict(filtered,strict=True)
        self.features.requires_grad_(False).eval()
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def train(self,mode=True):
        super().train(False); return self
    def extract(self,x):
        h=(x-self.mean)/self.std; features=[]
        for index,layer in enumerate(self.features):
            h=layer(h)
            if index in (3,8,17): features.append(h)
        return features
    def forward(self,pred,target):
        p=self.extract(pred)
        with torch.no_grad(): t=self.extract(target)
        return sum((a-b).square().mean() for a,b in zip(p,t))/3
