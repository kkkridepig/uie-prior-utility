import torch
from torch import nn
import torch.nn.functional as F
from mpa_diff.contracts import PriorBundle
from mpa_diff.models.unet import BetaUNet, UNet
from mpa_diff.priors.kernels import pad_image, haar, sobel, histogram, background
from mpa_diff.physics.renderer import render, invert
from mpa_diff.priors.depth import DepthAnythingV2Provider, SyntheticFixtureDepth
from mpa_diff.diffusion.core import Schedule, sample

class PriorProvider(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        mc=config['model']
        self.beta=BetaUNet(mc['base_channels'],mc['channel_mult'],mc['residual_blocks'],mc['dropout'])
        self.depth= SyntheticFixtureDepth() if config['depth']['provider']=='synthetic_fixture' else DepthAnythingV2Provider(config['depth'])
    @torch.no_grad()
    def static(self,image,depth=None):
        hc=self.config['histogram']
        depth=self.depth(image) if depth is None else depth
        hist=histogram(image,hc['bins'],hc['bandwidth'],hc['max_input_size'],hc['mode'])
        return {'depth':depth,'histogram':hist,'ambient':background(image),'edges':sobel(image),'wavelet':haar(image)[1]}
    def build(self,image,metadata=None,static=None):
        static=self.static(image) if static is None else static
        k=self.beta(image)
        _,fields=render(image,static['depth'].physical_coordinate,k,k,static['ambient'])
        physical,diagnostics=invert(image,fields)
        fields.update(diagnostics)
        return PriorBundle(image,static['depth'],physical,static['histogram'],static['edges'],static['wavelet'],fields,metadata or {})

class MPADiff(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        m=config['model']
        self.priors=PriorProvider(config)
        self.highfreq=nn.Sequential(nn.Conv2d(12,m['base_channels'],3,padding=1),nn.SiLU(),nn.Conv2d(m['base_channels'],m['base_channels'],3,padding=1))
        self.denoiser=UNet(m['base_channels'],m['channel_mult'],m['residual_blocks'],m['dropout'])
        d=config['diffusion']; self.schedule=Schedule(d['train_steps'],d['beta_start'],d['beta_end'])
    def condition(self,image,static=None):
        p=self.priors.build(image,static=static)
        shape=image.shape[-2:]
        cond={'image':image,'physical':p.physical_image,
              'histogram':F.interpolate(p.histogram,size=shape,mode='bilinear',align_corners=False),
              'highfreq':self.highfreq(torch.cat((F.interpolate(p.wavelet_high,size=shape,mode='bilinear',align_corners=False),p.edges),1))}
        return cond,p
    def forward(self,image,state,index,static=None):
        condition,_=self.condition(image,static)
        return self.denoiser(state,index,condition)
    @torch.no_grad()
    def enhance(self,image,seed=0,sampler=None):
        self.eval()
        padded,info=pad_image(image)
        cond,priors=self.condition(padded)
        sc=sampler or self.config['sampler']
        generator=torch.Generator(device=image.device).manual_seed(seed)
        result,diag=sample(self.denoiser,cond,padded.shape,self.schedule,generator,sc['name'],sc['steps'],sc.get('eta',0),clip=sc['clip_intermediate_x0'])
        h,w=info['hw']; diag['padding']=info; diag['seed']=seed
        return result[...,:h,:w].clamp(0,1),diag,priors
