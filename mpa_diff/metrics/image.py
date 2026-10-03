import math
import torch
import torch.nn.functional as F


def psnr(pred,target):
    mse=(pred.clamp(0,1)-target).double().square().mean().item()
    return float('inf') if mse==0 else -10*math.log10(mse)


def ssim(pred,target):
    """RGB population covariance; Gaussian 11, sigma 1.5, valid border."""
    if min(pred.shape[-2:])<11: raise ValueError('SSIM needs both dimensions >=11')
    x=pred.clamp(0,1).double()[None]; y=target.double()[None]
    q=torch.arange(11,device=x.device,dtype=x.dtype)-5
    g=torch.exp(-q.square()/(2*1.5**2));g=g/g.sum()
    kernel=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
    filt=lambda a:F.conv2d(a,kernel,groups=3)
    mx,my=filt(x),filt(y)
    vx,vy=filt(x*x)-mx*mx,filt(y*y)-my*my
    cov=filt(x*y)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean().item()


def finite_json(value):
    return value if math.isfinite(value) else ('+inf' if value>0 else '-inf')
