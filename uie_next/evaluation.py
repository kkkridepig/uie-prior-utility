import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .records import sha


def psnr(output,target):
    if not torch.isfinite(output).all() or not torch.isfinite(target).all():raise ValueError('nonfinite metric input')
    mse=(output.double()-target.double()).square().flatten(1).mean(-1)
    return [-10*math.log10(float(x)) if x>0 else '+inf' for x in mse],mse


def ssim(output,target):
    if min(output.shape[-2:])<11: raise ValueError('11x11 valid SSIM requires size >=11')
    x=output.double();y=target.double()
    coord=torch.arange(-5,6,device=x.device,dtype=x.dtype)
    kernel=torch.exp(-coord.square()/(2*1.5**2));kernel/=kernel.sum()
    window=(kernel[:,None]*kernel[None,:]).view(1,1,11,11).repeat(3,1,1,1)
    conv=lambda v:F.conv2d(v,window,groups=3)
    ux=conv(x);uy=conv(y)
    vx=conv(x*x)-ux*ux;vy=conv(y*y)-uy*uy;vxy=conv(x*y)-ux*uy
    value=((2*ux*uy+.01**2)*(2*vxy+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return value.flatten(1).mean(-1)


class VerifiedLPIPS:
    def __init__(self,vgg_path,device='cpu'):
        import lpips
        from importlib.metadata import version
        if version('lpips')!='0.1.4': raise ValueError('LPIPS version changed')
        path=Path(vgg_path)
        if not path.is_file(): raise FileNotFoundError('official VGG16 weights missing')
        weight_sha=sha(path)
        if not weight_sha.startswith('397923af'):raise ValueError('not the identified official VGG16 checkpoint')
        # Build offline and replace every feature parameter before the metric is usable.
        self.net=lpips.LPIPS(net='vgg',version='0.1',pnet_rand=True,pretrained=True,verbose=False)
        weights=torch.load(str(path),map_location='cpu')
        mapped={k:weights['features.'+'.'.join(k.split('.')[1:])] for k in self.net.net.state_dict()}
        self.net.net.load_state_dict(mapped,strict=True);self.net.eval().requires_grad_(False).to(device)
        calibrated=Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
        self.identity={'lpips_version':'0.1.4','vgg16_sha256':weight_sha,'calibration_sha256':sha(calibrated),'normalization':'2*RGB01-1'}

    def __call__(self,output,target):
        with torch.no_grad(): return self.net(2*output-1,2*target-1).flatten()


def bootstrap_group(differences,groups,seed=20261017,repeats=2000):
    values=np.asarray(differences,dtype=np.float64);names=np.asarray(groups)
    if not np.isfinite(values).all(): raise ValueError('undefined PSNR difference requires explicit investigation')
    unique=np.unique(names);rng=np.random.RandomState(seed)
    idx={g:np.flatnonzero(names==g) for g in unique}
    samples=[]
    for _ in range(repeats):
        selected=rng.choice(unique,len(unique),replace=True)
        samples.append(float(values[np.concatenate([idx[g] for g in selected])].mean()))
    return {'mean_image':float(values.mean()),'mean_group':float(np.mean([values[idx[g]].mean() for g in unique])),
            'ci95':np.quantile(samples,[.025,.975]).tolist(),'groups':len(unique),'repeats':repeats,'seed':seed}
