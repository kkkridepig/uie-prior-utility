"""Verified LPIPS-v0.1 VGG16, independent of the VGG19 training objective."""
from pathlib import Path
import inspect
import time
import torch
import lpips
from scripts.cde_v3.common import ROOT,RUN,sha,write
from mpa_diff.metrics.image import psnr,ssim

class Metrics:
    def __init__(self,device='cpu'):
        path=ROOT/'weights/cde_v3/vgg16-397923af.pth'
        h=sha(path)
        if not h.startswith('397923af'): raise ValueError('Official VGG16 filename hash mismatch')
        self.net=lpips.LPIPS(net='vgg',version='0.1',pnet_rand=True,verbose=False)
        state=torch.load(path,map_location='cpu')
        # pnet_rand avoids hidden network downloads; every used backbone weight is
        # immediately replaced strictly with the audited ImageNet VGG16 tensors.
        for name,module in self.net.net.named_children():
            mapped={k:state['features.'+k] for k in module.state_dict()}
            module.load_state_dict(mapped,strict=True)
        self.net.requires_grad_(False).eval().to(device); self.device=device
        head=Path(inspect.getfile(lpips)) .parent/'weights/v0.1/vgg.pth'
        identity={'package':'lpips==0.1.4','network':'VGG16','version':'0.1','backbone_source':'https://download.pytorch.org/models/vgg16-397923af.pth','backbone_sha256':h,'calibration_sha256':sha(head),'normalization':'2*RGB01-1 then official ScalingLayer','upstream_exposure':'ImageNet/BAPPS; specific UIEB scene exposure not provably exhaustive','metric_code_sha256':sha(Path(__file__))}
        write(RUN/'metric_identity.json',identity)
    @torch.no_grad()
    def __call__(self,pred,ref):
        if pred.ndim==3: pred=pred[None]; ref=ref[None]
        mse=(pred-ref).square().mean().item()
        return {'psnr':psnr(pred[0],ref[0]) if mse>0 else 'Inf','mse':mse,'ssim':ssim(pred[0],ref[0]),'lpips':self.net(2*pred.to(self.device)-1,2*ref.to(self.device)-1).mean().item()}
