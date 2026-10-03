import sys
from pathlib import Path
import torch
from torch import nn
from mpa_diff.contracts import DepthPrior
from mpa_diff.utils.io import verified_weight


def normalize_depth(raw, coordinate='thesis_raw_minmax', raw_kind='relative_inverse'):
    valid=torch.isfinite(raw)
    safe=torch.where(valid,raw,torch.zeros_like(raw))
    lo=raw.masked_fill(~valid,float('inf')).flatten(1).min(1)[0][:,None,None,None]
    hi=raw.masked_fill(~valid,float('-inf')).flatten(1).max(1)[0][:,None,None,None]
    span=hi-lo
    good=torch.isfinite(span)&(span>=1e-6)
    normalized=torch.where(good,(safe-lo)/span.clamp_min(1e-6),torch.zeros_like(safe))
    normalized=torch.where(valid & good,normalized,torch.zeros_like(normalized))
    proxy=1-normalized if raw_kind=='relative_inverse' else normalized
    proxy=torch.where(valid & good,proxy,torch.zeros_like(proxy))
    physical=normalized if coordinate=='thesis_raw_minmax' else proxy
    return DepthPrior(safe,proxy,physical,(valid & good).float(),valid & good,
                      {'units':'unitless','raw_kind':raw_kind,'coordinate_kind':'legacy_inverse_normalized' if coordinate=='thesis_raw_minmax' and raw_kind=='relative_inverse' else 'distance_proxy','is_metric':False})

class DepthAnythingV2Provider(nn.Module):
    def __init__(self, config):
        super().__init__()
        path=verified_weight(config['checkpoint'],config['checkpoint_sha256'])
        root=Path(config['source_root']).resolve()
        if not (root/'depth_anything_v2/dpt.py').exists(): raise FileNotFoundError('Official DA-V2 source missing: '+str(root))
        sys.path.insert(0,str(root))
        from depth_anything_v2.dpt import DepthAnythingV2
        self.model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
        self.model.load_state_dict(torch.load(str(path),map_location='cpu'),strict=True)
        self.model.requires_grad_(False).eval()
        self.config=config
    def train(self, mode=True):
        super().train(False)
        return self
    @torch.no_grad()
    def forward(self, image):
        # Official transforms, with explicit device selection (upstream auto-selects cuda).
        import cv2
        from torchvision.transforms import Compose
        from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
        transform=Compose([Resize(width=self.config['input_size'],height=self.config['input_size'],resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()])
        arrays=image.detach().cpu().clamp(0,1).mul(255).byte().permute(0,2,3,1).numpy()
        raw=[]
        for a in arrays:
            inp=torch.from_numpy(transform({'image':a/255.})['image']).unsqueeze(0).to(image.device)
            depth=self.model(inp)
            depth=torch.nn.functional.interpolate(depth[:,None],size=image.shape[-2:],mode='bilinear',align_corners=True)[0]
            raw.append(depth.float())
        result=normalize_depth(torch.stack(raw),self.config['physics_coordinate'],self.config['raw_kind'])
        result.metadata['checkpoint_hash']=self.config['checkpoint_sha256']
        return result

class SyntheticFixtureDepth(nn.Module):
    """TEST ONLY: deterministic luminance proxy, explicitly not estimated depth."""
    def forward(self,image):
        result=normalize_depth(image.mean(1,keepdim=True))
        result.metadata['test_double']=True
        return result
