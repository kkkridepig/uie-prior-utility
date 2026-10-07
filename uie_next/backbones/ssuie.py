import importlib
import subprocess
import sys
from pathlib import Path
import torch
from torch import nn
from ..records import sha

COMMIT='88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df'


class BackboneUnavailable(RuntimeError):
    pass


def postprocess(raw,policy):
    if not torch.isfinite(raw).all(): raise ValueError('nonfinite backbone output')
    if policy=='clip01': return raw.clamp(0,1)
    if policy=='official_minmax_float':
        lo=raw.amin((1,2,3),keepdim=True);hi=raw.amax((1,2,3),keepdim=True)
        return (raw-lo)/(hi-lo).clamp_min(1e-5)
    raise ValueError('unfrozen output policy')


class FrozenBackbone(nn.Module):
    def __init__(self,model,policy='clip01'):
        super().__init__();self.model=model;self.policy=policy
        self.model.requires_grad_(False);self.model.eval()

    def train(self,mode=True):
        super().train(mode);self.model.eval()
        return self

    def forward(self,image):
        if image.shape[1:]!=(3,256,256) or image.dtype!=torch.float32:
            raise ValueError('SS-UIE requires float32 RGB 256x256')
        with torch.no_grad(): return postprocess(self.model(image),self.policy)


def load_official(root,checkpoint,expected_sha=None,policy='clip01'):
    source=Path(root).resolve()/'third_party/ss_uie';path=Path(checkpoint)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=COMMIT: raise BackboneUnavailable('upstream commit mismatch')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise BackboneUnavailable('upstream source changed')
    if not path.is_file(): raise BackboneUnavailable('official SS-UIE checkpoint missing')
    h=sha(path)
    if expected_sha is not None and h!=expected_sha: raise BackboneUnavailable('backbone hash mismatch')
    # Absolute upstream imports must never resolve to an unrelated net package.
    for name in ['net','net.model','net.blocks']:
        mod=sys.modules.get(name)
        if mod is not None and source not in Path(mod.__file__).resolve().parents:
            raise BackboneUnavailable('net module collision: '+name)
    original=list(sys.path)
    original_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(source));module=importlib.import_module('net.model')
        if Path(module.__file__).resolve()!=source/'net/model.py': raise BackboneUnavailable('wrong official module')
        model=module.SS_UIE_model(in_channels=3,channels=16,num_resblock=4,num_memblock=4,H=256,W=256)
    except ImportError as exc: raise BackboneUnavailable(str(exc)) from exc
    finally:
        sys.path[:]=original
        sys.dont_write_bytecode=original_bytecode
    state=torch.load(str(path),map_location='cpu')
    if not isinstance(state,dict): raise BackboneUnavailable('not a state dictionary')
    if state and all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    frozen=FrozenBackbone(model,policy)
    return frozen,{'commit':COMMIT,'checkpoint_sha256':h,'size_bytes':path.stat().st_size,'module':str(module.__file__),'policy':policy}
