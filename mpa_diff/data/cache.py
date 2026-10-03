"""Per-image, lossless float32 depth cache; cheap input priors are recomputed.

Avoid caching combinatorial minibatches or ~11MB of redundant maps per image.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from mpa_diff.utils.io import canonical_hash
from mpa_diff.priors.depth import normalize_depth
from mpa_diff.contracts import DepthPrior


def cache_key(image, config):
    import hashlib
    return canonical_hash({'input_tensor_sha256':hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'shape':list(image.shape),'depth':config['depth'],'preprocess':'rgb-bilinear-pad8-min16-v1','calibration':'none','storage':'raw-valid-float32-npz-v2'})


@torch.no_grad()
def static_cached(provider,image,config):
    root=Path(config['runtime']['cache_dir'])/'depth_v2';root.mkdir(parents=True,exist_ok=True)
    depths=[]
    for item in image.split(1):
        key=cache_key(item,config);path=root/(key+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as record:
                if str(record['key'])!=key:raise ValueError('Stale prior cache: '+str(path))
                raw=torch.from_numpy(record['raw'].copy()).to(image.device)
                valid=torch.from_numpy(record['valid'].copy()).to(image.device)
                metadata=json.loads(str(record['metadata']))
            depth=normalize_depth(raw.masked_fill(~valid,float('nan')),config['depth']['physics_coordinate'],config['depth']['raw_kind'])
            depth.raw=raw;depth.metadata=metadata
        else:
            depth=provider.depth(item)
            temporary=path.with_suffix('.%d.tmp'%os.getpid())
            with temporary.open('wb') as f:
                np.savez_compressed(f,key=key,raw=depth.raw.cpu().numpy(),valid=depth.valid_mask.cpu().numpy(),metadata=json.dumps(depth.metadata))
            temporary.replace(path)
        depths.append(depth)
    combined=DepthPrior(**{name:torch.cat([getattr(d,name) for d in depths]) for name in ('raw','distance_proxy','physical_coordinate','confidence','valid_mask')},metadata=depths[0].metadata)
    return provider.static(image,depth=combined)
