from pathlib import Path
import cv2
import numpy as np
import torch
from ..records import sha


def image_tensor(path):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[2]!=3:
        raise ValueError('expected intact uint8 three-channel image: '+str(path))
    rgb=cv2.cvtColor(cv2.resize(raw,(256,256),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255.


def inference_input(path):
    """Deployment never accepts or resolves a reference path."""
    return image_tensor(Path(path))


def pair_tensor(guard,sample_id,operation):
    row=guard.check(sample_id,operation)
    for key in ['input','reference']:
        if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('audited image changed: '+sample_id)
    return image_tensor(row['input_path']),image_tensor(row['reference_path'])
