from pathlib import Path
import torch
from ..checkpoint import atomic_torch


REQUIRED={'input_sha256','preprocess','backbone_commit','backbone_weight_sha256','baseline_policy',
          'candidate_weight_sha256','prior_spec','intervention_spec','view_id','shape','dtype'}


def put(path,identity,tensors):
    if not REQUIRED.issubset(identity): raise ValueError('incomplete cache identity')
    if any(t.requires_grad or t.dtype!=torch.float32 for t in tensors.values()):
        raise ValueError('cache requires no-grad float32 tensors')
    atomic_torch(path,{'identity':identity,'tensors':{k:t.detach().cpu().clone() for k,t in tensors.items()}})


def get(path,identity):
    state=torch.load(str(Path(path)),map_location='cpu')
    if state['identity']!=identity: raise ValueError('stale cache rejected')
    if any(not torch.isfinite(t).all() for t in state['tensors'].values()): raise ValueError('nonfinite cache')
    return state['tensors']
