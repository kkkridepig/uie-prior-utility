import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from .records import sha,write


def rng_state(streams=None, include_cuda=False):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if include_cuda else None,
            'streams':{n:g.get_state() for n,g in (streams or {}).items()}}


def restore_rng(state,streams=None):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None: torch.cuda.set_rng_state_all(state['torch_cuda'])
    for n,g in (streams or {}).items(): g.set_state(state['streams'][n])


def atomic_torch(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=str(path.parent),prefix='.'+path.name)
    try:
        with os.fdopen(fd,'wb') as out:
            torch.save(state,out);out.flush();os.fsync(out.fileno())
        os.replace(temp,str(path))
        d=os.open(str(path.parent),os.O_DIRECTORY)
        try: os.fsync(d)
        finally: os.close(d)
    except BaseException:
        if os.path.exists(temp): os.unlink(temp)
        raise


def save(path,model,optimizer,scheduler,step,identity,streams=None,include_cuda=False,extra=None):
    state={'schema_version':1,'identity':identity,'model_state':model.state_dict(),
           'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),
           'global_step':step,'rng':rng_state(streams,include_cuda),'extra':extra or {}}
    atomic_torch(path,state)
    receipt={'sha256':sha(path),'size_bytes':Path(path).stat().st_size,'step':step,'identity':identity}
    write(str(path)+'.json',receipt)
    return receipt


def resume(path,model,optimizer,scheduler,identity,streams=None):
    state=torch.load(str(path),map_location='cpu')
    if state['identity']!=identity: raise ValueError('checkpoint identity changed')
    if sha(path)!=__import__('json').loads(Path(str(path)+'.json').read_text())['sha256']:
        raise ValueError('checkpoint checksum mismatch')
    model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state']);restore_rng(state['rng'],streams)
    return state
