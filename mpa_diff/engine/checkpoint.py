import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from mpa_diff.utils.io import canonical_hash,sha256


def seed_all(seed,threads=2):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def architecture_hash(config):
    return canonical_hash({k:config[k] for k in ('model','depth','physics','histogram','frequency','diffusion','extensions')})


def source_hash():
    root=Path(__file__).resolve().parents[1]
    return canonical_hash({str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))})


def provenance(config):
    try: head=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:head=None
    return {'command':sys.argv,'git_head':head,'source_sha256':source_hash(),'config_sha256':canonical_hash(config),'torch':torch.__version__,'torch_path':torch.__file__,'numpy':np.__version__,'device':torch.cuda.get_device_name(0) if config['runtime']['device'].startswith('cuda') else 'cpu','python':sys.version,'manifest_sha256':sha256(config['data']['manifest'])}


def save_checkpoint(path,model,optimizer,scheduler,step,config,order,cursor,best):
    path=Path(path)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':None,'rng':rng_state(),'step':step,'config':config,'architecture_hash':architecture_hash(config),'source_sha256':source_hash(),'manifest_sha256':sha256(config['data']['manifest']),'data_sampler':{'order':order,'cursor':cursor},'best_val_psnr':best}
    torch.save(state,str(path)+'.tmp');Path(str(path)+'.tmp').replace(path)


def load_model(path,device='cpu'):
    state=torch.load(path,map_location='cpu')
    config=state['config']
    if state['architecture_hash']!=architecture_hash(config):raise ValueError('Checkpoint contract mismatch')
    from mpa_diff.priors.provider import MPADiff
    model=MPADiff(config).to(device)
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),state
