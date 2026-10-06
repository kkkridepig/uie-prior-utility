"""V3 identity, atomic records, role guards and independent keyed random streams."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/prior_utility_cde_v3_20261004'
DOC = ROOT/'docs/experiments/PRIOR_UTILITY_CDE_V3_20261004'
PARENT = ROOT/'runs/explore_ag_single_seed_v2_20261003/parent_0.pt'
PARENT_HASH = '3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea'
MODES = ('null','physical','histogram','highfreq','all')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def append(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a') as f: f.write(json.dumps(obj,allow_nan=False)+'\n'); f.flush()
def key(*parts): return int(digest(parts)[:15],16)
def generator(seed,*parts): return torch.Generator(device='cpu').manual_seed(key('noise_protocol_v3',seed,*parts))
def noise(shape,seed,sid,role='initial',t=0,device='cpu'):
    return torch.randn(shape,generator=generator(seed,sid,role,t)).to(device)
def setup():
    os.chdir(ROOT); torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def roles(role, purpose='evaluate', branch=None):
    allowed={'enhancer':{'adapter_fit'},'labels':{'route_fit'},'calibration':{'route_cal'},'development':{'source_dev'},'evaluate':{'adapter_fit','route_fit','route_cal','source_dev'}}
    if purpose=='enhancer' and branch=='BASE_CONT_DATA_MATCHED': ok=role in ('adapter_fit','route_fit')
    else: ok=role in allowed.get(purpose,set())
    if not ok: raise ValueError('Data-role violation: '+str((role,purpose,branch)))
    return read(RUN/'data_roles.json')['roles'][role]

def code_hash():
    files=sorted((ROOT/'scripts/cde_v3').glob('*.py'))+sorted((ROOT/'mpa_diff').rglob('*.py'))
    return digest({str(p.relative_to(ROOT)):sha(p) for p in files})

def train_identity():
    return {p:sha(ROOT/p) for p in ['scripts/cde_v3/common.py','scripts/cde_v3/model.py','scripts/cde_v3/train_bank.py','scripts/cde_v3/sampling.py','mpa_diff/engine/losses.py','mpa_diff/priors/provider.py','mpa_diff/models/unet.py']}

def deadline():
    d=float(os.environ.get('CDE_DEADLINE','inf'))
    return time.time() >= d-120
