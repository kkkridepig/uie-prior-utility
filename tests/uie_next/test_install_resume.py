import os
import subprocess
import sys
import torch
from pathlib import Path
from uie_next.records import ROOT


def test_T38_install_outside_repository(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    got=subprocess.run([sys.executable,'-m','uie_next.cli','--help'],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert got.returncode==0,got.stderr
    assert 'verify-backbone' in got.stdout
    mod=subprocess.check_output([sys.executable,'-c','import uie_next; print(uie_next.__file__)'],cwd=tmp_path,env=env,text=True).strip()
    assert Path(mod).resolve()==ROOT/'uie_next/__init__.py'


def test_T32_fresh_process_resume(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    for mode in ['continuous','first','resume']:
        p=subprocess.run([sys.executable,'-m','uie_next.resume_probe',mode,'--directory',str(tmp_path)],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=60)
        assert p.returncode==0,p.stderr
    a=torch.load(tmp_path/'continuous.pt');b=torch.load(tmp_path/'resumed.pt')
    assert a['global_step']==b['global_step']==20
    assert all(torch.equal(v,b['model_state'][k]) for k,v in a['model_state'].items())
    assert a['scheduler_state']==b['scheduler_state']
    assert a['identity']==b['identity']
    assert a['rng']['python']==b['rng']['python']
    assert torch.equal(a['rng']['torch_cpu'],b['rng']['torch_cpu'])
    for n in a['rng']['streams']:
        assert torch.equal(a['rng']['streams'][n],b['rng']['streams'][n])
    for p,state in a['optimizer_state']['state'].items():
        assert all(torch.equal(value,b['optimizer_state']['state'][p][key]) for key,value in state.items())
