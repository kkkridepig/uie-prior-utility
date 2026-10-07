import subprocess
import sys
import time
from pathlib import Path
import torch
from .budget import DeviceBudget
from .records import write


def recovery(run,device='cpu'):
    directory=Path(run)/'tests'/('recovery_ppu' if device.startswith('cuda') else 'recovery_cpu')
    directory.mkdir(parents=True,exist_ok=True);start=time.monotonic();commands=[]
    for mode in ['continuous','first','resume']:
        command=[sys.executable,'-m','uie_next.resume_probe',mode,'--directory',str(directory),'--device',device]
        p=subprocess.run(command,text=True,capture_output=True,timeout=120)
        (directory/(mode+'.log')).write_text(p.stdout+p.stderr)
        commands.append({'args':command,'exit_code':p.returncode})
        if p.returncode:raise RuntimeError('recovery probe failed; inspect '+mode+'.log')
    full=torch.load(directory/'continuous.pt',map_location='cpu');resumed=torch.load(directory/'resumed.pt',map_location='cpu')
    error=max(float((v-resumed['model_state'][k]).abs().max()) for k,v in full['model_state'].items())
    result={'kind':'SYNTHETIC_HEAD_ONLY','official_backbone':False,'device':device,'max_weight_abs_error':error,
            'identity':full['identity'],
            'same_scheduler':full['scheduler_state']==resumed['scheduler_state'],'commands':commands,
            'seconds':time.monotonic()-start,'passed':error<=1e-5}
    write(directory/'receipt.json',result)
    if not result['passed']:raise RuntimeError('recovery tolerance failed')
    return result


def recovery_ppu(run):
    with DeviceBudget(run) as b:
        b.start('synthetic_PPU_fresh_process_recovery',120)
        return recovery(run,'cuda:0')
