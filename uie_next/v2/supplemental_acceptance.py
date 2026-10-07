"""V2 registry, cross-run lease and real-output fresh-process RNG recovery receipts."""
import subprocess
import sys
import copy
import os
import fcntl
import torch
from ..records import ROOT,read,write,sha
from ..checkpoint import atomic_torch
from ..priors.heuristic import interventions
from .context import METHOD_ORDER
from .diagnostics import checkpoint_model
from .evaluation import RegistryRuntime


def accept(s,d):
    path=s.run/'tests/supplemental_v2.json'
    if path.exists() and read(path)['passed']:return
    d.load();d.verify_reuse();directory=s.run/'tests/supplemental';directory.mkdir(parents=True,exist_ok=True)
    items=read(s.run/'diagnostics/checkpoint_inventory.json');entry=next(i for i in items if i['checkpoint_id']=='B1_004000')
    registry={}
    for method in METHOD_ORDER:
        p=s.run/'tests/real_matrix'/(method+'.pt');w={'checkpoint_id':method+'_fixture','checkpoint_sha256':sha(p),'path':str(p),'training_step':2,'head':'B3'}
        registry[method]={'kind':'fixed_candidate' if method=='B4' else 'controller','candidate':w if method=='B4' else entry,'controller':None if method=='B4' else w,'baseline_policy':'clip01'}
    scales=read(s.run/'tests/real_matrix/scales.json');errors=[]
    with s.device_job('V2_REGISTRY_AND_FRESH_PROCESS_RNG_ACCEPTANCE',600):
        rt=RegistryRuntime(d,registry,scales);ids=[r['sample_id'] for r in d.data.role('utility_fit')[:8]]+[r['sample_id'] for r in d.data.role('model_fit')[:8]]
        with torch.no_grad():
            for sid in ids:
                row=d.data.by_id[sid];batch=d.data.batch([sid],'utility_train' if row['role']=='utility_fit' else 'candidate_train');c,p=rt.features(batch['image'],batch['base'])
                for method in METHOD_ORDER:
                    policy={'alpha':.5} if method=='B4' else {'temperature':1.,'margin':.25} if method.startswith('G') else {'tau':1e-5,'lamb':1e-4}
                    cached,_=rt.output(method,policy,batch['image'],batch['base'],c,p);real=rt.deploy(method,policy,batch['image']);err=float((cached-real).abs().max());errors.append(err)
                    if err>1e-5:raise RuntimeError('registry mismatch '+method)
            b=d.data.batch(ids[:4],'utility_train');model=checkpoint_model(entry);fs=interventions(b['image']);b['candidates']=torch.stack([model(b['image'],b['base'],fs[k]['P'],fs[k]['V']) for k in s.config['prior']['train_views']],1)
            atomic_torch(directory/'fixture.pt',{'identity':{'engineering_fixture_only':True,'official_weight':s.config['backbone']['checkpoint_sha256'],'candidate':entry['checkpoint_sha256']},
              'batch':{k:v.cpu() for k,v in b.items()},'scales':scales})
        for mode in ['continuous','first','resume']:
            p=subprocess.run([sys.executable,'-m','uie_next.v2.recovery_probe',mode,'--directory',str(directory)],cwd='/tmp',env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
            (directory/(mode+'.log')).write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError('PPU fresh-process recovery failed '+mode)
        a=torch.load(directory/'continuous.pt',map_location='cpu');b=torch.load(directory/'resume.pt',map_location='cpu')
        err=max(float((v-b['model_state'][k]).abs().max()) for k,v in a['model_state'].items())
        if err>1e-5 or a['extra']['seen']!=b['extra']['seen'] or a['scheduler_state']!=b['scheduler_state']:raise RuntimeError('PPU parameter/LR/sample resume mismatch')
        for k in a['rng']['streams']:
            if not torch.equal(a['rng']['streams'][k],b['rng']['streams'][k]):raise RuntimeError('random stream mismatch')
        opt_error=0
        for key,state in a['optimizer_state']['state'].items():
            for field,v in state.items():
                other=b['optimizer_state']['state'][key][field]
                if torch.is_tensor(v):opt_error=max(opt_error,float((v-other).abs().max()))
                elif v!=other:raise RuntimeError('optimizer state mismatch')
        if opt_error>1e-5:raise RuntimeError('optimizer tensor mismatch')
    # Global device mutex: second process, different simulated run, cannot acquire.
    shared=(ROOT/'runs/.ssuie_ppu_device.lock').open('a+');fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code='import fcntl; f=open('+repr(str(ROOT/'runs/.ssuie_ppu_device.lock'))+',"a+");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: print("cross_run_blocked"); raise SystemExit(0)\nraise SystemExit(1)'
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);shared.close()
    if p.returncode:raise RuntimeError('cross-run shared device lease not exclusive')
    write(path,{'passed':True,'registry_cached_realtime_16_images_all_11':True,'registry_max_abs':max(errors),
       'real_PPU_4_vs_2_newprocess_2_model_max_abs':err,'optimizer_max_abs':opt_error,'RNG_and_source_view_and_LR_exact':True,
       'different_run_shared_device_lock':True,'engineering_fixture_only':True,'no_calibration_or_sealed_sample':True})
