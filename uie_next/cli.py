"""Inspection and dependency-safe resumption. Importing this module does no work."""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .records import ROOT,GUIDE,write,read,sha,digest,append
from .schema import load
from .pipeline import paths,advance
from .budget import DeviceBudget


def output(value):print(json.dumps(value,ensure_ascii=False,indent=2))


def inspect(config):
    import torch,torchvision
    from importlib.metadata import version
    run,doc=paths(config)
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT)]+list(args),text=True).strip()
    package_names=['numpy','scipy','Pillow','timm','einops','lpips','opencv-python-headless','PyYAML',
                   'setuptools','packaging','wheel','mamba-ssm','causal-conv1d','transformers','tokenizers','safetensors']
    packages={}
    for n in package_names:
        try:packages[n]=version(n)
        except Exception:packages[n]='not_installed'
    env={'python':sys.version,'executable':sys.executable,'torch_version':torch.__version__,'torch_path':torch.__file__,
         'torchvision_version':torchvision.__version__,'torchvision_path':torchvision.__file__,'cuda_api':torch.version.cuda,
         'torch_cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,'device_count':torch.cuda.device_count(),
         'devices':[{'name':torch.cuda.get_device_name(i),'memory_bytes':torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())],
         'disk_free_bytes':shutil.disk_usage(ROOT).free,'mount':subprocess.check_output(['df','-P',str(ROOT)],text=True),
         'packages':packages}
    state={'root':str(ROOT),'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
           'status':git('status','--porcelain'),'run_id':config['runtime']['run_id'],'config_sha256':digest(config),
           'vendor_torch_init_sha256':sha(torch.__file__),'guide_sha256':sha(GUIDE)}
    if not (run/'intake.json').exists():
        write(run/'intake.json',state);write(run/'environment.json',env)
        write(run/'source_state.json',state)
        legacy=list((ROOT/'docs/experiments').rglob('*'))
        old={str(p.relative_to(ROOT)):sha(p) for p in legacy if p.is_file() and doc not in p.parents}
        for p in [ROOT/'uie/data.py',ROOT/'uie/download.py',ROOT/'uie/engine.py']:
            old[str(p.relative_to(ROOT))]=sha(p)
        write(run/'legacy_preservation.json',old)
    else:
        before=read(run/'intake.json')
        if before['config_sha256']!=digest(config):raise ValueError('intake protocol identity changed')
        write(run/'environment_latest.json',env)
    DeviceBudget(run,config['budget']['max_device_hours'],config['budget']['final_reserved_device_hours'])
    output(state);return {'passed':True,'status':'S0_INTAKE'}


def verify(config):
    run,_=paths(config);reasons=[];ckpt=config['backbone']['checkpoint']
    if not ckpt or not Path(ckpt).is_file():reasons.append('official_SS_UIE_checkpoint_missing')
    for name in ['mamba_ssm','selective_scan_cuda','causal_conv1d']:
        if importlib.util.find_spec(name) is None:reasons.append('missing_dependency:'+name)
    identity=None
    if not reasons:
        try:
            provenance=read(run/'backbone_provenance.json')
            receipt=read(run/'backbone_upload_verification_20261007/receipt_retry.json')
            if not provenance['accepted'] or not receipt['passed'] or sha(ckpt)!=config['backbone']['checkpoint_sha256']:
                raise ValueError('official provenance or real device receipt mismatch')
            identity=provenance
        except Exception as exc:reasons.append(type(exc).__name__+':'+str(exc))
    result={'passed':not reasons,'status':'BACKBONE_ADMITTED' if not reasons else 'BLOCKED_BACKBONE','reasons':reasons,'identity':identity,
            'official_weight_obtained':bool(ckpt and Path(ckpt).is_file()),'formal_training_started':False}
    write(run/'backbone_verification.json',result);output(result);return result


def budget_plan(config):
    run,_=paths(config);audit=read(run/'cross_dataset_audit.json')
    utility=sum(audit['role_counts'][n]['pairs'] for n in ['utility_fit','utility_val','calibration','sealed_eval'])
    base_bytes=audit['pairs']*3*256*256*4
    candidate_bytes=utility*7*3*256*256*4
    jobs=['B1','B3','B4','G0','G1','G2','F0','R0','O','O-NI','O-NP','O-NS','O-ND']
    result={'status':'NOT_PROFILED_BLOCKED_BACKBONE','jobs':[{'method':n,'updates':None,'seconds_per_update':None,'validation_seconds':None} for n in jobs],
            'max_device_hours':config['budget']['max_device_hours'],'reserve_device_hours':config['budget']['final_reserved_device_hours'],
            'candidate_update_options':config['training']['candidate_update_options'],'utility_update_options':config['training']['utility_update_options'],
            'safety_factor':1.3,'cost_prediction':None,'reason':'No official backbone; synthetic single updates cannot set formal common steps.',
            'cache_payload_float32_bytes':base_bytes+candidate_bytes,'cache_with_25_percent_margin_bytes':int(1.25*(base_bytes+candidate_bytes)),
            'cache_exists':False,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    write(run/'budget_plan.json',result);output(result);return result


def command_run(config):
    from .experiment import Experiment
    return Experiment(config).execute()


def main(argv=None):
    parser=argparse.ArgumentParser(description='SS-UIE local utility audit, training and evidence closeout')
    parser.add_argument('command',choices=['inspect','audit-data','verify-backbone','smoke','plan-budget','run','status','closeout'])
    parser.add_argument('--config',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--local-heads',action='store_true',help='Synthetic new-head PPU checks only; not official integration')
    args=parser.parse_args(argv);config=load(args.config);run,doc=paths(config)
    if args.command=='inspect':inspect(config)
    elif args.command=='audit-data':
        from .data.audit import run_audit
        output(run_audit())
    elif args.command=='verify-backbone':return 0 if verify(config)['passed'] else 2
    elif args.command=='smoke':
        if args.local_heads:
            from .smoke import local_head_smoke
            output(local_head_smoke(run))
        else:
            from .experiment import Experiment
            from .integration import accept_real,supplemental_real
            e=Experiment(config);e.load();e.baseline();accept_real(e);supplemental_real(e)
            output(read(run/'tests/real_integration/receipt.json'))
    elif args.command=='plan-budget':
        from .experiment import Experiment
        from .integration import accept_real,supplemental_real
        from .profiling import profile_and_freeze
        e=Experiment(config);e.load();e.baseline();accept_real(e);supplemental_real(e);e.freeze_source();profile_and_freeze(e)
        output(read(run/'budget_plan.json'))
    elif args.command=='run':
        result=command_run(config);output(result)
        return 2 if result['state']['scientific_status'].startswith(('BLOCKED','FAILED')) else 0
    elif args.command=='status':
        output({'state':read(run/'state.json') if (run/'state.json').exists() else 'NOT_STARTED','budget':read(run/'budget.json')})
    elif args.command=='closeout':
        from .experiment import Experiment
        from .delivery import deliver
        output(deliver(Experiment(config)))
    return 0


if __name__=='__main__':raise SystemExit(main())
