"""Isolated V2 context, write guard, protocol and role permissions."""
import contextlib
import copy
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
import torch
import yaml
from ..records import ROOT,RunContext,sha,read,write,append,digest
from ..budget import DeviceBudget
from ..data.runtime import RuntimeData
from ..data.cache import get
from ..data.roles import RoleGuard

RUN_ID='ssuie_local_utility_v2_diagnostic_20261007'
PROTOCOL_SHA='d8b5d2046ebf09ad12295052701ea0d12218983258aac775fd31bb9af0479034'
OLD=ROOT/'runs/ssuie_local_utility_v1_20261007'
METHOD_ORDER=['G1','R0','O','B4','G0','G2','F0','O-NI','O-NP','O-NS','O-ND']
STAGES=['RECOVERY_AUDIT','ENGINEERING_ACCEPTANCE','D0_MODEL_DIAGNOSTICS','MODEL_PRODUCER_FREEZE','D0_UTILITY_DIAGNOSTICS',
        'ONE_RESCUE','RESCUE_MODEL_FREEZE','RESCUE_UTILITY_GATE','PRODUCER_FROZEN','UTILITY_CORE','UTILITY_REMAINING',
        'NETWORK_FREEZE','CALIBRATION','DEV_GATE','SEALED_ONCE','CLOSEOUT']


def config_template():
    reference=ROOT/'configs/uie_next/protocol_v2_reference.json'
    if reference.exists():
        return read(reference)
    c=yaml.safe_load((ROOT/'configs/uie_next/protocol.yaml').read_text())
    c['schema_version']=2;c['experiment']='ssuie_local_utility_v2_diagnostic'
    c['backbone']['output_policy']='clip01';c['training'].update(candidate_updates=12000,utility_updates=3000,
         candidate_update_options=[12000],utility_update_options=[3000])
    c['gates']['bootstrap_draws']=5000
    c['runtime'].update(run_id=RUN_ID,run_dir=str(ROOT/'runs'/RUN_ID))
    c['data'].update(manifest=str(ROOT/'runs'/RUN_ID/'roles.jsonl'),exposure_ledger=str(ROOT/'runs'/RUN_ID/'exposure_ledger.json'))
    c['v2']={'protocol_sha256':PROTOCOL_SHA,'inherited_budget_run':str(OLD),'max_rescue_device_hours':4,
             'method_order':METHOD_ORDER,'diagnostic_points':[1000,2000,3000,4000],
             'probe_groups':128,'probe_seed':20261008,'rescue_lr':1e-5,'LOG_eps':1e-6,
             'backend':'vendor_PPU_float32_cpu_float64_exact_oracle','final_reserved_seconds':12600,
             'coarse_H32':.15,'coarse_G32':.05,'pixel_Gp':.10,'fine_Hp':.15,
             'max_prior_response_min':1e-4,'network_checkpoint_steps':[0,750,1500,2250,3000]}
    return c


def load_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c!=config_template():raise ValueError('V2 structured protocol differs from archived fixed recipe')
    return c


class V2RoleGuard(RoleGuard):
    def __init__(self,rows,ctx):super().__init__(rows);self.ctx=ctx
    def check(self,sample_id,operation):
        row=self.rows[sample_id]
        if operation=='probe_diagnostic':
            if row['role']!='model_fit':raise PermissionError('probe only model_fit')
        elif operation=='model_diagnostic':
            if row['role']!='model_val':raise PermissionError('model diagnostics only model_val')
        else:
            if operation=='calibrate' and not (self.ctx.run/'selection/network_freeze_before_calibration.json').exists():
                raise PermissionError('calibration blocked before all networks frozen')
            super().check(sample_id,operation)
        append(self.ctx.run/'access_events.jsonl',{'sample_id':sample_id,'role':row['role'],'operation':operation})
        return row


class V2Data(RuntimeData):
    def __init__(self,config,backbone,ctx):
        super().__init__(config,backbone);self.ctx=ctx;self.guard=V2RoleGuard(self.rows,ctx);self.by_id=self.guard.rows
        self.policy='clip01';self.reuse_verified=False
    def base_pair(self,row,image=None):
        if row['role'] in ('calibration','sealed_eval'):
            self.guard.check(row['sample_id'],'calibrate' if row['role']=='calibration' else 'final')
        old_path=OLD/'cache/base'/('%s.pt'%digest(row['sample_id']))
        if self.reuse_verified and old_path.exists():return get(old_path,self.identity(row))
        return super().base_pair(row,image)
    def identity(self,row,candidate_hash=None,views='base_both_policies'):
        value=super().identity(row,candidate_hash,views)
        if candidate_hash:value.update(protocol_sha256=PROTOCOL_SHA,backend=self.config['v2']['backend'],role_manifest_sha256=sha(self.run/'roles.jsonl'))
        return value


class State:
    def __init__(self):
        self.ctx=RunContext(ROOT,RUN_ID,PROTOCOL_SHA);self.run=self.ctx.run;self.doc=self.ctx.doc
        for p in [self.run,self.doc]:p.mkdir(parents=True,exist_ok=True)
        self.config=config_template();self.path=self.run/'state.json';self.active_budget=None
        if not self.path.exists():
            self.ctx.write(self.path,{'schema_version':2,'run_id':RUN_ID,'config_hash':digest(self.config),'protocol_sha256':PROTOCOL_SHA,
               'completed':[],'scientific_status':'RECOVERY_AUDIT','sealed_eval_released':False,'formal_training_updates_v2':0,
               'independent_backup_verified':False,'stages':{s:{'status':'pending'} for s in STAGES}})
        self.state=read(self.path)
        if self.state['config_hash']!=digest(self.config):raise ValueError('run configuration identity mismatch')
    def live(self,**changes):
        self.state.update(changes,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid());self.ctx.write(self.path,self.state)
        budget=read(self.run/'budget.json') if (self.run/'budget.json').exists() else {}
        self.ctx.text(self.doc/'LIVE_STATUS.md','# V2 实时状态\n\n科学状态：'+self.state['scientific_status']+'\n\n阶段：'+str(self.state.get('current_job','RECOVERY_AUDIT'))+'\n\n累计设备小时：'+str(budget.get('used_device_seconds',0)/3600)+' / 16；保留3.5小时。\n\n封存评分解锁：'+str(self.state['sealed_eval_released'])+'；独立备份核验：false。\n')
    def complete(self,name,receipt):
        if name not in self.state['completed']:self.state['completed'].append(name)
        self.state['stages'][name]={'status':'complete','receipt':receipt}
        append(self.run/'events.jsonl',{'stage':name,'event':'complete','receipt':receipt});self.live()
        if name in ['D0_MODEL_DIAGNOSTICS','D0_UTILITY_DIAGNOSTICS','ONE_RESCUE','UTILITY_CORE','UTILITY_REMAINING','NETWORK_FREEZE','CALIBRATION']:
            from .backups import stage_backup
            stage_backup(self,name)
    @contextlib.contextmanager
    def device_job(self,name,estimate=600,final=False):
        shared=(ROOT/'runs/.ssuie_ppu_device.lock').open('a+')
        try:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with DeviceBudget(self.run) as b:
                b.start(name,estimate,final=final);self.active_budget=b;self.live(current_job=name)
                try:yield b
                finally:
                    torch.cuda.synchronize();self.active_budget=None
        finally:shared.close()
    def guard(self):
        if self.active_budget:
            self.active_budget.guard(10)
            if self.state.get('rescue_budget_active'):
                used=read(self.run/'budget.json')['used_device_seconds']
                if used-self.state['rescue_budget_start_seconds']>=14390:
                    from ..budget import BudgetStop
                    raise BudgetStop('single rescue package 4-device-hour limit reached')
            plan=self.run/'budget_plan.json'
            if plan.exists():
                b=read(self.run/'budget.json')
                reserved=read(plan).get('protected_final_seconds',12600)
                if b['active'] and not b['active']['final'] and 57600-reserved-b['used_device_seconds']<10:
                    from ..budget import BudgetStop
                    raise BudgetStop('profiled final reserve reached; save and stop')
    def source_snapshot(self):
        paths=list((ROOT/'uie_next').rglob('*.py'))+list((ROOT/'tests/uie_next').rglob('*.py'))+[ROOT/'configs/uie_next/protocol_v2.yaml',ROOT/'configs/uie_next/protocol_v2_reference.json']
        value={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
        self.ctx.write(self.run/'source_snapshot.json',value);return sha(self.run/'source_snapshot.json')
    def audit(self):
        if sha(self.run/'protocol_source.md')!=PROTOCOL_SHA:raise ValueError('protocol hash mismatch')
        from .diagnostics import inventory
        checkpoints=inventory();self.ctx.write(self.run/'diagnostics/checkpoint_inventory.json',checkpoints)
        snapshot=read(OLD/'source_snapshot.json');restored={p:{'expected':h,'actual':sha(ROOT/p)} for p,h in snapshot.items() if p.startswith('uie_next/data/')}
        if any(v['actual']!=v['expected'] for v in restored.values()):raise ValueError('original source identity changed')
        original=read(OLD/'budget.json')
        if original['active'] is not None:raise RuntimeError('old budget has unreconciled active job')
        if not (self.run/'budget.json').exists():
            self.ctx.write(self.run/'budget.json',{**original,'inherited_device_seconds':original['used_device_seconds'],'inherited_budget_sha256':sha(OLD/'budget.json')})
            append(self.run/'budget_ledger.jsonl',{'event':'inherit_v1','device_seconds':original['used_device_seconds'],'budget_sha256':sha(OLD/'budget.json')})
        for name in ['roles.jsonl','exposure_ledger.json','backbone_provenance.json','split_freeze.json']:
            src=OLD/name;dst=self.run/name
            if dst.exists() and sha(dst)!=sha(src):raise ValueError('data identity differs: '+name)
            if not dst.exists():shutil.copy2(src,dst)
        rows=[json.loads(x) for x in (self.run/'roles.jsonl').read_text().splitlines()]
        from collections import Counter
        counts=dict(Counter(r['role'] for r in rows))
        expected={'model_fit':3608,'model_val':671,'utility_fit':443,'utility_val':136,'calibration':133,'sealed_eval':177,'excluded_overlap':1}
        if counts!=expected:raise ValueError('role counts differ')
        groups={};errors=[]
        for row in rows:
            groups.setdefault(row['group_id'],set()).add(row['role'])
            for key in ['input','reference']:
                if sha(row[key+'_path'])!=row[key+'_sha256']:errors.append(row['sample_id']+':'+key)
        if errors or any(len(v-{'excluded_overlap'})>1 for v in groups.values()):raise ValueError('data identity or cross-role group mismatch '+str(errors))
        cpath=ROOT/'configs/uie_next/protocol_v2.yaml';cpath.write_text(yaml.safe_dump(self.config,sort_keys=False),encoding='utf-8')
        self.ctx.text(self.run/'protocol_resolved.yaml',cpath.read_text())
        write(self.run/'data_audit.json',{'role_counts':counts,'all_file_hashes_verified':True,'active_group_cross_role':False,
          'excluded_overlap_group':{g:sorted(v) for g,v in groups.items() if len(v)>1},
          'historical_exposure':True,'sealed_is_new_blind_test':False,'upstream_membership':'LSUI unknown; UIEB documented nonoverlap limited to declared LSUI source and local audit',
          'identity_only_sealed_read':True,'reference_scores_calibration_or_sealed':False})
        env={'python':sys.version,'executable':sys.executable,'torch_version':torch.__version__,'torch_path':torch.__file__,'torch_init_sha256':sha(torch.__file__),
             'device_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             'device_memory':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
             'disk_free':shutil.disk_usage(ROOT).free,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'vendor_torch_preserved':sha(torch.__file__)==read(OLD/'intake.json')['vendor_torch_init_sha256']}
        if not env['vendor_torch_preserved']:raise ValueError('vendor torch changed')
        self.ctx.write(self.run/'environment.json',env)
        self.complete('RECOVERY_AUDIT',{'data':sha(self.run/'data_audit.json'),'checkpoints':sha(self.run/'diagnostics/checkpoint_inventory.json')})
