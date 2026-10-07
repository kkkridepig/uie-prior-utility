"""V3-only state, read-only inherited artifacts, guarded roles and cumulative leases."""
import contextlib,fcntl,hashlib,json,os,platform,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch,yaml
from ..records import ROOT,RunContext,sha,read,write,append,digest
from ..budget import DeviceBudget,BudgetStop
from ..data.runtime import RuntimeData
from ..data.cache import get
from ..data.manifest import image_tensor

RUN_ID='ssuie_utility_v3_predictability_20261007'
PROTOCOL_SHA='181458b3b0d2acf8cb4d82069436dfd9bbe28e67436c03ff7aa0d76181624cc8'
OLD=ROOT/'runs/ssuie_local_utility_v2_diagnostic_20261007'
V1=ROOT/'runs/ssuie_local_utility_v1_20261007'
ROLE_SHA='e6176357cbccc2424c268c245ef7e8c3b7dd36c4ef7f2fbbd2fa9b4070b520ce'
PRODUCER_SHA='c1af746056f0419eeb5787283a5c5e28d15d1ceec95d1dd6cedd4ed7592879af'
STAGES=['RECOVERY_AUDIT','PROTOCOL_AND_DATA_FREEZE','STAGE1_ACCEPTANCE','D1_D6_DIAGNOSTICS','D7_OOF_AND_DEV_PROBES','STAGE1_DECISION','STAGE2_ACCEPTANCE_AND_PROFILE','TRAINING_SCHEDULE_FREEZE','COMPLETE_MATCHED_MATRIX','NETWORK_FREEZE','CALIBRATION','DEV_GATE','SEALED_ONCE','CLOSEOUT']


class State:
    def __init__(self):
        self.ctx=RunContext(ROOT,RUN_ID,PROTOCOL_SHA);self.run=self.ctx.run;self.doc=self.ctx.doc
        self.run.mkdir(parents=True,exist_ok=True);self.doc.mkdir(parents=True,exist_ok=True)
        self.path=self.run/'state.json';self.active=None;self.stage=None
        if not self.path.exists():write(self.path,{'run_id':RUN_ID,'protocol_sha256':PROTOCOL_SHA,'status':'RECOVERY_AUDIT','completed':[],
           'stages':{s:{'status':'pending'} for s in STAGES},'sealed_eval_released':False,'independent_backup_verified':False})
        self.state=read(self.path)
        if self.state['protocol_sha256']!=PROTOCOL_SHA:raise ValueError('protocol changed')
        self.config=yaml.safe_load((OLD/'protocol_resolved.yaml').read_text())
    def live(self,**changes):
        self.state.update(changes,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),pid=os.getpid());write(self.path,self.state)
        b=read(self.run/'budget.json') if (self.run/'budget.json').exists() else {}
        self.ctx.text(self.doc/'LIVE_STATUS.md','# V3 实时状态\n\n状态：'+self.state['status']+'\n\n当前任务：'+str(self.state.get('current_job'))+'\n\n累计设备小时：'+str(b.get('used_device_seconds',0)/3600)+' / 16。封存解锁：'+str(self.state['sealed_eval_released'])+'。独立备份：未验证。\n')
    def complete(self,name,receipt):
        if name not in self.state['completed']:self.state['completed'].append(name)
        self.state['stages'][name]={'status':'complete','receipt':receipt}
        append(self.run/'events.jsonl',{'stage':name,'event':'complete','receipt':receipt});self.live()
    @contextlib.contextmanager
    def device_job(self,name,estimate=120,stage=1,final=False):
        with (ROOT/'runs/.ssuie_ppu_device.lock').open('a+') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with DeviceBudget(self.run) as b:
                budgets=read(self.run/'stage_budget.json')
                orphan=budgets.pop('_active',None)
                if orphan:
                    charged=read(self.run/'budget.json')['used_device_seconds']-orphan['baseline_seconds']
                    key=str(orphan['stage']);budgets[key]=budgets.get(key,0)+charged
                    append(self.run/'device_events.jsonl',{'job':orphan['job'],'stage':orphan['stage'],'device_seconds':charged,'reconciled_orphan_conservative':True})
                    write(self.run/'stage_budget.json',budgets)
                cap={1:3600,2:14400}.get(stage)
                if cap and budgets.get(str(stage),0)+estimate>cap:raise BudgetStop('stage cap insufficient for predicted package')
                before=read(self.run/'budget.json')['used_device_seconds'];self.stage=stage
                b.start(name,estimate,final=final);self.active=b
                budgets['_active']={'job':name,'stage':stage,'baseline_seconds':before};write(self.run/'stage_budget.json',budgets)
                self.live(current_job=name)
                try:yield b
                finally:
                    torch.cuda.synchronize();b.stop();self.active=None
                    after=read(self.run/'budget.json')['used_device_seconds'];budgets[str(stage)]=budgets.get(str(stage),0)+after-before
                    budgets.pop('_active',None);write(self.run/'stage_budget.json',budgets);append(self.run/'device_events.jsonl',{'job':name,'stage':stage,'device_seconds':after-before,'cumulative_seconds':after})
    def guard(self):
        if self.active:
            self.active.guard(15)
            budgets=read(self.run/'stage_budget.json');b=read(self.run/'budget.json')
            pending=max(0,time.time()-b['active']['last_accounted_at']) if b['active'] else 0
            event=read(self.run/'device_events.jsonl') if False else None
            # cumulative current-stage cost includes the active job via a fixed baseline.
            cap={1:3600,2:14400}.get(self.stage)
            ledger=[json.loads(x) for x in (self.run/'device_events.jsonl').read_text().splitlines()] if (self.run/'device_events.jsonl').exists() else []
            completed=sum(x['device_seconds'] for x in ledger)
            active=b['used_device_seconds']-b['inherited_device_seconds']-completed+pending
            if cap and budgets.get(str(self.stage),0)+active+15>=cap:raise BudgetStop('stage deadline; preserve state')
    def snapshot(self):
        paths=list((ROOT/'uie_next').rglob('*.py'))+list((ROOT/'tests/ssuie_v3').rglob('*.py'))+list((ROOT/'scripts').glob('ssuie_v3*.py'))+[ROOT/'configs/ssuie_utility_v3.yaml']
        v={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.exists()};write(self.run/'source_snapshot.json',v);return sha(self.run/'source_snapshot.json')
    def audit(self):
        if sha(self.doc/'protocol_source.md')!=PROTOCOL_SHA:raise ValueError('uploaded protocol mismatch')
        shutil.copy2(self.doc/'protocol_source.md',self.run/'protocol_source.md')
        self.ctx.text(self.doc/'protocol_sha256.txt',PROTOCOL_SHA+'  protocol_source.md\n')
        if sha(OLD/'roles.jsonl')!=ROLE_SHA:raise ValueError('role identity mismatch')
        budget=read(OLD/'budget.json')
        if budget['active'] is not None:raise ValueError('unreconciled old job')
        if budget['used_device_seconds']<15623.031541585922:raise ValueError('budget rollback')
        if not (self.run/'budget.json').exists():
            write(self.run/'budget.json',{**budget,'inherited_device_seconds':budget['used_device_seconds'],'inherited_budget_sha256':sha(OLD/'budget.json')})
            append(self.run/'budget_ledger.jsonl',{'event':'inherit_v2_includes_v1','device_seconds':budget['used_device_seconds'],'source_sha256':sha(OLD/'budget.json')})
            write(self.run/'stage_budget.json',{'1':0.,'2':0.,'3':0.})
        for name in ['roles.jsonl','split_freeze.json','exposure_ledger.json','backbone_provenance.json']:
            dst=self.run/name
            if dst.exists() and sha(dst)!=sha(OLD/name):raise ValueError('inherited file mismatch')
            if not dst.exists():shutil.copy2(OLD/name,dst)
        originals={}
        for old in [V1,OLD]:
            originals.update({str(p.relative_to(ROOT)):sha(p) for p in old.rglob('*') if p.is_file()})
        before=self.run/'inherited_files_sha256.json'
        if before.exists() and read(before)!=originals:raise ValueError('old artifacts changed')
        if not before.exists():write(before,originals)
        rows=[json.loads(x) for x in (self.run/'roles.jsonl').read_text().splitlines()]
        counts=dict(Counter(r['role'] for r in rows));groups={}
        for r in rows:
            groups.setdefault(r['group_id'],set()).add(r['role'])
            for k in ['input','reference']:
                if sha(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('data file changed')
        if any(len(v-{'excluded_overlap'})>1 for v in groups.values()):raise ValueError('group crosses roles')
        checkpoints={};registry=read(OLD/'method_registry.json')
        for m,v in registry.items():
            for k in ['candidate','controller']:
                if v[k]:
                    p=Path(v[k]['path']);h=sha(p)
                    if h!=v[k]['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
                    checkpoints[str(p.relative_to(ROOT))]=h
        for m in ['O','O-NP','O-NI','O-ND']:
            p=OLD/'checkpoints'/m/'step_003000.pt';checkpoints[str(p.relative_to(ROOT))]=sha(p)
        c=self.config
        if sha(c['backbone']['checkpoint'])!=c['backbone']['checkpoint_sha256']:raise ValueError('backbone mismatch')
        checkpoints[str(Path(c['backbone']['checkpoint']).relative_to(ROOT))]=sha(c['backbone']['checkpoint'])
        for p,h in read(OLD/'source_snapshot.json').items():
            if sha(ROOT/p)!=h:raise ValueError('frozen scientific source changed: '+p)
        if sha(V1/'checkpoints/B1/step_003000.pt')!=PRODUCER_SHA:raise ValueError('producer mismatch')
        env={'torch':torch.__version__,'torch_file':torch.__file__,'python':sys.version,'device_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0),
             'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'disk':list(shutil.disk_usage(ROOT)),
             'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
             'upstream_commit':subprocess.check_output(['git','-C',str(ROOT/'third_party/ss_uie'),'rev-parse','HEAD'],text=True).strip(),
             'vendor_preserved':sha(torch.__file__)==read(V1/'intake.json')['vendor_torch_init_sha256']}
        if not env['vendor_preserved']:raise ValueError('vendor environment changed')
        write(self.run/'environment.json',env)
        audit={'weights':checkpoints,'roles_sha256':ROLE_SHA,'role_counts':counts,'all_data_hashes_verified':True,
          'old_artifact_count':len(originals),'old_source_count':len(read(OLD/'source_snapshot.json')),'budget':budget,
          'sealed_scored':False,'upstream_membership':'unknown author per-image membership','historical_exposure':True,
          'near_duplicate_groups':'inherited content proxies; not human-verified scenes','review_missing':'optional independent review not uploaded; frozen raw records used'}
        write(self.run/'recovery_audit.json',audit);write(self.run/'data_exposure_ledger.json',audit)
        resolved={'protocol_sha256':PROTOCOL_SHA,'run_id':RUN_ID,'inherited_run':OLD.name,'seed':20261007,'producer_sha256':PRODUCER_SHA,
           'budget':{'maximum':57600,'stage1':3600,'stage2':14400,'reserve':12600,'safety_factor':1.3},'ridge_penalty':.01,'folds':5,
           'stage1_gates':{'base_and_fixed':.03,'region_minus_global':.02,'block_oracle_minus_global':.05,'harm_slack':.05},
           'candidate_updates':[6000,12000],'kappa':[0,.25,.5,.75,1],'formal_gates':{'quality':.10,'mechanism':.03,'ssim':-.001,'lpips':.002},
           'preprocess':c['preprocess'],'inherited_config':c}
        text=yaml.safe_dump(resolved,allow_unicode=True,sort_keys=False);self.ctx.text(self.run/'protocol_resolved.yaml',text)
        (ROOT/'configs/ssuie_utility_v3.yaml').write_text(text)
        self.ctx.text(self.doc/'CURRENT_STATE_DIFF.md','# 现场审计\n\n基准提交0521bfc；新独立分支，无原用户工作区改动。旧V2科学停止保留。继承累计'+str(budget['used_device_seconds'])+'设备秒，已包含V1。厂商环境保留；数据、旧源码、权重哈希均核对。旧独立审阅文章未上传，使用原始冻结记录。新输出仅写V3。\n')
        self.complete('RECOVERY_AUDIT',{'audit':sha(self.run/'recovery_audit.json')});self.complete('PROTOCOL_AND_DATA_FREEZE',{'protocol':PROTOCOL_SHA,'roles':ROLE_SHA})


class Data:
    def __init__(self,s):
        self.s=s;self.rows=[json.loads(x) for x in (s.run/'roles.jsonl').read_text().splitlines()];self.by_id={r['sample_id']:r for r in self.rows}
        self.legacy=RuntimeData(s.config,None);self.legacy.policy='clip01'
    def role(self,name):return sorted([r for r in self.rows if r['role']==name],key=lambda r:r['sample_id'])
    def check(self,row,operation='diagnostic'):
        role=row['role'];allowed=role in ['utility_fit','utility_val']
        if operation=='train':allowed=role=='utility_fit'
        if role=='calibration':allowed=operation=='calibrate' and (self.s.run/'selection/network_freeze.json').exists()
        if role=='sealed_eval':allowed=operation=='final' and self.s.state.get('status')=='DEV_PASS' and (self.s.run/'selection/selection_freeze_before_sealed.json').exists()
        if not allowed:raise PermissionError('V3 role denied '+role+' '+operation)
        append(self.s.run/'access_events.jsonl',{'sample_id':row['sample_id'],'role':role,'operation':operation})
    def sample(self,row,views=False,operation='diagnostic'):
        self.check(row,operation)
        image=image_tensor(row['input_path']);target=image_tensor(row['reference_path'])
        ident=self.legacy.identity(row)
        filename=digest(row['sample_id'])+'.pt'
        basepath=V1/'cache/base'/filename
        if not basepath.exists():basepath=OLD/'cache/base'/filename
        base=get(basepath,ident)['clip01']
        ident=self.legacy.identity(row,PRODUCER_SHA,'seven_field_recomputed_views')
        ident.update(protocol_sha256='d8b5d2046ebf09ad12295052701ea0d12218983258aac775fd31bb9af0479034',backend=self.s.config['v2']['backend'],role_manifest_sha256=ROLE_SHA)
        c=get(OLD/'cache/candidates'/PRODUCER_SHA/(digest(row['sample_id'])+'.pt'),ident)['candidates']
        return {'image':image,'base':base,'target':target,'candidate':c[0],'views':c if views else None}
    def probe(self):
        rows=self.role('utility_fit');groups=sorted({r['group_id'] for r in rows},key=lambda g:hashlib.sha256(('v3-fit-probe-20261007|'+g).encode()).digest())[:32]
        out=[r for r in rows if r['group_id'] in groups]
        write(self.s.run/'diagnostics/probe_manifest.json',{'groups':groups,'sample_ids':[r['sample_id'] for r in out],'gradient_groups':groups[:8],'engineering_groups':groups[:4],'selection_uses_quality':False})
        return out
