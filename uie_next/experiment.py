"""Resumable single-seed scientific execution on the official public backbone."""
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .backbones.ssuie import load_official, postprocess
from .budget import BudgetStop, DeviceBudget
from .checkpoint import resume, save
from .data.audit import run_audit
from .data.runtime import RuntimeData
from .evaluation import VerifiedLPIPS, bootstrap_group, psnr, ssim
from .math.utility import labels, oracle
from .models.candidate import Candidate
from .models.controls import Controller
from .priors.heuristic import interventions, make_prior, rgb_prior, stress_views
from .records import ROOT, GUIDE,append, digest, jsonl, read, sha, write
from .training import ORDER, GroupSampler, drop_prior, initialize, optimizer, pair_objective, streams


class ScientificStop(RuntimeError):
    def __init__(self,status,reason):
        super().__init__(reason);self.status=status


def tensor_state_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((tuple(value.shape),str(value.dtype))).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def average_psnr(output,target):
    values,_=psnr(output,target)
    if any(not isinstance(x,float) for x in values):raise ValueError('Nonfinite PSNR requires explicit investigation.')
    return float(np.mean(values))


class Experiment:
    def __init__(self,config):
        self.config=config;self.run=Path(config['runtime']['run_dir'])
        self.doc=ROOT/'docs/experiments'/config['runtime']['run_id']
        self.doc.mkdir(parents=True,exist_ok=True)
        self.state=read(self.run/'state.json')
        self.backbone=None;self.data=None;self.lpips=None
        self.stop_requested=False
        self.budget=None
        self._samplers={}

    def live(self,**changes):
        self.state.update(changes,last_update_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
        write(self.run/'state.json',self.state)
        b=read(self.run/'budget.json')
        lines=['# Live status', '', 'Backbone: SS-UIE official public simplified implementation; not the complete paper model.',
               '', 'Scientific status: '+self.state['scientific_status'],
               'Stage/job: '+str(self.state.get('current_job','preflight')),
               'Completed: '+', '.join(self.state.get('completed',[])),
               'Step: '+str(self.state.get('global_step',0)),
               'Used device hours: %.6f / 16; reserve: 3.5.'%(b['used_device_seconds']/3600),
               'Sealed evaluation released: '+str(self.state.get('sealed_eval_released',False)),
               'Independent backup verified: false.',
               'Updated UTC: '+self.state['last_update_utc']]
        (self.doc/'LIVE_STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    @contextlib.contextmanager
    def device_job(self,name,estimate=600,final=False):
        with DeviceBudget(self.run) as budget:
            budget.start(name,max(estimate,1),final=final)
            self.budget=budget
            self.live(current_job=name,scientific_status='RUNNING')
            try:yield budget
            finally:self.budget=None

    def guard(self,save_seconds=5):
        if self.stop_requested:raise BudgetStop('Graceful signal received; save and stop.')
        if self.budget:self.budget.guard(save_seconds)

    def load(self):
        if self.backbone is not None:return
        if not torch.cuda.is_available():raise ScientificStop('BLOCKED_BACKBONE','Real PPU/GPU is unavailable; no CPU substitution.')
        torch.set_num_threads(2)
        torch.manual_seed(self.config['seed']);torch.cuda.manual_seed_all(self.config['seed'])
        torch.backends.cudnn.benchmark=False
        with self.device_job('official_backbone_load_and_device_setup',30):
            self.backbone,_=load_official(ROOT,self.config['backbone']['checkpoint'],self.config['backbone']['checkpoint_sha256'])
            self.backbone.to('cuda:0').eval()
            self.data=RuntimeData(self.config,self.backbone)
        if (self.run/'baseline_policy.json').exists():
            self.data.policy=read(self.run/'baseline_policy.json')['selected']
            self.backbone.policy=self.data.policy

    def metrics(self,output,target):
        if self.lpips is None:self.lpips=VerifiedLPIPS(ROOT/'weights/cde_v3/vgg16-397923af.pth','cuda:0')
        p,m=psnr(output,target)
        if any(not isinstance(x,float) for x in p):raise ValueError('Undefined PSNR value.')
        # CPU float64 SSIM retains the tested metric on devices with slow FP64.
        ss=ssim(output.cpu(),target.cpu()).tolist()
        lp=self.lpips(output,target).cpu().tolist()
        return [{'psnr':p[i],'mse':float(m[i]),'ssim':float(ss[i]),'lpips':float(lp[i])} for i in range(len(p))]

    def baseline(self):
        self.load()
        if self.data.policy:return
        with self.device_job('S1_baseline_model_val_policy_selection',1800):
            rows=[]
            for i,row in enumerate(self.data.role('model_val')):
                self.guard()
                target=self.data.target(row,'candidate_select')[None].to('cuda:0')
                image=self.data.input(row)
                outputs=self.data.base_pair(row,image)
                for policy,output in outputs.items():
                    metric=self.metrics(output[None].to('cuda:0'),target)[0]
                    rows.append({'sample_id':row['sample_id'],'group_id':row['group_id'],'role':'model_val','policy':policy,**metric})
                if i%50==0:self.live(completed_images=i+1)
            jsonl(self.run/'metrics/baseline_policy_model_val.jsonl',rows)
            means={policy:float(np.mean([r['psnr'] for r in rows if r['policy']==policy])) for policy in ['clip01','official_minmax_float']}
            selected='official_minmax_float' if means['official_minmax_float']>means['clip01']+1e-8 else 'clip01'
            result={'selected':selected,'mean_psnr':means,'selection_role':'model_val','image_count':len(rows)//2,
                    'checkpoint_sha256':self.config['backbone']['checkpoint_sha256'],'rows_sha256':sha(self.run/'metrics/baseline_policy_model_val.jsonl'),
                    'tie_rule':'within 1e-8 dB choose clip01','paper_complete_model':False,'metric_identity':self.lpips.identity}
            write(self.run/'baseline_policy.json',result)
            self.data.policy=selected;self.backbone.policy=selected

    def identity(self,method):
        values={'run_id':self.config['runtime']['run_id'],'method_id':method,'seed':self.config['seed'],
                'scientific_config_hash':digest(self.config),'backbone_commit':self.config['backbone']['commit'],
                'backbone_weight_sha256':self.config['backbone']['checkpoint_sha256'],'backbone_label':'SS-UIE official public simplified implementation',
                'baseline_policy_hash':sha(self.run/'baseline_policy.json'),'role_manifest_hash':sha(self.run/'roles.jsonl'),
                'exposure_ledger_hash':sha(self.run/'exposure_ledger.json'),'prior_spec_hash':digest(self.config['prior']),
                'intervention_spec_hash':digest(self.config['prior']['train_views'])}
        for name,key in [('source_snapshot.json','source_commit_or_snapshot_hash'),('normalization_stats.json','normalization_stats_hash')]:
            values[key]=sha(self.run/name) if (self.run/name).exists() else None
        values['candidate_checkpoint_sha256']=self.data.candidate_hash if method in ORDER and method!='B4' else None
        values['resolved_budget_plan_sha256']=sha(self.run/'budget_plan.json') if (self.run/'protocol_resolved.yaml').exists() else None
        if method=='B4':values['RGB_initialization_checkpoint_sha256']=read(self.run/'checkpoints/B3/selection.json')['selected_sha256']
        values['guide_v1_1_sha256']=sha(__import__('uie_next.records',fromlist=['GUIDE']).GUIDE)
        return values

    def factory(self,method,rng):
        return initialize(Candidate if method in ['B1','B3','B4'] else lambda:Controller(method),rng['init']).to('cuda:0')

    def fresh_streams(self):
        rng=streams(self.config['seed'])
        rng['model_data']=streams(self.config['seed']+101)['data']
        return rng

    def update(self,method,model,opt,schedule,rng,temporary=None,scales=None):
        opt.zero_grad(set_to_none=True)
        if method in ['B1','B3','B4']:
            if temporary is None:
                if method=='B4':
                    us=self.sampler('utility_fit',rng['data'])
                    ms=self.sampler('model_fit',rng['model_data'])
                    batch=self.data.batch(ms.sample(4)+us.sample(4),'matched_train')
                else:
                    sampler=self.sampler('model_fit',rng['data'])
                    batch=self.data.batch(sampler.sample(8),'candidate_train')
            else:batch=temporary
            if method in ['B3','B4']:P,V=rgb_prior(batch['image'])
            else:
                f=make_prior(batch['image']);P,V=f['P'],f['V']
            P,V,missing=drop_prior(P,V,rng['missing_prior'])
            out=model(batch['image'],batch['base'],P,V)
            loss=(out-batch['target']).square().mean()
            parts={'mse':loss.detach(),'missing_count':missing.sum().to(loss.device)}
        else:
            if temporary is None:
                ids=self.sampler('utility_fit',rng['data']).sample(4)
                batch=self.data.batch(ids,'utility_train',True)
                views=torch.randint(1,7,(4,),generator=rng['view']).to('cuda:0')
                indices=torch.arange(4,device='cuda:0')
                candidates=torch.stack([batch['candidates'][:,0],batch['candidates'][indices,views]],1)
            else:
                batch=temporary;candidates=batch['candidates']
            if method=='O-NI':candidates=candidates[:,:1].repeat(1,2,1,1,1)
            loss,parts=pair_objective(model,batch['image'],batch['base'],candidates,batch['target'],scales)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss: '+method)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite training gradient: '+method)
        lr=opt.param_groups[0]['lr'];opt.step();schedule.step()
        return {'loss':float(loss.detach()),'gradient_norm':float(norm),'learning_rate':lr,
                **{k:float(v) for k,v in parts.items() if not k.endswith('_live')}}

    def sampler(self,role,generator):
        key=(role,id(generator))
        if key not in self._samplers:self._samplers[key]=GroupSampler(self.data.role(role),generator)
        return self._samplers[key]

    def validate(self,method,model):
        # B4 is a candidate head initialized from B3 but trained with the
        # matched model_fit+utility_fit stream.  Its checkpoint must still be
        # selected on model_val, as required by the protocol; it must not leak
        # utility_val into candidate checkpoint selection.
        operation='candidate_select' if method in ['B1','B3','B4'] else 'utility_select'
        role='model_val' if method in ['B1','B3','B4'] else 'utility_val'
        values=[];model.eval()
        with torch.no_grad():
            rows=self.data.role(role)
            for start in range(0,len(rows),8):
                self.guard()
                batch=self.data.batch([r['sample_id'] for r in rows[start:start+8]],operation,method not in ['B1','B3','B4'])
                if method in ['B1','B3','B4']:
                    out=self.data.real_candidate(batch['image'],batch['base'],model,method in ['B3','B4'])
                else:out=model(batch['image'],batch['base'],batch['candidates'][:,0],se2=read(self.run/'normalization_stats.json')['s_e2'])['output']
                p,_=psnr(out,batch['target']);values.extend(p)
        model.train()
        return float(np.mean(values))

    def selected(self,method):
        directory=self.run/'checkpoints'/method
        record=read(directory/'selection.json');path=directory/record['selected_file']
        if sha(path)!=record['selected_sha256']:raise ValueError('Selected checkpoint changed.')
        rng=self.fresh_streams();model=self.factory(method,rng)
        state=torch.load(path,map_location='cpu');model.load_state_dict(state['model_state'],strict=True)
        model.requires_grad_(False).eval()
        return model,record

    def set_candidate(self):
        self.data.candidate,record=self.selected('B1')
        self.data.candidate_hash=record['selected_sha256']

    def train(self,method,total):
        directory=self.run/'checkpoints'/method;directory.mkdir(parents=True,exist_ok=True)
        if (directory/'selection.json').exists():self.selected(method);return
        self._samplers.clear()
        rng=self.fresh_streams();model=self.factory(method,rng)
        if method=='B4':
            init,_=self.selected('B3');model.load_state_dict(init.state_dict(),strict=True);del init
        opt,schedule=optimizer(model,total)
        identity=self.identity(method);history=[];step=0
        latest=directory/'latest.pt';start_state=tensor_state_hash(model)
        if latest.exists():
            restored=resume(latest,model,opt,schedule,identity,rng);step=restored['global_step'];history=restored['extra']['history']
            start_state=restored['extra']['initial_state_hash']
        else:write(directory/'initialization.json',{'initial_state_hash':start_state,'seed':self.config['seed'],'parameters':sum(p.numel() for p in model.parameters()),'identity':identity})
        points=sorted(set(int(total*f) for f in [0,.25,.5,.75,1]))
        estimate=read(self.run/'budget_plan.json')['jobs'][method]['predicted_total_seconds']
        scales=read(self.run/'normalization_stats.json') if method not in ['B1','B3','B4'] else None
        used_before=read(self.run/'budget.json')['used_device_seconds'];begin=time.monotonic()
        with self.device_job('train_'+method,max(estimate*(total-step)/total,60)):
            try:
                for j in range(step,total+1):
                    if j in points and not any(h['step']==j for h in history):
                        metric=self.validate(method,model)
                        entry={'step':j,'mean_psnr':metric,'selection_role':'model_val' if method in ['B1','B3','B4'] else 'utility_val'}
                        history.append(entry)
                        extra={'history':history,'initial_state_hash':start_state,'completed_source_samples':j*(8 if method in ['B1','B3','B4'] else 4),
                               'elapsed_device_seconds':read(self.run/'budget.json')['used_device_seconds']-used_before}
                        save(directory/('step_%06d.pt'%j),model,opt,schedule,j,identity,rng,True,extra)
                        save(latest,model,opt,schedule,j,identity,rng,True,extra)
                        self.live(global_step=j,checkpoint_validation=entry)
                    if j==total:break
                    self.guard(10)
                    row=self.update(method,model,opt,schedule,rng,scales=scales)
                    step=j+1
                    if step%50==0:
                        self.budget.tick();row.update(step=step,method=method,wall_seconds=time.monotonic()-begin,
                                                     source_samples=step*(8 if method in ['B1','B3','B4'] else 4))
                        completed_updates=sum(read(p)['updates'] for p in (self.run/'checkpoints').glob('*/selection.json'))
                        append(self.run/'logs'/('train_'+method+'.jsonl'),row);self.live(global_step=step,training_last=row,formal_training_updates=completed_updates+step)
                        print(json.dumps(row),flush=True)
                    if step%250==0:
                        save(latest,model,opt,schedule,step,identity,rng,True,{'history':history,'initial_state_hash':start_state,'completed_source_samples':step*(8 if method in ['B1','B3','B4'] else 4),
                                                                           'elapsed_device_seconds':read(self.run/'budget.json')['used_device_seconds']-used_before})
            except BaseException:
                save(latest,model,opt,schedule,step,identity,rng,True,{'history':history,'initial_state_hash':start_state,'completed_source_samples':step*(8 if method in ['B1','B3','B4'] else 4),
                                                                    'elapsed_device_seconds':read(self.run/'budget.json')['used_device_seconds']-used_before})
                raise
        best=history[0]
        for entry in history[1:]:
            if entry['mean_psnr']>best['mean_psnr']+1e-8:best=entry
        path=directory/('step_%06d.pt'%best['step'])
        record={'method':method,'updates':total,'selected_step':best['step'],'selected_file':path.name,
                'selected_sha256':sha(path),'initial_state_hash':start_state,'history':history,
                'convergence_unresolved':best['step']==total and history[-1]['mean_psnr']-history[-2]['mean_psnr']>.05,
                'device_seconds':read(self.run/'budget.json')['used_device_seconds']-used_before,'identity':identity,
                'completed_source_samples':total*(8 if method in ['B1','B3','B4'] else 4)}
        write(directory/'selection.json',record)
        self.backup_weights(method)

    def backup_weights(self,stage):
        from .reporting import archive
        paths=[p for p in (self.run/'checkpoints').rglob('*') if p.is_file() and (p.name.startswith('latest.') or p.name in ['selection.json','initialization.json'])]
        for selection in (self.run/'checkpoints').glob('*/selection.json'):
            record=read(selection)
            paths.extend([selection.parent/record['selected_file'],selection.parent/'step_000000.pt',
                          selection.parent/('step_%06d.pt'%record['updates'])])
        out=ROOT.parent/('ssuie_local_utility_v1_20261007_%s_weights_recovery.zip'%stage)
        receipt=archive(out,sorted(set(paths)),ROOT)
        write(self.run/'backups'/(stage+'.json'),{**receipt,'independent_backup_verified':False,'same_server_disk_only':True})

    def diagnostics(self):
        self.set_candidate();rgb,_=self.selected('B3');rows=[];sensitivity={n:[] for n in self.config['prior']['train_views'][1:]}
        with self.device_job('S5_candidate_diagnostics',1200):
            with torch.no_grad():
                for row in self.data.role('utility_val'):
                    self.guard();batch=self.data.batch([row['sample_id']],'candidate_diagnostic',True)
                    image,base,target=batch['image'],batch['base'],batch['target'];candidate=batch['candidates'][:,0]
                    variants={'B0':base,'B1':candidate,'B3':self.data.real_candidate(image,base,rgb,True)}
                    for alpha in [0,.25,.5,.75,1]:variants['fixed_'+str(alpha)]=base+alpha*(candidate-base)
                    variants['oracle_pixel']=oracle(base,candidate,target)[0]
                    variants['oracle_block32']=oracle(base,candidate,target,32)[0]
                    variants['oracle_hard']=candidate if average_psnr(candidate,target)>average_psnr(base,target) else base
                    for method,output in variants.items():
                        metric=self.metrics(output,target)[0]
                        rows.append({'sample_id':row['sample_id'],'group_id':row['group_id'],'role':'utility_val','method':method,
                                     'uses_reference':method.startswith('oracle'),'candidate_sha256':self.data.candidate_hash,**metric})
                    nominal=labels(base,candidate,target)
                    rows[-1].update(nominal_U_positive_fraction=float((nominal['U']>0).float().mean()),
                                    nominal_U_negative_fraction=float((nominal['U']<0).float().mean()),
                                    nominal_active_fraction=float(nominal['active'].float().mean()))
                    for vi,name in enumerate(self.config['prior']['train_views'][1:],1):
                        sensitivity[name].append(float((batch['candidates'][:,vi]-candidate).abs().mean()))
            jsonl(self.run/'metrics/candidate_diagnostics.jsonl',rows)
            means={m:float(np.mean([r['psnr'] for r in rows if r['method']==m])) for m in sorted(set(r['method'] for r in rows))}
            fixed=max(means[m] for m in means if m.startswith('fixed_'))
            result={'means_psnr':means,'oracle_vs_base_db':means['oracle_block32']-means['B0'],
                    'oracle_vs_best_fixed_db':means['oracle_block32']-fixed,'diagnostic_best_fixed_only_not_deployment':True,
                    'prior_sensitivity':{n:{'mean_mae':float(np.mean(v)),'quantiles':np.quantile(v,[0,.1,.5,.9,1]).tolist()} for n,v in sensitivity.items()},
                    'candidate_sha256':self.data.candidate_hash,'role':'utility_val','sealed_eval_read':False}
            result['sensitivity_pass']=any(v['mean_mae']>=1e-4 for v in result['prior_sensitivity'].values())
            result['headroom_pass']=result['oracle_vs_base_db']>=.15 and result['oracle_vs_best_fixed_db']>=.05
            result['passed']=result['sensitivity_pass'] and result['headroom_pass']
            result['nominal_label_distribution']={k:float(np.mean([r[k] for r in rows if k in r])) for k in ['nominal_U_positive_fraction','nominal_U_negative_fraction','nominal_active_fraction']}
            write(self.run/'candidate_diagnostics.json',result)
            from .visuals import export_candidates
            export_candidates(self,rows,rgb)
            if not result['sensitivity_pass']:raise ScientificStop('STOP_PRIOR_UNUSED','No registered intervention produces mean candidate MAE >=1e-4; CPU field/path tests passed.')
            if not result['headroom_pass']:raise ScientificStop('STOP_CANDIDATE_NO_HEADROOM','Registered coarse oracle headroom thresholds not met.')

    def utility_cache(self):
        self.set_candidate();stats={'v2':[],'u2':[],'e2':[],'active':[],'positive':[],'negative':[]}
        with self.device_job('S6_seven_view_utility_cache_and_fit_only_scales',1800):
            for role,op in [('utility_fit','label_scales'),('utility_val','utility_select')]:
                for i,row in enumerate(self.data.role(role)):
                    self.guard();batch=self.data.batch([row['sample_id']],op,True)
                    if role=='utility_fit':
                        for vi in range(7):
                            lab=labels(batch['base'],batch['candidates'][:,vi],batch['target']);active=lab['active']
                            if active.any():stats['v2'].append(float(lab['v'][active].square().mean()))
                            stats['u2'].append(float(lab['U'].square().mean()));stats['active'].append(float(active.float().mean()))
                            stats['positive'].append(float((lab['U']>0).float().mean()));stats['negative'].append(float((lab['U']<0).float().mean()))
                        stats['e2'].append(float((batch['base']-batch['target']).square().mean()))
                    if i%50==0:self.live(completed_images=i+1,cache_role=role)
            if not stats['v2']:raise ScientificStop('STOP_CANDIDATE_NO_HEADROOM','No active utility labels.')
            values={'s_v':max(math.sqrt(np.mean(stats['v2'])),1e-3),'s_U':max(math.sqrt(np.mean(stats['u2'])),1e-4),
                    's_e2':max(float(np.mean(stats['e2'])),1e-4),'source_count':len(stats['e2']),
                    'active_image_view_count':len(stats['v2']),'active_fraction':float(np.mean(stats['active'])),
                    'U_positive_fraction':float(np.mean(stats['positive'])),'U_negative_fraction':float(np.mean(stats['negative'])),
                    'role':'utility_fit','candidate_sha256':self.data.candidate_hash,'normalization_is_image_view_equal':True}
            write(self.run/'normalization_stats.json',values)
            write(self.run/'normalization_fit_distributions.json',{'role':'utility_fit','candidate_sha256':self.data.candidate_hash,'raw_image_view_statistics':stats})

    def freeze_source(self):
        paths=list((ROOT/'uie_next').rglob('*.py'))+list((ROOT/'tests/uie_next').rglob('*.py'))+[ROOT/'configs/uie_next/protocol.yaml']
        record={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
        path=self.run/'source_snapshot.json'
        if path.exists() and read(path)!=record:raise ValueError('Frozen source changed; explicit engineering amendment required.')
        write(path,record)
        if not (self.run/'guide_v1_1_admission.json').exists():
            write(self.run/'guide_v1_1_admission.json',{'path':str(GUIDE),'sha256':sha(GUIDE),'version':'1.1',
                  'scientific_network_recipe_changed':False,'budget_reset':False,'original_guide_preserved':True})
            append(self.run/'protocol_amendments.jsonl',{'kind':'uploaded_guide_v1_1_and_pretraining_dispatch_completion',
                  'guide_path':str(GUIDE),'guide_sha256':sha(GUIDE),'previous_guide_preserved':'protocol_source.md and audit_history',
                  'updates_before_admission':0,'networks_and_thresholds_changed':False,'budget_reset':False,
                  'engineering_fixes':['B4 selects model_val','resumable S2-S11 dispatch','raw loss logging','complete deploy timing and evidence exports']})
        from .reporting import archive
        receipt=archive(ROOT.parent/'ssuie_local_utility_v1_20261007_official_resume_code_protocol.zip',paths+[self.run/'roles.jsonl',self.run/'exposure_ledger.json'],ROOT)
        write(self.run/'backups/source_protocol.json',receipt)
        backbone_package=ROOT.parent/'ssuie_local_utility_v1_20261007_official_backbone.zip'
        if not backbone_package.exists():
            record=archive(backbone_package,[__import__('pathlib').Path(self.config['backbone']['checkpoint']),self.run/'backbone_provenance.json'],ROOT)
            write(self.run/'backups/official_backbone.json',{**record,'independent_backup_verified':False,'same_server_disk_only':True})

    def complete(self,name):
        if name not in self.state['completed']:self.state['completed'].append(name)
        append(self.run/'dispatch_events.jsonl',{'stage':name,'completed_at':datetime.now(timezone.utc).isoformat()})
        self.live()

    def execute(self):
        lock=(self.run/'.pipeline.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();raise RuntimeError('An active dispatcher already owns this run.')
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda signum,frame:setattr(self,'stop_requested',True))
        append(self.run/'commands.jsonl',{'argv':sys.argv,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()})
        try:
            if self.state['scientific_status'] in ['STOP_PRIOR_UNUSED','STOP_CANDIDATE_NO_HEADROOM','STOP_MECHANISM_NOT_SUPPORTED',
                                                 'QUALITY_ONLY_NO_MECHANISM_EVIDENCE','CONFIRMATION_FAIL','CONFIRMATION_PASS_SINGLE_SEED','INCONCLUSIVE_CONFIRMATION','INCONCLUSIVE_BUDGET']:
                from .delivery import deliver
                return deliver(self)
            audit=run_audit()
            if audit['minimum_deficits']:raise ScientificStop('BLOCKED_DATA',json.dumps(audit['minimum_deficits']))
            self.load();self.baseline();self.complete('S1_DATA_AND_BACKBONE_READY')
            if 'S2_IMPLEMENTED_AND_TESTED' not in self.state['completed']:
                from .integration import accept_real,supplemental_real
                accept_real(self);supplemental_real(self);self.freeze_source();self.complete('S2_IMPLEMENTED_AND_TESTED')
            else:self.freeze_source()
            if 'S3_BUDGET_FROZEN' not in self.state['completed']:
                from .profiling import profile_and_freeze
                profile_and_freeze(self);self.complete('S3_BUDGET_FROZEN')
            resolved=yaml.safe_load((self.run/'protocol_resolved.yaml').read_text())
            for method in ['B1','B3']:self.train(method,resolved['training']['candidate_updates'])
            self.complete('S4_CANDIDATES_TRAINED')
            if 'S5_CANDIDATE_GATE_PASSED' not in self.state['completed']:self.diagnostics();self.complete('S5_CANDIDATE_GATE_PASSED')
            self.set_candidate()
            if 'S6_UTILITY_CACHE_READY' not in self.state['completed']:self.utility_cache();self.complete('S6_UTILITY_CACHE_READY')
            for method in ORDER:self.train(method,resolved['training']['utility_updates'])
            self.complete('S7_METHODS_TRAINED')
            from .scientific_evaluation import calibrate, develop, final_evaluate
            if 'S8_CALIBRATION_FROZEN' not in self.state['completed']:calibrate(self);self.complete('S8_CALIBRATION_FROZEN')
            if 'S9_DEV_GATE_PASSED' not in self.state['completed']:develop(self);self.complete('S9_DEV_GATE_PASSED')
            if 'S10_SEALED_EVAL_COMPLETE' not in self.state['completed']:final_evaluate(self);self.complete('S10_SEALED_EVAL_COMPLETE')
        except ScientificStop as exc:self.live(scientific_status=exc.status,stop_reason=str(exc))
        except BudgetStop as exc:self.live(scientific_status='INCONCLUSIVE_BUDGET',stop_reason=str(exc))
        except BaseException as exc:
            self.live(scientific_status='FAILED_IMPLEMENTATION',stop_reason=type(exc).__name__+': '+str(exc))
            append(self.run/'engineering_failures.jsonl',{'error':self.state['stop_reason'],'current_job':self.state.get('current_job'),'utc':datetime.now(timezone.utc).isoformat()})
            import traceback
            (self.run/'logs/last_engineering_traceback.log').write_text(traceback.format_exc())
        finally:lock.close()
        from .delivery import deliver
        return deliver(self)
