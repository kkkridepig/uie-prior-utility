"""Full finite-pool diagnostics; exact oracles use actual float32 outputs in CPU float64."""
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import convolve1d
from ..records import ROOT,sha,read,write,jsonl,digest
from ..models.candidate import Candidate
from ..data.cache import get
from ..backbones.ssuie import load_official
from ..evaluation import VerifiedLPIPS
from ..priors.heuristic import interventions
from ..math.utility import oracle,labels
from .context import OLD,PROTOCOL_SHA,V2Data
from .selection import choose_producer,standalone,rescue_choice
from .statistics import paired_stats

HASHES={
'B1':[('001000','c78961b97dfd7302c6fb1e4f01f4f20f6edfe19786fee5586245434cc3dcb1ea'),('002000','f0074ee1def232f1790a309f1e899b109d7066bfa2c336d1e2d303f9642c08d7'),('003000','c1af746056f0419eeb5787283a5c5e28d15d1ceec95d1dd6cedd4ed7592879af'),('004000','29c77e1a7def6c7ba1a596a919b2e945f127b9349f47b07e8658c51e1193e47e')],
'B3':[('001000','bf973a9a97636c2b5d97e5c61c1633d46b82772eab8a88bad546725e07f7a28f'),('002000','17972e27de546dc6222fcf9cd1903c3b2a653f208cc4d3ecea3d6fbd634ffd37'),('003000','75178a852c8f7078a50a7c8c592427c7ccea608f210b41dcd1a1084d293c4901'),('004000','52183ff21e71a3e301b8cb29264a03c54ddb3c094bab76d15ac3843526eecea1')]}


def inventory():
    out=[]
    for head,values in HASHES.items():
        for num,expected in [('000000',None)]+values:
            path=OLD/'checkpoints'/head/('step_'+num+'.pt');actual=sha(path);side=read(str(path)+'.json')
            if actual!=side['sha256'] or (expected is not None and actual!=expected):raise ValueError('checkpoint identity mismatch '+str(path))
            out.append({'checkpoint_id':head+'_'+num,'head':head,'training_step':int(num),'checkpoint_sha256':actual,'path':str(path),
                       'sidecar_sha256':sha(str(path)+'.json'),'configuration_id':'V1_MSE','official_commit':side['identity']['backbone_commit']})
    return out


def checkpoint_model(item,device='cuda:0'):
    if sha(item['path'])!=item['checkpoint_sha256']:raise ValueError('weight identity changed')
    state=torch.load(item['path'],map_location='cpu')
    if state['global_step']!=item['training_step']:raise ValueError('wrong checkpoint step')
    model=Candidate();model.load_state_dict(state['model_state'],strict=True)
    return model.to(device).eval().requires_grad_(False)


def exact_strategies(base,candidate,target):
    # Conversion occurs before arithmetic. The residual is AFTER candidate clipping.
    b=base.detach().cpu().double();j=candidate.detach().cpu().double();y=target.detach().cpu().double();r=j-b
    zero=torch.zeros_like(b[:,:1]);one=torch.ones_like(zero)
    out={'B0':(b,zero),'endpoint':(j,one)}
    for a in [0,.25,.5,.75,1]:out['fixed_'+str(a)]=(b+a*r,zero+a)
    out['oracle_pixel']=oracle(b,j,y,1,eps=0)
    out['oracle_pixel_active']=oracle(b,j,y,1,eps=1e-6)
    out['oracle_block32']=oracle(b,j,y,32,eps=0)
    out['oracle_image_continuous']=oracle(b,j,y,256,eps=0)
    l0=(b-y).square().flatten(1).mean(1);l1=(j-y).square().flatten(1).mean(1)
    hard=(l1<l0).view(-1,1,1,1);out['oracle_hard']=(torch.where(hard,j,b),torch.where(hard,one,zero))
    mse={k:(v[0]-y).square().flatten(1).mean(1) for k,v in out.items()}
    errors=[]
    for left,right in [('oracle_pixel','oracle_block32'),('oracle_block32','oracle_image_continuous'),('oracle_image_continuous','oracle_hard')]+[('oracle_image_continuous','fixed_'+str(a)) for a in [0,.25,.5,.75,1]]:
        errors.append(float((mse[left]-mse[right]).max()))
    if max(errors)>1e-10:raise ValueError('oracle ordering failed')
    return out,max(errors)


def numpy_ssim(x,y):
    """Same population/valid Gaussian RGB SSIM; separable float64 convolution."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    coord=np.arange(-5,6,dtype=np.float64);k=np.exp(-coord**2/(2*1.5**2));k/=k.sum()
    def filt(v):return convolve1d(convolve1d(v,k,axis=-1,mode='constant'),k,axis=-2,mode='constant')[...,5:-5,5:-5]
    ux=filt(x);uy=filt(y);vx=filt(x*x)-ux*ux;vy=filt(y*y)-uy*uy;cov=filt(x*y)-ux*uy
    return (((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))).reshape(len(x),-1).mean(1)


def metrics(output,target):
    x=output.detach().cpu().double().numpy();y=target.detach().cpu().double().numpy()
    mse=((x-y)**2).reshape(len(x),-1).mean(1);ss=numpy_ssim(x,y)
    return [{'mse':float(m),'log_mse':float(np.log(m)) if m>0 else '-inf','log_mse_eps':float(np.log(m+1e-6)),
             'psnr':float(-10*np.log10(m)) if m>0 else '+inf','ssim':float(s),
             'metric_status':'finite' if m>0 else 'exact_zero_mse'} for m,s in zip(mse,ss)]


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r));tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows);f.flush();__import__('os').fsync(f.fileno())
    tmp.replace(path)


class Diagnostics:
    def __init__(self,state):
        self.s=state;self.run=state.run;self.config=state.config;self.backbone=None;self.data=None;self.lpips=None
    def load(self):
        if self.backbone is not None:return
        if not torch.cuda.is_available():raise RuntimeError('PPU unavailable; no CPU substitution')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        with self.s.device_job('official_load',60):
            self.backbone,self.backbone_identity=load_official(ROOT,self.config['backbone']['checkpoint'],self.config['backbone']['checkpoint_sha256'])
            self.backbone.to('cuda:0');self.data=V2Data(self.config,self.backbone,self.s.ctx)
            self.lpips=VerifiedLPIPS(ROOT/'weights/cde_v3/vgg16-397923af.pth','cuda:0')
            write(self.run/'backbone_verification.json',{'strict_load':True,'identity':self.backbone_identity,'metric_identity':self.lpips.identity})
    def verify_reuse(self):
        path=self.run/'tests/base_reuse.json'
        if path.exists():
            rec=read(path)
            if rec['backbone_sha256']!=self.config['backbone']['checkpoint_sha256']:raise ValueError('base reuse identity mismatch')
            self.data.reuse_verified=True;return
        rows=self.data.role('model_fit')[:8]+self.data.role('utility_fit')[:8];diff=[]
        with self.s.device_job('verify_16_base_cache',300):
            for row in rows:
                self.data.guard.check(row['sample_id'],'candidate_train' if row['role']=='model_fit' else 'utility_train')
                image=self.data.input(row)
                with torch.no_grad():out=self.backbone(image[None].to('cuda:0'))[0].cpu()
                p=OLD/'cache/base'/('%s.pt'%digest(row['sample_id']))
                cached=get(p,self.data.identity(row))['clip01'];v=float((out-cached).abs().max());diff.append(v)
                if v>1e-5:raise ValueError('base cache differs from real inference')
        write(path,{'passed':True,'sample_ids':[r['sample_id'] for r in rows],'max_abs':max(diff),
                    'backbone_sha256':self.config['backbone']['checkpoint_sha256'],'source_cache_read_only':True})
        self.data.reuse_verified=True
    def probe(self):
        path=self.run/'diagnostics/probe_manifest.json'
        rows=self.data.role('model_fit');groups=sorted({r['group_id'] for r in rows},key=lambda g:__import__('hashlib').sha256(('20261008|'+g).encode()).hexdigest())[:128]
        ids=sorted(r['sample_id'] for r in rows if r['group_id'] in groups)
        rec={'groups':groups,'sample_ids':ids,'n_groups':128,'n_images':len(ids),'role':'model_fit_probe','identity':digest(ids),'selection_uses_quality':False}
        if path.exists() and read(path)!=rec:raise ValueError('probe identity changed')
        write(path,rec);return [self.data.by_id[i] for i in ids]
    def scan(self,item,role,rows,operation,sensitivity=False):
        directory=self.run/'diagnostics/parts'/item['checkpoint_id'];directory.mkdir(parents=True,exist_ok=True)
        path=directory/(role+'.json');identity={'checkpoint_sha256':item['checkpoint_sha256'],'role_manifest':sha(self.run/'roles.jsonl'),
             'protocol':PROTOCOL_SHA,'source':self.s.state.get('D0_initial_snapshot_sha256',sha(self.run/'source_snapshot.json')),'ids':digest([r['sample_id'] for r in rows]),'metric':sha(ROOT/'uie_next/v2/diagnostics.py')}
        prior=read(path) if path.exists() else {'identity':identity,'metrics':[],'residuals':[],'sensitivity':[],'done':[],'complete':False}
        if prior['identity']!=identity:raise ValueError('diagnostic pairing identity mismatch')
        if prior['complete']:
            if set(prior['done'])!={r['sample_id'] for r in rows}:raise ValueError('diagnostic count mismatch')
            return prior
        model=checkpoint_model(item);todo=[r for r in rows if r['sample_id'] not in prior['done']]
        start=time.monotonic()
        with self.s.device_job('D0_'+item['checkpoint_id']+'_'+role,max(120,len(todo)*.3)):
            for n in range(0,len(todo),8):
                self.s.guard();part=todo[n:n+8];batch=self.data.batch([r['sample_id'] for r in part],operation)
                with torch.no_grad():
                    candidate=self.data.real_candidate(batch['image'],batch['base'],model,item['head']=='B3')
                    variants,maxerr=exact_strategies(batch['base'],candidate,batch['target'])
                    endpoint_lp=self.lpips(torch.cat([batch['base'],candidate]),batch['target'].repeat(2,1,1,1)).cpu().tolist()
                    lab=labels(batch['base'].cpu().double(),candidate.cpu().double(),batch['target'].cpu().double())
                    baseline=metrics(batch['base'],batch['target'])
                    for kind,(output,alpha) in variants.items():
                        met=baseline if kind=='B0' else metrics(output,batch['target'])
                        for i,row in enumerate(part):
                            m=met[i];b=baseline[i]
                            if not isinstance(m['psnr'],float) or not isinstance(b['psnr'],float):raise ValueError('BLOCKED_NONFINITE_METRIC')
                            lp=endpoint_lp[i] if kind=='B0' else endpoint_lp[len(part)+i] if kind=='endpoint' else None
                            prior['metrics'].append({'run_id':self.s.ctx.run_id,'protocol_sha256':PROTOCOL_SHA,
                              'checkpoint_id':item['checkpoint_id'],'checkpoint_sha256':item['checkpoint_sha256'],'training_step':item['training_step'],
                              'head':item['head'],'role':role,'sample_id':row['sample_id'],'group_id':row['group_id'],
                              'input_sha256':row['input_sha256'],'reference_sha256':row['reference_sha256'],
                              'prediction_kind':kind,'alpha_fixed':float(kind[6:]) if kind.startswith('fixed_') else None,
                              'uses_reference':kind.startswith('oracle'),'n_pixels':65536,**m,'lpips':lp,
                              'lpips_status':'computed' if lp is not None else 'not_scheduled_in_D0','baseline_mse':b['mse'],
                              'baseline_psnr':b['psnr'],'delta_mse':m['mse']-b['mse'],'delta_psnr_db':m['psnr']-b['psnr'],
                              'delta_ssim':m['ssim']-b['ssim'],'finite_pass':True,'alpha_mean':float(alpha[i].mean()),
                              'alpha_zero_fraction':float((alpha[i]==0).double().mean()),'alpha_one_fraction':float((alpha[i]==1).double().mean()),
                              'boundary_fraction':float(((output[i]==0)|(output[i]==1)).double().mean()),'oracle_order_max_error':maxerr})
                    for i,row in enumerate(part):
                        record={'checkpoint_id':item['checkpoint_id'],'checkpoint_sha256':item['checkpoint_sha256'],'training_step':item['training_step'],
                            'role':role,'sample_id':row['sample_id'],'group_id':row['group_id'],'residual_mae':float(lab['r'][i].abs().mean()),
                            'residual_rms':float(lab['r'][i].square().mean().sqrt()),'active_fraction':float(lab['active'][i].double().mean())}
                        for k in ['a','b','U']:
                            ar=lab[k][i].numpy();record[k+'_mean']=float(ar.mean());record[k+'_positive_fraction']=float((ar>0).mean())
                            for q,v in zip(['05','10','50','90','95'],np.quantile(ar,[.05,.1,.5,.9,.95])):record[k+'_pixel_q'+q]=float(v)
                        prior['residuals'].append(record)
                    if sensitivity and item['training_step']>0:
                        fields=interventions(batch['image'])
                        for name in self.config['prior']['train_views'][1:]:
                            f=fields[name];out=model(batch['image'],batch['base'],f['P'],f['V'])
                            mae=(out-candidate).abs().flatten(1).mean(1).cpu().tolist()
                            for i,row in enumerate(part):prior['sensitivity'].append({'checkpoint_id':item['checkpoint_id'],'checkpoint_sha256':item['checkpoint_sha256'],
                               'role':role,'sample_id':row['sample_id'],'group_id':row['group_id'],'intervention_id':name,
                               'view_spec_hash':digest({'name':name,'prior':self.config['prior'],'implementation':sha(ROOT/'uie_next/priors/heuristic.py')}),'mae_vs_nominal':mae[i]})
                prior['done'].extend(r['sample_id'] for r in part);write(path,prior)
                if n%80==0:self.s.live(current_checkpoint=item['checkpoint_id'],diagnostic_role=role,completed_images=len(prior['done']))
        prior.update(complete=True,wall_seconds=time.monotonic()-start);write(path,prior)
        del model;return prior
    def summary(self,record,item,role):
        rows=record['metrics'];means={kind:{k:float(np.mean([r[k] for r in rows if r['prediction_kind']==kind])) for k in ['psnr','mse','ssim']} for kind in sorted({r['prediction_kind'] for r in rows})}
        fixed=max(means['fixed_'+str(a)]['psnr'] for a in [0,.25,.5,.75,1])
        names={r['intervention_id'] for r in record['sensitivity']}
        S=max([float(np.mean([r['mae_vs_nominal'] for r in record['sensitivity'] if r['intervention_id']==name])) for name in names] or [0])
        return {'checkpoint_id':item['checkpoint_id'],'checkpoint_sha256':item['checkpoint_sha256'],'head':item['head'],'configuration_id':item['configuration_id'],
             'training_step':item['training_step'],'role':role,'baseline_psnr':means['B0']['psnr'],'baseline_mse':means['B0']['mse'],
             'endpoint_psnr':means['endpoint']['psnr'],'endpoint_mse':means['endpoint']['mse'],'best_fixed_psnr':fixed,
             'oracle_block32_psnr':means['oracle_block32']['psnr'],'oracle_pixel_psnr':means['oracle_pixel']['psnr'],
             'H32':means['oracle_block32']['psnr']-means['B0']['psnr'],'G32':means['oracle_block32']['psnr']-fixed,
             'Hp':means['oracle_pixel']['psnr']-means['B0']['psnr'],'Gp':means['oracle_pixel']['psnr']-fixed,'S':S,
             'finite_pass':True,'n_images':len(record['done']),'n_groups':len({r['group_id'] for r in rows}),'aggregation':'image_weighted',
             'mean_residual_mae':float(np.mean([r['residual_mae'] for r in record['residuals']]))}
    def export(self):
        paths=sorted((self.run/'diagnostics/parts').glob('*/*.json'));parts=[read(p) for p in paths if read(p).get('complete')]
        allmetrics=[r for p in parts for r in p['metrics']];res=[r for p in parts for r in p['residuals']];sens=[r for p in parts for r in p['sensitivity']]
        for name,rows in [('checkpoint_image_metrics',allmetrics),('checkpoint_residual_diagnostics',res),('checkpoint_prior_sensitivity',sens)]:
            csv_write(self.run/'diagnostics'/(name+'.csv'),rows);jsonl(self.run/'diagnostics'/(name+'.jsonl'),rows)
        summaries=[];intervals={};comparisons=[]
        for p in parts:
            first=p['metrics'][0];cid=first['checkpoint_id'];role=first['role'];item=next(x for x in read(self.run/'diagnostics/checkpoint_inventory.json') if x['checkpoint_id']==cid)
            summ=self.summary(p,item,role);summaries.append(summ)
            for kind in sorted({r['prediction_kind'] for r in p['metrics']}):
                rr=[r for r in p['metrics'] if r['prediction_kind']==kind]
                stat=paired_stats([r['delta_psnr_db'] for r in rr],[r['group_id'] for r in rr]);intervals[cid+'|'+role+'|'+kind]=stat
                comparisons.append({'checkpoint_id':cid,'role':role,'comparison':kind+'-B0','ci95_lower':stat['ci95'][0],'ci95_upper':stat['ci95'][1],**{k:v for k,v in stat.items() if not isinstance(v,list)}})
        # Same role and checkpoint-step paired comparisons. Never pair different images.
        for role in sorted({r['role'] for r in allmetrics}):
            for step in sorted({r['training_step'] for r in allmetrics}):
                for kind in ['endpoint','oracle_block32','oracle_pixel']:
                    a={r['sample_id']:r for r in allmetrics if r['role']==role and r['training_step']==step and r['head']=='B1' and r['prediction_kind']==kind}
                    b={r['sample_id']:r for r in allmetrics if r['role']==role and r['training_step']==step and r['head']=='B3' and r['prediction_kind']==kind}
                    if a and set(a)==set(b):
                        ids=sorted(a);key=role+'|'+str(step)+'|'+kind+'|B1-B3';intervals[key]=paired_stats([a[i]['psnr']-b[i]['psnr'] for i in ids],[a[i]['group_id'] for i in ids])
                        st=intervals[key];comparisons.append({'checkpoint_id':str(step),'role':role,'comparison':kind+' B1-B3','ci95_lower':st['ci95'][0],'ci95_upper':st['ci95'][1],**{k:v for k,v in st.items() if not isinstance(v,list)}})
            for head in ['B1','B3']:
                for other in [2000,3000]:
                    a={r['sample_id']:r for r in allmetrics if r['role']==role and r['training_step']==4000 and r['head']==head and r['prediction_kind']=='endpoint'}
                    b={r['sample_id']:r for r in allmetrics if r['role']==role and r['training_step']==other and r['head']==head and r['prediction_kind']=='endpoint'}
                    if a and set(a)==set(b):
                        ids=sorted(a);key=role+'|'+head+'|4000-'+str(other);intervals[key]=paired_stats([a[i]['psnr']-b[i]['psnr'] for i in ids],[a[i]['group_id'] for i in ids])
                        st=intervals[key];comparisons.append({'checkpoint_id':head,'role':role,'comparison':'4000-'+str(other),'ci95_lower':st['ci95'][0],'ci95_upper':st['ci95'][1],**{k:v for k,v in st.items() if not isinstance(v,list)}})
        write(self.run/'diagnostics/checkpoint_summary.json',summaries);csv_write(self.run/'diagnostics/checkpoint_summary.csv',summaries)
        csv_write(self.run/'diagnostics/checkpoint_comparisons.csv',comparisons);write(self.run/'diagnostics/paired_intervals.json',intervals)
        groups={}
        for r in allmetrics:groups.setdefault((r['checkpoint_id'],r['role'],r['prediction_kind'],r['group_id']),[]).append(r)
        csv_write(self.run/'diagnostics/per_content_group.csv',[{'checkpoint_id':key[0],'role':key[1],'strategy':key[2],'group_id':key[3],
             'n_images':len(part),'mean_psnr':float(np.mean([r['psnr'] for r in part])),'mean_delta_db':float(np.mean([r['delta_psnr_db'] for r in part])),
             'group_quality':'content_group_proxy','seed':20261007} for key,part in groups.items()])
        return summaries
