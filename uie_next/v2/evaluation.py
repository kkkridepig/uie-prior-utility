"""Registry-driven V2 calibration, deployment and guarded once-only evaluation."""
import math
import time
import numpy as np
import torch
from ..records import ROOT,read,write,sha,digest,jsonl
from ..models.candidate import Candidate
from ..models.controls import Controller
from ..math.utility import decision,labels
from ..priors.heuristic import make_prior,recompute,shift_field,rgb_prior
from ..scientific_evaluation import prediction_utility_statistics
from .context import METHOD_ORDER
from .diagnostics import checkpoint_model,metrics,exact_strategies,csv_write
from .selection import grid,choose_grid,standalone
from .statistics import paired_stats
from .cli import Stop


def apply_policy(method,policy,base,candidate,prediction):
    zero=torch.zeros_like(base[:,:1])
    if policy.get('return_base'):return base,zero
    if method.startswith('B'):
        a=zero+policy.get('alpha',1.);return (base+a*(candidate-base)).clamp(0,1),a
    if method.startswith('G'):
        a=((torch.sigmoid(prediction['logit']/policy.get('temperature',1.))-policy.get('margin',0))/(1-policy.get('margin',0))).clamp(0,1)
        return (base+a*(candidate-base)).clamp(0,1),a
    if method=='F0':
        c00,c11,c01=prediction['C'].split(1,1)
        a=((c00-c01-policy.get('tau',0))/(c00+c11-2*c01+policy.get('lamb',1e-4))).clamp(0,1)
        return (base+a*(candidate-base)).clamp(0,1),a
    return decision(base,prediction,prediction['b_hat'],policy.get('tau',0),policy.get('lamb',1e-4))


def stress_field(image,condition):
    f=make_prior(image)
    if condition=='nominal':return f
    if condition in ['tau_050','tau_150']:return recompute(image,f['d'],f['A'],f['tau']*(.5 if condition=='tau_050' else 1.5))
    if condition=='ambient_green':return recompute(image,f['d'],(f['A']+image.new_tensor([0,.06,0]).view(1,3,1,1)).clamp(.05,.95))
    if condition=='shift_right_4':
        d,V=shift_field(f['d'],4,axis=-1);return recompute(image,d,f['A'],V=V)
    raise ValueError('unregistered stress')


def build_registry(s):
    path=s.run/'method_registry.json'
    inventory=read(s.run/'diagnostics/checkpoint_inventory.json');p=read(s.run/'selection/producer_freeze.json')['producer'];stand=read(s.run/'selection/standalone_selection.json')
    def item(cid):return next(i for i in inventory if i['checkpoint_id']==cid)
    producer=item(p['checkpoint_id']);matched=next(i for i in inventory if i['head']=='B3' and i['training_step']==producer['training_step'] and i['configuration_id']==producer['configuration_id'])
    registry={}
    for policy in ['clip01','official_minmax_float']:
        registry['B0_'+policy]={'kind':'backbone','baseline_policy':policy,'candidate':None,'controller':None,'grid':[{}],'primary_eligible':True,'uses_reference':False,'oracle_space':None}
    endpoints={'B1_producer':producer,'B1_standalone_best':item(stand['B1']['checkpoint_id']),
               'B3_standalone_best':item(stand['B3']['checkpoint_id']),'B3_matched':matched}
    if producer['configuration_id'].startswith('LOSS_COMPARISON'):
        summaries=read(s.run/'diagnostics/checkpoint_summary.json')
        for recipe in ['LOSS_COMPARISON_MSE','LOSS_COMPARISON_LOG']:
            for head in ['B1','B3']:
                chosen=standalone([r for r in summaries if r['role']=='model_val' and r['head']==head and r['configuration_id']==recipe])
                endpoints[head+'_'+recipe+'_standalone']=item(chosen['checkpoint_id'])
    for method,entry in endpoints.items():
        registry[method]={'kind':'candidate','baseline_policy':'clip01','candidate':entry,'controller':None,'grid':[{}],
              'primary_eligible':True,'uses_reference':False,'oracle_space':entry['checkpoint_sha256']}
    fixed={'B2':producer,'B2_RGB':endpoints['B3_standalone_best'],'B2_RGB_matched':matched}
    for method,entry in endpoints.items():
        if 'LOSS_COMPARISON' in method:fixed[method+'_fixed']=entry
    for method,entry in fixed.items():registry[method]={'kind':'fixed_candidate','baseline_policy':'clip01','candidate':entry,'controller':None,'grid':grid('B2'),
          'primary_eligible':True,'uses_reference':False,'oracle_space':entry['checkpoint_sha256']}
    for method in METHOD_ORDER:
        rec=read(s.run/'checkpoints'/method/'selection.json');weight=s.run/'checkpoints'/method/rec['selected_file']
        if sha(weight)!=rec['selected_sha256']:raise ValueError('frozen network mismatch')
        entry={'path':str(weight),'checkpoint_sha256':rec['selected_sha256'],'head':'B3','training_step':rec['selected_step'],'checkpoint_id':method+'_%06d'%rec['selected_step']}
        registry[method]={'kind':'fixed_candidate' if method=='B4' else 'controller','baseline_policy':'clip01','candidate':entry if method=='B4' else producer,
          'controller':None if method=='B4' else entry,'grid':grid(method),'primary_eligible':not method.startswith('O'),
          'uses_reference':False,'oracle_space':entry['checkpoint_sha256'] if method=='B4' else producer['checkpoint_sha256']}
    write(path,registry);return registry


class RegistryRuntime:
    def __init__(self,d,registry,scales=None):
        self.d=d;self.registry=registry;self.candidates={};self.controllers={}
        for method,r in registry.items():
            if r['candidate']:
                key=r['candidate']['checkpoint_sha256']
                if key not in self.candidates:self.candidates[key]=checkpoint_model(r['candidate'])
            if r['controller']:
                state=torch.load(r['controller']['path'],map_location='cpu');model=Controller(method);model.load_state_dict(state['model_state'],strict=True)
                self.controllers[method]=model.to('cuda:0').eval().requires_grad_(False)
        self.scales=scales if scales is not None else read(d.run/'normalization_stats.json')
    def features(self,image,base,condition='nominal'):
        outputs={};preds={};f=stress_field(image,condition)
        for key,model in self.candidates.items():
            entry=next(r['candidate'] for r in self.registry.values() if r['candidate'] and r['candidate']['checkpoint_sha256']==key)
            P,V=rgb_prior(image) if entry['head']=='B3' else (f['P'],f['V'])
            outputs[key]=model(image,base,P,V)
        for method,model in self.controllers.items():
            key=self.registry[method]['candidate']['checkpoint_sha256'];preds[method]=model(image,base,outputs[key],se2=self.scales['s_e2'])
        return outputs,preds
    def output(self,method,policy,image,base,candidates,preds,bases=None):
        r=self.registry[method]
        if r['kind']=='backbone':return bases[r['baseline_policy']],torch.zeros_like(base[:,:1])
        candidate=candidates[r['candidate']['checkpoint_sha256']]
        return apply_policy(method,policy,base,candidate,preds.get(method))
    def deploy(self,method,policy,image,condition='nominal'):
        """Only I; no reference, metadata or cache. Computes only selected route."""
        r=self.registry[method];base=self.d.backbone(image)
        if policy.get('return_base'):return base
        if r['kind']=='backbone':
            if r['baseline_policy']=='clip01':return base
            from ..backbones.ssuie import postprocess
            return postprocess(self.d.backbone.model(image),r['baseline_policy'])
        entry=r['candidate'];model=self.candidates[entry['checkpoint_sha256']]
        if entry['head']=='B3':P,V=rgb_prior(image)
        else:f=stress_field(image,condition);P,V=f['P'],f['V']
        candidate=model(image,base,P,V);pred=self.controllers[method](image,base,candidate,se2=self.scales['s_e2']) if method in self.controllers else None
        return apply_policy(method,policy,base,candidate,pred)[0]


def registry_acceptance(s,d,rt):
    path=s.run/'tests/registry_realtime_pairing.json'
    if path.exists():return
    differences=[]
    with s.device_job('REGISTRY_16_REALTIME_OUTPUT_PAIRING',600,final=True):
        with torch.no_grad():
            for row in d.data.role('utility_fit')[:16]:
                batch=d.data.batch([row['sample_id']],'utility_train');c,p=rt.features(batch['image'],batch['base']);bases={k:v[None].to('cuda:0') for k,v in d.data.base_pair(row).items()}
                for method,r in rt.registry.items():
                    policy=r['grid'][-1];a,_=rt.output(method,policy,batch['image'],batch['base'],c,p,bases);b=rt.deploy(method,policy,batch['image'])
                    diff=float((a-b).abs().max());differences.append(diff)
                    if diff>1e-5:raise ValueError('cache vs realtime differs for '+method)
    write(path,{'passed':True,'n_images':16,'method_ids':list(rt.registry),'max_abs':max(differences),'uses_reference_at_inference':False})


def score_table(s,d,rt,role,operation,selection=None,condition='nominal'):
    path=s.run/'metrics/parts'/(role+'_'+condition+('_cal_grid' if selection is None else '')+'.json')
    ident={'registry':sha(s.run/'method_registry.json'),'roles':sha(s.run/'roles.jsonl'),'source_snapshot':sha(s.run/'source_snapshot.json'),
           'selection':digest(selection),'condition':condition,'role':role,'metric_impl':sha(ROOT/'uie_next/evaluation.py')}
    rec=read(path) if path.exists() else {'identity':ident,'rows':[],'done':[]}
    if rec['identity']!=ident:raise ValueError('evaluation identity mismatch')
    with torch.no_grad():
        for row in d.data.role(role):
            if row['sample_id'] in rec['done']:continue
            s.guard();batch=d.data.batch([row['sample_id']],operation);base=batch['base'];image=batch['image'];target=batch['target'];c,p=rt.features(image,base,condition)
            bases={k:v[None].to('cuda:0') for k,v in d.data.base_pair(row).items()};base_m=metrics(base,target)[0];base_lp=float(d.lpips(base,target)[0]);oracles={}
            for key,j in c.items():oracles[key]=float((exact_strategies(base,j,target)[0]['oracle_pixel'][0]-target.cpu().double()).square().mean())
            for method,r in rt.registry.items():
                policies=r['grid'] if selection is None else [selection[method]['policy']]
                for index,policy in enumerate(policies):
                    output,alpha=rt.output(method,policy,image,base,c,p,bases);met=metrics(output,target)[0];lp=float(d.lpips(output,target)[0])
                    truth=labels(base,c[r['candidate']['checkpoint_sha256']],target) if r['candidate'] else None
                    rec['rows'].append({'run_id':s.ctx.run_id,'protocol_sha256':s.ctx.protocol_sha256,'role':role,'sample_id':row['sample_id'],'group_id':row['group_id'],
                       'method':method,'condition':condition,'strategy_index':index,'policy':policy,**met,'lpips':lp,'delta_psnr_db':met['psnr']-base_m['psnr'],
                       'ssim_delta':met['ssim']-base_m['ssim'],'lpips_delta':lp-base_lp,'alpha_mean':float(alpha.mean()),
                       'coverage_over_0_05':float((alpha>.05).float().mean()),'oracle_regret_mse':met['mse']-oracles[r['oracle_space']] if r['oracle_space'] else None,
                       'regret_status':'exact_same_candidate_space' if r['oracle_space'] else 'not_in_candidate_fusion_space',
                       **(prediction_utility_statistics(p.get(method,{}),truth) if truth else {'U_hat_mse':None,'U_hat_true_correlation':None,'U_hat_status':'not_applicable'}),
                       'input_sha256':row['input_sha256'],'reference_sha256':row['reference_sha256'],'candidate_sha256':r['oracle_space'],
                       'controller_sha256':r['controller']['checkpoint_sha256'] if r['controller'] else None,'uses_reference_at_inference':False,'identity':digest(ident)})
            rec['done'].append(row['sample_id']);write(path,rec)
    return rec['rows']


def method_summary(rows):
    summary={}
    for method in sorted({r['method'] for r in rows}):
        part=[r for r in rows if r['method']==method]
        summary[method]={k:float(np.mean([r[k] for r in part])) for k in ['psnr','ssim','lpips','mse','alpha_mean','coverage_over_0_05']}
        summary[method]['paired_vs_base']=paired_stats([r['delta_psnr_db'] for r in part],[r['group_id'] for r in part])
    return summary


def gate(s,rows,selection,confirm=False):
    summary=method_summary(rows);primary=selection['primary_control'];ablation=selection['primary_mechanism_ablation'];reg=read(s.run/'method_registry.json')
    by={m:{r['sample_id']:r for r in rows if r['method']==m} for m in summary};ids=sorted(by['O']);comp={}
    for method in summary:
        if set(by[method])!=set(ids):raise ValueError('missing paired methods')
        comp[method]=paired_stats([by['O'][i]['psnr']-by[method][i]['psnr'] for i in ids],[by['O'][i]['group_id'] for i in ids])
    checks={'base_gain':comp['B0_clip01']['mean_image']>=.1,'primary_gain':comp[primary]['mean_image']>=.1,
       'all_strong_controls_positive':all(comp[m]['mean_image']>0 for m,r in reg.items() if r['primary_eligible']),
       'SSIM':summary['O']['ssim']-summary[primary]['ssim']>=-.001,'LPIPS':summary['O']['lpips']-summary[primary]['lpips']<=.002,
       'ablation_gain':comp[ablation]['mean_image']>=.03,'all_main_ablations_positive':all(comp[m]['mean_image']>0 for m in ['O-NI','O-NP','O-NS']),
       'harm_rate':summary['O']['paired_vs_base']['harm_rate']<=summary[primary]['paired_vs_base']['harm_rate'],
       'all_controls_complete':all((s.run/'checkpoints'/m/'selection.json').exists() for m in METHOD_ORDER)}
    unresolved=[]
    for m in ['O','B4','G1','G2','F0','R0','O-NI','O-NP','O-NS']:
        rec=read(s.run/'checkpoints'/m/'selection.json');best={step:max(r['psnr'] for r in rec['all_grid_scores'] if r['step']==step) for step in [0,750,1500,2250,3000]}
        if rec['selected_step']==3000 and best[3000]-best[2250]>.05 and best[2250]-best[1500]>.05:unresolved.append(m)
    if confirm:checks.update(primary_CI_positive=comp[primary]['ci95'][0]>0,ablation_CI_positive=comp[ablation]['ci95'][0]>0)
    result={'passed':all(checks.values()),'checks':checks,'summary':summary,'paired':comp,'unresolved_optimization':unresolved,
          'historical_exposure':True,'single_module_seed':20261007,'independent_pretrain_repeats':False,'primary_control':primary,'primary_mechanism_ablation':ablation}
    return result


def evaluate_all(s,d):
    registry=build_registry(s);rt=RegistryRuntime(d,registry);registry_acceptance(s,d,rt)
    selection_path=s.run/'selection/calibration_selection.json'
    if not selection_path.exists():
        with s.device_job('CALIBRATION_ALL_METHODS',5400,final=True):rows=score_table(s,d,rt,'calibration','calibrate')
        csv_write(s.run/'selection/calibration_all_strategy_per_image.csv',rows);chosen={}
        for method,r in registry.items():
            table=[]
            for index,policy in enumerate(r['grid']):
                part=[p for p in rows if p['method']==method and p['strategy_index']==index]
                table.append({'step':0,'strategy_index':index,'return_base':policy.get('return_base',False),**{k:np.mean([p[k] for p in part]) for k in ['psnr','ssim_delta','lpips_delta']}})
            best=choose_grid(table) if len(table)>1 else table[0];chosen[method]={'policy':r['grid'][best['strategy_index']],'summary':best,'all_grid_scores':table}
        controls=sorted(m for m,r in registry.items() if r['primary_eligible']);primary=controls[0]
        for m in controls[1:]:
            if chosen[m]['summary']['psnr']>chosen[primary]['summary']['psnr']+1e-8:primary=m
        ablation='O-NI'
        for m in ['O-NP','O-NS']:
            if chosen[m]['summary']['psnr']>chosen[ablation]['summary']['psnr']+1e-8:ablation=m
        write(selection_path,{'methods':chosen,'primary_control':primary,'primary_mechanism_ablation':ablation,'role':'calibration','n_images':133,'no_dev_hint_cherry_pick':True,
               'network_freeze':sha(s.run/'selection/network_freeze_before_calibration.json'),'registry':sha(s.run/'method_registry.json')})
    selection=read(selection_path);s.complete('CALIBRATION',{'selection':sha(selection_path)})
    with s.device_job('DEV_AND_FOUR_REGISTERED_STRESS',5400,final=True):
        rows=score_table(s,d,rt,'utility_val','develop',selection['methods']);result=gate(s,rows,selection)
        write(s.run/'development_gate.json',result);csv_write(s.run/'metrics/development_per_image.csv',rows)
        stress=[]
        for condition in ['tau_050','tau_150','ambient_green','shift_right_4']:
            sr=score_table(s,d,rt,'utility_val','develop',selection['methods'],condition);su=method_summary(sr)
            flag='ROBUSTNESS_CLAIM_FAIL' if su['O']['psnr']-su['B0_clip01']['psnr']<-.1 or su['O']['psnr']-su[selection['primary_control']]['psnr']<-.1 else 'NO_PREDEFINED_FAILURE_NOT_GENERAL_ROBUSTNESS_PROOF'
            stress.append({'condition':condition,'status':flag,'summary':su})
        write(s.run/'development_stress.json',stress)
    from .delivery import visual_method_panels,benchmark
    visual_method_panels(s,d,rt,rows,selection,'utility_val');benchmark(s,d,rt,selection)
    if not result['passed']:
        if result['unresolved_optimization']:status='INCONCLUSIVE_OPTIMIZATION'
        elif result['checks']['base_gain'] and result['checks']['primary_gain']:status='QUALITY_ONLY_NO_MECHANISM_EVIDENCE'
        else:status='STOP_CURRENT_RECIPE_NOT_SUPPORTED'
        raise Stop(status,'完整开发闸门未通过，保留177对封存数据不评分。')
    s.complete('DEV_GATE',{'result':sha(s.run/'development_gate.json')})
    # Reverify every code, data, checkpoint and selection identity before release.
    for p,h in read(s.run/'source_snapshot.json').items():
        if sha(ROOT/p)!=h:raise Stop('BLOCKED_ENGINEERING','解封前源码身份改变 '+p)
    for m,r in registry.items():
        for field in ['candidate','controller']:
            if r[field] and sha(r[field]['path'])!=r[field]['checkpoint_sha256']:raise Stop('BLOCKED_ENGINEERING','解封前权重身份改变')
    frozen={'development_status':'DEV_PASS','protocol':s.ctx.protocol_sha256,'source_snapshot':sha(s.run/'source_snapshot.json'),'registry':sha(s.run/'method_registry.json'),
       'calibration':sha(selection_path),'roles':sha(s.run/'roles.jsonl'),'metrics':sha(ROOT/'uie_next/evaluation.py'),'development':sha(s.run/'development_gate.json'),
       'network_freeze':sha(s.run/'selection/network_freeze_before_calibration.json'),'backbone':s.config['backbone']['checkpoint_sha256'],'historical_exposure':True}
    fp=s.run/'selection/selection_freeze_before_eval.json'
    if fp.exists() and read(fp)!=frozen:raise Stop('BLOCKED_ENGINEERING','一次封存freeze不一致')
    write(fp,frozen);d.data.guard.final_freeze=frozen;s.live(sealed_eval_released=True)
    with s.device_job('SEALED_ONCE',5400,final=True):
        sealed=score_table(s,d,rt,'sealed_eval','final',selection['methods']);confirmation=gate(s,sealed,selection,True)
        write(s.run/'confirmation_results.json',confirmation);csv_write(s.run/'metrics/sealed_per_image.csv',sealed)
        allstress=[]
        for condition in ['tau_050','tau_150','ambient_green','shift_right_4']:
            table=score_table(s,d,rt,'sealed_eval','final',selection['methods'],condition);allstress.append({'condition':condition,'summary':method_summary(table)})
        write(s.run/'confirmation_stress.json',allstress)
    visual_method_panels(s,d,rt,sealed,selection,'sealed_eval');s.complete('SEALED_ONCE',{'result':sha(s.run/'confirmation_results.json')})
    status='CONFIRMATION_PASS_SINGLE_SEED_HISTORICAL_EXPOSURE' if confirmation['passed'] else 'CONFIRMATION_FAIL' if confirmation['paired'][selection['primary_control']]['mean_image']<0 else 'INCONCLUSIVE_CONFIRMATION'
    s.live(scientific_status=status)
