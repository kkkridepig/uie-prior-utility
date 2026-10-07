"""Frozen checkpoint calibration, development gates and one final release."""
import math
import json
import time

import numpy as np
import torch

from .calibration import grid, select
from .experiment import ScientificStop
from .math.utility import decision, geometry, labels, oracle
from .priors.heuristic import stress_views
from .records import ROOT, digest, jsonl, read, sha, write
from .training import ORDER

CONTROLS=['B0_clip01','B0_official_minmax_float','B1','B2','B3','B4','G0','G1','G2','F0','R0']
ABLATIONS=['O-NI','O-NP','O-NS','O-ND']
METHODS=CONTROLS+['O']+ABLATIONS


def prediction_utility_statistics(prediction, lab):
    """Absent predicted utility is NA, never replaced with supervision."""
    actual = lab['U'].flatten()
    result = {'U_hat_mse': None, 'U_hat_true_correlation': None,
              'U_hat_status': 'not_provided_by_method',
              'U_positive_fraction': float((actual > 0).float().mean()),
              'U_negative_fraction': float((actual < 0).float().mean())}
    if 'U_hat' in prediction:
        predicted = prediction['U_hat'].flatten()
        result.update(U_hat_status='provided_by_method',
                      U_hat_mse=float((predicted-actual).square().mean()))
        if float(predicted.std(unbiased=False)) > 0 and float(actual.std(unbiased=False)) > 0:
            result['U_hat_true_correlation'] = float(((predicted-predicted.mean())*(actual-actual.mean())).mean() /
                (predicted.std(unbiased=False)*actual.std(unbiased=False)))
    return result


def load_models(e):
    models={}
    for method in ['B1','B3']+ORDER:models[method]=e.selected(method)[0]
    return models


def decode_policy(method,pred,base,candidate,policy):
    g=geometry(candidate-base)
    if policy.get('return_base'):
        return base,torch.zeros_like(g['a'])
    if method in ['B1','B2','B3','B4']:
        alpha=torch.full_like(g['a'],policy.get('alpha',1.))
        return base+alpha*g['r'],alpha
    if method.startswith('G'):
        margin=policy.get('margin',0.);temperature=policy.get('temperature',1.)
        alpha=((torch.sigmoid(pred['logit']/temperature)-margin)/(1-margin)).clamp(0,1)
        return (base+alpha*g['r']).clamp(0,1),alpha
    if method=='F0':
        c00,c11,c01=pred['C'].split(1,1)
        alpha=((c00-c01-policy.get('tau',0.))/(c00+c11-2*c01+policy.get('lambda',1e-4))).clamp(0,1)
        return (base+alpha*g['r']).clamp(0,1),alpha
    return decision(base,g,pred['b_hat'],policy.get('tau',0.),policy.get('lambda',1e-4))


def image_features(e,models,row,operation,condition='nominal'):
    batch=e.data.batch([row['sample_id']],operation,True)
    image,base,target=batch['image'],batch['base'],batch['target']
    candidate=batch['candidates'][:,0]
    if condition!='nominal':
        fields=stress_views(image)[condition]
        candidate=models['B1'](image,base,fields['P'],fields['V'])
    rgb=e.data.real_candidate(image,base,models['B3'],True)
    matched=e.data.real_candidate(image,base,models['B4'],True)
    preds={m:models[m](image,base,candidate,se2=read(e.run/'normalization_stats.json')['s_e2']) for m in ORDER if m!='B4'}
    bases=e.data.base_pair(row)
    return batch,candidate,rgb,matched,preds,bases


def get_output(e,method,batch,candidate,rgb,matched,preds,bases,policy):
    base=batch['base']
    if method.startswith('B0_'):
        return bases[method[3:]][None].to('cuda:0'),torch.zeros_like(base[:,:1])
    j1=rgb if method=='B3' else matched if method=='B4' else candidate
    return decode_policy(method,preds.get(method),base,j1,policy)


def summarize(rows,selected_base):
    by_method={m:{r['sample_id']:r for r in rows if r['method']==m} for m in sorted(set(r['method'] for r in rows))}
    base=by_method[selected_base];out={}
    for method,items in by_method.items():
        ids=sorted(items);delta=np.array([items[i]['psnr']-base[i]['psnr'] for i in ids])
        groups=[items[i]['group_id'] for i in ids]
        out[method]={'images':len(ids),'psnr':float(np.mean([items[i]['psnr'] for i in ids])),
                     'ssim':float(np.mean([items[i]['ssim'] for i in ids])),
                     'lpips':float(np.mean([items[i]['lpips'] for i in ids])),
                     'mse':float(np.mean([items[i]['mse'] for i in ids])),
                     'delta_vs_base':float(delta.mean()),'median_delta':float(np.median(delta)),
                     'delta_quantiles':np.quantile(delta,[.05,.1,.5,.9,.95]).tolist(),
                     'harm_rate_over_0_1':float(np.mean(delta<-.1)),'improvement_rate_over_0_1':float(np.mean(delta>.1)),
                     'worst_10_percent_mean_delta':float(np.mean(np.sort(delta)[:max(1,math.ceil(.1*len(delta)))])),
                     'group_bootstrap_vs_base':__import__('uie_next.evaluation',fromlist=['bootstrap_group']).bootstrap_group(delta,groups),
                     'alpha_mean':float(np.mean([items[i]['alpha_mean'] for i in ids])),
                     'alpha_over_0_05_fraction':float(np.mean([items[i]['alpha_over_0_05_fraction'] for i in ids]))}
    return out


def paired(rows,first,second):
    from .evaluation import bootstrap_group
    a={r['sample_id']:r for r in rows if r['method']==first};b={r['sample_id']:r for r in rows if r['method']==second}
    if set(a)!=set(b):raise ValueError('Paired evaluation sample mismatch.')
    ids=sorted(a)
    return bootstrap_group([a[i]['psnr']-b[i]['psnr'] for i in ids],[a[i]['group_id'] for i in ids])


def calibrate(e):
    if (e.run/'calibration_selection.json').exists():return
    models=load_models(e);methods=['B2','B4']+[m for m in ORDER if m!='B4']
    grids={m:grid(m) for m in methods};write(e.run/'calibration_grid.json',grids)
    all_rows=[]
    with e.device_job('S8_fixed_calibration_matrix',5400,final=True):
        with torch.no_grad():
            for count,row in enumerate(e.data.role('calibration')):
                e.guard();batch,j1,rgb,matched,preds,bases=image_features(e,models,row,'calibrate')
                common={'sample_id':row['sample_id'],'group_id':row['group_id'],'role':'calibration','condition':'nominal'}
                baseline_metrics=e.metrics(batch['base'],batch['target'])[0]
                for method in METHODS:
                    options=grids[method] if method in grids else [{}]
                    for index,policy in enumerate(options):
                        out,alpha=get_output(e,method,batch,j1,rgb,matched,preds,bases,policy)
                        metric=baseline_metrics if policy.get('return_base') or (method in ['B2','B4'] and policy.get('alpha')==0) else e.metrics(out,batch['target'])[0]
                        all_rows.append({**common,'method':method,'strategy_index':index,'policy':policy,**metric,
                                         'alpha_mean':float(alpha.mean()),'alpha_over_0_05_fraction':float((alpha>.05).float().mean()),
                                         'ssim_delta':metric['ssim']-baseline_metrics['ssim'],'lpips_delta':metric['lpips']-baseline_metrics['lpips']})
                if count%10==0:e.budget.tick();e.live(completed_images=count+1)
            jsonl(e.run/'metrics/calibration_per_image.jsonl',all_rows)
            selection={};summary={}
            for method in METHODS:
                options=grids.get(method,[{}]);table=[]
                for index,policy in enumerate(options):
                    part=[r for r in all_rows if r['method']==method and r['strategy_index']==index]
                    table.append({k:float(np.mean([r[k] for r in part])) for k in ['psnr','ssim','lpips','ssim_delta','lpips_delta']})
                chosen=select(table) if method in grids else 0
                selection[method]={'policy':options[chosen],'strategy_index':chosen,'summary':table[chosen]};summary[method]=table
            from .timing import tie_costs
            costs=tie_costs(e,models,selection)
            primary=CONTROLS[0]
            for method in CONTROLS[1:]:
                gap=selection[method]['summary']['psnr']-selection[primary]['summary']['psnr']
                if gap>1e-8 or abs(gap)<=1e-8 and costs[method]<costs[primary]:primary=method
            ablation='O-NI'
            for method in ['O-NP','O-NS']:
                if selection[method]['summary']['psnr']>selection[ablation]['summary']['psnr']+1e-8:ablation=method
            result={'methods':selection,'primary_control':primary,'primary_mechanism_ablation':ablation,
                    'calibration_manifest_sha256':digest([r['sample_id'] for r in e.data.role('calibration')]),
                    'all_strategy_tables':summary,'tie_costs_service_seconds':costs,'selection_uses_only_calibration':True,
                    'selected_checkpoints':{m:read(e.run/'checkpoints'/m/'selection.json')['selected_sha256'] for m in models},
                    'metric_identity':e.lpips.identity,'source_snapshot_sha256':sha(e.run/'source_snapshot.json')}
            write(e.run/'calibration_selection.json',result)


def eval_table(e,role,operation,models,condition='nominal'):
    selection=read(e.run/'calibration_selection.json')['methods']
    path=e.run/'metrics'/('%s_%s.jsonl'%(role,condition))
    freeze_identity={'roles':sha(e.run/'roles.jsonl'),'calibration_selection':sha(e.run/'calibration_selection.json'),
                     'checkpoint_selections':{m:sha(e.run/'checkpoints'/m/'selection.json') for m in ['B1','B3']+ORDER},
                     'condition':condition,'role':role,'source_snapshot':sha(e.run/'source_snapshot.json')}
    identity_path=path.with_suffix('.identity.json')
    if identity_path.exists() and read(identity_path)!=freeze_identity:raise ValueError('Cached evaluation identity changed.')
    write(identity_path,freeze_identity)
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed={r['sample_id'] for r in rows}
    for sample_id in completed:
        subset=[r for r in rows if r['sample_id']==sample_id]
        if set(r['method'] for r in subset)!=set(METHODS):raise ValueError('Incomplete sample evaluation receipt.')
    for count,row in enumerate(e.data.role(role)):
        if row['sample_id'] in completed:continue
        e.guard()
        with torch.no_grad():
            batch,j1,rgb,matched,preds,bases=image_features(e,models,row,operation,condition)
            lab=labels(batch['base'],j1,batch['target']);qstar=oracle(batch['base'],j1,batch['target'])[0]
            qloss=float((qstar-batch['target']).square().mean())
            records=[]
            for method in METHODS:
                policy=selection[method]['policy']
                out,alpha=get_output(e,method,batch,j1,rgb,matched,preds,bases,policy)
                metric=e.metrics(out,batch['target'])[0]
                item={'run_id':e.config['runtime']['run_id'],'sample_id':row['sample_id'],'group_id':row['group_id'],
                      'role':role,'condition':condition,'method':method,'policy':policy,**metric,
                      'alpha_mean':float(alpha.mean()),'alpha_over_0_05_fraction':float((alpha>.05).float().mean()),
                      'oracle_regret_mse':metric['mse']-qloss,'uses_reference_at_inference':False,
                      'input_sha256':row['input_sha256'],'backbone_sha256':e.config['backbone']['checkpoint_sha256'],
                      'candidate_sha256':e.data.candidate_hash,'evaluation_identity':digest(freeze_identity),'error':None}
                if method in preds and 'v_hat' in preds[method]:
                    p=preds[method];mask=lab['active']
                    item['v_hat_mse_active']=float((p['v_hat'][mask]-lab['v'][mask]).square().mean()) if mask.any() else None
                    item['b_hat_mse']=float((p['b_hat']-lab['b']).square().mean())
                if method in preds:
                    item.update(prediction_utility_statistics(preds[method], lab))
                item['checkpoint_sha256']=e.config['backbone']['checkpoint_sha256'] if method.startswith('B0_') else read(e.run/'checkpoints'/('B1' if method=='B2' else method)/'selection.json')['selected_sha256']
                item['input_path']=row['input_path'];item['reference_path']=row['reference_path']
                records.append(item)
            rows.extend(records)
            # Complete all methods of one image before the atomic progress commit.
            jsonl(path,rows)
        if count%10==0:e.budget.tick();e.live(completed_images=count+1,evaluation_role=role,condition=condition)
    return rows


def gate(e,rows,confirmation=False):
    selection=read(e.run/'calibration_selection.json');base='B0_'+e.data.policy;primary=selection['primary_control'];ablation=selection['primary_mechanism_ablation']
    summary=summarize(rows,base);o=summary['O'];control=summary[primary]
    comparisons={m:paired(rows,'O',m) for m in set([base,primary,ablation]+['G1','G2','F0','R0','B4','O-NI','O-NP','O-NS'])}
    checks={'vs_selected_base_db':comparisons[base]['mean_image']>=.1,'vs_primary_control_db':comparisons[primary]['mean_image']>=.1,
            'vs_key_controls':all(comparisons[m]['mean_image']>0 for m in ['G1','G2','F0','R0','B4']),
            'ssim_guard':o['ssim']-control['ssim']>=-.001,'lpips_guard':o['lpips']-control['lpips']<=.002,
            'primary_ablation_db':comparisons[ablation]['mean_image']>=.03,
            'all_main_ablations_positive':all(comparisons[m]['mean_image']>0 for m in ['O-NI','O-NP','O-NS']),
            'harm_rate_guard':o['harm_rate_over_0_1']<=control['harm_rate_over_0_1']}
    unresolved=[m for m in ['O','B4','G1','G2','F0','R0','O-NI','O-NP','O-NS'] if read(e.run/'checkpoints'/m/'selection.json')['convergence_unresolved']]
    checks['training_not_obviously_unresolved']=not unresolved
    if confirmation:
        checks['primary_control_ci_lower_positive']=comparisons[primary]['ci95'][0]>0
        checks['primary_ablation_ci_lower_positive']=comparisons[ablation]['ci95'][0]>0
    return {'passed':all(checks.values()),'checks':checks,'summary':summary,'paired_comparisons':comparisons,
            'primary_control':primary,'primary_mechanism_ablation':ablation,'selected_base':base,
            'convergence_unresolved_methods':unresolved,'historically_analyzed':True,
            'scene_group_quality':'content_group_proxy','training_seed':e.config['seed'],'independent_pretraining_repeats':False}


def develop(e):
    models=load_models(e)
    with e.device_job('S9_development_and_registered_stress',5400,final=True):
        rows=eval_table(e,'utility_val','develop',models)
        result=gate(e,rows);write(e.run/'development_gate.json',result)
        stress=[]
        for condition in ['tau_050','tau_150','ambient_green','shift_down_4','blur_d_9','central_missing_64']:
            table=eval_table(e,'utility_val','develop',models,condition)
            stress.append({'condition':condition,'summary':summarize(table,'B0_'+e.data.policy)})
        write(e.run/'development_stress.json',stress)
        from .visuals import export_panels
        export_panels(e,models,rows,'utility_val')
        if not result['passed']:
            if result['convergence_unresolved_methods']:status='INCONCLUSIVE_BUDGET'
            elif result['checks']['vs_selected_base_db'] and result['checks']['vs_primary_control_db']:status='QUALITY_ONLY_NO_MECHANISM_EVIDENCE'
            else:status='STOP_MECHANISM_NOT_SUPPORTED'
            raise ScientificStop(status,'Development gates failed; sealed_eval remains unopened. '+json.dumps(result['checks']))


def final_evaluate(e):
    models=load_models(e)
    freeze_path=e.run/'selection_freeze_before_eval.json'
    if not freeze_path.exists():
        write(freeze_path,{'development_status':'DEV_PASS','development_gate_sha256':sha(e.run/'development_gate.json'),
                          'calibration_selection_sha256':sha(e.run/'calibration_selection.json'),
                          'source_snapshot_sha256':sha(e.run/'source_snapshot.json'),
                          'training_manifest_hashes_sha256':sha(e.run/'training_manifest_hashes.json'),
                          'role_manifest_sha256':sha(e.run/'roles.jsonl'),'metric_implementation_sha256':sha(ROOT/'uie_next/evaluation.py'),
                          'checkpoints':{m:read(e.run/'checkpoints'/m/'selection.json')['selected_sha256'] for m in ['B1','B3']+ORDER},
                          'historical_exposure':True,'independent_blind_test':False})
    freeze=read(freeze_path)
    if freeze['calibration_selection_sha256']!=sha(e.run/'calibration_selection.json') or freeze['role_manifest_sha256']!=sha(e.run/'roles.jsonl'):
        raise ValueError('Final release freeze changed.')
    e.data.guard.final_freeze=freeze;e.live(sealed_eval_released=True)
    with e.device_job('S10_once_frozen_final_evaluation_and_stress',5400,final=True):
        rows=eval_table(e,'sealed_eval','final',models);result=gate(e,rows,True)
        write(e.run/'confirmation_results.json',result)
        stress=[]
        for condition in ['tau_050','tau_150','ambient_green','shift_down_4','blur_d_9','central_missing_64']:
            table=eval_table(e,'sealed_eval','final',models,condition)
            stress.append({'condition':condition,'summary':summarize(table,'B0_'+e.data.policy)})
        write(e.run/'confirmation_stress.json',stress)
        from .timing import benchmark
        benchmark(e,models,read(e.run/'calibration_selection.json')['methods'])
        from .visuals import export_panels
        export_panels(e,models,rows,'sealed_eval')
        if result['passed']:status='CONFIRMATION_PASS_SINGLE_SEED'
        elif result['paired_comparisons'][result['primary_control']]['mean_image']<0:status='CONFIRMATION_FAIL'
        else:status='INCONCLUSIVE_CONFIRMATION'
        e.live(scientific_status=status)
