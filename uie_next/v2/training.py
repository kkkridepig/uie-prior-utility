"""Finite, paired candidate rescue and 11-method local-utility training."""
import copy
import json
import math
import time
import signal
from pathlib import Path
import numpy as np
import torch
from ..records import ROOT,read,write,sha,digest,append
from ..checkpoint import save,resume,restore_rng
from ..training import streams,initialize,optimizer,GroupSampler,drop_prior,pair_objective
from ..models.candidate import Candidate
from ..models.controls import Controller
from ..priors.heuristic import make_prior,interventions,rgb_prior
from ..math.utility import labels
from .context import OLD,METHOD_ORDER
from .diagnostics import checkpoint_model,metrics,exact_strategies,csv_write
from .selection import eligibility,choose_producer,standalone,grid,choose_grid
from .errors import Stop


def profile_plan(s,d):
    path=s.run/'budget_plan.json'
    if path.exists():return read(path)
    d.load();d.verify_reuse();items=read(s.run/'diagnostics/checkpoint_inventory.json');producer=checkpoint_model(next(x for x in items if x['checkpoint_id']=='B1_004000'))
    records=read(s.run/'tests/real_matrix/profile.json');prof={};seconds=0
    with s.device_job('PROFILE_D0_AND_FULL_MATRIX',600):
        rowids=[r['sample_id'] for r in d.data.role('utility_fit')[:4]];batch=d.data.batch(rowids,'utility_train');fields=interventions(batch['image'])
        with torch.no_grad():cand=torch.stack([producer(batch['image'],batch['base'],fields[k]['P'],fields[k]['V']) for k in s.config['prior']['train_views']],1)
        scales=read(s.run/'tests/real_matrix/scales.json')
        for method in METHOD_ORDER:
            rng=streams(2026100799);model=initialize(Candidate if method=='B4' else lambda:Controller(method),rng['init']).to('cuda:0');opt,sched=optimizer(model,3000)
            times=[]
            for step in range(12):
                start=time.monotonic();opt.zero_grad(set_to_none=True)
                if method=='B4':
                    ids=[r['sample_id'] for r in d.data.role('model_fit')[:4]]+rowids;b=d.data.batch(ids,'matched_train');P,V=rgb_prior(b['image']);out=model(b['image'],b['base'],P,V);loss=(out-b['target']).square().mean()
                else:
                    b=d.data.batch(rowids,'utility_train');loss,_=pair_objective(model,b['image'],b['base'],cand[:,:2] if method!='O-NI' else cand[:,:1].repeat(1,2,1,1,1),b['target'],scales)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step();torch.cuda.synchronize()
                if step>=2:times.append(time.monotonic()-start)
            prof[method]={'measured_seconds_per_update':float(np.mean(times)),'measured_updates':10,'warmup_updates':2,
                  'predicted_training_seconds':float(np.mean(times))*3000*1.3,'temporary_fixture_only':True,'parameters':sum(p.numel() for p in model.parameters())}
            del model,opt,sched
        start=time.monotonic();base=batch['base'];target=batch['target']
        with torch.no_grad():
            nominal=producer(batch['image'],base,fields['nominal']['P'],fields['nominal']['V']);variants,_=exact_strategies(base,nominal,target)
            d.lpips(torch.cat([base,nominal]),target.repeat(2,1,1,1))
            for out,a in variants.values():metrics(out,target)
            for f in list(fields.values())[1:]:producer(batch['image'],base,f['P'],f['V'])
        torch.cuda.synchronize();sec=(time.monotonic()-start)/4
        probe=d.probe();d0count=(len(probe)+671+136)*8+2*(len(probe)+671+136)
        d0=sec*d0count*1.3
        # Every checkpoint's full 136 x ten strategies quality table, including LPIPS.
        validation=sec*136*5*10*11*1.3/4
        matrix=sum(r['predicted_training_seconds'] for r in prof.values())+validation
        cache=sec*(443+136)*2*1.3
    used=read(s.run/'budget.json')['used_device_seconds'];final=max(12600.,sec*(133+136+177)*11*10*1.3)
    plan={'status':'PROFILED','profile':prof,'D0_seconds_per_image_including_six_views_metrics':sec,'D0_predicted_seconds':d0,
          'full_matrix_training_and_validation_predicted_seconds':matrix,'cache_predicted_seconds':cache,
          'rescue_predicted_seconds':max(prof['B4']['measured_seconds_per_update']*16000*1.3,1200)+d0,
          'protected_final_seconds':final,'used_device_seconds_at_plan':used,'safety_factor':1.3,'all_11_methods_3000_updates':True,
          'complete_without_rescue_predicted_seconds':used+d0+matrix+cache+final,'maximum_device_seconds':57600,
          'all_profile_is_temporary_and_counted':True,'CPU_bootstrap_not_device_runtime':True}
    write(path,plan);write(s.run/'profile_measurements.json',{'profiles':prof,'measured_diagnostic_seconds_per_image':sec})
    if used+d0+matrix+cache+final>57600:raise Stop('INCONCLUSIVE_BUDGET','profile预测完整诊断与11控制矩阵超过继承预算，未删除控制或缩数据。')
    return plan


def identity(s,method,extra=None):
    return {'run_id':s.ctx.run_id,'protocol_sha256':s.ctx.protocol_sha256,'config_sha256':digest(s.config),'method':method,'seed':20261007,
       'roles':sha(s.run/'roles.jsonl'),'backbone_sha256':s.config['backbone']['checkpoint_sha256'],
       'producer':sha(s.run/'selection/producer_freeze.json') if (s.run/'selection/producer_freeze.json').exists() else None,
       'normalization':sha(s.run/'normalization_stats.json') if (s.run/'normalization_stats.json').exists() else None,
       'source_snapshot':sha(s.run/'source_snapshot.json'),'extra':extra}


def candidate_streams():
    """Exactly the legacy four-stream contract used by both original heads."""
    return streams(20261007)


def train_candidate(s,d,method,total,route,source,sref=None):
    directory=s.run/'checkpoints'/method;directory.mkdir(parents=True,exist_ok=True)
    head='B3' if method.startswith('B3') else 'B1';rgb=head=='B3';losskind='LOG' if method.endswith('_LOG') else 'MSE'
    # Original candidate checkpoints contain precisely these four streams.
    # model_data is utility B4-specific and must not be requested from V1 RNG.
    rng=candidate_streams()
    model=Candidate().to('cuda:0');opt,sched=optimizer(model,total);source_state=torch.load(source['path'],map_location='cpu')
    extra={'route':route,'source_checkpoint_sha256':source['checkpoint_sha256'],'loss':losskind,'s_ref':sref,
        'source_scheduler_state':source_state['scheduler_state'],'group_sampler':'group_uniform_then_image_uniform_rng_only_no_hidden_state'}
    ident=identity(s,method,extra);latest=directory/'latest.pt';step=0;restored=None
    if latest.exists():restored=resume(latest,model,opt,sched,ident,rng);step=restored['global_step']
    else:
        model.load_state_dict(source_state['model_state'],strict=True);restore_rng(source_state['rng'],rng)
        if route=='LOW_LR_CONTINUATION':opt.load_state_dict(source_state['optimizer_state']);step=4000
        for group in opt.param_groups:
            if route=='LOW_LR_CONTINUATION':group['lr']=1e-5
        save(directory/('step_%06d.pt'%step),model,opt,sched,step,ident,rng,include_cuda=True,extra=extra)
    if route=='LOW_LR_CONTINUATION':
        for group in opt.param_groups:group['lr']=1e-5
        points=[4000,6000,8000,10000,12000]
    else:points=[0,1000,2000,4000,8000,12000]
    def output_inventory():
        return [{'checkpoint_id':method+'_%06d'%n,'head':head,'training_step':n,'checkpoint_sha256':sha(directory/('step_%06d.pt'%n)),
                 'path':str(directory/('step_%06d.pt'%n)),'configuration_id':route+'_'+losskind,'method_id':method} for n in points]
    if step==total:
        return output_inventory()
    previous=restored['extra'] if restored is not None else {}
    sampler=GroupSampler(d.data.role('model_fit'),rng['data'])
    logs=previous.get('unfinished_loss_window',[]);gradnorms=previous.get('unfinished_gradient_window',[]);seen=set(previous.get('seen_source_ids',[]))
    def recoverable_extra():
        return {**extra,'seen_source_ids':sorted(seen),'unfinished_loss_window':logs,'unfinished_gradient_window':gradnorms}
    predict=read(s.run/'budget_plan.json')['rescue_predicted_seconds']/(2 if route=='LOW_LR_CONTINUATION' else 4)
    with s.device_job('rescue_train_'+method,max(predict*(total-step)/total,60)):
        try:
            for update in range(step+1,total+1):
                s.guard();ids=sampler.sample(8);batch=d.data.batch(ids,'candidate_train');opt.zero_grad(set_to_none=True)
                if rgb:P,V=rgb_prior(batch['image'])
                else:f=make_prior(batch['image']);P,V=f['P'],f['V']
                P,V,missing=drop_prior(P,V,rng['missing_prior']);out=model(batch['image'],batch['base'],P,V)
                mi=(out-batch['target']).square().flatten(1).mean(1)
                loss=mi.mean() if losskind=='MSE' else sref*torch.log((mi+1e-6)/(sref+1e-6)).mean()
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not torch.isfinite(loss) or not torch.isfinite(gn):raise RuntimeError('nonfinite candidate loss/gradient')
                lr=opt.param_groups[0]['lr'];opt.step()
                if route!='LOW_LR_CONTINUATION':sched.step()
                step=update;logs.append(float(loss));gradnorms.append(float(gn));seen.update(ids);s.state['formal_training_updates_v2']=s.state.get('formal_training_updates_v2',0)+1
                if step%50==0:
                    append(directory/'training.jsonl',{'step':step,'loss_current':float(loss),'last50_mean':float(np.mean(logs)),'last50_min':min(logs),'last50_max':max(logs),
                         'learning_rate':lr,'gradient_norm':float(gn),'clipped':float(gn)>1,'last50_gradient_norm_mean':float(np.mean(gradnorms)),
                         'last50_gradient_clip_fraction':float(np.mean(np.array(gradnorms)>1)),'source_presentations':8*(step-(4000 if route=='LOW_LR_CONTINUATION' else 0)),
                         'distinct_sources_cumulative':len(seen),'source_ids_current':ids,'source_rng_state_hash':digest(rng['data'].get_state().tolist()),'missing_count':int(missing.sum()),'loss_kind':losskind});logs=[];gradnorms=[]
                    s.live(global_step=step,current_method=method)
                if step%250==0:save(latest,model,opt,sched,step,ident,rng,include_cuda=True,extra=recoverable_extra())
                if step in points:save(directory/('step_%06d.pt'%step),model,opt,sched,step,ident,rng,include_cuda=True,extra=recoverable_extra())
        finally:save(latest,model,opt,sched,step,ident,rng,include_cuda=True,extra=recoverable_extra())
    del model,opt,sched
    return output_inventory()


def rescue(s,d):
    choice=read(s.run/'selection/rescue_choice.json');route=choice['route'];plan=read(s.run/'budget_plan.json')
    cost=plan['rescue_predicted_seconds']*(2 if route=='LOSS_COMPARISON' else 1)
    remaining=57600-read(s.run/'budget.json')['used_device_seconds']
    if 'ONE_RESCUE' not in s.state['completed'] and (cost>14400 or cost+plan['full_matrix_training_and_validation_predicted_seconds']+plan['protected_final_seconds']>remaining):
        raise Stop('INCONCLUSIVE_BUDGET','唯一救援包（全部匹配臂与诊断）profile预算不足，未运行。')
    if not s.state.get('rescue_budget_active'):
        s.live(rescue_budget_active=True,rescue_budget_start_seconds=read(s.run/'budget.json')['used_device_seconds'])
    inventory=read(s.run/'diagnostics/checkpoint_inventory.json');probe=d.probe();new=[]
    sref=None
    if route=='LOSS_COMPARISON':
        records=read(s.run/'diagnostics/checkpoint_summary.json');sref=max(next(r for r in records if r['head']=='B1' and r['role']=='model_fit_probe' and r['training_step']==0)['baseline_mse'],1e-4)
    for head in ['B1','B3']:
        losses=['MSE','LOG'] if route=='LOSS_COMPARISON' else ['MSE']
        for loss in losses:
            method=head+'_'+loss if route=='LOSS_COMPARISON' else head+'_LOWLR'
            src=next(i for i in inventory if i['head']==head and i['training_step']==(0 if route=='LOSS_COMPARISON' else 4000))
            new.extend(train_candidate(s,d,method,12000,route,src,sref))
    from .context import merge_inventory
    inventory=merge_inventory(inventory,new);write(s.run/'diagnostics/checkpoint_inventory.json',inventory)
    for item in new:
        for role,rows,op in [('model_fit_probe',probe,'probe_diagnostic'),('model_val',d.data.role('model_val'),'model_diagnostic')]:
            d.scan(item,role,rows,op,sensitivity=item['head']=='B1' and role=='model_val')
    summaries=d.export();mv=[r for r in summaries if r['role']=='model_val']
    pool=[r for r in mv if r['head']=='B1' and (r['configuration_id'].startswith('LOSS_COMPARISON') if route=='LOSS_COMPARISON' else True)]
    selection=choose_producer(pool);write(s.run/'selection/rescue_producer_selection.json',selection)
    s.complete('ONE_RESCUE',{'route':route,'selection':sha(s.run/'selection/rescue_producer_selection.json')})
    s.live(rescue_budget_active=False)
    s.complete('RESCUE_MODEL_FREEZE',{'producer':selection['selected']})
    if not selection['selected']:raise Stop('STOP_NO_USABLE_PRODUCER','唯一有限救援后仍无符合model_val资格的非零producer，不追加训练。')
    selected=selection['selected'];item=next(i for i in inventory if i['checkpoint_id']==selected['checkpoint_id']);recipe=selected['configuration_id']
    matching=[r for r in mv if r['configuration_id']==recipe] if route=='LOSS_COMPARISON' else mv
    write(s.run/'selection/standalone_selection.json',{h:standalone([r for r in matching if r['head']==h]) for h in ['B1','B3']})
    d.scan(item,'utility_val',d.data.role('utility_val'),'candidate_diagnostic',True);summaries=d.export()
    transfer=next(r for r in summaries if r['checkpoint_id']==item['checkpoint_id'] and r['role']=='utility_val');q=eligibility(transfer)
    write(s.run/'selection/producer_transfer_gate.json',{'passed':q!='NO_QUALIFIED_PRODUCER_ON_MODEL_VAL','scores':transfer,'qualification':q})
    s.complete('RESCUE_UTILITY_GATE',{'gate':sha(s.run/'selection/producer_transfer_gate.json')})
    if q=='NO_QUALIFIED_PRODUCER_ON_MODEL_VAL':raise Stop('STOP_PRODUCER_TRANSFER_GATE','唯一救援producer未通过utility_val迁移，其他新点未评分、不改选。')
    write(s.run/'selection/producer_freeze.json',{'producer':selected,'transfer':transfer,'route':route,'s_ref':sref,'protocol':s.ctx.protocol_sha256})
    s.complete('PRODUCER_FROZEN',{'freeze':sha(s.run/'selection/producer_freeze.json')})


def producer_setup(s,d):
    item=read(s.run/'selection/producer_freeze.json')['producer'];inventory=read(s.run/'diagnostics/checkpoint_inventory.json');entry=next(x for x in inventory if x['checkpoint_id']==item['checkpoint_id'])
    d.data.candidate=checkpoint_model(entry);d.data.candidate_hash=entry['checkpoint_sha256']
    return entry


def prepare_scales(s,d):
    if (s.run/'normalization_stats.json').exists():return read(s.run/'normalization_stats.json')
    stats={'v2_per_source':[],'U2_per_source':[],'e2_per_source':[],'active_fraction':[]}
    with s.device_job('UTILITY_CACHE_AND_FIT_ONLY_SCALES',read(s.run/'budget_plan.json')['cache_predicted_seconds']):
        for row in d.data.role('utility_fit'):
            s.guard();batch=d.data.batch([row['sample_id']],'label_scales',True);vs=[];us=[]
            for vi in range(7):
                lab=labels(batch['base'],batch['candidates'][:,vi],batch['target']);act=lab['active']
                if act.any():vs.append(float(lab['v'][act].square().mean()))
                us.append(float(lab['U'].square().mean()));stats['active_fraction'].append(float(act.float().mean()))
            if vs:stats['v2_per_source'].append(float(np.mean(vs)))
            stats['U2_per_source'].append(float(np.mean(us)));stats['e2_per_source'].append(float((batch['base']-batch['target']).square().mean()))
        if not stats['v2_per_source']:raise Stop('BLOCKED_ENGINEERING','全数据无active效用标签')
        scales={'s_v':max(math.sqrt(np.mean(stats['v2_per_source'])),1e-3),'s_U':max(math.sqrt(np.mean(stats['U2_per_source'])),1e-4),
           's_e2':max(float(np.mean(stats['e2_per_source'])),1e-4),'role':'utility_fit','source_count':443,'source_equal_weight':True,
           'candidate_sha256':d.data.candidate_hash,'roles_sha256':sha(s.run/'roles.jsonl')}
        write(s.run/'normalization_stats.json',scales);write(s.run/'normalization_fit_distribution.json',stats)
    return scales


def checkpoint_grid(s,d,method,model,step,scales):
    path=s.run/'selection/grid_parts'/method/('step_%06d.json'%step)
    if path.exists():return read(path)['scores']
    table={i:[] for i in range(len(grid(method)))};model.eval()
    with torch.no_grad():
        for start in range(0,136,8):
            batch=d.data.batch([r['sample_id'] for r in d.data.role('utility_val')[start:start+8]],'utility_select',method!='B4')
            if method=='B4':P,V=rgb_prior(batch['image']);candidate=model(batch['image'],batch['base'],P,V)
            else:candidate=batch['candidates'][:,0]
            base_metrics=metrics(batch['base'],batch['target']);base_lp=d.lpips(batch['base'],batch['target'])
            # Prediction cached once; grid changes only the specified decision rule.
            pred=None if method=='B4' else model(batch['image'],batch['base'],candidate,se2=scales['s_e2'])
            from .evaluation import apply_policy
            for i,policy in enumerate(grid(method)):
                out,alpha=apply_policy(method,policy,batch['base'],candidate,pred)
                met=metrics(out,batch['target']);lp=d.lpips(out,batch['target'])
                for n,m in enumerate(met):table[i].append({'psnr':m['psnr'],'ssim_delta':m['ssim']-base_metrics[n]['ssim'],'lpips_delta':float(lp[n]-base_lp[n])})
    scores=[{'step':step,'strategy_index':i,'policy':grid(method)[i],'return_base':grid(method)[i].get('return_base',False),
              **{k:float(np.mean([v[k] for v in rows])) for k in ['psnr','ssim_delta','lpips_delta']},'method':method,'n_images':136,'role':'utility_val'} for i,rows in table.items()]
    write(path,{'scores':scores,'producer_hash':d.data.candidate_hash,'scales_hash':sha(s.run/'normalization_stats.json')});model.train();return scores


def utility_matrix(s,d):
    producer_setup(s,d);scales=prepare_scales(s,d);plan=read(s.run/'budget_plan.json');remain=57600-read(s.run/'budget.json')['used_device_seconds']
    if 'UTILITY_CORE' not in s.state['completed'] and plan['full_matrix_training_and_validation_predicted_seconds']+plan['protected_final_seconds']>remain:
        raise Stop('INCONCLUSIVE_BUDGET','启动前完整11方法3000更新矩阵预算不足，不缩减对照。')
    allscores=[];inventory=read(s.run/'diagnostics/checkpoint_inventory.json');stand=read(s.run/'selection/standalone_selection.json')
    for method in METHOD_ORDER:
        directory=s.run/'checkpoints'/method;directory.mkdir(parents=True,exist_ok=True);selection_path=directory/'selection.json'
        ident=identity(s,method);rng=streams(20261007);rng['model_data']=streams(20261108)['data']
        model=initialize(Candidate if method=='B4' else lambda:Controller(method),rng['init']).to('cuda:0');opt,schedule=optimizer(model,3000)
        if selection_path.exists():
            rec=read(selection_path);selected=directory/rec['selected_file']
            if sha(selected)!=rec['selected_sha256'] or rec['identity']!=ident:raise ValueError('method checkpoint identity mismatch')
            allscores.extend(rec['all_grid_scores']);continue
        if method=='B4':
            src=next(i for i in inventory if i['checkpoint_id']==stand['B3']['checkpoint_id']);model.load_state_dict(torch.load(src['path'],map_location='cpu')['model_state'],strict=True)
        latest=directory/'latest.pt';step=0;history=[]
        saved_extra={}
        if latest.exists():
            state=resume(latest,model,opt,schedule,ident,rng);step=state['global_step'];saved_extra=state['extra'];history=saved_extra['grid_scores']
        sampler=GroupSampler(d.data.role('utility_fit'),rng['data']);ms=GroupSampler(d.data.role('model_fit'),rng['model_data'])
        logs=saved_extra.get('unfinished_loss_window',[]);gradnorms=saved_extra.get('unfinished_gradient_window',[]);seen=set(saved_extra.get('seen_source_ids',[]))
        def recoverable_extra():
            return {'grid_scores':history,'seen_source_ids':sorted(seen),'unfinished_loss_window':logs,'unfinished_gradient_window':gradnorms}
        estimate=plan['full_matrix_training_and_validation_predicted_seconds']/11
        with s.device_job('utility_train_'+method,max(estimate*(3000-step)/3000,60)):
            try:
                for update in range(step,3001):
                    if update in [0,750,1500,2250,3000] and not any(h['step']==update for h in history):
                        history.extend(checkpoint_grid(s,d,method,model,update,scales))
                        save(directory/('step_%06d.pt'%update),model,opt,schedule,update,ident,rng,include_cuda=True,extra=recoverable_extra())
                    if update==3000:break
                    s.guard();opt.zero_grad(set_to_none=True)
                    if method=='B4':
                        ids=ms.sample(4)+sampler.sample(4);batch=d.data.batch(ids,'matched_train');P,V=rgb_prior(batch['image']);P,V,_=drop_prior(P,V,rng['missing_prior'])
                        out=model(batch['image'],batch['base'],P,V);mi=(out-batch['target']).square().flatten(1).mean(1)
                        freeze=read(s.run/'selection/producer_freeze.json')
                        if freeze.get('route')=='LOSS_COMPARISON' and freeze['producer']['configuration_id'].endswith('LOG'):
                            ref=freeze['s_ref'];loss=ref*torch.log((mi+1e-6)/(ref+1e-6)).mean()
                        else:loss=mi.mean()
                    else:
                        ids=sampler.sample(4);batch=d.data.batch(ids,'utility_train',True);view=torch.randint(1,7,(4,),generator=rng['view']).to('cuda:0');idx=torch.arange(4,device='cuda:0')
                        c=torch.stack([batch['candidates'][:,0],batch['candidates'][idx,view]],1)
                        if method=='O-NI':c=c[:,:1].repeat(1,2,1,1,1)
                        loss,parts=pair_objective(model,batch['image'],batch['base'],c,batch['target'],scales)
                    loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    if not torch.isfinite(loss) or not torch.isfinite(gn):raise RuntimeError('nonfinite utility loss or gradient')
                    lr=opt.param_groups[0]['lr'];opt.step();schedule.step();step=update+1;logs.append(float(loss));gradnorms.append(float(gn));seen.update(ids)
                    s.state['formal_training_updates_v2']=s.state.get('formal_training_updates_v2',0)+1
                    if step%50==0:
                        append(directory/'training.jsonl',{'step':step,'loss_current':float(loss),'last50_mean':np.mean(logs),'last50_min':min(logs),'last50_max':max(logs),
                            'learning_rate':lr,'gradient_norm':float(gn),'clipped':float(gn)>1,'last50_gradient_norm_mean':float(np.mean(gradnorms)),
                            'last50_gradient_clip_fraction':float(np.mean(np.array(gradnorms)>1)),'source_presentations':step*(8 if method=='B4' else 4),
                            'view_presentations':step*8,'distinct_sources_cumulative':len(seen),'source_ids_current':ids});logs=[];gradnorms=[]
                        s.live(current_method=method,global_step=step)
                    if step%250==0:save(latest,model,opt,schedule,step,ident,rng,include_cuda=True,extra=recoverable_extra())
            finally:save(latest,model,opt,schedule,step,ident,rng,include_cuda=True,extra=recoverable_extra())
        chosen=choose_grid(history);filename='step_%06d.pt'%chosen['step'];rec={'identity':ident,'selected_file':filename,'selected_sha256':sha(directory/filename),'updates':3000,
            'selected_step':chosen['step'],'development_policy_hint':chosen['policy'],'all_grid_scores':history,'selection_role':'utility_val','network_only_frozen':True}
        write(selection_path,rec);allscores.extend(history);del model,opt,schedule
        if method=='O':s.complete('UTILITY_CORE',{'methods':['G1','R0','O']})
    csv_write(s.run/'selection/utility_checkpoint_grid_scores.csv',allscores)
    s.complete('UTILITY_REMAINING',{'methods':METHOD_ORDER,'scores':sha(s.run/'selection/utility_checkpoint_grid_scores.csv')})
    write(s.run/'selection/network_freeze_before_calibration.json',{'protocol':s.ctx.protocol_sha256,'source_snapshot':sha(s.run/'source_snapshot.json'),
        'producer':sha(s.run/'selection/producer_freeze.json'),'normalization':sha(s.run/'normalization_stats.json'),'roles':sha(s.run/'roles.jsonl'),
        'methods':{m:sha(s.run/'checkpoints'/m/'selection.json') for m in METHOD_ORDER}})
    s.complete('NETWORK_FREEZE',{'freeze':sha(s.run/'selection/network_freeze_before_calibration.json')})
