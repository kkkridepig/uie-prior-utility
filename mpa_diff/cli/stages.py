"""Durable S2 queue: separate datasets / 3 independent seeds / frozen protocol."""
import argparse,copy,gc,json,os,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from mpa_diff.config import load_config
from mpa_diff.data.manifest import audit,read_manifest
from mpa_diff.engine.trainer import train
from mpa_diff.engine.checkpoint import load_model,source_hash
from mpa_diff.engine.evaluator import evaluate_rows
from mpa_diff.engine.reporting import update_report,report_path
from mpa_diff.utils.io import verified_weight,write_json,sha256

SEEDS=[20260927,20260928,20260929]


def data_spec(dataset,profile):
    return yaml.safe_load(Path('configs/data/'+dataset+'_'+profile+'.yaml').read_text())


def preflight(config,profile='grouped',datasets=('uieb','lsui')):
    errors=[];results={}
    for section,p,h in [('depth','checkpoint','checkpoint_sha256'),('loss','vgg_checkpoint','vgg_sha256')]:
        try:verified_weight(config[section][p],config[section][h])
        except (ValueError,FileNotFoundError) as e:errors.append(str(e))
    for dataset in datasets:
        spec=data_spec(dataset,profile)
        try:
            result=audit(spec['manifest'],spec['data_root'])
            expected=dict(zip(('train','val','test'),spec['counts']))
            if result['samples']!=sum(spec['counts']):raise ValueError('Pair count mismatch')
            if profile=='recon' and result['splits']!=expected:raise ValueError('Split count mismatch')
            if any(result['splits'].get(k,0)==0 for k in expected):raise ValueError('Empty split')
            if profile=='grouped' and (not spec['groups'] or 'grouped' not in spec['manifest']):raise ValueError('Group split must be named and auditable')
            result['group_map_sha256']=sha256(spec['groups']) if spec['groups'] else None
            results[dataset]=result
        except (ValueError,FileNotFoundError) as e:
            errors.append(dataset+': '+str(e));results[dataset]={'status':'missing_or_invalid','source':spec['source_url']}
    return {'ready':not errors,'errors':errors,'datasets':results,'profile':profile,'training_seeds':SEEDS,'steps_per_run':config['train']['total_steps'],'runs':len(datasets)*len(SEEDS),'results':None}


def execute(c,root,report,profile,datasets):
    # flock prevents duplicate queues from writing to the same run/checkpoint.
    import fcntl
    lock=(root/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An S2 queue already owns this output')
    protocol=Path('configs/protocols/paired.yaml')
    frozen={'protocol':yaml.safe_load(protocol.read_text()),'protocol_sha256':sha256(protocol),'datasets':report['datasets'],'source_sha256':source_hash(),'profile':profile}
    path=root/'frozen_protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen protocol/data/source changed; use a new output')
    write_json(path,frozen)
    jobs=[]
    for dataset in datasets:
        spec=data_spec(dataset,profile)
        for seed in SEEDS:
            cfg=copy.deepcopy(c);cfg['experiment'].update(seed=seed,name='mpa_recon_v1_'+dataset+'_'+profile)
            cfg['data'].update(manifest=spec['manifest'],data_root=spec['data_root']);out=root/dataset/('seed_'+str(seed));cfg['runtime']['output']=str(out)
            if not (out/'progress.json').exists():update_report(cfg,'queued')
            jobs.append((dataset,seed,cfg,out,spec))
    summaries=[]
    for dataset,seed,cfg,out,spec in jobs:
        if source_hash()!=frozen['source_sha256']:raise RuntimeError('Source changed while queue was running; refusing mixed-code experiment')
        write_json(root/'queue_status.json',{'status':'running','pid':os.getpid(),'current_dataset':dataset,'current_seed':seed,'completed_runs':len(summaries),'total_runs':len(jobs),'report':str(report_path(cfg))})
        try:
            # Resume from last without changing optimizer/RNG/data order.
            last=out/'last.pt';train(cfg,str(last) if last.exists() else None)
            update_report(cfg,'testing')
            model,state=load_model(out/'best.pt',cfg['runtime']['device'])
            test=[r for r in read_manifest(spec['manifest']) if r['split']=='test']
            result,_=evaluate_rows(model,test,spec['data_root'],cfg['data']['resize_hw'],seed,out/'test')
            entry={'dataset':dataset,'seed':seed,'checkpoint_step':state['step'],**result}
            summaries.append(entry);write_json(root/'runs.json',summaries);update_report(cfg,'completed',evaluation=entry)
            del model,state;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except BaseException as error:
            update_report(cfg,'failed',error=type(error).__name__+': '+str(error))
            write_json(root/'queue_status.json',{'status':'failed','pid':os.getpid(),'current_dataset':dataset,'current_seed':seed,'completed_runs':len(summaries),'error':str(error)})
            raise
    aggregated=[]
    for dataset in datasets:
        values=[r for r in summaries if r['dataset']==dataset]
        aggregated.append({'dataset':dataset,**{metric:{'mean':float(np.mean([r[metric] for r in values])),'sample_std':float(np.std([r[metric] for r in values],ddof=1))} for metric in ('mean_psnr','mean_ssim')}})
    write_json(root/'aggregate.json',aggregated)
    write_json(root/'queue_status.json',{'status':'completed','completed_runs':len(summaries),'total_runs':len(jobs)})
    Path('docs/experiments/S2_'+root.name+'_AGGREGATE.md').write_text('# S2 三种子汇总\n\n数据采用 '+profile+' 协议，与原论文划分不同。\n\n```json\n'+json.dumps(aggregated,indent=2,ensure_ascii=False)+'\n```\n\n逐图结果、失败例和各实验配置见 '+str(root)+'。仅对实际完成的三种子统计样本标准差；未实现的指标保持缺测。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/base/mpa_recon_v1.yaml');p.add_argument('--execute',action='store_true');p.add_argument('--output',default='runs/s2_grouped_v1');p.add_argument('--profile',choices=['grouped','recon'],default='grouped');p.add_argument('--datasets',nargs='+',choices=['uieb','lsui'],default=['uieb','lsui']);a=p.parse_args()
    c=load_config(a.config);root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    report=preflight(c,a.profile,a.datasets);write_json(root/'readiness.json',report)
    print(json.dumps({'ready':report['ready'],'errors':report['errors'],'splits':{k:v.get('splits') for k,v in report['datasets'].items()}},indent=2),flush=True)
    if not a.execute:return
    if not report['ready']:raise SystemExit('S2 blocked by listed dependencies; no synthetic substitution')
    execute(c,root,report,a.profile,a.datasets)
if __name__=='__main__':main()
