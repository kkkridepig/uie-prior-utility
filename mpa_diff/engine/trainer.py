import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from mpa_diff.priors.provider import MPADiff
from mpa_diff.priors.kernels import pad_image
from mpa_diff.data.manifest import audit,read_manifest
from mpa_diff.data.cache import static_cached
from mpa_diff.engine.losses import masked_mse,VGG19Loss
from mpa_diff.engine.checkpoint import seed_all,save_checkpoint,restore_rng,architecture_hash,provenance,source_hash
from mpa_diff.engine.evaluator import evaluate_rows
from mpa_diff.engine.reporting import update_report
from mpa_diff.utils.io import read_image,write_json,sha256


def train(config,resume=None,stop_after=None):
    c=config;t=c['train'];dev=c['runtime']['device'];out=Path(c['runtime']['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not resume: raise FileExistsError('Run exists; explicitly resume or select another output')
    seed_all(c['experiment']['seed'],c['runtime']['threads'])
    report=audit(c['data']['manifest'],c['data']['data_root'])
    rows=read_manifest(c['data']['manifest']); train_rows=[r for r in rows if r['split']=='train']; val_rows=[r for r in rows if r['split']=='val']
    if not train_rows or not val_rows:raise ValueError('Train and validation splits are required')
    if not c['runtime']['test_fixture'] and any(r['dataset']=='synthetic_test_fixture' for r in rows):raise ValueError('Synthetic fixtures forbidden in formal experiment')
    model=MPADiff(c).to(dev)
    perceptual=VGG19Loss(c['loss']['vgg_checkpoint'],c['loss']['vgg_sha256']).to(dev) if c['loss']['perceptual_weight'] else None
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=t['learning_rate'],betas=tuple(t['betas']),weight_decay=t['weight_decay'])
    def rate(step):
        if step<=t['decay_start']:return 1.
        f=min(1.,(step-t['decay_start'])/(t['total_steps']-t['decay_start']))
        return (1-f)+f*t['final_learning_rate']/t['learning_rate']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    order=list(range(len(train_rows)));random.shuffle(order);cursor=0;step=0;best=float('-inf')
    if resume:
        state=torch.load(resume,map_location='cpu')
        if state['architecture_hash']!=architecture_hash(c) or state['manifest_sha256']!=sha256(c['data']['manifest']):raise ValueError('Resume architecture/data mismatch')
        for key in ('train','loss','experiment','data'):
            if state['config'][key]!=c[key]:raise ValueError('Resume config mismatch: '+key)
        if state['source_sha256']!=source_hash():raise ValueError('Resume source changed; fork a new experiment explicitly')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step=state['step'];order=state['data_sampler']['order'];cursor=state['data_sampler']['cursor'];best=state['best_val_psnr'];restore_rng(state['rng'])
    write_json(out/'resolved_config.json',c);write_json(out/'provenance.json',provenance(c));write_json(out/'data_audit.json',report)
    groups={name:sum(p.numel() for p in module.parameters() if p.requires_grad) for name,module in [('beta',model.priors.beta),('denoiser',model.denoiser),('highfreq',model.highfreq),('depth',model.priors.depth)]}
    write_json(out/'parameter_groups.json',groups);print(json.dumps({'trainable_parameters':groups}),flush=True)
    end=min(t['total_steps'],stop_after) if stop_after else t['total_steps']
    curve=[]
    update_report(c,"training", {"step":step})
    while step<end:
        model.train();optimizer.zero_grad(set_to_none=True)
        if dev.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter();loss_value=0.;pixel_value=0.;outside=0.
        for _ in range(t['gradient_accumulation']):
            chosen=[]
            for _ in range(t['micro_batch']):
                if cursor==len(order):random.shuffle(order);cursor=0
                chosen.append(train_rows[order[cursor]]);cursor+=1
            x=torch.stack([read_image(Path(c['data']['data_root'])/r['image_path'],c['data']['resize_hw']) for r in chosen]).to(dev)
            y=torch.stack([read_image(Path(c['data']['data_root'])/r['reference_path'],c['data']['resize_hw']) for r in chosen]).to(dev)
            padded,info=pad_image(x);yp,_=pad_image(y)
            # Cache one lossless raw depth per input; recompute cheap static priors.
            static=static_cached(model.priors,padded,c)
            index=torch.randint(model.schedule.steps,(x.shape[0],),device=dev);noise=torch.randn_like(yp)
            state=model.schedule.q_sample(yp,index,noise)
            pred=model(padded,state,index,static)[...,:y.shape[-2],:y.shape[-1]]
            pixel=masked_mse(pred,y);loss=pixel
            if perceptual is not None:loss=loss+c['loss']['perceptual_weight']*perceptual(pred,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; refusing to continue')
            (loss/t['gradient_accumulation']).backward()
            loss_value+=loss.item()/t['gradient_accumulation'];pixel_value+=pixel.item()/t['gradient_accumulation'];outside+=((pred<0)|(pred>1)).float().mean().item()/t['gradient_accumulation']
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();scheduler.step();step+=1
        if dev.startswith('cuda'):torch.cuda.synchronize()
        record={'step':step,'loss':loss_value,'pixel_mse':pixel_value,'lr':optimizer.param_groups[0]['lr'],'prediction_outside_fraction':outside,'seconds':time.perf_counter()-start}
        if 10 < step <= 110: curve.append(record)
        if step == 110:
            seconds=float(np.mean([r['seconds'] for r in curve]))
            write_json(out/'timing.json', {'stable_steps':len(curve),'mean_step_seconds':seconds,'estimated_400000_hours':seconds*400000/3600,'resize_hw':c['data']['resize_hw'],'excludes':'validation/evaluation; observed steps may include depth cache misses'})
        with (out/'train.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        if step==1 or step%25==0:
            print(json.dumps(record),flush=True)
            update_report(c,"training",record)
        if step%t['validation_every']==0 or step==t['total_steps']:
            summary,_=evaluate_rows(model,val_rows,c['data']['data_root'],c['data']['resize_hw'],c['experiment']['seed'],out/('validation_%07d'%step),save_images=False)
            score=summary['mean_psnr'];score=float('inf') if score=='+inf' else score
            if score is not None and score>best:
                best=score;save_checkpoint(out/'best.pt',model,optimizer,scheduler,step,c,order,cursor,best)
        if step%t['checkpoint_every']==0 or step==end:save_checkpoint(out/'last.pt',model,optimizer,scheduler,step,c,order,cursor,best)
    stable=curve
    timing={'stable_steps':len(stable),'mean_step_seconds':float(np.mean([r['seconds'] for r in stable])) if stable else None,'estimated_400000_hours':float(np.mean([r['seconds'] for r in stable]))*400000/3600 if len(stable)==100 else None,'excludes':'validation/depth cache warmup/evaluation/other seeds'}
    if stable: write_json(out/'timing.json',timing)
    update_report(c,'training_complete_test_pending' if step==t['total_steps'] else 'paused_at_requested_step',record if curve else {'step':step})
    return {'step':step,'output':str(out),'timing':timing}
