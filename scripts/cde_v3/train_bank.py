"""Frozen-endpoint 5000-update bank/control training with stateless independent streams."""
import argparse, collections, copy, hashlib, json, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import V3Model
from mpa_diff.engine.losses import VGG19Loss
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image

BRANCHES=['BASE_CONT_V3','BASE_CONT_DATA_MATCHED','C_BANK','C_RGB_CONTROL','C_ALL_ONLY','D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']
stop_requested=False

def request_stop(*args):
    global stop_requested
    stop_requested=True

def select_batch(rows,seed,step,route_rows=None):
    rng=random.Random(key(seed,'data',step))
    if route_rows is None: return [rows[rng.randrange(len(rows))] for _ in range(4)]
    return [(lambda rr:rr[rng.randrange(len(rr))])(route_rows if rng.random()<2/9 else rows) for _ in range(4)]

def mode_at(seed,step,branch):
    if branch=='C_ALL_ONLY': return 'all'
    if branch.startswith(('BASE','D_')): return 'null'
    modes=list(MODES[1:]); random.Random(key(seed,'mode',step//4)).shuffle(modes)
    return modes[step%4]

def batch_random(shape,seed,step,ids,device):
    t=torch.tensor([key(seed,'diffusion_t',step,sid,i)%1000 for i,sid in enumerate(ids)],dtype=torch.long,device=device)
    n=torch.stack([noise(shape[1:],seed,sid,'training_noise_'+str(i),step,device) for i,sid in enumerate(ids)])
    return t,n

def tensor_hash(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def save(path,model,optimizer,step,identity,counts,last):
    state={'protocol':'prior_utility_cde_v3_20261004','delta':model.delta_state(),'optimizer':optimizer.state_dict(),'step':step,'identity':identity,'mode_counts':dict(counts),'last_record':last,'rng_policy':'stateless_by_seed_step_role_sample; next_step=step','rng_global_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp=Path(str(path)+'.tmp'); torch.save(state,tmp); tmp.replace(path)

def train(args):
    setup(); signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    if args.finetune_seed not in (20261004,20261005,20261006): raise ValueError('Unfrozen fine-tune seed')
    if args.steps>5000 or args.steps<0 or args.micro_batch not in (1,2,4): raise ValueError('Frozen endpoint/micro batch')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    parent=torch.load(PARENT,map_location='cpu'); assert sha(PARENT)==PARENT_HASH
    model=V3Model(parent,args.branch,args.finetune_seed).to(args.device); del parent
    cfg=model.config; rows=roles('adapter_fit','enhancer',args.branch)
    route=roles('route_fit','enhancer',args.branch) if args.branch=='BASE_CONT_DATA_MATCHED' else None
    loss_vgg=VGG19Loss(cfg['loss']['vgg_checkpoint'],cfg['loss']['vgg_sha256']).to(args.device)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5 if args.branch.startswith('BASE') else 1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    identity={'parent_sha256':PARENT_HASH,'branch':args.branch,'finetune_seed':args.finetune_seed,'parent_pretrain_seed':20260927,'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'data_roles_sha256':sha(RUN/'data_roles.json'),'implementation':train_identity(),'micro_batch':args.micro_batch,'schedule_endpoint':5000}
    if args.branch.startswith('D_'): identity['d_rms_sha256']=sha(RUN/'d_rms.json')
    step=0; counts=collections.Counter(); record={}; last=out/'last.pt'
    if last.exists():
        cp=torch.load(last,map_location='cpu'); assert cp['identity']==identity,'Resume identity changed'
        model.load_delta(cp['delta']); opt.load_state_dict(cp['optimizer']); step=cp['step']; counts.update(cp['mode_counts']); record=cp['last_record']
        if step==args.steps and (out/'progress.json').exists() and read(out/'progress.json').get('status')=='complete':
            print('Already complete with matching identity:',out); return 0
        if step>args.steps: raise ValueError('Refusing to rewind completed training')
    else:
        save(last,model,opt,0,identity,counts,record); os.link(last,out/'step_00000.pt')
    write(out/'run_config.json',dict(identity,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),max_updates=5000))
    # Verification output uses fixed train-only sample and a fixed t/noise, not source_dev selection.
    probe=rows[0]; px=read_image(ROOT/cfg['data']['data_root']/probe['image_path'],cfg['data']['resize_hw'])[None].to(args.device)
    ps=static_cached(model.base.priors,px,cfg); pt=torch.tensor([499],device=args.device); pn=noise(px.shape,17,probe['sample_id'],'profile_probe',499,args.device)
    with torch.no_grad():
        model.eval(); pc,pe=model.prepare(px,ps); initial=model.predict(pn,pt,pc,model.encode(pe),'all' if model.adapters else 'null').detach()
    start=time.time(); runtimes=[]
    while step<args.steps and not deadline() and not stop_requested:
        begin=time.perf_counter(); model.train(); opt.zero_grad(set_to_none=True)
        for g in opt.param_groups: g['lr']=(1e-5 if args.branch.startswith('BASE') else 1e-4)*(1-.9*step/5000)
        batch=select_batch(rows,args.finetune_seed,step,route); mode=mode_at(args.finetune_seed,step,args.branch)
        x=torch.stack([read_image(ROOT/cfg['data']['data_root']/r['image_path'],cfg['data']['resize_hw']) for r in batch]).to(args.device)
        y=torch.stack([read_image(ROOT/cfg['data']['data_root']/r['reference_path'],cfg['data']['resize_hw']) for r in batch]).to(args.device)
        ids=[r['sample_id'] for r in batch]; index,n=batch_random(y.shape,args.finetune_seed,step,ids,args.device)
        audit={'step':step+1,'sample_ids':ids,'t':index.cpu().tolist(),'noise_sha256':tensor_hash(n),'mode':mode,'dropout_seed':key(args.finetune_seed,'dropout',step)}
        total=0.
        for offset in range(0,4,args.micro_batch):
            sl=slice(offset,offset+args.micro_batch)
            # Dropout is isolated from all common streams; fixed micro-batch defines its identity.
            ds=key(args.finetune_seed,'dropout',step,offset); torch.manual_seed(ds)
            if args.device=='cuda': torch.cuda.manual_seed_all(ds)
            static=static_cached(model.base.priors,x[sl],cfg); cond,extra=model.prepare(x[sl],static)
            xt=model.schedule.q_sample(y[sl],index[sl],n[sl]); prediction=model.predict(xt,index[sl],cond,model.encode(extra),mode)
            loss=(prediction-y[sl]).square().mean()+.1*loss_vgg(prediction,y[sl])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
            (loss*args.micro_batch/4).backward(); total+=loss.item()*args.micro_batch/4
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient')
        opt.step(); step+=1; counts[mode]+=1
        if args.device=='cuda': torch.cuda.synchronize()
        seconds=time.perf_counter()-begin; runtimes.append(seconds)
        record={'step':step,'loss':total,'seconds':seconds,'lr':opt.param_groups[0]['lr'],'mode':mode}
        append(out/'train.jsonl',record); append(out/'rng_pairing.jsonl',audit)
        if step%10==0: write(out/'progress.json',{'status':'training','step':step,'target':args.steps,'last':record}); print(record,flush=True)
        if step%100==0 or step in (20,2000,5000):
            save(last,model,opt,step,identity,counts,record)
            if step in (2000,5000):
                dest=out/('step_%05d.pt'%step)
                if not dest.exists(): os.link(last,dest)
    save(last,model,opt,step,identity,counts,record)
    with torch.no_grad():
        model.eval(); pc,pe=model.prepare(px,ps); final=model.predict(pn,pt,pc,model.encode(pe),'all' if model.adapters else 'null')
    summary={'status':'complete' if step==args.steps else 'safely_stopped','step':step,'target':args.steps,'elapsed_seconds':time.time()-start,'median_step_seconds':float(np.median(runtimes)) if runtimes else None,'p95_step_seconds':float(np.quantile(runtimes,.95)) if runtimes else None,'probe_output_rms_change':(final-initial).square().mean().sqrt().item(),'mode_counts':dict(counts),'last':record,'checkpoint_sha256':sha(last)}
    write(out/'progress.json',summary); print(summary)
    return 0 if step==args.steps else 75

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--branch',choices=BRANCHES,required=True); p.add_argument('--finetune-seed',type=int,required=True); p.add_argument('--steps',type=int,default=5000); p.add_argument('--output',required=True); p.add_argument('--micro-batch',type=int,default=4); p.add_argument('--device',default='cuda'); a=p.parse_args(); sys.exit(train(a))
if __name__=='__main__': main()
