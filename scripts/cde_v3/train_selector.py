"""Same-capacity utility MSE, winner CE and shuffled-label input-ablation selectors."""
import argparse,signal,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import Selector

stop_requested=False

def request_stop(*args):
    global stop_requested
    stop_requested=True

def save_selector(out,identity,model,optimizer,step):
    tmp=out/'last.pt.tmp'
    torch.save({'identity':identity,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng_policy':'stateless_gate_data_by_finetune_seed_step'},tmp)
    tmp.replace(out/'last.pt')

def selector_step(model,opt,image,summaries,stats,target,kind):
    pred=model(image,summaries,stats)
    target=target.detach()
    loss=torch.nn.functional.cross_entropy(pred,target.argmax(1)) if kind=='WINNER_CE' else (pred[:,1:]-target[:,1:]).square().mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite selector loss')
    opt.zero_grad(); loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite selector gradient')
    opt.step()
    return loss.item()

def choose(scores,stats,threshold=0.,kind='UTILITY'):
    if kind=='WINNER_CE' and threshold!=0: raise ValueError('CE logits are not decibels')
    valid=stats[:,:3]>0
    allowed=torch.cat((torch.ones_like(valid[:,:1]),valid,valid.any(1,keepdim=True)),1)
    scores=scores.masked_fill(~allowed,float('-inf'))
    selected=scores.argmax(1)
    if threshold: selected=torch.where(scores.max(1).values>=threshold,selected,torch.zeros_like(selected))
    return selected

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--labels',required=True); p.add_argument('--kind',choices=['UTILITY','WINNER_CE','SHUFFLED'],required=True); p.add_argument('--finetune-seed',type=int,required=True); p.add_argument('--output',required=True); a=p.parse_args(); setup()
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    assert a.finetune_seed in (20261004,20261005,20261006)
    data=torch.load(a.labels,map_location='cpu'); assert data['identity']['role']=='route_fit' and data['identity']['allowed_use']=='selector_optimization'
    assert data['identity']['protocol_sha256']==sha(RUN/'protocol_frozen.yaml')
    assert data['identity']['finetune_seed']==a.finetune_seed
    for field,path in [('sampler_sha256','scripts/cde_v3/sampling.py'),('preprocess_sha256','mpa_diff/utils/io.py'),('inference_runtime_sha256','scripts/cde_v3/inference.py')]:
        assert data['identity'][field]==sha(ROOT/path),'Stale utility identity: '+field
    entries=data['entries']; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    image=torch.stack([r['image64'] for r in entries]).cuda(); summaries=torch.stack([r['summaries'] for r in entries]).cuda(); stats=torch.stack([r['stats'] for r in entries]).cuda(); u=torch.stack([r['utility'] for r in entries]).cuda()
    if a.kind=='SHUFFLED': u=u[torch.randperm(len(entries),generator=generator(a.finetune_seed,'shuffle_vectors')).cuda()]
    torch.manual_seed(key(a.finetune_seed,'gate_init')); model=Selector().cuda(); opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    ident={'label_sha256':sha(a.labels),'bank_sha256':data['identity']['bank_sha256'],'kind':a.kind,'finetune_seed':a.finetune_seed,'gate_seed':key(a.finetune_seed,'gate_init'),'protocol_sha256':sha(RUN/'protocol_frozen.yaml'),'code_sha256':sha(Path(__file__)),'updates':2000}
    start=0
    if (out/'last.pt').exists():
        cp=torch.load(out/'last.pt',map_location='cpu'); assert cp['identity']==ident; model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer']); start=cp['step']
    for step in range(start,2000):
        if deadline() or stop_requested:
            save_selector(out,ident,model,opt,step)
            write(out/'progress.json',{'step':step,'target':2000,'status':'safely_stopped'})
            sys.exit(75)
        idx=torch.randint(len(entries),(32,),generator=generator(a.finetune_seed,'gate_data',step)).cuda()
        loss=selector_step(model,opt,image[idx],[summaries[idx,i] for i in range(3)],stats[idx],u[idx],a.kind)
        append(out/'train.jsonl',{'step':step+1,'loss':loss})
        if (step+1)%100==0:
            save_selector(out,ident,model,opt,step+1)
            write(out/'progress.json',{'step':step+1,'target':2000,'status':'complete' if step==1999 else 'training'})
    write(out/'identity.json',dict(ident,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(out/'last.pt')))
if __name__=='__main__': main()
