"""Four real-output PPU updates vs two + fresh-process two, engineering only."""
import argparse
import torch
from pathlib import Path
from ..checkpoint import save,resume
from ..training import streams,initialize,optimizer,pair_objective
from ..models.controls import Controller


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['continuous','first','resume']);p.add_argument('--directory',required=True);a=p.parse_args()
    directory=Path(a.directory);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    fixture=torch.load(directory/'fixture.pt',map_location='cpu');ident=fixture['identity'];rng=streams(20261007)
    model=initialize(lambda:Controller('O'),rng['init']).to('cuda:0');opt,sched=optimizer(model,3000);start=0;seen=[]
    if a.mode=='resume':
        st=resume(directory/'first.pt',model,opt,sched,ident,rng);start=st['global_step'];seen=st['extra']['seen']
    b={k:v.to('cuda:0') for k,v in fixture['batch'].items()}
    for step in range(start,2 if a.mode=='first' else 4):
        ids=torch.randperm(4,generator=rng['data']);view=torch.randint(1,7,(4,),generator=rng['view']);seen.append({'source_order':ids.tolist(),'views':view.tolist(),'lr':opt.param_groups[0]['lr']})
        ids=ids.to('cuda:0');vv=view.to('cuda:0');c=torch.stack([b['candidates'][ids,0],b['candidates'][ids,vv]],1)
        opt.zero_grad(set_to_none=True);loss,_=pair_objective(model,b['image'][ids],b['base'][ids],c,b['target'][ids],fixture['scales'])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step()
    save(directory/(a.mode+'.pt'),model,opt,sched,2 if a.mode=='first' else 4,ident,rng,include_cuda=True,extra={'seen':seen,'engineering_fixture_only':True})


if __name__=='__main__':main()
