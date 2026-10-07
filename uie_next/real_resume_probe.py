"""Fresh-process PPU recovery of a head using real frozen official outputs."""
import argparse
from pathlib import Path

import torch

from .checkpoint import resume, save
from .models.controls import Controller
from .records import digest
from .training import initialize, optimizer, pair_objective, streams


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['continuous','first','resume']);parser.add_argument('--directory',required=True)
    args=parser.parse_args();directory=Path(args.directory)
    torch.set_num_threads(2);torch.manual_seed(20261007);torch.cuda.manual_seed_all(20261007)
    fixture=torch.load(directory/'recovery_fixture.pt',map_location='cpu')
    identity={'kind':'temporary_real_official_output_head_recovery','backbone_sha256':fixture['backbone_weight_sha256'],
              'policy_hash':fixture['baseline_policy_hash'],'not_formal_training':True}
    rng=streams(20261007);model=initialize(lambda:Controller('O'),rng['init']).to('cuda:0')
    opt,sched=optimizer(model,1000);start=0
    if args.mode=='resume':start=resume(directory/'first.pt',model,opt,sched,identity,rng)['global_step']
    batch={k:v.to('cuda:0') for k,v in fixture['utility_batch'].items()}
    end=10 if args.mode=='first' else 20
    for step in range(start,end):
        indices=torch.randperm(4,generator=rng['data']).to('cuda:0')
        opt.zero_grad(set_to_none=True)
        loss,_=pair_objective(model,batch['image'][indices],batch['base'][indices],batch['candidates'][indices],batch['target'][indices],fixture['scales'])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step()
    target='resumed.pt' if args.mode=='resume' else args.mode+'.pt'
    save(directory/target,model,opt,sched,end,identity,rng,True)
    print('PASS',args.mode,'global_step',end)


if __name__=='__main__':main()
