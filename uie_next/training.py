"""Shared optimizer, independent recoverable streams and paired head objectives."""
import hashlib
import math
import torch
from .losses import huber,utility_loss
from .math.utility import labels

ORDER=['B4','G0','G1','G2','F0','R0','O','O-NI','O-NP','O-NS','O-ND']


def streams(seed):
    result={}
    for name in ['init','data','view','missing_prior']:
        value=int.from_bytes(hashlib.sha256(('%d:%s'%(seed,name)).encode()).digest()[:8],'big')%(2**63-1)
        result[name]=torch.Generator().manual_seed(value)
    return result


def initialize(factory,init_stream):
    old=torch.get_rng_state()
    try:
        torch.set_rng_state(init_stream.get_state());model=factory()
        init_stream.set_state(torch.get_rng_state())
    finally:torch.set_rng_state(old)
    return model


class GroupSampler:
    def __init__(self,rows,generator):
        self.generator=generator;self.groups={}
        for r in rows:self.groups.setdefault(r['group_id'],[]).append(r['sample_id'])
        self.keys=sorted(self.groups)
        if not self.keys:raise ValueError('empty role')
        for g in self.keys:self.groups[g].sort()

    def sample(self,n):
        ids=[]
        for _ in range(n):
            group=self.keys[int(torch.randint(len(self.keys),(1,),generator=self.generator))]
            options=self.groups[group]
            ids.append(options[int(torch.randint(len(options),(1,),generator=self.generator))])
        return ids


def drop_prior(P,V,generator,probability=.2):
    missing=torch.rand(P.shape[0],generator=generator)<probability
    keep=(~missing).to(P.device,dtype=P.dtype).view(-1,1,1,1)
    return P*keep,V*keep,missing


def optimizer(model,total_updates):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    warm=max(1,math.ceil(.05*total_updates))
    def multiplier(step):
        update=step+1
        if update<=warm:return update/warm
        q=min(1.,(update-warm)/max(1,total_updates-warm))
        return .1+.9*.5*(1+math.cos(math.pi*q))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,multiplier)


def frozen_candidate(candidate):
    candidate.requires_grad_(False);candidate.eval();return candidate


def pair_objective(controller,image,base,candidates,target,scales):
    """Exactly two views per source; candidates and supervision are detached."""
    if candidates.shape[1]!=2:raise ValueError('two paired views required')
    c=candidates.detach();base=base.detach();target=target.detach()
    results=[];labs=[];eh=None
    if controller.name=='R0': eh=controller.net(torch.cat([image,base],1))
    for view in range(2):
        results.append(controller(image,base,c[:,view],se2=scales['s_e2'],error_hat=eh))
        with torch.no_grad(): labs.append(labels(base,c[:,view],target))
    pred={k:torch.stack([r[k] for r in results],1) for k in results[0]}
    lab={k:torch.stack([l[k] for l in labs],1) for k in labs[0]};lab['target']=target[:,None].expand_as(c)
    name=controller.name;se2=scales['s_e2']
    if name.startswith('G'):
        dec=(pred['output']-lab['target']).square().flatten(1).mean(-1).mean()/se2
        return dec,{'dec':dec.detach(),'dec_raw':(dec*se2).detach()}
    if name=='F0':
        e0=lab['target']-base[:,None];e1=lab['target']-c
        truth=torch.cat([e0.square().mean(2,keepdim=True),e1.square().mean(2,keepdim=True),(e0*e1).mean(2,keepdim=True)],2)
        proj=huber((pred['C']-truth)/se2).flatten(1).mean(-1).mean()
        mask=(lab['active'][:,0]|lab['active'][:,1]).float()
        delta=(pred['U_hat'][:,0]-pred['U_hat'][:,1])-(lab['U'][:,0]-lab['U'][:,1])
        pair=(huber(delta/scales['s_U'])*mask).flatten(1).sum(-1)/mask.flatten(1).sum(-1).clamp_min(1)
        valid=(mask.flatten(1).sum(-1)>0).float()
        pair=(pair*valid).sum()/valid.sum().clamp_min(1.)
        dec=(pred['output']-lab['target']).square().mean()/se2
        raw_pair=(huber(delta)*mask).flatten(1).sum(-1)/mask.flatten(1).sum(-1).clamp_min(1)
        raw_pair=(raw_pair*valid).sum()/valid.sum().clamp_min(1.)
        return proj+.25*pair+.1*dec,{'proj':proj.detach(),'pair':pair.detach(),'dec':dec.detach(),
                                  'proj_raw':huber(pred['C']-truth).mean().detach(),'pair_raw':raw_pair.detach(),'dec_raw':(dec*se2).detach()}
    loss,parts=utility_loss(pred,lab,scales,eta_pair=0 if name=='O-NP' else .25,eta_dec=0 if name=='O-ND' else .1)
    if name=='R0':
        proj=huber((eh-(target-base))/math.sqrt(se2)).flatten(1).mean(-1).mean()
        loss=proj+.25*parts['pair_live']+.1*parts['dec_live']
        parts['proj']=proj.detach()
        parts['proj_raw']=huber(eh-(target-base)).mean().detach()
    return loss,parts
