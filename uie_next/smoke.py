"""Real-device head tests; explicitly not an official backbone integration."""
import copy
import time
import torch
from .budget import DeviceBudget
from .checkpoint import atomic_torch
from .models.candidate import Candidate
from .models.controls import Controller
from .priors.heuristic import make_prior
from .training import ORDER,pair_objective,optimizer
from .records import write


def local_head_smoke(run):
    if not torch.cuda.is_available(): raise RuntimeError('PPU unavailable; do not silently time CPU')
    rows=[]
    with DeviceBudget(run) as budget:
        budget.start('synthetic_PPU_local_heads_only',180)
        device='cuda:0';torch.manual_seed(20261007);torch.cuda.manual_seed_all(20261007)
        image=torch.rand(1,3,256,256,device=device);base=(.1+.8*image).detach()
        target=(base+.04*torch.randn_like(base)).clamp(0,1).detach()
        candidates=(base[:,None]+.03*torch.randn(1,2,3,256,256,device=device)).clamp(0,1).detach()
        torch.cuda.synchronize();start=time.monotonic()
        for name in ['B1','B3']+ORDER:
            torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
            if name in ['B1','B3','B4']:
                m=Candidate().to(device);o,s=optimizer(m,1000)
                f=make_prior(image);p=f['P'];v=f['V']
                if name!='B1':
                    from .priors.heuristic import rgb_prior
                    p,v=rgb_prior(image)
                loss=(m(image,base,p,v)-target).square().mean()
            else:
                m=Controller(name).to(device);o,s=optimizer(m,1000)
                c=candidates if name!='O-NI' else candidates[:,0:1].repeat(1,2,1,1,1)
                loss,_=pair_objective(m,image,base,c,target,{'s_v':.04,'s_U':.003,'s_e2':.003})
            o.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm) or norm<=0:raise RuntimeError('nonfinite/no gradient '+name)
            o.step();s.step();torch.cuda.synchronize()
            rows.append({'method':name,'loss':float(loss),'gradient_norm':float(norm),'updates':1,
                         'seconds':time.monotonic()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'trainable_parameters':sum(p.numel() for p in m.parameters()),'official_backbone_used':False,
                         'source':'synthetic','input_size':[256,256]})
            budget.guard(30)
            del m,o,s,loss
        elapsed=time.monotonic()-start
    result={'status':'LOCAL_HEADS_PASS','official_integration':'BLOCKED_BACKBONE','device':torch.cuda.get_device_name(0),
            'rows':rows,'seconds':elapsed,'not_a_throughput_budget_profile':True}
    write(run/'tests/ppu_local_heads.json',result)
    return result
