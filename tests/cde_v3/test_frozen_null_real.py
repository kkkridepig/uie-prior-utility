"""Fast CPU synthetic-conditioned topology check; real PPU acceptance is separate."""
import copy
import torch
from mpa_diff.config import DEFAULT
from mpa_diff.priors.provider import MPADiff
from scripts.cde_v3.model import V3Model

def test_hooks_target_quarter_eighth_and_persist_null_after_training():
    cfg=copy.deepcopy(DEFAULT); cfg['depth']['provider']='synthetic_fixture'
    parent=MPADiff(cfg); cp={'config':cfg,'model':parent.state_dict()}
    m=V3Model(cp,'C_BANK',20261004); m.eval(); image=torch.rand(1,3,32,32); state=torch.randn_like(image); t=torch.tensor([99])
    cond,extra=m.prepare(image); enc=m.encode(extra)
    before=m.base.denoiser(state,t,cond).detach()
    assert torch.equal(before,m.predict(state,t,cond,enc,'all'))
    opt=torch.optim.Adam(m.adapters.parameters(),lr=.001)
    for _ in range(3):
        opt.zero_grad(); y=m.predict(state,t,cond,m.encode(extra),'all'); y.square().mean().backward(); opt.step()
    assert torch.equal(before,m.predict(state,t,cond,None,'null'))
    assert all(p.grad is None for p in m.base.parameters())
    observed=[]
    def hook(mod,inp,out): observed.append(out.shape[-2:])
    hooks=[m.base.denoiser.encoder.layers[j].register_forward_hook(hook) for j in (4,6)]
    m.predict(state,t,cond,m.encode(extra),'all')
    for h in hooks: h.remove()
    assert observed==[(8,8),(4,4)]
