import torch
from scripts.cde_v3.corruption import corrupt
from scripts.cde_v3.model import CrossInjection

def example():
    image=torch.rand(1,3,32,32); d=torch.rand(1,1,32,32); k=torch.ones_like(image)*.2; ambient=torch.ones_like(image)*.5
    t=(-k*d).exp(); b=ambient*(1-(-k*d).exp()); raw=(image-b)/t
    valid=torch.ones(1,1,32,32,dtype=torch.bool)
    physical=torch.cat((raw.clamp(0,1),d,k,k,ambient,t,b,valid.float()),1)
    hist=torch.ones(1,3,64,64)/(3*64*64); high=torch.randn(1,32,32,32)
    return image,([physical,hist,high],[valid,torch.ones(1,1,64,64,dtype=torch.bool),valid],torch.ones(1,4))

def test_missing_masks_summary_inputs_and_parent_not_mutated():
    image,extra=example(); old=extra[0][0].clone(); x,m,s=corrupt(extra,image,'physical_missing','a',torch.ones(1,32,1,1))
    assert torch.equal(extra[0][0],old) and not m[0].any() and x[0].count_nonzero()==0 and s[0,0]==0
    bank=CrossInjection(96); encoded=bank.encode(x,m)
    assert not encoded[0][1].any() and encoded[0][0].count_nonzero()==0

def test_shift_recomputes_field_and_noise_paired():
    image,extra=example(); rms=torch.ones(1,32,1,1)
    x,m,s=corrupt(extra,image,'depth_shift_005','a',rms)
    assert torch.allclose(x[0][:,13:16],(-x[0][:,4:7]*x[0][:,3:4]).exp())
    a=corrupt(extra,image,'highfreq_noise_03','a',rms); torch.rand(100)
    b=corrupt(extra,image,'highfreq_noise_03','a',rms)
    assert torch.equal(a[0][2],b[0][2])
