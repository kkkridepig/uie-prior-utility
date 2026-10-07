from uie_next.evaluation import VerifiedLPIPS
from uie_next.records import ROOT
import torch


def test_T36_real_lpips_local_official_weights():
    metric=VerifiedLPIPS(ROOT/'weights/cde_v3/vgg16-397923af.pth')
    assert metric.identity['vgg16_sha256'].startswith('397923af')
    torch.manual_seed(36);x=torch.rand(1,3,32,32)
    same=metric(x,x);different=metric(x,1-x)
    assert same.abs().max()<1e-7 and different.min()>0
