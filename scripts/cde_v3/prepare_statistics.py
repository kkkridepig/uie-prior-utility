"""Freeze training-only high-frequency RMS and D feature RMS; no development data."""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import V3Model,phase_feature
from mpa_diff.utils.io import read_image
from mpa_diff.priors.kernels import haar,sobel

@torch.no_grad()
def main():
    setup(); parent=torch.load(PARENT,map_location='cpu'); model=V3Model(parent,'C_BANK',20261004).cuda().eval(); cfg=model.config
    total={k:torch.zeros(3,device='cuda') for k in ['D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']}; hf=torch.zeros(32,device='cuda'); n=0; tau=[]; imag=[]
    rows=roles('adapter_fit','enhancer','C_BANK')
    for r in rows:
        if deadline(): sys.exit(75)
        x=read_image(ROOT/cfg['data']['data_root']/r['image_path'],cfg['data']['resize_hw'])[None].cuda()
        for name,kind in [('D_SOBEL_STD','sobel'),('D_PHASE_STD','phase'),('D_SOFTPHASE_STD','softphase')]:
            feat,diag=phase_feature(x,kind); total[name]+=feat.square().mean((0,2,3))
            if diag: tau.extend(diag['tau'].flatten().cpu().tolist()); imag.append(diag['imaginary_max'].item())
        feature=model.base.highfreq(torch.cat((torch.nn.functional.interpolate(haar(x)[1],x.shape[-2:],mode='bilinear',align_corners=False),sobel(x)),1))
        hf+=feature.square().mean((0,2,3)); n+=1
    result={'rms':{k:(v/n).sqrt().clamp_min(.001).cpu().tolist() for k,v in total.items()},'highfreq_rms':(hf/n).sqrt().cpu().tolist(),'training_role':'adapter_fit','rows_sha256':digest(rows),'samples':n,'tau_min':min(tau),'tau_max':max(tau),'imaginary_max':max(imag),'constant_features_zeroed':True,'parent_hash':PARENT_HASH}
    if (RUN/'d_rms.json').exists(): assert read(RUN/'d_rms.json')==result
    else: write(RUN/'d_rms.json',result)
    print({k:v for k,v in result.items() if k!='highfreq_rms'})
if __name__=='__main__': main()
