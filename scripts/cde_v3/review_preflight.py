"""PPU correctness and measured operation counters without training selection."""
import sys,json,time,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import V3Model,phase_feature
from scripts.cde_v3.metrics import Metrics
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image

@torch.no_grad()
def main():
    setup(); metric=Metrics('cuda'); x=noise((1,3,32,32),101,'metric_preflight').sigmoid().cuda(); y=x*.9
    identity=metric(x,x); a=metric(x,y); b=metric(y,x)
    assert abs(identity['lpips'])<1e-6 and abs(a['lpips']-b['lpips'])<1e-5 and a['lpips']>0
    write(RUN/'metric_preflight.json',{'status':'passed','identity_zero_lpips':identity['lpips'],'symmetry_error':abs(a['lpips']-b['lpips']),'nonzero_distance':a['lpips'],'device':'PPU-ZW810E','identity':read(RUN/'metric_identity.json')})
    parent=torch.load(PARENT,map_location='cpu'); counts={}
    for branch in ['C_BANK','C_RGB_CONTROL']:
        m=V3Model(parent,branch,20261004).cuda().eval(); cfg=m.config; row=roles('adapter_fit')[0]
        x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],cfg['data']['resize_hw'])[None].cuda(); static=static_cached(m.base.priors,x,cfg)
        cond,extra=m.prepare(x,static); count=[0]; handles=[]
        def hook(module,args,out):
            if isinstance(module,torch.nn.Linear): count[0]+=out.numel()*module.in_features
            elif isinstance(module,torch.nn.Conv2d): count[0]+=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            elif isinstance(module,torch.nn.ConvTranspose2d): count[0]+=args[0].numel()*(module.out_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        for mod in m.modules():
            if isinstance(mod,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.ConvTranspose2d)): handles.append(mod.register_forward_hook(hook))
        enc=m.encode(extra); encode=count[0]; count[0]=0
        xt=noise(x.shape,101,row['sample_id'],device='cuda'); t=torch.tensor([999],device='cuda'); m.predict(xt,t,cond,enc,'all')
        # Two matrix multiplications QK and AV per prior, head and spatial injection.
        attention_macs=2*2*3*4*256*256*16
        counts[branch]={'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in m.parameters()),'encoded_token_linear_macs':encode,'denoiser_linear_conv_macs_per_call':count[0],'attention_matmul_macs_per_call':attention_macs,'DDIM20_estimated_flops_excluding_norm_softmax_prior':2*(encode+20*(count[0]+attention_macs)),'definition':'MAC*2; excludes norm/activation/softmax/interpolation and prior extraction; not total hardware operations'}
        for h in handles: h.remove()
        del m; torch.cuda.empty_cache()
    assert abs(counts['C_BANK']['trainable_parameters']-counts['C_RGB_CONTROL']['trainable_parameters'])<=.05*counts['C_BANK']['trainable_parameters']
    write(RUN/'parameter_and_flop_audit.json',counts); print(counts)
if __name__=='__main__': main()
