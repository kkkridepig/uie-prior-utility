"""Three-step real-parent gradient, full-null trajectory and PPU metric preflight."""
import argparse, copy, hashlib, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.model import V3Model
from scripts.cde_v3.sampling import sample
from mpa_diff.data.cache import static_cached
from mpa_diff.utils.io import read_image

def params_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); a=p.parse_args(); setup()
    assert sha(PARENT)==PARENT_HASH
    parent=torch.load(PARENT,map_location='cpu'); model=V3Model(parent,'C_BANK',20261004).to(a.device).eval()
    if a.device=='cpu':
        # Official plain QK softmax fallback: xformers build has no CPU kernel.
        for name,module in list(sys.modules.items()):
            if name.startswith('depth_anything_v2.') and name.endswith('.attention'):
                module.XFORMERS_AVAILABLE=False
    row=roles('adapter_fit','enhancer','C_BANK')[0]; cfg=model.config
    # Real model weights, real source training image; small CPU integration uses 32 pixels.
    size=[32,32] if a.device=='cpu' else [336,336]
    x=read_image(ROOT/cfg['data']['data_root']/row['image_path'],size)[None].to(a.device)
    static=static_cached(model.base.priors,x,cfg)
    cond,extras=model.prepare(x,static); encoded=model.encode(extras); t=torch.tensor([450],device=a.device); state=noise(x.shape,20261004,row['sample_id'],'train',0,a.device)
    before=params_hash(model.base)
    with torch.no_grad(): ref=model.base.denoiser(state,t,cond); zero=model.predict(state,t,cond,encoded,'all')
    assert torch.equal(ref,zero),'zero initialization'
    opt=torch.optim.Adam(model.adapters.parameters(),lr=1e-4)
    for _ in range(3):
        opt.zero_grad(); y=model.predict(state,t,cond,model.encode(extras),'all'); (y-x).square().mean().backward(); opt.step()
    grads={k:float(v.grad.abs().sum()) for k,v in model.adapters.named_parameters()}
    assert all(v>0 for v in grads.values()),'adapter gradient missing'
    assert before==params_hash(model.base),'frozen parameter/stat changed'
    null=lambda xt,idx,c:model.predict(xt,idx,c,None,'null')
    out,d=sample(null,cond,x.shape,model.schedule,row['sample_id'],101,20)
    baseline,db=sample(model.base.denoiser,cond,x.shape,model.schedule,row['sample_id'],101,20)
    assert torch.equal(out,baseline),'trained null differs from parent trajectory'
    result={'status':'passed','device':a.device,'device_name':torch.cuda.get_device_name(0) if a.device=='cuda' else 'CPU','torch':torch.__version__,'parent_frozen_sha256':before,'parent_unchanged':True,'gradient_sums':grads,'zero_init_exact':True,'trained_null_DDIM20_exact':True,'shape':list(x.shape),'adapter_parameters':sum(p.numel() for p in model.adapters.parameters()),'perceptual_metric':'pending_independent_check'}
    write(RUN/('device_preflight.json' if a.device=='cuda' else 'cpu_preflight.json'),result); print(result)
if __name__=='__main__': main()
