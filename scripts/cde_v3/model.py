"""Independent C_ADD_BANK_SELECT_V3: frozen parent with true cross-token attention."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from mpa_diff.priors.provider import MPADiff
from mpa_diff.priors.kernels import sobel
from scripts.cde_v3.common import MODES,key


def coords(h,w,device,dtype):
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=device,dtype=dtype),torch.linspace(-1,1,w,device=device,dtype=dtype),indexing='ij')
    return torch.stack((x,y),-1).reshape(1,h*w,2)

def pool_tokens(x,valid,mass=False):
    x=torch.where(valid.expand_as(x),torch.nan_to_num(x),torch.zeros_like(x))
    h,w=x.shape[-2:]; size=(min(16,h),min(16,w))
    if mass:
        if h%size[0] or w%size[1]: raise ValueError('Color bin mass pooling requires divisible bins')
        out=F.avg_pool2d(x,(h//size[0],w//size[1]))*(h//size[0])*(w//size[1])
    else: out=F.adaptive_avg_pool2d(x,size)
    # Any invalid pixels in a bin invalidate that token (conservative pooling).
    mask=(F.adaptive_avg_pool2d(valid.float(),size)>=1).flatten(2).squeeze(1)
    pos=coords(*size,x.device,x.dtype).expand(x.shape[0],-1,-1)
    return torch.cat((out.flatten(2).transpose(1,2),pos),-1),mask


def fields(cond,p,kind='C_BANK'):
    image=cond['image']; b,_,h,w=image.shape
    expand=lambda x:x.expand(b,x.shape[1],h,w)
    f=p.fields; raw=f['unclipped']; d=p.depth.physical_coordinate
    phys=torch.cat((p.physical_image,d,expand(f['kappa_D']),expand(f['kappa_B']),expand(f['A']),f['t_D'],f['b']),1)
    v=p.depth.valid_mask.bool() & torch.isfinite(phys).all(1,keepdim=True) & (f['t_D']>=1e-6).all(1,keepdim=True)
    hist=p.histogram
    hv=(torch.isfinite(hist)&(hist>=0)).flatten(1).all(1) & (torch.nan_to_num(hist).sum((1,2,3))>0)
    hf=cond['highfreq']; fv=torch.isfinite(hf).all(1,keepdim=True)
    finite_raw=torch.isfinite(raw).all(1,keepdim=True)
    clipfrac=(((raw<0)|(raw>1)).any(1,keepdim=True)&finite_raw).float().sum((1,2,3))/finite_raw.float().sum((1,2,3)).clamp_min(1)
    phys=torch.cat((phys,v.float()),1)
    inputs=[phys,hist,hf]; masks=[v,hv[:,None,None,None].expand(b,1,*hist.shape[-2:]),fv]
    stats=torch.stack((v.float().mean((1,2,3)),hv.float(),fv.float().mean((1,2,3)),clipfrac),1)
    if kind=='C_RGB_CONTROL':
        # Same actively used parameter topology, but all key inputs come only from RGB.
        # Channel tiling uses no extra learned parameters and never reads target or priors.
        inputs=[image.repeat(1,7,1,1)[:,:20],image,image.repeat(1,11,1,1)[:,:32]]
        masks=[torch.isfinite(z).all(1,keepdim=True) for z in inputs]
    return inputs,masks,stats

class CrossInjection(nn.Module):
    def __init__(self,channels):
        super().__init__(); self.heads=4; self.width=64
        self.in_proj=nn.Linear(channels,64); self.xy=nn.Linear(2,64,bias=False); self.qnorm=nn.LayerNorm(64); self.query=nn.Linear(64,64,bias=False)
        self.tokens=nn.ModuleList([nn.Sequential(nn.Linear(c+2,64),nn.SiLU(),nn.Linear(64,64)) for c in (20,3,32)])
        self.norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(3)])
        self.keys=nn.ModuleList([nn.Linear(64,64,bias=False) for _ in range(3)])
        self.values=nn.ModuleList([nn.Linear(64,64,bias=False) for _ in range(3)])
        self.outputs=nn.ModuleList([nn.Linear(64,channels,bias=False) for _ in range(3)])
        for out in self.outputs: nn.init.zeros_(out.weight)
    def encode(self,inputs,masks,rgb=False):
        result=[]
        for k in range(3):
            raw,mask=pool_tokens(inputs[k],masks[k],mass=k==1 and not rgb)
            token=self.tokens[k](raw); token=torch.where(mask[:,:,None],token,torch.zeros_like(token))
            result.append((token,mask))
        return result
    def forward(self,h,index,encoded,mode):
        if mode=='null': return h
        b,c,hh,ww=h.shape; qh,qw=min(16,hh),min(16,ww)
        q=F.adaptive_avg_pool2d(h,(qh,qw)).flatten(2).transpose(1,2)
        freqs=torch.exp(torch.arange(32,device=h.device,dtype=h.dtype)*(-math.log(10000)/31))
        phase=index.to(h.dtype)[:,None]*freqs[None]
        te=torch.cat((phase.sin(),phase.cos()),-1)[:,None,:]
        q=self.query(self.qnorm(self.in_proj(q)+self.xy(coords(qh,qw,h.device,h.dtype))+te)).view(b,-1,4,16).transpose(1,2)
        total=torch.zeros_like(h); n=h.new_zeros(b,1,1,1)
        for k in (range(3) if mode=='all' else [MODES.index(mode)-1]):
            token,mask=encoded[k]; valid=mask.any(1); z=self.norms[k](token)
            keys=self.keys[k](z).view(b,-1,4,16).transpose(1,2); vals=self.values[k](z).view(b,-1,4,16).transpose(1,2)
            logits=(q@keys.transpose(-1,-2))/4
            logits=logits.masked_fill(~mask[:,None,None,:],-1e4)
            attn=logits.softmax(-1)*mask[:,None,None,:].to(logits.dtype)
            mix=(attn@vals).transpose(1,2).reshape(b,qh*qw,64)
            update=self.outputs[k](mix).transpose(1,2).reshape(b,c,qh,qw)
            update=F.interpolate(update,size=(hh,ww),mode='bilinear',align_corners=False)*valid[:,None,None,None]
            total=total+update; n=n+valid[:,None,None,None]
        return h+total/n.clamp_min(1)

class V3Model(nn.Module):
    def __init__(self,parent,branch,seed):
        super().__init__(); self.branch=branch; self.config=parent['config']; self.mode='null'; self.context=None; self.index=None
        # Initialization scoped separately from all data/t/noise/dropout streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(key(seed,'adapter_init'))
            self.base=MPADiff(self.config); self.base.load_state_dict(parent['model'],strict=True)
            self.adapters=nn.ModuleList([CrossInjection(96),CrossInjection(128)]) if branch.startswith('C_') else nn.ModuleList()
            if branch.startswith('D_'):
                self.phase_adapter=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1,bias=False))
                nn.init.zeros_(self.phase_adapter[-1].weight)
                from scripts.cde_v3.common import RUN,read
                self.register_buffer('frozen_rms',torch.tensor(read(RUN/'d_rms.json')['rms'][branch]).view(1,3,1,1))
        self.base.requires_grad_(False).eval()
        if branch.startswith('BASE_CONT'):
            self.base.denoiser.requires_grad_(True); self.base.highfreq.requires_grad_(True); self.base.priors.beta.requires_grad_(True)
        self.handles=[]
        if self.adapters:
            for i,j in enumerate((4,6)):
                def hook(module,args,output,i=i):
                    if self.mode=='null': return output
                    return self.adapters[i](output,self.index,self.context[i],self.mode)
                self.handles.append(self.base.denoiser.encoder.layers[j].register_forward_hook(hook))
    @property
    def schedule(self): return self.base.schedule
    def train(self,mode=True):
        super().train(mode); self.base.eval()
        if self.branch.startswith('BASE_CONT') and mode:
            self.base.denoiser.train(); self.base.highfreq.train(); self.base.priors.beta.train()
        return self
    def prepare(self,x,static=None):
        if self.branch.startswith('BASE_CONT'): cond,p=self.base.condition(x,static)
        else:
            with torch.no_grad(): cond,p=self.base.condition(x,static)
        extras=fields(cond,p,self.branch)
        if self.branch.startswith('D_'):
            kind={'D_SOBEL_STD':'sobel','D_PHASE_STD':'phase','D_SOFTPHASE_STD':'softphase'}[self.branch]
            feature,_=phase_feature(x,kind)
            cond=dict(cond,highfreq=cond['highfreq']+self.phase_adapter(feature/self.frozen_rms.clamp_min(.001)))
        return cond,extras
    def encode(self,extras): return [a.encode(*extras[:2],rgb=self.branch=='C_RGB_CONTROL') for a in self.adapters]
    def predict(self,x,t,cond,encoded=None,mode='null'):
        self.mode=mode; self.index=t; self.context=encoded
        try: return self.base.denoiser(x,t,cond)
        finally: self.mode='null'; self.context=None; self.index=None
    def delta_state(self):
        if self.branch.startswith('D_'): return self.phase_adapter.state_dict()
        if self.branch.startswith('BASE_CONT'):
            return {k:v for k,v in self.state_dict().items() if k.startswith(('base.denoiser.','base.highfreq.','base.priors.beta.'))}
        return self.adapters.state_dict()
    def load_delta(self,state):
        if self.branch.startswith('D_'):
            self.phase_adapter.load_state_dict(state,strict=True); return
        if self.branch.startswith('BASE_CONT'):
            result=self.load_state_dict(state,strict=False)
            if result.unexpected_keys: raise ValueError(result.unexpected_keys)
        else: self.adapters.load_state_dict(state,strict=True)

class Selector(nn.Module):
    def __init__(self):
        super().__init__(); self.cnn=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU())
        self.summary=nn.ModuleList([nn.Sequential(nn.Linear(128,16),nn.SiLU()) for _ in range(3)])
        self.head=nn.Sequential(nn.Linear(116,64),nn.SiLU(),nn.Linear(64,4))
    def forward(self,image,token_summaries,stats):
        z=self.cnn(F.interpolate(image,(64,64),mode='bilinear',align_corners=False)).mean((2,3))
        z=torch.cat([z]+[m(s)*(stats[:,i:i+1]>0) for i,(m,s) in enumerate(zip(self.summary,token_summaries))]+[stats],1)
        return torch.cat((z.new_zeros(z.shape[0],1),self.head(z)),1)

def token_summary(encoded):
    out=[]
    for token,mask in encoded:
        n=mask.sum(1,keepdim=True).clamp_min(1).float(); m=(token*mask[:,:,None]).sum(1)/n
        var=((token-m[:,None]).square()*mask[:,:,None]).sum(1)/n
        out.append(torch.cat((m,var.sqrt()),1).detach())
    return out

def phase_feature(x,kind):
    if kind=='sobel': z=sobel(x); return z-z.mean((-2,-1),keepdim=True),{}
    centered=x.float()-x.float().mean((-2,-1),keepdim=True)
    constant=(x.amax((-2,-1),keepdim=True)==x.amin((-2,-1),keepdim=True))
    centered=torch.where(constant,torch.zeros_like(centered),centered)
    fft=torch.fft.fft2(centered,norm='ortho'); fft=fft.clone(); fft[...,0,0]=0
    amp=fft.abs(); non_dc=amp.flatten(-2)[...,1:]
    tau=non_dc.median(-1).values.clamp_min(1e-6)[...,None,None]
    if kind=='softphase': unit=fft/(amp.square()+tau.square()).sqrt()
    else: unit=torch.where(amp>1e-6,fft/amp.clamp_min(1e-6),torch.zeros_like(fft))
    result=torch.fft.ifft2(unit,norm='ortho'); real=result.real
    return real-real.mean((-2,-1),keepdim=True),{'tau':tau,'imaginary_max':result.imag.abs().max()}
