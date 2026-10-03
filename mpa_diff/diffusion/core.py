import torch
from torch import nn

class Schedule(nn.Module):
    def __init__(self, steps=1000, beta_start=1e-6, beta_end=.02):
        super().__init__()
        if not (steps>0 and 0<beta_start<=beta_end<1):
            raise ValueError('Invalid diffusion schedule')
        b=torch.linspace(beta_start,beta_end,steps,dtype=torch.float64)
        self.register_buffer('beta',b.float())
        self.register_buffer('alpha_bar_math',torch.cat((torch.ones(1,dtype=torch.float64),(1-b).cumprod(0))).float())
        self.steps=steps

    def coefficients(self,index):
        if bool(((index<0)|(index>=self.steps)).any()):
            raise ValueError('Network time index outside [0,T-1]')
        ab=self.alpha_bar_math[index+1].view(-1,1,1,1)
        return ab.sqrt(),(1-ab).sqrt()

    def q_sample(self,x,index,noise):
        a,s=self.coefficients(index)
        return a*x+s*noise

    def to_x0(self,pred,xt,index,kind='x0'):
        a,s=self.coefficients(index)
        if kind=='x0': return pred
        if kind=='epsilon': return (xt-s*pred)/a
        if kind=='v': return a*xt-s*pred
        raise ValueError('Unknown prediction type')

    def epsilon(self,xt,x0,index):
        a,s=self.coefficients(index)
        return (xt-a*x0)/s


def time_grid(total, steps):
    if not 1<=steps<=total: raise ValueError('Sampling steps must be in [1,T]')
    return torch.linspace(total,1,steps).round().long().tolist()+[0]


@torch.no_grad()
def sample(denoiser, condition, shape, schedule, generator, name='ddpm', steps=None, eta=0., initial=None, clip=False):
    steps=steps or schedule.steps
    if name not in ('ddpm','ddim'): raise ValueError('Only verified DDPM/DDIM supported')
    if name=='ddpm' and steps!=schedule.steps: raise ValueError('DDPM requires all training timesteps')
    grid=time_grid(schedule.steps,steps)
    x=initial.clone() if initial is not None else torch.randn(shape,device=schedule.beta.device,generator=generator)
    nfe=0
    for t,s in zip(grid[:-1],grid[1:]):
        index=torch.full((shape[0],),t-1,device=x.device,dtype=torch.long)
        x0=denoiser(x,index,condition); nfe+=1
        if clip: x0=x0.clamp(0,1)
        at,ass=schedule.alpha_bar_math[t],schedule.alpha_bar_math[s]
        if s==0:
            x=x0
        elif name=='ddim':
            eps=schedule.epsilon(x,x0,index)
            sigma=eta*((1-ass)/(1-at)*(1-at/ass)).sqrt()
            x=ass.sqrt()*x0+(1-ass-sigma.square()).clamp_min(0).sqrt()*eps
            if eta: x=x+sigma*torch.randn(x.shape,device=x.device,generator=generator)
        else:
            b=schedule.beta[t-1]; alpha=1-b
            mean=ass.sqrt()*b/(1-at)*x0+alpha.sqrt()*(1-ass)/(1-at)*x
            variance=b*(1-ass)/(1-at)
            x=mean+variance.sqrt()*torch.randn(x.shape,device=x.device,generator=generator)
    return x, {'solver':name,'nfe':nfe,'math_timesteps':grid,'eta':eta,'prediction_type':'x0','variance':'fixed_small','clip_intermediate_x0':clip}
