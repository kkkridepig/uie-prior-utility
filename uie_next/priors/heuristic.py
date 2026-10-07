import torch
import torch.nn.functional as F


def _blur(x, k=9):
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode='replicate'), k, stride=1)


def recompute(image, d, A, tau=None, V=None):
    k = image.new_tensor([2.4, 1.2, .8]).view(1, 3, 1, 1)
    tau = k * d if tau is None else tau
    t = torch.exp(-tau).clamp_min(.05)
    qraw = (image - A * (1 - t)) / t
    Q = qraw.clamp(0, 1)
    q = ((qraw < 0) | (qraw > 1)).to(image.dtype).mean(1, keepdim=True)
    V = torch.ones_like(q) if V is None else V
    P = torch.cat([Q, t, q], 1) * V
    return {'d': d, 'tau': tau, 't': t, 'A': A, 'Q': Q, 'q': q, 'P': P, 'V': V}


def make_prior(image):
    """Return the nominal seven-channel prior P and validity V (not depth)."""
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError('expected floating NCHW RGB')
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('invalid source image; never silently replace it')
    blur = _blur(image, 9)
    d = (.5 + blur[:,1:2] - blur[:,0:1] - (image-blur).abs().mean(1,keepdim=True)).clamp(0,1)
    A = blur.amax((2,3),keepdim=True).clamp(.05,.95)
    return recompute(image, d, A)


def interventions(image):
    """Field-level counterfactual views; each view recomputes all dependent fields."""
    nominal = make_prior(image)
    views = {'nominal': nominal}
    for name, scale in [('tau_075', .75), ('tau_125', 1.25)]:
        views[name] = recompute(image, nominal['d'], nominal['A'], nominal['tau']*scale)
    for name, off in [('ambient_rb_plus', [.04,0,-.04]), ('ambient_rb_minus',[-.04,0,.04])]:
        A = (nominal['A']+image.new_tensor(off).view(1,3,1,1)).clamp(.05,.95)
        views[name] = recompute(image, nominal['d'], A)
    for name, shift in [('shift_right_2',2),('shift_left_2',-2)]:
        d,V = shift_field(nominal['d'], shift, axis=-1)
        views[name] = recompute(image,d,nominal['A'],V=V)
    return views


def shift_field(d, pixels, axis=-1):
    out=torch.zeros_like(d); V=torch.zeros_like(d)
    dst=[slice(None)]*d.ndim;src=list(dst)
    if pixels>0: dst[axis]=slice(pixels,None);src[axis]=slice(None,-pixels)
    elif pixels<0: dst[axis]=slice(None,pixels);src[axis]=slice(-pixels,None)
    else: return d.clone(),torch.ones_like(d)
    out[tuple(dst)]=d[tuple(src)];V[tuple(dst)]=1
    return out,V


def stress_views(image):
    n=make_prior(image);views={}
    for name,s in [('tau_050',.5),('tau_150',1.5)]:
        views[name]=recompute(image,n['d'],n['A'],n['tau']*s)
    A=(n['A']+image.new_tensor([0,.06,0]).view(1,3,1,1)).clamp(.05,.95)
    views['ambient_green']=recompute(image,n['d'],A)
    d,V=shift_field(n['d'],4,axis=-2)
    views['shift_down_4']=recompute(image,d,n['A'],V=V)
    views['blur_d_9']=recompute(image,_blur(n['d']),n['A'])
    missing={k:v.clone() for k,v in n.items()};h,w=image.shape[-2:]
    if h<64 or w<64: raise ValueError('central 64 stress requires size >=64')
    for k in ['P','V']: missing[k][...,(h-64)//2:(h+64)//2,(w-64)//2:(w+64)//2]=0
    views['central_missing_64']=missing
    return views


def rgb_prior(image):
    return torch.cat([image,torch.ones_like(image),torch.zeros_like(image[:,:1])],1),torch.ones_like(image[:,:1])
