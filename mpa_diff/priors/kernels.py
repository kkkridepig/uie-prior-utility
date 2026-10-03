"""Independent implementation of architecture §4; no upstream code copied."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def pad_image(x, multiple=8, minimum=16):
    h, w = x.shape[-2:]
    ph = max(minimum, ((h + multiple - 1) // multiple) * multiple) - h
    pw = max(minimum, ((w + multiple - 1) // multiple) * multiple) - w
    mode = 'reflect' if ph < h and pw < w else 'replicate'
    return F.pad(x, (0, pw, 0, ph), mode=mode), {'hw': [h, w], 'padding': [0, pw, 0, ph], 'mode': mode}


def haar(x):
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError('Haar requires even dimensions; pad explicitly')
    a,b,c,d = x[...,::2,::2],x[...,::2,1::2],x[...,1::2,::2],x[...,1::2,1::2]
    return (a+b+c+d)/2, torch.cat(((a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2),1)


def inverse_haar(low, high):
    hx,hy,hd = high.chunk(3,1)
    out = low.new_empty(low.shape[0], low.shape[1], low.shape[2]*2, low.shape[3]*2)
    out[...,::2,::2]=(low+hx+hy+hd)/2
    out[...,::2,1::2]=(low-hx+hy-hd)/2
    out[...,1::2,::2]=(low+hx-hy-hd)/2
    out[...,1::2,1::2]=(low-hx-hy+hd)/2
    return out


def sobel(x):
    k = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]) / 8
    xp = F.pad(x,(1,1,1,1),mode='reflect' if min(x.shape[-2:]) > 1 else 'replicate')
    gx = F.conv2d(xp,k[None,None].repeat(x.shape[1],1,1,1),groups=x.shape[1])
    gy = F.conv2d(xp,k.t()[None,None].repeat(x.shape[1],1,1,1),groups=x.shape[1])
    return (gx.square()+gy.square()+1e-12).sqrt()


def histogram(x, bins=64, bandwidth=.02, max_size=150, mode='histogan_rgbuv'):
    if max(x.shape[-2:]) > max_size:
        x = F.interpolate(x,size=(max_size,max_size),mode='bilinear',align_corners=False)
    x = x.float().flatten(2)
    log = (x+1e-6).log()
    q = torch.linspace(-3,3,bins,device=x.device)[None,None,:]
    intensity = (x.square().sum(1)+1e-6).sqrt()
    planes=[]
    for c, other in enumerate(((1,2),(0,2),(0,1))):
        if mode == 'histogan_rgbuv':
            u,v,w = log[:,c]-log[:,other[0]],log[:,c]-log[:,other[1]],intensity
        elif mode == 'thesis_shared_uv_rgb_weight':
            u,v,w = log[:,0]-log[:,1],log[:,2]-log[:,1],x[:,c]
        else:
            raise ValueError('Unknown histogram mode')
        h=x.new_zeros(x.shape[0],bins,bins)
        for start in range(0,x.shape[-1],4096):
            ku = 1/(1+((u[:,start:start+4096,None]-q)/bandwidth).square())
            kv = 1/(1+((v[:,start:start+4096,None]-q)/bandwidth).square())
            h = h + (ku*w[:,start:start+4096,None]).transpose(1,2).bmm(kv)
        planes.append(h)
    h=torch.stack(planes,1)
    return h/(h.sum((1,2,3),keepdim=True)+1e-6)


def background(x):
    result=[]
    for a in x.detach().cpu():
        im=Image.fromarray(a.clamp(0,1).mul(255).byte().permute(1,2,0).numpy())
        im=im.filter(ImageFilter.GaussianBlur(sum(im.size)/2))
        result.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255)
    return torch.stack(result).to(x.device)
