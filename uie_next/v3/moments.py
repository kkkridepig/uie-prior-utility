"""CPU float64 exact objectives and deployable relative decisions."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


def aggregate(x, block):
    n,c,h,w=x.shape
    if h%block or w%block: raise ValueError('block must divide image')
    return x.reshape(n,c,h//block,block,w//block,block).mean((3,5))


def expand(x,block):
    return np.repeat(np.repeat(x,block,-2),block,-1)


def geometry(base,candidate,target=None,block=1):
    b=np.asarray(base,dtype=np.float64);r=np.asarray(candidate,dtype=np.float64)-b
    a=(r*r).mean(1,keepdims=True);out={'r':r,'a':a,'A':aggregate(a,block)}
    if target is not None:
        e=np.asarray(target,dtype=np.float64)-b
        bb=(e*r).mean(1,keepdims=True)
        out.update(b=bb,B=aggregate(bb,block),mse0=(e*e).mean((1,2,3)))
    return out


def oracle_alpha(A,B):
    return np.where(A>0,np.clip(B/np.where(A>0,A,1),0,1),0)


def relative_alpha(A,C,alpha_ref,lamb):
    if lamb<=0 or not 0<alpha_ref<1: raise ValueError('invalid relative policy')
    return np.where(A>0,np.clip(alpha_ref+C/(A+lamb),0,1),alpha_ref)


def regret(A,B,alpha):
    star=oracle_alpha(A,B);d=alpha-star
    square=A*d*d;boundary=2*(A*star-B)*d
    return square+boundary,square,boundary


def mse_from(mse0,A,B,alpha):
    return mse0+np.mean(A*alpha*alpha-2*B*alpha,axis=tuple(range(1,A.ndim)))


def psnr(mse):
    if np.any(np.asarray(mse)<=0):raise ValueError('nonpositive MSE; report exact-zero separately')
    return -10*np.log10(mse)


def fixed_alpha(mse0,A,B):
    A=A.reshape(len(mse0),-1).mean(1);B=B.reshape(len(mse0),-1).mean(1)
    grid=np.arange(10001,dtype=np.float64)/10000
    scores=psnr(np.asarray(mse0)[:,None]+A[:,None]*grid**2-2*B[:,None]*grid).mean(0)
    best=int(np.flatnonzero(scores>=scores.max()-1e-12)[0])
    return float(grid[best]),scores


def seed_for(prefix,sample_id):
    return int.from_bytes(hashlib.sha256((prefix+sample_id).encode('utf-8')).digest()[:8],'big')


def spatial_variants(alpha,a,sample_id):
    order=np.random.Generator(np.random.PCG64(seed_for('v3-alpha-shuffle-20261007|',sample_id))).permutation(alpha.size)
    den=float(a.sum());weighted=float((a*alpha).sum()/den) if den>0 else 0.
    return {'original':alpha,'mean':np.full_like(alpha,alpha.mean()),'energy_mean':np.full_like(alpha,weighted),'shuffle':alpha.reshape(-1)[order].reshape(alpha.shape)}


def matched_null(base,r,sample_id,index):
    rng=np.random.Generator(np.random.PCG64(seed_for('v3-null-'+str(index)+'|',sample_id)))
    q=rng.standard_normal(r.shape);q/=np.sqrt(np.mean(q*q,axis=0,keepdims=True))
    c=np.sqrt(np.mean(r*r,axis=0,keepdims=True));delta=c*q
    bound=np.where(delta>0,(1-base)/np.where(delta>0,delta,1),np.where(delta<0,-base/np.where(delta<0,delta,1),1))
    eta=np.minimum(1,np.maximum(0,bound.min(axis=0,keepdims=True)))
    return eta*delta,eta*r,eta


def torch_relative(A,C,alpha_ref,lamb):
    return torch.where(A>0,(alpha_ref+C/(A+lamb)).clamp(0,1),torch.full_like(A,alpha_ref))
