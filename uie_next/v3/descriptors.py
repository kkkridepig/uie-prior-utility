"""Deployment-visible 13-channel population moments; no reference input."""
import numpy as np
import torch
from .moments import aggregate


def descriptors(image,base,candidate):
    I=np.asarray(image,dtype=np.float64);J=np.asarray(base,dtype=np.float64);r=np.asarray(candidate,dtype=np.float64)-J
    a=np.mean(r*r,1,keepdims=True);z=np.concatenate([I,J,r,np.abs(r),a],1)
    def stats(block):
        mu=aggregate(z,block);var=np.maximum(aggregate(z*z,block)-mu*mu,0)
        return np.concatenate([mu,np.sqrt(var)],1)
    return stats(32).astype(np.float32),stats(256).astype(np.float32)


def torch_descriptors(image,base,candidate):
    r=candidate-base;a=r.square().mean(1,keepdim=True);z=torch.cat([image,base,r,r.abs(),a],1)
    def stats(block):
        n,c,h,w=z.shape
        tiles=z.reshape(n,c,h//block,block,w//block,block)
        mu=tiles.mean((3,5))
        # Population variance, algebraically E[Z²]-E[Z]²; centered summation
        # avoids cancellation and sequential avg_pool accumulation error.
        var=(tiles-mu[:,:,:,None,:,None]).square().mean((3,5))
        return torch.cat([mu,var.sqrt()],1)
    return stats(32),stats(256)


def fit_standardization(x):
    x=np.asarray(x,dtype=np.float64)
    return {'mean':x.mean(axis=0).tolist(),'std':np.maximum(x.std(axis=0),1e-6).tolist()}


def standardize(x,stats):
    return (np.asarray(x,dtype=np.float64)-np.array(stats['mean']))/np.array(stats['std'])
