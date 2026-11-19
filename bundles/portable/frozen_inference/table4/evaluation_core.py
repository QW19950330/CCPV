"""Exact original fixed-noise reshape and complete-sample qualification."""
import numpy as np

def grid(z):
 u,e=z['mixture_uniform'],z['normal']
 if u.shape!=(32,32,20,23) or e.shape!=(32,32,20,23,2):raise ValueError('Fixedprevious32x32bank')
 return np.ascontiguousarray(u.reshape(1024,20,23).transpose(1,0,2)),np.ascontiguousarray(e.reshape(1024,20,23,2).transpose(1,0,2,3))

def required_samples(z,S):
 x=z['states'].astype(np.float64);support=z['support'].astype(bool);done=z['completed_steps']
 if x.shape!=(len(S),32,20,23,4) or support.shape!=x.shape[:-1] or done.shape!=(len(S),32):raise ValueError('Fixed M32 sample layout')
 ok=np.isfinite(x).all((2,4))&((support|~S[:,None,None]).all(2))&(done==20)[...,None]
 return np.where(ok[:,:,None,:,None],x[...,:2],np.nan)
