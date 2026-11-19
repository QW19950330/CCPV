"""Exact original Table4-relevant FP64 scoring functions."""
import math
import numpy as np

ENERGY_NEGATIVE_FACTOR=1e-10

def jackknife_se(loo):
    a = np.asarray(loo, np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all(): return math.nan
    return float(np.sqrt((len(a)-1)/len(a) * np.square(a-a.mean()).sum()))

def strict_macro(window_values, match_ids):
    a = np.asarray(window_values, np.float64); ids = np.asarray(match_ids).astype(str)
    if a.shape[0] != len(ids): raise ValueError('Macro shape')
    if not np.isfinite(a).all(): return math.nan if a.ndim == 1 else np.full(a.shape[1:], np.nan)
    if len(a) == 0: return math.nan
    return np.mean([a[ids == k].mean(axis=0) for k in sorted(set(ids))], axis=0)

def source_conditional_macro(values, match_ids, eligible):
    """Only source truth eligibility can be excluded; missing whole match staysNA."""
    a = np.asarray(values, np.float64); ids = np.asarray(match_ids).astype(str); e = np.asarray(eligible, bool)
    result = []
    for k in sorted(set(ids)):
        chosen = (ids == k) & e
        if not chosen.any() or not np.isfinite(a[chosen]).all(): return math.nan if a.ndim == 1 else np.full(a.shape[1:], np.nan)
        result.append(a[chosen].mean(axis=0))
    return np.mean(result, axis=0)

def actor_mean(values, mask):
    values=np.asarray(values,np.float64); mask=np.asarray(mask,bool); count=mask.sum(-1)
    if not np.isfinite(values[mask]).all(): raise FloatingPointError('Required actor score nonfinite')
    return np.divide(np.where(mask,values,0.).sum(-1),count,out=np.full(count.shape,np.nan),where=count>0)

def window_average(values,mask):
    n=mask.sum(-1)
    return np.divide(np.where(mask,values,0.).sum(-1),n,out=np.full(n.shape,np.nan),where=n>0)

def energy_components(samples,truth,mask):
    n=int(mask.sum());M=len(samples)
    if not n:return np.full(M,np.nan),np.full((M,M),np.nan)
    x=np.asarray(samples[:,mask],np.float64);y=np.asarray(truth[mask],np.float64)
    if not np.isfinite(y).all():raise FloatingPointError('Observed xy nonfinite')
    return np.sqrt(np.square(x-y).sum((1,2))/n),np.sqrt(np.square(x[:,None]-x[None,:]).sum((2,3))/n)

def energy_from_components(a,pair):
    M=len(a)
    if M<3 or not(np.isfinite(a).all() and np.isfinite(pair).all()):return dict(score=math.nan,loo=np.full(M,np.nan),negative_roundoff=False)
    A=a.sum();B=pair.sum();first=A/M;second=B/(2*M*(M-1));value=first-second
    if value < -ENERGY_NEGATIVE_FACTOR*(1+abs(first)+abs(second)):raise FloatingPointError('Negative cached norm energy')
    return dict(score=float(value),loo=(A-a)/(M-1)-(B-2*pair.sum(1))/(2*(M-1)*(M-2)),negative_roundoff=bool(value<0))
