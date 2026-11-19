from __future__ import annotations
import hashlib,json,os,random,sys,time,uuid
from pathlib import Path
import numpy as np
import torch
from consistent_core.models import WorldModel,_last_observed,forecast_validity
from consistent_core.gap_model import PersistentGapAwareWorldModel,initial_observation_age
from motion_models import VelocityResidualMDN,current_anchor,DT,FPS,VELOCITY_SCALE

ARMS={'A_raw4d':dict(kind='raw4d',teacher_forcing=True),
      'B_free2d':dict(kind='free2d',teacher_forcing=True),
      'C_coupled_tf':dict(kind='coupled2d',teacher_forcing=True),
      'D_coupled_ar':dict(kind='coupled2d',teacher_forcing=False)}
SEEDS=tuple(range(5))
MASK_SEED=2026102417
AUGMENTATION_SEED=2026102418
MODELS=tuple(ARMS)+('A_integrate','cv','smooth_cv')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def configure(device=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    return device or ('cuda' if torch.cuda.is_available() else 'cpu')

def state_hash(mapping):
    h=hashlib.sha256()
    for name,value in sorted(mapping.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def common_hash(model):
    return state_hash({k:v for k,v in model.state_dict().items()
                       if not any(x in k for x in ('loc_head','log_scale_head','position_residual_head'))})

def velocity_head_hash(model):
    state=model.state_dict();result={}
    if isinstance(model,VelocityResidualMDN):
        for head in ('loc','log_scale'):
            for param in ('weight','bias'):result[head+'.'+param]=state[f'velocity_{head}_head.{param}']
    else:
        k=model.num_mixtures
        for head in ('loc','log_scale'):
            result[head+'.weight']=state[head+'_head.weight'].reshape(k,4,-1)[:,2:].reshape(k*2,-1)
            result[head+'.bias']=state[head+'_head.bias'].reshape(k,4)[:,2:].reshape(k*2)
    return state_hash(result)

def make_model(arm,seed):
    if arm=='A_integrate':arm='A_raw4d'
    spec=ARMS[arm];seed_all(100_003_003+int(seed))
    kwargs=dict(hidden_dim=32,num_mixtures=3,use_temporal_attention=True,use_spatial=True,
                neighborhood_radius=20.,fps=10.,deterministic=False)
    model=(PersistentGapAwareWorldModel(**kwargs) if spec['kind']=='raw4d'
           else VelocityResidualMDN(free_position=spec['kind']=='free2d',**kwargs))
    model.pilot_arm=arm;return model

def initial_provenance(valid):return valid.clone(),initial_observation_age(valid,FPS)

def advance_provenance(observed,age,next_observation=None):
    if next_observation is None:next_observation=torch.zeros_like(observed[:,-1])
    return (torch.cat((observed[:,1:],next_observation[:,None]),1),
            torch.where(next_observation,torch.zeros_like(age),age+DT))

def output_mean(model,output):
    if isinstance(model,VelocityResidualMDN):return model.mean_state(output)
    return model.denormalize((output['logits'].softmax(-1)[...,None]*output['loc']).sum(-2))

def step(model,history,valid,teams,observation_valid,observation_age,mode='mean',noise=None,integrate=False):
    output=model(history,valid,teams,observation_valid=observation_valid,observation_age=observation_age)
    if mode=='mean':state=output_mean(model,output)
    elif mode=='sample' and isinstance(model,VelocityResidualMDN):state=model.sample_state(output,noise)
    else:raise ValueError('Only mean rollout, or explicit 2D velocity sample is supported')
    if integrate:
        base,present,_=current_anchor(history,valid)
        state=torch.cat((base[...,:2]+DT*state[...,2:],state[...,2:]),-1)
        state=torch.where(present[...,None],state,0.)
    return state

def predict(model,history,valid,teams,horizon=20,integrate=False):
    observed,age=initial_provenance(valid);states=[]
    for _ in range(horizon):
        state=step(model,history,valid,teams,observed,age,integrate=integrate)
        if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite generated state')
        usable=valid.any(1);states.append(state)
        history=torch.cat((history[:,1:],state[:,None]),1);valid=torch.cat((valid[:,1:],usable[:,None]),1)
        observed,age=advance_provenance(observed,age)
    return torch.stack(states,1)

def mask_history(history,valid,teams,indices,condition='natural',seed=MASK_SEED,epoch=0):
    usable=valid.clone();chosen=torch.zeros_like(valid[:,0])
    if condition not in ('natural','tailgap2'):raise ValueError('Unknown condition')
    if condition=='tailgap2':
        for b,index in enumerate(indices):
            tm=teams if teams.ndim==1 else teams[b]
            eligible=torch.where((tm!=2)&valid[b,-1]&valid[b,:-2].any(0))[0].cpu().numpy()
            n=int(np.floor(.2*len(eligible)+.5))
            if n:
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(index),int(epoch)]));selection=rng.choice(eligible,n,replace=False)
                usable[b,-2:,selection]=False;chosen[b,selection]=True
    if not torch.equal(usable.any(1),valid.any(1)):raise AssertionError('Mask changed cohort')
    return torch.where(usable[...,None],history,0.),usable,chosen

def analytic_forecast(arm,history,valid,horizon=20):
    last,present,index=_last_observed(torch.where(valid[...,None],history,0.),valid);velocity=last[...,2:]
    if arm=='smooth_cv':
        rank=valid.long().flip(1).cumsum(1).flip(1);m=valid&(rank<=5)
        velocity=torch.where(m[...,None],history[...,2:],0.).sum(1)/m.sum(1).clamp_min(1)[...,None]
    elif arm!='cv':raise ValueError('Unknown analytic baseline')
    base,_,_=current_anchor(history,valid)
    elapsed=torch.arange(1,horizon+1,device=history.device,dtype=history.dtype)[None,:,None]/FPS
    xy=base[:,None,:,:2]+elapsed[...,None]*velocity[:,None]
    return torch.where(present[:,None,:,None],torch.cat((xy,velocity[:,None].expand(-1,horizon,-1,-1)),-1),0.)
