"""Velocity-only densities and affine position pushforwards, never a 4D density."""
from __future__ import annotations
import math
import torch
from torch import nn
from consistent_core.gap_model import PersistentGapAwareWorldModel
from consistent_core.models import _last_observed,_teams,_validate

FPS=10.
DT=1./FPS
SIGMA_FLOOR_M_S=.1
VELOCITY_SCALE=10.5

def current_anchor(history,valid):
    """Current latent base from latest usable state, not real-observation age.

    A generated state at the current history end has usable_age=0 even while its
    true-observation age keeps increasing. No hidden coordinate is consulted.
    """
    last,present,index=_last_observed(torch.where(valid[...,None],history,0.),valid)
    age=(history.shape[1]-1-index.clamp_min(0)).to(history.dtype)/FPS
    base=last.clone()
    base_xy=last[...,:2]+age[...,None]*last[...,2:]
    base=torch.cat((base_xy,last[...,2:]),-1)
    return torch.where(present[...,None],base,0.),present,age

class VelocityResidualMDN(PersistentGapAwareWorldModel):
    """2D normalized-velocity Gaussian mixture plus an explicit point xy loss.

    B has a deterministic learned xy residual, zero initially. C/D fix it to
    zero. Joint (xy,v) is an affine rank-two pushforward conditional on history;
    no full-rank 4D Gaussian likelihood is defined or used.
    """
    def __init__(self,free_position=False,**kwargs):
        super().__init__(**kwargs)
        k,h=self.num_mixtures,self.logits_head.in_features
        old_loc=self.loc_head;old_scale=self.log_scale_head
        self.velocity_loc_head=nn.Linear(h,k*2)
        self.velocity_log_scale_head=nn.Linear(h,k*2)
        with torch.no_grad():
            self.velocity_loc_head.weight.copy_(old_loc.weight.reshape(k,4,h)[:,2:].reshape(k*2,h))
            self.velocity_loc_head.bias.copy_(old_loc.bias.reshape(k,4)[:,2:].reshape(k*2))
            self.velocity_log_scale_head.weight.copy_(old_scale.weight.reshape(k,4,h)[:,2:].reshape(k*2,h))
            self.velocity_log_scale_head.bias.copy_(old_scale.bias.reshape(k,4)[:,2:].reshape(k*2))
        del self.loc_head,self.log_scale_head
        self.position_residual_head=nn.Linear(h,2) if free_position else None
        if self.position_residual_head is not None:
            nn.init.zeros_(self.position_residual_head.weight);nn.init.zeros_(self.position_residual_head.bias)
        self.model_config.update(free_position=bool(free_position),velocity_sigma_floor_m_s=SIGMA_FLOOR_M_S,
                                 position_huber_scale_m=1.,position_huber_lambda=1.)

    def forward(self,history,valid,teams,*,observation_valid,observation_age):
        _validate(history,valid)
        b,_,n,_=history.shape;types=_teams(teams,b,n,history.device)
        clean=torch.where(valid[...,None],history,0.)
        last,present,_=_last_observed(clean,valid)
        encoded=self.temporal(self.normalize(clean),valid,types,
            observation_valid=observation_valid,observation_age=observation_age)
        if self.spatial is not None:encoded=self.spatial(encoded,last[...,:2],present)
        anchor,_,usable_age=current_anchor(history,valid)
        loc=anchor[...,None,2:]/VELOCITY_SCALE+self.velocity_loc_head(encoded).reshape(b,n,self.num_mixtures,2)
        log_scale=self.velocity_log_scale_head(encoded).reshape(b,n,self.num_mixtures,2).clamp(
            math.log(SIGMA_FLOOR_M_S/VELOCITY_SCALE),1.5)
        residual=(self.position_residual_head(encoded) if self.position_residual_head is not None
                  else torch.zeros_like(anchor[...,:2]))
        # Position residual is physical metres; component positions are readouts,
        # not a second stochastic density or a non-singular joint Gaussian.
        component_xy=anchor[...,None,:2]+DT*(loc*VELOCITY_SCALE)+residual[...,None,:]
        return dict(velocity_logits=self.logits_head(encoded),velocity_loc=loc,
                    velocity_log_scale=log_scale,position_component=component_xy,
                    position_residual=residual,anchor=anchor,present=present,usable_age=usable_age)

    def velocity_nll(self,output,target_velocity,mask):
        clean=torch.where(mask[...,None],target_velocity,0.)/VELOCITY_SCALE
        z=(clean[...,None,:]-output['velocity_loc'])*torch.exp(-output['velocity_log_scale'])
        log_density=(-.5*z.square()-output['velocity_log_scale']-.5*math.log(2*math.pi)).sum(-1)
        log_prob=torch.logsumexp(output['velocity_logits'].log_softmax(-1)+log_density,-1)
        return torch.where(mask,-log_prob,0.).sum()/mask.sum().clamp_min(1)

    def nll(self,*args,**kwargs):
        raise RuntimeError('This model has no non-singular 4D NLL; use velocity_nll explicitly')

    def mean_state(self,output):
        weights=output['velocity_logits'].softmax(-1)
        velocity=(weights[...,None]*output['velocity_loc']).sum(-2)*VELOCITY_SCALE
        xy=output['anchor'][...,:2]+DT*velocity+output['position_residual']
        state=torch.cat((xy,velocity),-1)
        return torch.where(output['present'][...,None],state,0.)

    def sample_state(self,output,noise):
        weights=output['velocity_logits'].softmax(-1);b,n,k=weights.shape
        u=noise['mixture_uniform'];eps=noise['normal']
        if u.shape!=(b,n) or eps.shape!=(b,n,2):raise ValueError('Velocity noise must be uniform[B,N],normal[B,N,2]')
        if not torch.isfinite(u).all() or not ((u>=0)&(u<1)).all() or not torch.isfinite(eps).all():
            raise ValueError('Explicit sample noise must be finite, with uniform entries in[0,1)')
        index=(u[...,None]>weights.cumsum(-1)).sum(-1).clamp_max(k-1)
        ix=index[...,None,None].expand(-1,-1,1,2)
        loc=output['velocity_loc'].gather(-2,ix).squeeze(-2)
        sigma=output['velocity_log_scale'].gather(-2,ix).squeeze(-2).exp()
        velocity=(loc+sigma*eps)*VELOCITY_SCALE
        xy=output['anchor'][...,:2]+DT*velocity+output['position_residual']
        return torch.where(output['present'][...,None],torch.cat((xy,velocity),-1),0.)
