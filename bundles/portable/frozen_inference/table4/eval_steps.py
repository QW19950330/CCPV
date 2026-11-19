"""Forward-once wrappers, no file/load/fit. Inputs never include scoring truth."""
import numpy as np
import torch
CONTEXTS=('rolling_observed_history','original_t0_mean_feedback')
def collect_block(api,model,states,valid,teams,context):
 if context not in CONTEXTS or states.shape[1:]!=(30,23,4) or states.dtype!=torch.float32 or valid.shape!=states.shape[:-1]:raise ValueError('Literal context/30frame frozen data')
 # Whole states belongs to the factual cache. Only[t:t+10] past can enter a
 # rolling call; originalt0 meanfeedback never reads truefuture states.
 h=states[:,:10].clone();v=valid[:,:10].clone();seen=v.any(1);observed,age=api.initial_provenance(v);rows=[]
 for t in range(20):
  if context=='rolling_observed_history':h=states[:,t:t+10].clone();v=valid[:,t:t+10].clone();observed,age=api.initial_provenance(v)
  out=model(h,v,teams,observation_valid=observed,observation_age=age)
  mean=model.mean_state(out)
  if not torch.isfinite(mean).all():raise FloatingPointError('Nonfinite mean path; no scientificmissing fallback')
  rows.append(dict(logits=out['velocity_logits'].detach().numpy().copy(),loc=out['velocity_loc'].detach().numpy().copy(),log_scale=out['velocity_log_scale'].detach().numpy().copy(),mean_state=mean.detach().numpy().copy(),anchor=out['anchor'].detach().numpy().copy(),present=out['present'].detach().numpy().copy(),input_valid=v.numpy().copy(),observation_valid=observed.numpy().copy(),observation_age=age.numpy().copy()))
  if context=='original_t0_mean_feedback':
   h=torch.cat((h[:,1:],mean[:,None]),1);v=torch.cat((v[:,1:],seen[:,None]),1);observed,age=api.advance_provenance(observed,age)
 return {k:np.stack([r[k] for r in rows],axis=1) for k in rows[0]}
def sample_block(api,model,h,v,teams,noise):
 if h.shape!=(1024,10,23,4) or noise[0].shape!=(20,1024,23) or noise[1].shape!=(20,1024,23,2):raise ValueError('FixedMC32scene32draw B1024')
 h=h.clone();v=v.clone();seen=v.any(1);observed,age=api.initial_provenance(v);rows=[];states=[]
 for t in range(20):
  out=model(h,v,teams,observation_valid=observed,observation_age=age)
  state=model.sample_state(out,dict(mixture_uniform=noise[0][t],normal=noise[1][t]));mean=model.mean_state(out)
  if not torch.isfinite(state).all() or not torch.isfinite(mean).all():raise FloatingPointError('Nonfinite requiredMC path; no redraw or clipping')
  rows.append(dict(logits=out['velocity_logits'].detach().numpy().copy(),loc=out['velocity_loc'].detach().numpy().copy(),log_scale=out['velocity_log_scale'].detach().numpy().copy(),mean_state=mean.detach().numpy().copy(),anchor=out['anchor'].detach().numpy().copy(),present=out['present'].detach().numpy().copy(),input_valid=v.numpy().copy(),observation_valid=observed.numpy().copy(),observation_age=age.numpy().copy()))
  states.append(state.numpy().copy());h=torch.cat((h[:,1:],state[:,None]),1);v=torch.cat((v[:,1:],seen[:,None]),1);observed,age=api.advance_provenance(observed,age)
 result={k:np.stack([r[k] for r in rows],axis=1) for k in rows[0]};result['states']=np.stack(states,axis=1);result['support']=np.broadcast_to(seen.numpy()[:,None],(1024,20,23)).copy();result['completed_steps']=np.full(1024,20,np.int64)
 return result
