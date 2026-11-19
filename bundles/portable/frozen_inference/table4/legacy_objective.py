"""Exact old objective source segment; bind globals only under root runtime authorization."""
import torch
from torch.nn import functional as F
def bind_api(api):
 global VelocityResidualMDN,initial_provenance,advance_provenance,output_mean,ARMS
 VelocityResidualMDN=api.VelocityResidualMDN;initial_provenance=api.initial_provenance;advance_provenance=api.advance_provenance;output_mean=api.output_mean;ARMS=api.ARMS

def objective(model,batch,arm,return_predictions=False):
    history,valid=batch['history'],batch['valid'];seen=batch['history_seen'];teams=batch['teams']
    observed,age=initial_provenance(valid)
    nll_total=history.new_zeros(());position_total=history.new_zeros(())
    nll_count,position_count=0,0;predictions=[]
    for t in range(20):
        out=model(history,valid,teams,observation_valid=observed,observation_age=age)
        state=output_mean(model,out)
        full=batch['target_valid'][:,t]&seen
        pos=batch['position_valid'][:,t]&seen
        count=int(full.sum());pc=int(pos.sum())
        if isinstance(model,VelocityResidualMDN):
            loss=model.velocity_nll(out,batch['target'][:,t,:,2:],full)
            # Replace missing raw xy before arithmetic; actual position-only
            # observations remain in the loss even without a velocity label.
            truth=torch.where(pos[...,None],batch['position'][:,t],0.)
            safe=torch.where(pos[...,None],state[...,:2],0.)
            per_actor=F.smooth_l1_loss(safe/1.,truth/1.,reduction='none',beta=1.).mean(-1)
            position_total=position_total+torch.where(pos,per_actor,0.).sum();position_count+=pc
        else:loss=model.nll(out,batch['target'][:,t],full)
        nll_total=nll_total+loss*count;nll_count+=count
        if return_predictions:predictions.append(state)
        real=full if ARMS[arm]['teacher_forcing'] else torch.zeros_like(full)
        actual=torch.where(full[...,None],batch['target'][:,t],0.)
        feedback=torch.where(real[...,None],actual,state)
        # All and only originally seen actors stay model-usable. Missing real
        # states use prediction; full-state-invalid xy never enters feedback.
        feedback=torch.where(seen[...,None],feedback,0.)
        history=torch.cat((history[:,1:],feedback[:,None]),1)
        valid=torch.cat((valid[:,1:],seen[:,None]),1)
        observed,age=advance_provenance(observed,age,real)
    if nll_count<=0:raise ValueError('No true velocity/full-state supervision in a batch')
    loss=nll_total/nll_count
    if isinstance(model,VelocityResidualMDN):
        if position_count<=0:raise ValueError('No real xy targets')
        loss=loss+position_total/position_count
    diagnostics=dict(nll_count=nll_count,position_count=position_count,
        nll=float((nll_total/nll_count).detach()),position_huber=float((position_total/max(position_count,1)).detach()),
        loss=float(loss.detach()))
    if return_predictions:diagnostics['predictions']=torch.stack(predictions,1)
    return loss,diagnostics
