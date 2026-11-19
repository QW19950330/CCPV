"""Original paired 20-step loss and data batch; no optimizer or training entry point."""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from runtime import *

class Windows(Dataset):
    def __init__(self,arrays):self.a=arrays;self.indices=arrays['train_indices'];self.epoch=0
    def __len__(self):return len(self.indices)
    def __getitem__(self,item):
        ix=int(self.indices[item]);a=self.a
        h=torch.tensor(a['states'][ix,:10],dtype=torch.float32);v=torch.tensor(a['valid'][ix,:10],dtype=torch.bool)
        teams=torch.tensor(a['teams'] if a['teams'].ndim==1 else a['teams'][ix],dtype=torch.long)
        clean,usable,_=mask_history(h[None],v[None],teams,[int(a['source_indices'][ix])],
            condition='tailgap2',seed=AUGMENTATION_SEED,epoch=self.epoch)
        return dict(history=clean[0],valid=usable[0],history_seen=v.any(0),teams=teams,
            target=torch.tensor(a['states'][ix,10:30],dtype=torch.float32),
            target_valid=torch.tensor(a['valid'][ix,10:30],dtype=torch.bool),
            position=torch.tensor(a['position_targets'][ix,10:30],dtype=torch.float32),
            position_valid=torch.tensor(a['position_valid'][ix,10:30],dtype=torch.bool),
            item_index=torch.tensor(item,dtype=torch.long))

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
