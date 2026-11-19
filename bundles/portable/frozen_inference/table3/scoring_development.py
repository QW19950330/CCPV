"""Original fixed-support development metrics."""
import numpy as np
HORIZONS=(5,10,15,20)

def score(prediction,positions,position_valid,states,full_valid,seen,teams,chosen,anchor,acceleration_bound):
    pred=np.asarray(prediction);truth=np.asarray(positions);b,t,n,_=pred.shape
    target=position_valid&seen[:,None];velocity_target=full_valid&seen[:,None]
    if not np.isfinite(truth[target]).all() or not np.isfinite(pred[...,:2][target]).all():raise FloatingPointError('Invalid fixed supported position')
    if not np.isfinite(pred).all():raise FloatingPointError('Invalid generated velocity/state')
    types=np.broadcast_to(teams,(b,n)) if teams.ndim==1 else teams;player=types!=2
    groups=dict(players=player,hidden_players=player&chosen,visible_players=player&~chosen,ball=types==2)
    distance=np.linalg.norm(pred[...,:2]-truth,axis=-1)
    velocity_squared=np.square(pred[...,2:]-states[...,2:]).sum(-1)
    previous=np.concatenate((anchor[:,None,:,:2],pred[:,:-1,:,:2]),axis=1)
    residual=np.linalg.norm(pred[...,:2]-previous-.1*pred[...,2:],axis=-1)
    geometric_velocity=(pred[...,:2]-previous)/.1
    previous_velocity=np.concatenate((anchor[:,None,:,2:],pred[:,:-1,:,2:]),axis=1)
    previous_geometric=np.concatenate((anchor[:,None,:,2:],geometric_velocity[:,:-1]),axis=1)
    head_speed=np.linalg.norm(pred[...,2:],axis=-1);geometric_speed=np.linalg.norm(geometric_velocity,axis=-1)
    head_acceleration=np.linalg.norm(pred[...,2:]-previous_velocity,axis=-1)/.1
    geometric_acceleration=np.linalg.norm(geometric_velocity-previous_geometric,axis=-1)/.1
    rows=[]
    for horizon in HORIZONS:
        for group,membership in groups.items():
            m=target[:,:horizon]&membership[:,None];count=m.sum((1,2));last=m[:,-1];terminal=last.sum(1)
            vm=velocity_target[:,:horizon]&membership[:,None];vc=vm.sum((1,2))
            cm=np.broadcast_to((seen&membership)[:,None],(b,horizon,n));cc=cm.sum((1,2))
            def mean(values,mask,denominator):
                return np.divide(np.where(mask,values,0.).sum((1,2)),denominator,out=np.full(b,np.nan),where=denominator>0)
            ade=mean(distance[:,:horizon],m,count)
            fde=np.divide(np.where(last,distance[:,horizon-1],0.).sum(1),terminal,out=np.full(b,np.nan),where=terminal>0)
            vrmse=np.sqrt(mean(velocity_squared[:,:horizon],vm,vc))
            consistency=mean(residual[:,:horizon],cm,cc)
            max_cons=np.where(cc>0,np.where(cm,residual[:,:horizon],-np.inf).max((1,2)),np.nan)
            violation=mean((residual[:,:horizon]>1e-4).astype(float),cm,cc)
            physical={}
            for name,value,bound in (('head_speed',head_speed,10.5),('geometric_speed',geometric_speed,10.5),
                                     ('head_acceleration',head_acceleration,acceleration_bound),
                                     ('geometric_acceleration',geometric_acceleration,acceleration_bound)):
                physical[name+'_mean']=mean(value[:,:horizon],cm,cc)
                # These pre-fixed thresholds describe player sensitivity only.
                # Ball speeds/accelerations remain descriptive measurements.
                physical[name+'_exceedance_fraction']=(np.full(b,np.nan) if group=='ball' else
                    mean((value[:,:horizon]>bound+1e-4).astype(float),cm,cc))
            rows.append((horizon,group,dict(ade_m=ade,fde_m=fde,velocity_rmse_m_s=vrmse,**physical,
                consistency_mean_m=consistency,consistency_max_m=max_cons,consistency_gt_1e4_fraction=violation,
                observed_points=count,terminal_agents=terminal,velocity_targets=vc,consistency_actor_steps=cc)))
    return rows
