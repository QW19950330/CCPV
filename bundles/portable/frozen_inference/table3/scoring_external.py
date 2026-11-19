"""Original exact-time external metrics with explicitly injected fixed diagnostic bound."""
import numpy as np
HORIZONS=(4,8,12,16,20)

def score(pred,truth,position_valid,states,full_valid,seen,teams,chosen,anchor,acceleration_bound):
    """Same old equal-actor-time score, only horizons and exact-time support differ."""
    b,_,n,_=pred.shape;target=position_valid&seen[:,None];vmask=full_valid&seen[:,None]
    types=np.broadcast_to(teams,(b,n)) if teams.ndim==1 else teams
    groups={'players':types!=2,'hidden_players':(types!=2)&chosen,'visible_players':(types!=2)&~chosen,'ball':types==2}
    assert np.isfinite(pred).all() and np.isfinite(truth[target]).all()
    distance=np.linalg.norm(pred[...,:2]-truth,axis=-1)
    velocity2=np.square(pred[...,2:]-states[...,2:]).sum(-1)
    previous=np.concatenate([anchor[:,None,:,:2],pred[:,:-1,:,:2]],1)
    residual=np.linalg.norm(pred[...,:2]-previous-.1*pred[...,2:],axis=-1)
    gv=(pred[...,:2]-previous)/.1;pv=np.concatenate([anchor[:,None,:,2:],pred[:,:-1,:,2:]],1)
    pgv=np.concatenate([anchor[:,None,:,2:],gv[:,:-1]],1)
    physical={'head_speed':np.linalg.norm(pred[...,2:],axis=-1),'geometric_speed':np.linalg.norm(gv,axis=-1),
        'head_acceleration':np.linalg.norm(pred[...,2:]-pv,axis=-1)/.1,'geometric_acceleration':np.linalg.norm(gv-pgv,axis=-1)/.1}
    rows=[]
    for horizon in HORIZONS:
        for group,membership in groups.items():
            mask=target[:,:horizon]&membership[:,None];count=mask.sum((1,2));last=mask[:,-1];terminal=last.sum(1)
            vm=vmask[:,:horizon]&membership[:,None];vc=vm.sum((1,2))
            cm=np.broadcast_to((seen&membership)[:,None],(b,horizon,n));cc=cm.sum((1,2))
            def mean(value,m,den):return np.divide(np.where(m,value,0.).sum((1,2)),den,out=np.full(b,np.nan),where=den>0)
            result=dict(ade_m=mean(distance[:,:horizon],mask,count),
                fde_m=np.divide(np.where(last,distance[:,horizon-1],0.).sum(1),terminal,out=np.full(b,np.nan),where=terminal>0),
                velocity_rmse_m_s=np.sqrt(mean(velocity2[:,:horizon],vm,vc)),
                consistency_mean_m=mean(residual[:,:horizon],cm,cc),
                consistency_max_m=np.where(cc>0,np.where(cm,residual[:,:horizon],-np.inf).max((1,2)),np.nan),
                consistency_gt_1e4_fraction=mean((residual[:,:horizon]>1e-4).astype(float),cm,cc),
                observed_points=count,terminal_agents=terminal,velocity_targets=vc,consistency_actor_steps=cc)
            for key,value in physical.items():
                result[key+'_mean']=mean(value[:,:horizon],cm,cc)
                bound=10.5 if 'speed' in key else acceleration_bound
                result[key+'_exceedance_fraction']=(np.full(b,np.nan) if group=='ball' else mean((value[:,:horizon]>bound+1e-4).astype(float),cm,cc))
            rows.append((horizon,group,result))
    return rows
