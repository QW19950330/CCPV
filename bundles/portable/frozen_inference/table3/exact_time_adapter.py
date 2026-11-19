"""Original pure 25 Hz-to-nominal-10 Hz exact-source array construction."""
import numpy as np

HISTORY_SOURCE=np.array([4,9,14,19,24])+1
HISTORY_SLOTS=np.array([1,3,5,7,9])
TARGET_SOURCE=np.arange(29,75,5)+1
TARGET_STEPS=np.arange(2,21,2)

def exact_arrays(native_positions):
    """native[0] is one causal pad frame; native[1:76] is main75 block."""
    p=np.asarray(native_positions,dtype=np.float32)
    assert p.shape==(76,23,2)
    seen=np.isfinite(p).all(-1)
    h=np.zeros((10,23,4),np.float32);v=np.zeros((10,23),bool)
    for slot,index in zip(HISTORY_SLOTS,HISTORY_SOURCE):
        support=seen[index]&seen[index-5]
        h[slot,:,:2]=np.where(support[:,None],p[index],0.)
        h[slot,:,2:]=np.where(support[:,None],(p[index]-p[index-5])/.2,0.)
        v[slot]=support
    truth=np.full((20,23,2),np.nan,np.float32);pv=np.zeros((20,23),bool)
    target=np.zeros((20,23,4),np.float32);tv=np.zeros((20,23),bool)
    for step,index in zip(TARGET_STEPS,TARGET_SOURCE):
        point=seen[index];support=point&seen[index-5]
        truth[step-1]=p[index];pv[step-1]=point;tv[step-1]=support
        target[step-1,:,:2]=np.where(support[:,None],p[index],0.)
        target[step-1,:,2:]=np.where(support[:,None],(p[index]-p[index-5])/.2,0.)
    return dict(history=h,history_valid=v,position_targets=truth,position_valid=pv,target_states=target,target_valid=tv)
