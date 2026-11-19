"""Verbatim executed fixed-history and quantization mathematics."""
import numpy as np

OFFSETS=np.array([-25,-20,-15,-10,-5,0],dtype=np.int64)

TIMES=OFFSETS/25.

CENTRED=TIMES-TIMES.mean()

CONDITIONS=('native','grid')

def quantize(x):return np.rint(np.asarray(x,dtype=np.float64)/np.array([1.05,.68]))*np.array([1.05,.68])

def slopes(history):
    assert history.ndim==3 and history.shape[1:]==(6,2)
    v200=(history[:,-1]-history[:,-2])/.2
    ols=np.sum(history*CENTRED[None,:,None],axis=1)/.7
    return v200,ols

def from_history(old,xy,player_times,frame_times,t0,L,condition):
    """Current geometry uses every valid current player; velocity uses exact six-point common support."""
    if t0<0:return np.full((2,16),np.nan),np.zeros(len(xy),bool),np.zeros(len(xy),bool)
    expected=t0+OFFSETS*40_000_000
    current=(player_times[:,-1]==t0)&np.isfinite(xy[:,-1]).all(1)
    history=current&np.all(player_times==expected[None,:],axis=1)&np.all(frame_times==expected)&np.isfinite(xy).all((1,2))
    if not current.any():return np.full((2,16),np.nan),current,history
    positions=xy if condition=='native' else quantize(xy)
    v200=np.full((len(xy),2),np.nan);vols=v200.copy()
    if history.any():
        v200[history],vols[history]=slopes(positions[history])
        assert np.isfinite(v200[history]).all() and np.isfinite(vols[history]).all(),'nonfinite velocity arithmetic'
    result=np.stack([old.geometry(positions[current,-1],v[current],L) for v in (v200,vols)])
    assert np.array_equal(result[0,:13],result[1,:13],equal_nan=True),'within-condition availability mismatch'
    return result,current,history
