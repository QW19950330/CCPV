"""Synthetic statistical checks only; never opens real predictions or raw data."""
from pathlib import Path
import hashlib,json,importlib.util,sys
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resp',P/'analyze_responses.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
n=3;s=np.zeros((n,6,20,23,4),np.float32);support=np.ones((n,6,20,23),bool)
t=np.arange(1,21,dtype=np.float32)/10
s[:,2,:,0,0]=t;s[:,3,:,0,0]=2*t;s[:,4,:,0,0]=.25*t;s[:,5,:,0,0]=-.5*t
steps=np.full((n,6),20,np.int64);status=np.full((n,6),'COMPLETED',dtype='<U32')
s[0,3,1:]=np.nan;support[0,3,1:]=False;steps[0,3]=1;status[0,3]='REQUEST_REJECTED'
s[2]=np.nan;support[2]=False;steps[2]=0;status[2]='UNSUPPORTED_PREPARED'
v=np.full((n,6,20,2),np.nan,np.float32)
for i,(l,r) in enumerate(m.PAIRS):
 ok=support[:,l,:,0]&support[:,r,:,0];raw=s[:,l,:,0,:2]-s[:,r,:,0,:2];v[:,i]=np.where(ok[...,None],raw,np.nan)
a=dict(states=s,support=support,completed_steps=steps,status=status,source_indices=np.arange(n),match_ids=np.array([m.MATCHES[0]]*n),actor_slots=np.array([0,0,-1]),technical_unsupported=np.array([False,False,True]),actor_difference_vectors=v,actor_distance=np.linalg.norm(v,axis=-1),actor_direction_projection=v[...,0],other_player_rms=np.where(np.isfinite(v[...,0]),0.,np.nan))
x,ok=m.validate_block(a)
assert ok[0,0,19] and not ok[0,1,19], 'persistent failure wrongly removed first'
assert np.isnan(x[2]).all() and m.statistics(x[:,0,19,2])['n_na']==1
assert np.allclose((x[:,0,:,:2]-x[:,4,:,:2]-x[:,3,:,:2])[ok[:,0]&ok[:,4]&ok[:,3]],0,atol=1e-6)
assert abs(x[0,0,19,2]-(x[0,4,19,2]+x[0,3,19,2]))>1, 'norms incorrectly treated additive'
# Unequal window counts deliberately distinguish equal-match macro from pooling.
mid=np.array(sum(([mm]*(i+1) for i,mm in enumerate(m.MATCHES)),[]));levels=np.concatenate([np.repeat(i+1,i+1) for i in range(4)])
mac={}
for seed,cell in enumerate(m.CELLS):
 vals=levels.astype(float)+(seed-1 if seed else 0)
 data=np.broadcast_to(vals[:,None,None,None],(10,6,20,5)).copy()
 _,mm=m.fixture_curves(cell,dict(metrics=data,match_ids=mid));mac[cell]=mm
 assert abs(mm[('first-natural',0,'actor_distance')]-(2.5+(seed-1 if seed else 0)))<1e-12
r=m.seed_summary(mac,'synthetic')[0]
assert abs(r['D_five_seed_mean']-4.5)<1e-12 and abs(r['D_seed_sd']-np.sqrt(2.5))<1e-12
assert m.strict_macro([1,2,3,np.nan])!=m.strict_macro([1,2,3,np.nan])
mac[m.CELLS[2]][('first-natural',0,'actor_dx')]=np.nan
r=m.seed_summary(mac,'synthetic')[0];assert r['n_seed_complete_four_matches']==4 and not np.isfinite(r['D_five_seed_mean'])
# A supported numerical zero stays zero; NA is not filled or omitted from denominator.
z=m.statistics([0.,np.nan]);assert z['n_planned']==2 and z['n_supported']==1 and z['mean']==0
out=dict(status='PASS_SYNTHETIC_RESPONSE_STATISTICS',checks=['contrast_specific_partial_support','technical_NA_not_zero','vector_sham_identity_not_norm_additivity','unequal_fixture_counts_equal_weight_macro','five_seed_ddof1_SD','strict_missing_fixture_seed_macro','zero_distinguished_from_NA'],real_prediction_reads=0,raw_future_reads=0,model_forwards=0,numpy_version=np.__version__,code_sha256=m.sha(P/'analyze_responses.py'))
m.dump(P/(sys.argv[1] if len(sys.argv)>1 else 'response_synthetic_validation.json'),out)
print(json.dumps(out,ensure_ascii=False))
