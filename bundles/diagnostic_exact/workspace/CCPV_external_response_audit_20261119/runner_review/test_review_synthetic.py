import sys,importlib.util,json,copy
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);O=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('review_runner_new',R/'run_skillcorner_response.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks={}
b=O/'synthetic_resume_payload.npz';rows=dict(source_indices=np.array([3,8]),match_ids=np.array(['synthetic:a','synthetic:b']),actor_slots=np.array([0,2]),technical_unsupported=np.array([False,True]))
m.npz(b,**rows)
kwargs=dict(arm='D_coupled_ar',seed=2,start=32,stop=34,input_sha='synthetic-input',implementation_sha='synthetic-lock',expected_rows=rows)
rec=dict(arm='D_coupled_ar',seed=2,start=32,stop=34,path=b.name,input_sha256='synthetic-input',implementation_sha256='synthetic-lock',sha256=m.sha(b))
m.validate_resume_block(rec,b,**kwargs);checks['valid_resume_block_passes']=True
for key,value in [('arm','CV'),('seed',3),('start',0),('stop',35),('path','wrong.npz'),('input_sha256','different'),('implementation_sha256','different'),('sha256','different')]:
 old={**rec,key:value}
 try:m.validate_resume_block(old,b,**kwargs)
 except ValueError:checks['rejects_wrong_'+key]=True
 else:raise AssertionError(key)
badrows={**rows,'source_indices':np.array([3,9])}
try:m.validate_resume_block(rec,b,**{**kwargs,'expected_rows':badrows})
except ValueError:checks['rejects_changed_source_rows']=True
else:raise AssertionError('wrong rows')
empty,reasons,_=m.compact({},np.array([],dtype=int),3,np.zeros((3,10,23),bool),[0,4,-1],np.empty((0,2,2),np.float32),np.empty((0,2,2),np.float32))
assert np.isnan(empty['states']).all() and np.isnan(empty['actor_distance']).all() and not empty['support'].any()
assert (empty['completed_steps']==0).all() and (empty['status']=='UNSUPPORTED_PREPARED').all()
checks['entire_unsupported_block_retains_NA_and_zero_support']=True
torch,api,ad=m.runtime();P=m.read(R/'skillcorner_response_protocol.json')
h=torch.zeros((2,10,23,4),dtype=torch.float32);h[...,0]=torch.arange(23);h[...,2]=1.
h[1,:,0,2]=10.5
v=torch.ones((2,10,23),dtype=torch.bool);v[0,:,22]=False
tm=torch.tensor([0]*11+[1]*11+[2],dtype=torch.long)
ids=[[f'synthetic-{i}' for i in range(22)]+['synthetic-ball'] for _ in range(2)]
status=[['VERIFIED_SOURCE_PLAYER_TEAM']*22+['EXPLICIT_MODEL_BALL_SOURCE_ID'] for _ in range(2)]
regs=ad.make_registries(ids,status,tm);targets=[[r[0],r[1]] for r in ids]
rr,cmd,v0=m.paths(torch,api,ad,None,h,v,tm,regs,targets,[0,0],P)
assert rr['first_A']['completed_steps'][1]==0 and rr['natural']['completed_steps'][1]==20
assert rr['first_A']['scene_status'][1]=='REQUEST_REJECTED'
assert not rr['first_A']['support'][1].any()
checks['rejected_first_reset_retained_while_natural_completes']=True
# Simulate a later-only persistent rejection as a compact() input; do not require this path to succeed for first-natural.
rr['persistent_A']['support'][0,5:]=False;rr['persistent_A']['states'][0,5:]=float('nan')
rr['persistent_A']['completed_steps'][0]=5;rr['persistent_A']['scene_status'][0]='REQUEST_REJECTED'
full,_,_=m.compact(rr,np.arange(2),2,v.numpy(),[0,0],cmd,v0)
assert np.isfinite(full['actor_distance'][0,0]).all()
assert np.isnan(full['actor_distance'][0,1,5:]).all()
assert np.isnan(full['actor_distance'][1,0]).all()
assert np.isnan(full['actor_difference_vectors'][1,4]).all()
assert np.isfinite(full['actor_distance'][1,3]).all()
checks['pairwise_support_independent_of_persistent_and_sham_does_not_rescue_first']=True
# The vector decomposition is algebraic only where all three vectors exist; norms are not used as a decomposition.
assert np.max(np.abs(full['actor_difference_vectors'][0,0]-(full['actor_difference_vectors'][0,4]+full['actor_difference_vectors'][0,3])))<1e-6
checks['first_sham_natural_vector_decomposition']=True
record={'status':'PASS_TARGETED_RUNNER_REVIEW_SYNTHETIC','checks':checks,'code_sha256':m.sha(R/'run_skillcorner_response.py'),'new_real_prefix_reads':0,'saved_checkpoint_loads':0,'real_model_forwards':0,'training_updates':0,'scope':'Synthetic arrays, CV and receipt fixtures only; earlier root synthetic separately covers random D.'}
m.write(O/'targeted_synthetic.json',record)
print(json.dumps({'status':record['status'],'checks':len(checks)}))
