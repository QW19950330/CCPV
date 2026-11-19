"""Synthetic only: raw selector/missing truth and matched factual scoring."""
from pathlib import Path
import json,tempfile,sys
import numpy as np
from recover_truth import detected_players,recover_file
from score_factual import row_scores,validate_natural_population
from analyze_responses import sha,dump
P=Path(__file__).resolve().parent
meta={'pitch_length':100.,'pitch_width':50.,'players':[{'id':1,'team_id':1},{'id':2,'team_id':2}]}
p=detected_players({'player_data':[{'player_id':1,'is_detected':True,'x':0,'y':0},{'player_id':2,'is_detected':False,'x':9999,'y':9999}]},{'1','2'},100,50)
assert np.array_equal(p['1'],[0,0]) and '2' not in p
try:detected_players({'player_data':[{'player_id':1},{'player_id':1}]},{'1'},100,50)
except ValueError:pass
else:raise AssertionError('Duplicate identity accepted')
with tempfile.TemporaryDirectory(prefix='synthetic_truth_',dir=P) as td:
 raw=Path(td)/'mock.jsonl';records=[]
 for fr in range(9,41):records.append({'frame':fr,'period':1,'player_data':[{'player_id':1,'is_detected':fr!=20,'x':10.,'y':5.},{'player_id':2,'is_detected':True,'x':None,'y':4.}]})
 raw.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf-8')
 ids=np.full((1,23),'',dtype='<U8');ids[0,0]='1';ids[0,11]='2';ids[0,22]='ball'
 pos,valid,found,rec=recover_file(raw,sha(raw),meta,np.array([0]),ids,np.array([1]),np.arange(10,40)[None,:])
 assert found.all() and rec['selected_frames']==30 and rec['lines_streamed']==32
 assert np.array_equal(pos[0,0,0],np.array([10.5,6.8],np.float32))
 assert not valid[0,10,0] and np.isnan(pos[0,10,0]).all() and np.isnan(pos[0,:,11]).all()
 assert not valid[:,:,22].any(), 'Unrequested ball must stay unscored'
 raw.write_text(''.join(json.dumps(x)+'\n' for x in records if x['frame']!=20),encoding='utf-8')
 try:recover_file(raw,sha(raw),meta,np.array([0]),ids,np.array([1]),np.arange(10,40)[None,:])
 except ValueError as e:assert 'Requested continuous cache frame missing' in str(e)
 else:raise AssertionError('Missing frame incorrectly treated as ordinary NA truth')
e=np.full((2,20,22),np.nan);e[1,0,0]=1;e[1,9,0]=3
s,n=row_scores(e);assert np.isnan(s[0]).all() and n[0].sum()==0 and s[1,0]==2 and s[1,1]==3 and np.isnan(s[1,2])
d=np.full((1,20,22),np.nan);c=d.copy();d[0,0,:2]=[1,100];c[0,0,0]=2
common=np.isfinite(d)&np.isfinite(c);ds,_=row_scores(np.where(common,d,np.nan));cs,_=row_scores(np.where(common,c,np.nan))
assert ds[0,0]-cs[0,0]==-1, 'D/CV must use same actor-time support'

hs=np.ones((2,22),bool);elig=np.array([True,False]);sup=np.broadcast_to(hs[:,None,:]&elig[:,None,None],(2,20,22)).copy();st=np.array([20,0]);status=np.array(['COMPLETED','UNSUPPORTED_PREPARED'])
validate_natural_population(sup,st,status,hs,elig)
bad=sup.copy();bad[0,3,7]=False
for su,steps,stat in [(bad,st,status),(sup,np.array([19,0]),status),(np.zeros_like(sup),np.array([0,0]),np.array(['UNSUPPORTED_PREPARED']*2))]:
 try:validate_natural_population(su,steps,stat,hs,elig)
 except ValueError:pass
 else:raise AssertionError('Fixed historical population failure was silently scored')

out=dict(status='PASS_SYNTHETIC_TRUTH_AND_FACTUAL_SCORING',checks=['detected_true_origin_is_valid','undetected_finite_not_truth','duplicate_id_rejected','selected_frames_only_deserialized','known_coordinate_scaling','NaN_missing_position_and_ball_unscored','missing_requested_frame_is_technical_failure','row_ADE_FDE_denominators','paired_D_CV_same_actor_time_support','prelocked_unsupported_allowed','historical_support_loss_rejected','partial_natural_path_rejected','shared_false_unsupported_rejected'],real_raw_reads=0,real_prediction_reads=0,model_calls=0,code_sha256={f:sha(P/f) for f in ('recover_truth.py','score_factual.py','analyze_responses.py')})
dump(P/(sys.argv[1] if len(sys.argv)>1 else 'truth_synthetic_validation.json'),out);print(json.dumps(out,ensure_ascii=False))
