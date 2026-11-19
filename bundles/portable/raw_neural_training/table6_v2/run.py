"""Root-gated zero-NN recorded-kinematics reference. No CP/event/model APIs."""
import argparse,csv,hashlib,json,os,sys,time,traceback
from os import getpid
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from metrics import (DT,LIMITS,GROUPS,anchor_from_history,kinematics,recorded_metrics,
    paired_observed_mask,fixed_groups,metric_summary,matched_metric,joint_summary)
from bindings import J, resolve_binding, preflight
DERIVS=('xy','head');METRICS=('speed','acceleration','turn')
DEV_KEYS=('states','valid','position_valid','teams','entity_ids','match_ids','source_indices','train_indices','source_frame_ids','frame_times_s')
TRUTH_KEYS=('positions','position_valid','source_indices','match_ids','source_frame_ids','entity_ids')
PREFIX_KEYS=('history_states','history_valid','teams','entity_ids','identity_status','operator_candidate_ids','operator_candidate_slots','source_indices','match_ids','selected','prefix_supported')
STATE_KEYS=('states','support','completed_steps','scene_status','scene_indices')
EXPECTED_ROWS=dict(alignment=24,agent=44160,recorded=864,recorded_joint=144,
 own=54432,own_joint=9072,matched=54432,matched_joint=9072,
 by_match=27648,pooled=2304,macro=2304,interop=54432,five_seed_macro_summary=432)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(r):
 if sha(r['path'])!=r['sha256']:raise ValueError('Pinned input changed: '+r['path'])
def plain(x):
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def put(p,z):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(plain(z),f,ensure_ascii=False,indent=2,allow_nan=False)
class Table:
 def __init__(self,p):self.f=Path(p).open('x',encoding='utf-8-sig',newline='');self.w=None;self.n=0
 def add(self,r):
  r=plain(r)
  if self.w is None:self.w=csv.DictWriter(self.f,fieldnames=list(r));self.w.writeheader()
  self.w.writerow(r);self.n+=1
 def close(self):self.f.close()
def load(r,expected):
 if tuple(r['members'])!=tuple(expected) or Path(r['path']).suffix.lower()!='.npz':raise PermissionError('Exact member/file allowlist')
 check(r)
 with np.load(r['path'],allow_pickle=False) as z:return {k:z[k] for k in expected}
def unique_map(a):
 keys=[(str(m),int(s)) for m,s in zip(a['match_ids'],a['source_indices'])]
 if len(set(keys))!=len(keys):raise ValueError('Duplicate physical window source key')
 return {k:i for i,k in enumerate(keys)}

def validate_case_contract(binding,freeze,metadata_reader=J):
 """Metadata-only exact cohort/case closure, before opening any NPZ member."""
 cases=binding['cases']
 worlds=[('CV',-1)]+[(w,s) for w in ('D','E') for s in range(5)]
 expected={(w,s,policy,mode,path) for w,s in worlds
  for mode in (('mean',) if w=='CV' else ('mean','sample'))
  for policy,path in [('input_reset','n_empty'),('input_reset','AB_shared'),('runner_output','AB_shared')]}
 keys=[(c['world'],c['seed'],c['policy'],c['mode'],c['path']) for c in cases]
 if len(cases)!=63 or len(set(keys))!=63 or set(keys)!=expected:raise ValueError('Exact63 scientific case product')
 if len({c['case_id'] for c in cases})!=63 or len({c['asset']['path'] for c in cases})!=63:raise ValueError('Duplicate case/file identity')
 if freeze['status']!='ALL_PLANNED_TRAJECTORIES_FROZEN_NO_VALUE' or len(freeze['manifests'])!=334:raise ValueError('Frozen334 original trajectory scope')
 frozen={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['manifests']}
 for c in cases:
  role='PRIMARY_NATURAL' if c['path']=='n_empty' else 'SECONDARY_CONTROLLED_NOT_FACTUAL_ERROR'
  if c['role']!=role:raise ValueError('Primary/controlled role contract')
  if frozen.get(str(Path(c['manifest']['path']).resolve()))!=c['manifest']['sha256']:raise ValueError('Case not in original334 freeze')
  m=metadata_reader(c['manifest']['path'])
  if (m['cell'],m['seed'],m['policy'],m['mode'],m['path'])!=(c['world'],c['seed'],c['policy'],c['mode'],c['path']) or m.get('proof') is not None:raise ValueError('Literal manifest scientific identity')
  if m['lock_sha256']!=binding['runner_lock']['sha256']:raise ValueError('Manifest oldsource lock')
  if (m['states']['path'],m['states']['sha256'])!=(c['asset']['path'],c['asset']['sha256']):raise ValueError('Manifest/asset pin mismatch')
  if tuple(c['asset']['members'])!=STATE_KEYS:raise PermissionError('Only saved kinematics five members')
 return True

def validate_selected(d,t,p,selected):
 if d['states'].shape!=(1024,30,23,4) or d['states'].dtype!=np.float32 or t['positions'].shape!=(1024,30,23,2) or t['positions'].dtype!=np.float32:raise ValueError('Pinned1024 cache shape/F32')
 if d['valid'].shape!=(1024,30,23) or t['position_valid'].shape!=(1024,30,23) or d['valid'].dtype!=bool or t['position_valid'].dtype!=bool:raise ValueError('Explicit boolean masks')
 if d['teams'].shape!=(23,) or p['teams'].shape!=(23,) or not np.array_equal(d['teams'],np.array([0]*11+[1]*11+[2])):raise ValueError('Fixed both-team/BALL slots')
 if d['source_frame_ids'].shape!=(1024,30) or d['frame_times_s'].shape!=(1024,30):raise ValueError('Explicit frame/time axes')
 dm=unique_map(d);tm=unique_map(t);mapped=[];rows=[]
 for ix in selected:
  key=(str(p['match_ids'][ix]),int(p['source_indices'][ix]))
  if key not in dm or key not in tm:raise ValueError('Missing exact source key')
  di,ti=dm[key],tm[key]
  if di not in set(d['train_indices'].tolist()):raise ValueError('Original24 must remainTRAIN diagnostics')
  for name in ['entity_ids','source_frame_ids']:
   if not np.array_equal(d[name][di],t[name][ti]):raise ValueError('Truth/development identity: '+name)
  if not np.array_equal(d['entity_ids'][di].astype(str),p['entity_ids'][ix].astype(str)):raise ValueError('Prefix/cache entity-slot identity')
  if not np.array_equal(d['teams'],p['teams']):raise ValueError('Team slots changed')
  if not np.array_equal(d['states'][di,:10],p['history_states'][ix],equal_nan=True) or not np.array_equal(d['valid'][di,:10],p['history_valid'][ix]):raise ValueError('Prefix history != recorded past; no repair')
  if not np.array_equal(d['position_valid'][di],t['position_valid'][ti]):raise ValueError('Position mask alignment')
  frame=d['source_frame_ids'][di];clock=d['frame_times_s'][di]
  if not np.all(np.diff(frame)==1) or not np.allclose(np.diff(clock),DT,rtol=0.,atol=1e-10):raise ValueError('Actual cached clock not contiguous10Hz')
  vv=d['valid'][di];pv=t['position_valid'][ti];pos=t['positions'][ti];state=d['states'][di]
  if np.any(vv&~pv) or not np.isfinite(pos[pv]).all() or np.isfinite(pos[~pv]).any() or not np.isfinite(state[vv]).all():raise ValueError('Position/fullstate validity inconsistent')
  if np.any(state[~vv]!=0):raise ValueError('Legacy zero-erased fullstate convention changed')
  if not np.array_equal(state[...,:2][vv],pos[vv]):raise ValueError('Recorded state/truth XY scale or identity changed')
  if not np.array_equal(vv[1:],pv[1:]&pv[:-1]):raise ValueError('Selected fullstate!=consecutiveposition source contract')
  derived=(pos[1:]-pos[:-1])*np.float32(10.)
  if not np.array_equal(state[1:,:,2:][vv[1:]],derived[vv[1:]]):raise ValueError('RecordedV lineage changed')
  slots=p['operator_candidate_slots'][ix]
  if not vv[9,slots].all():raise ValueError('Originaltwo targets require actualfullstate t0')
  if [str(d['entity_ids'][di,j]) for j in slots]!=[str(x) for x in p['operator_candidate_ids'][ix]]:raise ValueError('Typed targetsourceIDs not exact')
  mapped.append((int(ix),di,ti))
  rows.append(dict(scene_index=int(ix),development_row=di,truth_row=ti,match_id=key[0],source_index=key[1],
   first_source_frame=int(frame[0]),t0_source_frame=int(frame[9]),last_source_frame=int(frame[29]),
   position_only_history_actor_frames=int((pv[:10]&~vv[:10]).sum()),position_only_future_actor_frames=int((pv[10:]&~vv[10:]).sum()),
   exact_history=True,exact_entity_and_target_slots=True,physical_observed_realizations=1,kinematic_groundtruth_not_latent_physiology=True))
 return mapped,rows
def common(case,ix,mid,source,coords,g,deriv=None,metric=None):
 r=dict(case_id=case['case_id'],role=case['role'],world=case['world'],seed=case['seed'],policy=case['policy'],mode=case['mode'],path=case['path'],scene_index=ix,match_id=mid,source_index=source,coordinates=coords,fixed_group=g)
 if deriv is not None:r['derivative']=deriv
 if metric is not None:r['metric']=metric
 return r
def masked(z,mask):return np.where(mask,z,np.nan)
def vals(kin,deriv,metric,g):
 z=np.asarray(kin[deriv][metric]);sc=np.asarray(kin[deriv]['turn_screen' if metric=='turn' else metric])
 return z[...,g],sc[...,g]
def strict_macro(v,expected):
 if set(v)!=set(expected) or any(x is None or not np.isfinite(x) for x in v.values()):return None
 return float(np.mean([v[x] for x in expected]))
def aggregate_buffers(buffers,case,physical_matches,tables):
 macro_rows=[]
 for key,buckets in sorted(buffers.items()):
  coords,g,deriv,metric=key;by={}
  for mid,parts in sorted(buckets.items()):
   arr=[np.concatenate([p[i].reshape(-1) for p in parts]) for i in range(len(parts[0]))]
   if len(arr)==2:
    z=metric_summary(arr[0],arr[1],metric)
    row={**{'factual_'+k:v for k,v in z.items()},**{'model_'+k:None for k in z}}
   else:
    row={}
    for prefix,z,s in [('factual_',arr[0],arr[1]),('model_',arr[2],arr[3])]:row.update({prefix+k:v for k,v in metric_summary(z,s,metric).items()})
   base=dict(case_id=case['case_id'],role=case['role'],world=case['world'],seed=case['seed'],policy=case['policy'],mode=case['mode'],path=case['path'],coordinates=coords,fixed_group=g,derivative=deriv,metric=metric,match_id=mid,aggregation='pooled_defined_agent_steps',realization_reused_by_model_draws=len(arr)==4)
   tables['by_match'].add(dict(**base,**row));by[mid]=row
  allparts=[p for parts in buckets.values() for p in parts]
  arr=[np.concatenate([p[i].reshape(-1) for p in allparts]) for i in range(len(allparts[0]))]
  row={}
  if len(arr)==2:
   z=metric_summary(arr[0],arr[1],metric)
   row={**{'factual_'+k:v for k,v in z.items()},**{'model_'+k:None for k in z}}
  else:
   for prefix,z,s in [('factual_',arr[0],arr[1]),('model_',arr[2],arr[3])]:row.update({prefix+k:v for k,v in metric_summary(z,s,metric).items()})
  base=dict(case_id=case['case_id'],role=case['role'],world=case['world'],seed=case['seed'],policy=case['policy'],mode=case['mode'],path=case['path'],coordinates=coords,fixed_group=g,derivative=deriv,metric=metric,match_id='ALL12',aggregation='pooled_defined_agent_steps',realization_reused_by_model_draws=len(arr)==4)
  tables['pooled'].add(dict(**base,**row))
  mr=dict(case_id=case['case_id'],role=case['role'],world=case['world'],seed=case['seed'],policy=case['policy'],mode=case['mode'],path=case['path'],coordinates=coords,fixed_group=g,derivative=deriv,metric=metric,planned_matches=len(physical_matches),matches_with_any_defined=sum((r['factual_defined_units']>0) or (r['model_defined_units'] is not None and r['model_defined_units']>0) for r in by.values()),aggregation='equal12match_means_no_survivor_renormalization',quantiles_not_averaged=True)
  prefixes=['factual_','model_']
  for prefix in prefixes:
   for name in ['mean','strict_rate','screen_rate']:
    mr[prefix+name+'_macro']=strict_macro({m:r[prefix+name] for m,r in by.items()},physical_matches)
  tables['macro'].add(mr);macro_rows.append(mr)
 return macro_rows
def run(args):
 contract,binding,output=preflight(args)
 validate_case_contract(binding,J(binding['runner_freeze']['path']))
 if np.__version__!='1.26.4':raise ValueError('Pinned NumPy1.26.4')
 for key in ['runner_freeze','runner_lock','selected_metadata','pitch_metadata']:check(binding[key])
 if J(binding['runner_freeze']['path'])['lock_sha256']!=binding['runner_lock']['sha256']:raise ValueError('Runner source freeze differs')
 for c in binding['cases']:check(c['manifest'])
 output.mkdir(exist_ok=False);started=time.monotonic()
 put(output/'execution_start.json',dict(pid=getpid(),stage='saved_world_reference',input_files=len(contract['files']),NN_forwards=0,fits=0))
 a=binding['assets'];d=load(a['development'],DEV_KEYS);t=load(a['position_truth'],TRUTH_KEYS);p=load(a['prefix'],PREFIX_KEYS)
 selmeta=J(binding['selected_metadata']['path'])['selected'];selected=p['selected']
 if selected.shape!=(24,) or not p['prefix_supported'][selected].all() or len(selmeta)!=24:raise ValueError('Fixed24 sourceprefix')
 for ix,r in zip(selected,selmeta):
  if int(ix)!=r['scene_index'] or str(p['match_ids'][ix])!=r['match_id'] or int(p['source_indices'][ix])!=r['source_index']:raise ValueError('Source-only selected24 order')
 mapped,ledger=validate_selected(d,t,p,selected)
 tables={k:Table(output/n) for k,n in dict(alignment='cohort_alignment.csv',agent='recorded_agent_step_metrics.csv',recorded='recorded_metric_windows.csv',recorded_joint='recorded_joint_windows.csv',own='model_own_anchor_windows.csv',own_joint='model_own_anchor_joint.csv',matched='matched_observability_windows.csv',matched_joint='matched_joint_windows.csv',by_match='metric_by_match.csv',pooled='metric_pooled.csv',macro='metric_equal_match_macro.csv',interop='old_metric_interoperability.csv').items()}
 observed_case=dict(case_id='RECORDED_ONCE',world='RECORDED',seed=-1,policy='observed',mode='one_realization',path='observed_natural',role='RECORDED_REFERENCE_PHYSICAL_ONCE')
 refs={};fbuf={};macro_rows=[];interop_count=0;maximum_error=0.
 check(binding['old_summary_csv']);oldrows={}
 with Path(binding['old_summary_csv']['path']).open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['path'] in ('n_empty','AB_shared') and r['fixed_group'] in GROUPS:oldrows[(r['world'],int(r['seed']),r['policy'],r['mode'],r['path'],int(r['scene_index']),r['coordinates'],r['fixed_group'])]=r
 try:
  for led,(ix,di,ti) in zip(ledger,mapped):
   tables['alignment'].add(led);mid=led['match_id'];si=led['source_index']
   h=p['history_states'][ix];hv=p['history_valid'][ix];anc,present,t0,age=anchor_from_history(h,hv)
   groups=fixed_groups(hv,p['identity_status'][ix],p['teams'],p['operator_candidate_slots'][ix])
   ref={}
   for coord,scale in [('native',np.asarray(binding['pitch_transforms'][mid]['standard_to_native'],np.float64)),('standard',np.ones(4))]:
    rec=recorded_metrics(d['states'][di],d['valid'][di],t['positions'][ti],t['position_valid'][ti],scale);ref[coord]=rec
    for der in DERIVS:
     for step in range(20):
      for slot in range(23):
       known=bool(groups['all_known_history_players'][slot]);descriptor_player=bool(p['identity_status'][ix,slot]=='VERIFIED_SOURCE_PLAYER_TEAM' and p['teams'][slot] in (0,1))
       r=dict(scene_index=ix,match_id=mid,source_index=si,slot=slot,entity_id=str(p['entity_ids'][ix,slot]),coordinates=coord,derivative=der,step=step+1,time_s=(step+1)*DT,initial_known_player=known,verified_recorded_player=descriptor_player,target=bool(groups['all_two_t0_valid_targets'][slot]),true_t0_fullvalid=bool(t0[slot]),stale_fullstate_anchor=bool(present[slot] and not t0[slot]),physical_realizations=1)
       for metric in METRICS:r[metric]=float(rec[der][metric][step,slot]) if descriptor_player and np.isfinite(rec[der][metric][step,slot]) else None
       r['velocity_supported']=descriptor_player and bool(rec[der]['valid'][step,slot]);r['acceleration_pair_supported']=descriptor_player and bool(rec[der]['pair_valid'][step,slot]);r['no_neural_or_physiological_groundtruth']=True;tables['agent'].add(r)
    for gn,g in groups.items():
     co=common(observed_case,ix,mid,si,coord,gn)
     tables['recorded_joint'].add(dict(**co,**joint_summary(rec,g)))
     for der in DERIVS:
      for metric in METRICS:
       vv,ss=vals(rec,der,metric,g);base=common(observed_case,ix,mid,si,coord,gn,der,metric)
       tables['recorded'].add(dict(**base,planned_agent_steps=20*int(g.sum()),physical_realizations=1,t0_fullvalid_players=int((g&t0).sum()),stale_anchor_players=int((g&~t0).sum()),position_only_first_speed_descriptors=int((g&rec['position_only_first_speed']).sum()) if der=='xy' and metric=='speed' else 0,**metric_summary(vv,ss,metric)))
       fbuf.setdefault((coord,gn,der,metric),{}).setdefault(mid,[]).append((vv,ss))
   refs[ix]=dict(ref=ref,anchor=anc,present=present,t0=t0,groups=groups,mid=mid,source=si)
  matches=sorted({r['match_id'] for r in ledger});macro_rows+=aggregate_buffers(fbuf,observed_case,matches,tables)
  for case in binding['cases']:
   check(case['manifest']);x=load(case['asset'],STATE_KEYS)
   M=32 if case['mode']=='sample' else 1
   if x['states'].shape!=(24*M,20,23,4) or x['states'].dtype!=np.float32 or x['support'].shape!=(24*M,20,23) or x['support'].dtype!=bool or not np.array_equal(x['scene_indices'],np.repeat(selected,M)):raise ValueError('Frozen case dtype/shape/source-draw order')
   if x['completed_steps'].shape!=(24*M,) or x['scene_status'].shape!=(24*M,) or not np.issubdtype(x['completed_steps'].dtype,np.integer) or not np.isin(x['completed_steps'],np.arange(21)).all():raise ValueError('Saved completion metadata axes/range')
   states=x['states'].reshape(24,M,20,23,4);sp=x['support'].reshape(24,M,20,23);steps=x['completed_steps'].reshape(24,M)
   if np.any((x['scene_status']=='COMPLETED')&(x['completed_steps']!=20)):raise ValueError('Falsecomplete')
   buf={}
   for j,ix0 in enumerate(selected):
    ix=int(ix0);ref=refs[ix];mid=ref['mid'];si=ref['source']
    for coord,scale in [('native',np.asarray(binding['pitch_transforms'][mid]['standard_to_native'],np.float64)),('standard',np.ones(4))]:
     kin=kinematics(states[j],sp[j],steps[j],ref['anchor'],ref['present'],scale);rec=ref['ref'][coord]
     for gn,g in ref['groups'].items():
      co=common(case,ix,mid,si,coord,gn);ownj=joint_summary(kin,g);factj=joint_summary(rec,g)
      tables['own_joint'].add(dict(**co,**ownj))
      comparable=bool(factj['complete_required_draws']==1)
      tables['matched_joint'].add(dict(**co,factual_physical_realizations=1,factual_full20_roster_observed=comparable,model_planned_draws=M,model_own_complete_draws=ownj['complete_required_draws'],factual_strict_pass=factj['strict_pass_draws'] if comparable else None,factual_screen_pass=factj['screen_pass_draws'] if comparable else None,model_strict_pass_draws=ownj['strict_pass_draws'] if comparable else None,model_screen_pass_draws=ownj['screen_pass_draws'] if comparable else None,missing_reference_is_unknown_not_failed=True,controlled_is_not_counterfactual_truth_error=case['role']!='PRIMARY_NATURAL'))
      old=oldrows[(case['world'],case['seed'],case['policy'],case['mode'],case['path'],ix,coord,gn)]
      if ownj['strict_pass_draws']!=int(old['joint_strict_engineering_reference_pass_draws']) or ownj['screen_pass_draws']!=int(old['joint_screen_engineering_reference_pass_draws']):raise ValueError('Oldroot joint semantic interoperability')
      for der in DERIVS:
       for metric in METRICS:
        vv,ss=vals(kin,der,metric,g);os=metric_summary(vv,ss,metric);base=common(case,ix,mid,si,coord,gn,der,metric)
        tables['own'].add(dict(**base,model_planned_draws=M,own_anchor_semantic='original fullstate causalanchor; stale firstbridge is modelonly diagnostic',**os))
        for field,oldfield in [('defined_units',f'{der}_{metric}_comparisons'),('strict_violations',f'{der}_{metric}_violations'),('screen_defined_units',f'{der}_{metric}_screen_comparisons'),('screen_violations',f'{der}_{metric}_screen_violations')]:
         if os[field]!=int(old[oldfield]):raise ValueError('Oldmetric exactcount '+oldfield)
        a=os['maximum'];b=old[f'{der}_{metric}_maximum']
        if (a is None)!=(b==''):raise ValueError('Old maximum NA contract')
        err=abs(a-float(b)) if a is not None else 0.;maximum_error=max(maximum_error,err)
        if a is not None and err>5e-12+5e-13*abs(float(b)):raise ValueError('Old maximum float interoperability')
        tables['interop'].add(dict(**base,exact_counts=True,maximum_absolute_difference=err,abs_tolerance=5e-12,rel_tolerance=5e-13));interop_count+=1
        matched,arr=matched_metric(kin,rec,g,der,metric,case['mode'],case['world'])
        tables['matched'].add(dict(**base,controlled_is_not_counterfactual_truth_error=case['role']!='PRIMARY_NATURAL',**matched))
        buf.setdefault((coord,gn,der,metric),{}).setdefault(mid,[]).append(arr)
   macro_rows+=aggregate_buffers(buf,case,matches,tables)
 finally:
  for tb in tables.values():tb.close()
 # Five seeds are repeated model pipelines on the same observed24, not realN.
 st=Table(output/'five_seed_macro_summary.csv')
 try:
  keyed={}
  for r in macro_rows:
   if r['world'] not in ('D','E'):continue
   k=tuple(r[x] for x in ['world','policy','mode','path','coordinates','fixed_group','derivative','metric'])
   keyed.setdefault(k,[]).append(r)
  for key,rr in sorted(keyed.items()):
   if len(rr)!=5 or {r['seed'] for r in rr}!=set(range(5)):raise ValueError('All5seed macro rows')
   row=dict(zip(['world','policy','mode','path','coordinates','fixed_group','derivative','metric'],key));row.update(model_seeds=5,observed_physical_windows=24,observed_matches=12,seed_SD_not_data_CI=True)
   for name in ['model_mean_macro','model_strict_rate_macro','model_screen_rate_macro']:
    v=[r[name] for r in rr]
    row[name+'_mean']=float(np.mean(v)) if all(x is not None for x in v) else None
    row[name+'_sample_SD']=float(np.std(v,ddof=1)) if all(x is not None for x in v) else None
   st.add(row)
 finally:st.close()
 if {**{k:t.n for k,t in tables.items()},'five_seed_macro_summary':st.n}!=EXPECTED_ROWS:raise ValueError('All13 planned table row counts must hold, including NA rows')
 for rec in binding['assets'].values():check(rec)
 for case in binding['cases']:check(case['manifest']);check(case['asset'])
 for key in ['runner_freeze','runner_lock','selected_metadata','pitch_metadata','old_summary_csv']:check(binding[key])
 for rec in contract['files']:
  check(dict(path=str(Path(args.input_root).resolve()/rec['path']),sha256=rec['sha256']))
 comparison=[]
 for name,ref in contract['reference_csvs'].items():
  p=output/name;comparison.append(dict(file=name,bytes=p.stat().st_size,sha256=sha(p),reference_exact=sha(p)==ref['sha256']))
 ok=all(r['reference_exact'] for r in comparison)
 put(output/'receipt.json',dict(status='EXACT_SAVED_WORLD_REFERENCE' if ok else 'DIFFERENCE_REQUIRES_REVIEW',
  elapsed_seconds=time.monotonic()-started,comparisons=comparison,model_cases=63,physical_windows=24,physical_matches=12,
  row_counts={**{k:t.n for k,t in tables.items()},'five_seed_macro_summary':st.n},
  old_metric_interoperability=dict(rows=interop_count,maximum_float64_error=maximum_error),
  NN_forwards=0,fits=0,source_files_unchanged=True,
  scope='Recompute from saved worlds and same-window recorded kinematics; no new world generation, labels or intervention effect validation.'))
 print('EXACT_SAVED_WORLD_REFERENCE' if ok else 'DIFFERENCE_REQUIRES_REVIEW',round(time.monotonic()-started,3),flush=True)
 if not ok:raise SystemExit(2)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input-root',required=True);ap.add_argument('--output',required=True)
 args=ap.parse_args()
 try:run(args)
 except BaseException as e:
  o=Path(args.output)
  if o.is_dir() and (o/'execution_start.json').is_file() and not (o/'receipt.json').exists():
   put(o/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
  raise
