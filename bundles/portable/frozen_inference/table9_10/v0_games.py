"""Original maintained-v0 full game computation; explicit readouts and coverage-only old mask."""
from collections import defaultdict
import itertools
import numpy as np
import game_math as V
from game_math import V0_PATHS as PATHS,exact_v0_game
from portable_io import save,table,write as put,output_record as record,now

def description(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return dict(count=0,mean=None,sd=None,p05=None,median=None,p95=None,min=None,max=None,positive_fraction=None,negative_fraction=None,zero_fraction=None)
 return dict(count=int(len(x)),mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None,p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()),positive_fraction=float(np.mean(x>0)),negative_fraction=float(np.mean(x<0)),zero_fraction=float(np.mean(x==0)))

def run(z,oldmask,out):
 if z['paths'].astype(str).tolist()!=list(PATHS) or z['q_logp'].shape!=(2,5,4,4,768):raise ValueError('Frozen4coalition dimensions')
 q=np.exp(z['q_logp']);source_support=z['supported'];game_ok=source_support.all(2)&np.isfinite(q).all(0).all(2)
 common=game_ok.all((0,1));mids=z['match_ids'].astype(str);matches=sorted(set(mids))
 out.mkdir(exist_ok=False)
 save(out/'masks.npz',game_supported=game_ok,common_all_world_seed_families=common,old_natural_common_reference_only=oldmask,source_indices=z['source_indices'],match_ids=z['match_ids'])
 scene=[];distribution=[];matchrows=[];seedrows=[];orders={};metrics=tuple(exact_v0_game(np.zeros((1,4))))
 for fi,family in enumerate(V.VALUE_FAMILIES):
  for seed in range(5):
   for ai,arm in enumerate(V.WORLD_ARMS):
    vals=q[fi,seed,ai].T;g=exact_v0_game(vals);gap=g['phi_A']-g['phi_B'];orders[fi,seed,ai]=np.sign(gap)
    ok=game_ok[seed,ai]
    for w in range(768):
     r=dict(value_family=family,world_arm=arm,seed=seed,scene_index=w,match_id=mids[w],source_index=int(z['source_indices'][w]),actor_A_source_id=str(z['target_source_ids'][w,0]),actor_B_source_id=str(z['target_source_ids'][w,1]),game_supported=bool(ok[w]),common_all_world_seed_families=bool(common[w]),reference='BOTH_ACTORS_MAINTAIN_T0_V0_NOT_NATURAL_EMPTY')
     r.update({f'q_{p}':float(vals[w,j]) if ok[w] else None for j,p in enumerate(PATHS)})
     r.update({k:float(g[k][w]) if ok[w] else None for k in metrics})
     r.update(actor_phi_A_minus_phi_B=float(gap[w]) if ok[w] else None,actor_order_sign=int(np.sign(gap[w])) if ok[w] else None,actor_order_exact_tie=bool(gap[w]==0) if ok[w] else None)
     scene.append(r)
    for scope,mask in [('available_cell',ok),('common_all_world_seed_families',common)]:
     for metric in metrics:
      x=g[metric];distribution.append(dict(value_family=family,world_arm=arm,seed=seed,scope=scope,metric=metric,planned=768,**description(x[mask])))
      means=[]
      for mid in matches:
       take=mask&(mids==mid);v=float(x[take].mean()) if take.any() else None
       matchrows.append(dict(value_family=family,world_arm=arm,seed=seed,scope=scope,metric=metric,match_id=mid,planned=64,supported=int(take.sum()),mean=v))
       if v is not None:means.append(v)
      seedrows.append(dict(value_family=family,world_arm=arm,seed=seed,scope=scope,metric=metric,planned=768,supported=int(mask.sum()),matches_supported=len(means),equal12match_macro=float(np.mean(means)) if len(means)==12 else None,available_match_mean=float(np.mean(means)) if means else None))
 summary=[];groups=defaultdict(list)
 for r in seedrows:groups[r['value_family'],r['world_arm'],r['scope'],r['metric']].append(r)
 for (family,arm,scope,metric),rr in groups.items():
  x=[r['equal12match_macro'] for r in rr if r['equal12match_macro'] is not None]
  summary.append(dict(value_family=family,world_arm=arm,scope=scope,metric=metric,seeds_supported=len(x),five_seed_mean=float(np.mean(x)) if len(x)==5 else None,paired_initialization_seed_SD=float(np.std(x,ddof=1)) if len(x)==5 else None,uncertainty='Initialization sensitivity, not MC or independent-match population CI; CVworld only once'))
 actor=[]
 def pair(kind,label,a,c):actor.append(dict(kind=kind,comparison=label,rank_unit='FIXED_ACTOR_A_VS_B_WITHIN_MAINTAINED_V0_GAME',scope='common_all_world_seed_families',planned=768,**V.actor_agreement(a,c,common)))
 for fi,ai in itertools.product(range(2),range(4)):
  for s,t in itertools.combinations(range(5),2):pair('seed',f'{V.VALUE_FAMILIES[fi]}/{V.WORLD_ARMS[ai]}/s{s}:s{t}',orders[fi,s,ai],orders[fi,t,ai])
 for s,ai in itertools.product(range(5),range(4)):pair('value_family',f's{s}/{V.WORLD_ARMS[ai]}',orders[0,s,ai],orders[1,s,ai])
 for fi,s in itertools.product(range(2),range(5)):
  for ai,bi in itertools.combinations(range(4),2):pair('world_arm',f'{V.VALUE_FAMILIES[fi]}/s{s}/{V.WORLD_ARMS[ai]}:{V.WORLD_ARMS[bi]}',orders[fi,s,ai],orders[fi,s,bi])
 files=[]
 for name,rr in [('game_scene.csv',scene),('distributions_by_seed.csv',distribution),('game_by_match.csv',matchrows),('game_by_seed.csv',seedrows),('game_summary.csv',summary),('actor_order_agreement.csv',actor)]:
  path=out/name;table(path,rr);files.append(record(path))
 files.append(record(out/'masks.npz'))
 eff=[abs(r['efficiency_error']) for r in scene if r['game_supported']]
 put(out/'summary.json',dict(status='COMPLETED_EXACT_MAINTAINED_V0_K2_GAME_MODEL_INTERNAL_ONLY',utc=now(),planned_game_rows=len(scene),planned_windows=768,common_supported=int(common.sum()),old_common_reference_only=int(oldmask.sum()),common_new_and_old=int((common&oldmask).sum()),new_only=int((common&~oldmask).sum()),old_only=int((oldmask&~common).sum()),max_efficiency_error=max(eff,default=0.),primary_family=V.VALUE_FAMILIES[0],secondary_family=V.VALUE_FAMILIES[1],all_negative_zero_positive_retained=True,no_guard_filter=True,no_natural_reference_relabel=True,no_physical_or_tactical_efficacy_claim=True,world_forwards=0,fits=0,labels_read=False))
 files.append(record(out/'summary.json'))
 if len(scene)!=30720:raise AssertionError('All planned games retained')
 return dict(scene_rows=len(scene),common=int(common.sum()),intersection=int((common&oldmask).sum()))
