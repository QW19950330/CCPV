"""Natural-path factual player errors on separately recovered detected-only truth."""
from pathlib import Path
import argparse
import numpy as np
from analyze_responses import sha,read,dump,csv_rows,frozen_run,MATCHES,CELLS,statistics,strict_macro
SCORES=('ADE20','FDE1s','FDE2s')
def validate_natural_population(support,steps,statuses,history_seen,expected_complete,reference_status=None):
 complete=statuses=='COMPLETED';unsupported=np.char.startswith(statuses.astype(str),'UNSUPPORTED')
 if not np.array_equal(complete,expected_complete):raise ValueError('Natural completion differs from prelocked input eligibility')
 if not np.all(complete|unsupported) or not np.all(steps[complete]==20) or not np.all(steps[unsupported]==0):raise ValueError('Natural path has unexplained partial failure')
 expected=history_seen[:,None,:]&complete[:,None,None]
 if not np.array_equal(support,np.broadcast_to(expected,support.shape)):raise ValueError('Natural support differs from fixed historical population')
 if reference_status is not None and not np.array_equal(statuses,reference_status):raise ValueError('Natural population differs between frozen D and CV')
 return statuses.copy()

def row_scores(errors):
 n=len(errors);out=np.full((n,3),np.nan);counts=np.zeros((n,3),np.int64)
 for k,part in enumerate((errors,errors[:,9],errors[:,19])):
  ax=tuple(range(1,part.ndim));count=np.isfinite(part).sum(axis=ax);total=np.nansum(part,axis=ax)
  counts[:,k]=count;np.divide(total,count,out=out[:,k],where=count>0)
 return out,counts

def summarize(scores,counts,match_ids):
 rows=[];macros={}
 for cell,x in scores.items():
  for k,name in enumerate(SCORES):
   means=[]
   for mid in MATCHES:
    ix=match_ids==mid;s=statistics(x[ix,k]);means.append(s['mean'])
    rows.append(dict(cell=cell,match_id=mid,metric=name,unit='standardized_metre_equivalent',scored_player_time_points=int(counts[cell][ix,k].sum()),**s))
   macros[(cell,name)]=strict_macro(means)
 seed=[]
 for name in SCORES:
  v=np.asarray([macros[(c,name)] for c in CELLS[1:]],float);ok=np.isfinite(v)
  seed.append(dict(metric=name,D_mean=float(v.mean()) if ok.all() else np.nan,D_seed_sd=float(v.std(ddof=1)) if ok.all() else np.nan,n_seed_with_four_matches=int(ok.sum()),CV_once=macros.get((CELLS[0],name),np.nan),note='Window means, then four equal-match means; seed SD is initialization variation.'))
 return rows,macros,seed

def run(run_dir,freeze_sha,truth_dir,truth_completion_sha,out):
 comp=frozen_run(run_dir,freeze_sha);truth_dir=Path(truth_dir)
 if sha(truth_dir/'completion.json')!=truth_completion_sha:raise ValueError('Explicit truth completion SHA mismatch')
 tr=read(truth_dir/'completion.json')
 if tr['status']!='COMPLETE_DETECTED_ONLY_PLAYER_TRUTH_AFTER_PREDICTION_FREEZE' or tr['prediction_completion_sha256']!=freeze_sha:raise ValueError('Truth not bound to same prediction freeze')
 tp=truth_dir/tr['path']
 if sha(tp)!=tr['sha256']:raise ValueError('Truth SHA changed')
 with np.load(tp,allow_pickle=False) as z:truth={k:z[k] for k in z.files}
 y=truth['positions'][:,:,:22].astype(np.float64);tv=truth['position_valid'][:,:,:22]
 if y.shape!=(4992,20,22,2) or tv.dtype!=np.bool_:raise ValueError('Truth axes/mask')
 if not np.isfinite(y[tv]).all() or np.isfinite(y[~tv]).any():raise ValueError('Invalid truth must be NA, observed truth finite')
 pp=Path(tr['prepared_prefix_path'])
 if sha(pp)!=tr['prepared_prefix_sha256']:raise ValueError('Prepared historical support SHA changed')
 with np.load(pp,allow_pickle=False) as z:
  history_seen=z['history_valid'].any(1)[:,:22];prefix_source=z['source_indices'];prefix_match=z['match_ids']
  expected_complete=(z['operator_candidate_slots'][:,0]>=0)&~z['technical_unsupported']
 if not np.array_equal(prefix_source,truth['source_indices']) or not np.array_equal(prefix_match,truth['match_ids']):raise ValueError('Historical support/truth identity mismatch')
 errors={};all_support={};sources=[];natural_reference_status=None
 for cr in comp['cells']:
  chunks=[];ids=[];mids=[];ss=[];step_chunks=[];status_chunks=[]
  for br in cr['blocks']:
   p=Path(run_dir)/cr['cell']/br['path']
   if sha(p)!=br['sha256']:raise ValueError('Prediction SHA changed')
   with np.load(p,allow_pickle=False) as z:
    pred=z['states'][:,0,:,:22,:2].astype(np.float64);sup=z['support'][:,0,:,:22];src=z['source_indices'];mid=z['match_ids']
   chunks.append(pred);ss.append(sup);ids.append(src);mids.append(mid)
   with np.load(p,allow_pickle=False) as z:
    step_chunks.append(z['completed_steps'][:,0]);status_chunks.append(z['status'][:,0])
   sources.append(dict(cell=cr['cell'],path=str(p),sha256=br['sha256']))
  pred=np.concatenate(chunks);sup=np.concatenate(ss);steps=np.concatenate(step_chunks);statuses=np.concatenate(status_chunks)
  natural_reference_status=validate_natural_population(sup,steps,statuses,history_seen,expected_complete,natural_reference_status)
  if not np.array_equal(np.concatenate(ids),truth['source_indices']) or not np.array_equal(np.concatenate(mids),truth['match_ids']):raise ValueError('Truth/prediction row identity mismatch')
  if not np.isfinite(pred[sup]).all():raise ValueError('Nonfinite supported natural path')
  mask=sup&tv;e=np.linalg.norm(pred-y,axis=-1);e[~mask]=np.nan
  errors[cr['cell']]=e;all_support[cr['cell']]=mask
 scores={};counts={};paired_scores={};paired_counts={};paired_components={}
 for cell,e in errors.items():scores[cell],counts[cell]=row_scores(e)
 for cell,e in errors.items():
  common=np.isfinite(e)&np.isfinite(errors[CELLS[0]])
  ds,dn=row_scores(np.where(common,e,np.nan));cs,cn=row_scores(np.where(common,errors[CELLS[0]],np.nan))
  assert np.array_equal(dn,cn)
  paired_scores[cell]=ds-cs;paired_counts[cell]=dn;paired_components[cell]=(ds,cs)
 rows,macro,seed=summarize(scores,counts,truth['match_ids']);pr,pm,ps=summarize(paired_scores,paired_counts,truth['match_ids'])
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 csv_rows(out/'natural_fixture_errors.csv',rows);csv_rows(out/'natural_equal_match_five_seed.csv',seed)
 csv_rows(out/'paired_D_minus_CV_fixture_errors.csv',pr);csv_rows(out/'paired_D_minus_CV_five_seed.csv',ps)
 csv_rows(out/'natural_seed_macros.csv',[dict(cell=c,metric=m,four_match_equal_mean=v) for (c,m),v in macro.items()])
 def window_rows():
  for cell,x in scores.items():
   for i,source in enumerate(truth['source_indices']):
    for k,name in enumerate(SCORES):
     ds,cs=paired_components[cell]
     yield dict(cell=cell,source_index=int(source),match_id=str(truth['match_ids'][i]),metric=name,value=x[i,k],n_scored_player_time_points=int(counts[cell][i,k]),paired_D_value=ds[i,k],paired_CV_value=cs[i,k],paired_D_minus_CV=paired_scores[cell][i,k],n_paired_player_time_points=int(paired_counts[cell][i,k]))
 csv_rows(out/'natural_window_errors.csv.gz',window_rows())
 curve=[]
 for cell,e in errors.items():
  for t in range(20):
   v=row_scores(np.repeat(e[:,t,None,:],20,axis=1))[0][:,0] # same per-window player mean, no time reweighting
   for mid in MATCHES:curve.append(dict(cell=cell,match_id=mid,time_s=(t+1)/10,n_scored_player_points=int(np.isfinite(e[truth['match_ids']==mid,t]).sum()),**statistics(v[truth['match_ids']==mid])))
 csv_rows(out/'natural_time_error_curves.csv',curve)
 report=['自然轨迹事实预测统计（独立detected-only真值）','仅使用预测冻结后恢复的原提供方detected=true、finite xy球员位置；不读事件、不从清零state恢复位置、不拟合模型。','ADE20为每窗20时刻全部共同有效球员-时刻距离均值；FDE1/2s为对应时刻共同有效球员均值。仅历史模型支持且未来真实位置存在的点可评分。','先每窗均值、再每场均值、四场等权；D五seed样本SD为初始化变化。不同指标支持数独立保留；D−CV在完全相同球员/时刻/窗口支持上计算。','坐标是105×68标准化米等价值，不能冒称新增原生物理精度认证。']
 for r in seed:report.append(f"{r['metric']}: D={r['D_mean']:.9g} ± {r['D_seed_sd']:.9g}；CV={r['CV_once']:.9g}。")
 report.append('负/零/不利结果全部保留；事实误差不验证输入reset后的真实反事实响应或Goal收益。')
 (out/'中文事实预测报告.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
 dump(out/'source_manifest.json',dict(prediction_completion_sha256=freeze_sha,truth_completion_sha256=truth_completion_sha,truth_sha256=tr['sha256'],code_sha256=sha(__file__),blocks=sources))
 dump(out/'completion.json',dict(status='COMPLETE_NATURAL_FACTUAL_SAVED_PREDICTION_SCORING',planned_windows=4992,matches=4,learned_seeds=5,CV_once=True,truth='detected-only independently restored player xy',model_calls=0,training_updates=0,event_labels_read=0,performance_gate=False))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--prediction-freeze-sha256',required=True);p.add_argument('--truth-dir',required=True);p.add_argument('--truth-completion-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.run_dir,a.prediction_freeze_sha256,a.truth_dir,a.truth_completion_sha256,a.out)
