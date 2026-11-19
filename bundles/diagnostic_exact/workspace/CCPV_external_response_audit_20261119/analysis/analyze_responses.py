"""Saved-only frozen response analysis. No model imports, training or truth reads."""
from pathlib import Path
import argparse, collections, csv, gzip, hashlib, json, math
import numpy as np
PATHS=('natural','none_every_step','first_A','persistent_A','maintained_v0_A','first_provenance_only_A')
PAIRS=((2,0),(3,0),(4,0),(5,0),(2,5),(3,4))
CONTRASTS=('first-natural','persistent-natural','v0-natural','sham-natural','first-sham','persistent-v0')
METRICS=('actor_dx','actor_dy','actor_distance','actor_projection','other_player_rms')
MATCHES=('skillcorner:1927964','skillcorner:2006363','skillcorner:2015213','skillcorner:2017461')
CELLS=('CV_s-1',)+tuple(f'D_coupled_ar_s{s}' for s in range(5))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def dump(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csv_rows(p,rows,fields=None):
 rows=iter(rows);first=next(rows,None)
 if first is None:
  if fields is None:raise ValueError('No CSV schema')
 else:fields=list(first) if fields is None else fields
 op=gzip.open if str(p).endswith('.gz') else open
 with op(p,'xt',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  def emit(r):w.writerow({k:('' if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in r.items()})
  if first is not None:emit(first)
  for row in rows:emit(row)
def frozen_run(root,expected):
 root=Path(root);cp=root/'completion.json'
 if not expected or sha(cp)!=expected:raise ValueError('Explicit prediction completion SHA required and must match')
 r=read(cp)
 if r.get('status')!='COMPLETE_FIXED4992_SIX_CELLS_SIX_PATHS' or r.get('planned_source_windows')!=4992 or tuple(r.get('paths',()))!=PATHS:raise ValueError('Prediction is not the required frozen full run')
 if [x['cell'] for x in r['cells']]!=list(CELLS):raise ValueError('Six frozen cells required')
 return r

def validate_block(a):
 n=len(a['source_indices']);expected={'states':(n,6,20,23,4),'support':(n,6,20,23),'completed_steps':(n,6),'status':(n,6),'actor_difference_vectors':(n,6,20,2),'actor_distance':(n,6,20),'actor_direction_projection':(n,6,20),'other_player_rms':(n,6,20),'actor_slots':(n,),'technical_unsupported':(n,),'match_ids':(n,)}
 for k,s in expected.items():
  if a[k].shape!=s:raise ValueError((k,a[k].shape,s))
 if a['support'].dtype!=np.bool_:raise ValueError('support must be bool')
 st=a['completed_steps'];slot=a['actor_slots'];support=a['support']
 if ((st<0)|(st>20)).any() or ((slot<-1)|(slot>21)).any():raise ValueError('Invalid step/actor')
 if (support & (np.arange(20)[None,None,:,None]>=st[:,:,None,None])).any():raise ValueError('Support after completed step')
 if support[a['technical_unsupported']].any():raise ValueError('Technical unsupported row has predicted support')
 if not np.isfinite(a['states'][support]).all():raise ValueError('Nonfinite supported generated state')
 ok=np.zeros((n,6,20),bool)
 for b in range(n):
  if slot[b]<0:continue
  for c,(l,r) in enumerate(PAIRS):
   ok[b,c]=support[b,l,:,slot[b]]&support[b,r,:,slot[b]]
   got=a['actor_difference_vectors'][b,c]
   expected_v=a['states'][b,l,:,slot[b],:2]-a['states'][b,r,:,slot[b],:2]
   if not np.array_equal(got[ok[b,c]],expected_v[ok[b,c]],equal_nan=True):raise ValueError('Saved actor vector does not match states')
 if np.isfinite(a['actor_difference_vectors'][~ok]).any():raise ValueError('Unsupported actor vector not NA')
 vv=a['actor_difference_vectors'];norm=np.linalg.norm(vv,axis=-1)
 if not np.allclose(norm,a['actor_distance'],atol=1e-6,rtol=1e-6,equal_nan=True):raise ValueError('Distance/vector mismatch')
 if (np.abs(a['actor_direction_projection'][ok])-norm[ok]>1e-5).any():raise ValueError('Projection exceeds distance')
 out=np.concatenate((vv,a['actor_distance'][...,None],a['actor_direction_projection'][...,None],a['other_player_rms'][...,None]),axis=-1).astype(np.float64)
 out[~ok]=np.nan
 return out,ok

def statistics(x):
 x=np.asarray(x);good=x[np.isfinite(x)];n=len(good)
 return dict(n_planned=len(x),n_supported=n,n_na=len(x)-n,mean=float(good.mean()) if n else np.nan,median=float(np.median(good)) if n else np.nan,q10=float(np.quantile(good,.1)) if n else np.nan,q90=float(np.quantile(good,.9)) if n else np.nan,window_sd=float(good.std(ddof=1)) if n>1 else np.nan)
def strict_macro(values):
 v=np.asarray(values,dtype=float);return float(v.mean()) if len(v)==4 and np.isfinite(v).all() else np.nan

def fixture_curves(cell,a,kind='response'):
 rows=[];macros={}
 for ci,cname in enumerate(CONTRASTS):
  for ti in range(20):
   for mi,mname in enumerate(METRICS):
    means=[]
    for mid in MATCHES:
     x=a['metrics'][a['match_ids']==mid,ci,ti,mi];s=statistics(x);means.append(s['mean'])
     rows.append(dict(cell=cell,kind=kind,match_id=mid,contrast=cname,time_s=round((ti+1)*.1,1),metric=mname,unit='standardized_metre_equivalent',**s))
    macros[(cname,ti,mname)]=strict_macro(means)
 return rows,macros

def seed_summary(macros,kind):
 rows=[]
 for key in macros[CELLS[0]]:
  c,t,m=key;v=np.array([macros[name][key] for name in CELLS[1:]],float);finite=np.isfinite(v)
  rows.append(dict(kind=kind,contrast=c,time_s=round((t+1)*.1,1),metric=m,n_seed_planned=5,n_seed_complete_four_matches=int(finite.sum()),D_five_seed_mean=float(v.mean()) if finite.all() else np.nan,D_seed_sd=float(v.std(ddof=1)) if finite.all() else np.nan,CV_once=macros[CELLS[0]][key],note='SD is initialization variation, not match uncertainty; strict four-match macro within every seed'))
 return rows

def load_cell(root,cr,ledger):
 parts=[];source=[]
 for br in cr['blocks']:
  bp=Path(root)/cr['cell']/br['path'];rp=bp.with_suffix('.json')
  if sha(bp)!=br['sha256']:raise ValueError('Prediction block SHA mismatch')
  rec=read(rp)
  if rec['sha256']!=br['sha256']:raise ValueError('Block receipt mismatch')
  with np.load(bp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
  metrics,ok=validate_block(a)
  if len(a['source_indices'])!=rec['stop']-rec['start']:raise ValueError('Block bounds')
  for b,src in enumerate(a['source_indices']):
   for k,pn in enumerate(PATHS):
    reason=rec['reasons'][b][k]
    ledger.append(dict(cell=cr['cell'],source_index=int(src),match_id=str(a['match_ids'][b]),actor_slot=int(a['actor_slots'][b]),technical_unsupported=bool(a['technical_unsupported'][b]),path=pn,completed_steps=int(a['completed_steps'][b,k]),status=str(a['status'][b,k]),reason=json.dumps(reason,ensure_ascii=False)))
  parts.append(dict(metrics=metrics,source_indices=a['source_indices'],match_ids=a['match_ids'],actor_slots=a['actor_slots'],technical_unsupported=a['technical_unsupported'],status=a['status'],completed_steps=a['completed_steps'],ok=ok))
  source.append(dict(path=str(bp),sha256=br['sha256'],receipt_sha256=sha(rp)))
 merged={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
 if len(merged['source_indices'])!=4992 or len(set(merged['source_indices'].tolist()))!=4992:raise ValueError('Full unique4992 required')
 if [int((merged['match_ids']==m).sum()) for m in MATCHES]!=[1308,1261,1312,1111]:raise ValueError('Fixed four-match counts changed')
 return merged,source

def endpoint_rows(cells):
 for cell,a in cells.items():
  for b,src in enumerate(a['source_indices']):
   for ci,c in enumerate(CONTRASTS):
    for ti in (1,9,19):
     yield dict(cell=cell,source_index=int(src),match_id=str(a['match_ids'][b]),contrast=c,time_s=(ti+1)/10,**{m:float(a['metrics'][b,ci,ti,j]) for j,m in enumerate(METRICS)})
def run(root,freeze_sha,out):
 completion=frozen_run(root,freeze_sha);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 cells={};ledger=[];sources=[]
 for cr in completion['cells']:
  a,pins=load_cell(root,cr,ledger);cells[cr['cell']]=a;sources+=pins
 cv=cells[CELLS[0]]
 for cell,a in cells.items():
  for k in ('source_indices','match_ids','actor_slots','technical_unsupported'):
   if not np.array_equal(a[k],cv[k]):raise ValueError('Cross-cell cohort alignment changed: '+k)
 csv_rows(out/'row_path_status.csv.gz',ledger)
 coverage=[]
 for cell,a in cells.items():
  for m in MATCHES:
   ix=a['match_ids']==m
   for pi,pn in enumerate(PATHS):
    counts=collections.Counter(a['status'][ix,pi]);n=int(ix.sum())
    for status,count in sorted(counts.items()):coverage.append(dict(cell=cell,match_id=m,path=pn,status=status,n_status=count,n_planned=n,n_actor_candidates=int((ix&(a['actor_slots']>=0)&~a['technical_unsupported']).sum()),n_technical_unsupported=int((ix&a['technical_unsupported']).sum())))
 csv_rows(out/'path_status_counts.csv',coverage)
 allrows=[];macro={}
 for cell,a in cells.items():
  rows,mm=fixture_curves(cell,a);allrows+=rows;macro[cell]=mm
 csv_rows(out/'fixture_curves.csv',allrows);csv_rows(out/'five_seed_equal_match_summary.csv',seed_summary(macro,'response'))
 csv_rows(out/'window_endpoints.csv.gz',endpoint_rows(cells))
 decay=[];decomposition=[];decompmax=0.
 for cell,a in cells.items():
  x=a['metrics'];res=x[:,0,:,:2]-x[:,4,:,:2]-x[:,3,:,:2]
  if np.isfinite(res).any():decompmax=max(decompmax,float(np.nanmax(np.abs(res))))
  for m in MATCHES:
   ix=a['match_ids']==m
   for ci,c in enumerate(CONTRASTS):decay.append(dict(cell=cell,match_id=m,contrast=c,metric='distance_2s_minus_0.2s',unit='standardized_metre_equivalent',**statistics(x[ix,ci,19,2]-x[ix,ci,1,2])))
   for ti in range(20):
    rr=res[ix,ti];good=np.isfinite(rr).all(-1)
    decomposition.append(dict(cell=cell,match_id=m,time_s=(ti+1)/10,n_planned=int(ix.sum()),n_triplet_supported=int(good.sum()),max_abs_vector_residual=float(np.abs(rr[good]).max()) if good.any() else np.nan,mean_distance_first_natural=statistics(x[ix,0,ti,2])['mean'],mean_distance_first_sham=statistics(x[ix,4,ti,2])['mean'],mean_distance_sham_natural=statistics(x[ix,3,ti,2])['mean'],note='Distance columns may have different pairwise support and are not additive; vector residual uses triple support.'))
 csv_rows(out/'fixture_decay.csv',decay);csv_rows(out/'sham_vector_decomposition.csv',decomposition)
 decay_lookup={(r['cell'],r['contrast'],r['match_id']):r['mean'] for r in decay}
 decay_macro={(cell,c):strict_macro([decay_lookup[(cell,c,m)] for m in MATCHES]) for cell in CELLS for c in CONTRASTS}
 decay_summary=[]
 for c in CONTRASTS:
  v=np.asarray([decay_macro[(cell,c)] for cell in CELLS[1:]],float);ok=np.isfinite(v)
  decay_summary.append(dict(contrast=c,metric='distance_2s_minus_0.2s',D_five_seed_mean=float(v.mean()) if ok.all() else np.nan,D_seed_sd=float(v.std(ddof=1)) if ok.all() else np.nan,n_seed_complete_four_matches=int(ok.sum()),CV_once=decay_macro[(CELLS[0],c)]))
 csv_rows(out/'decay_equal_match_five_seed_summary.csv',decay_summary)
 csv_rows(out/'decay_seed_macros.csv',[dict(cell=cell,contrast=c,four_match_equal_mean=v) for (cell,c),v in decay_macro.items()])

 pairedrows=[];pairedmacros={CELLS[0]:{k:0. if np.isfinite(v) else np.nan for k,v in macro[CELLS[0]].items()}}
 for cell in CELLS[1:]:
  a=cells[cell];p={**a,'metrics':a['metrics']-cv['metrics']};rows,mm=fixture_curves(cell,p,'D-minus-CV_paired');pairedrows+=rows;pairedmacros[cell]=mm
 csv_rows(out/'D_CV_paired_fixture_curves.csv',pairedrows);csv_rows(out/'D_CV_paired_five_seed_summary.csv',seed_summary(pairedmacros,'D-minus-CV_paired'))
 csv_rows(out/'seed_macros.csv',[dict(cell=cell,contrast=k[0],time_s=(k[1]+1)/10,metric=k[2],four_match_equal_mean=v) for cell,mm in macro.items() for k,v in mm.items()])
 report=['冻结轨迹响应统计报告','本报告仅分析保存预测；没有加载模型、训练或读取未来真值。','全4992计划窗、4场、D五种子与CV一次均保留；各contrast独立配对支持，首步主结果不要求persistent成功。','先窗内指标、每场均值，再四场等权；五种子样本SD只表示初始化变化，窗口不是独立比赛。某场无支持则严格四场macro为NA；某种子无完整macro则严格五种子汇总为NA。','四场是已暴露历史test cache，不能称新盲验；历史cache纳入不代表新前瞻prefix抽样。','first-natural包含速度和provenance改变；first-sham和sham-natural是模型内对照，欧氏范数不可相加。','下面仅列2s的配对主读数，不设效果通过门槛。']
 summary=seed_summary(macro,'response')
 for c in ('first-natural','first-sham','sham-natural','persistent-natural','persistent-v0'):
  row=next(r for r in summary if r['contrast']==c and r['time_s']==2. and r['metric']=='actor_distance')
  report.append(f"{c}: D五种子四场macro均值={row['D_five_seed_mean']:.9g}, seed SD={row['D_seed_sd']:.9g}; CV={row['CV_once']:.9g}。单位为标准坐标米等价值。")
 report+=['逐场/逐时NA和分母见fixture_curves；逐行状态与失败原因见row_path_status；0.2/1/2s全部计划行见window_endpoints；持续性变化见fixture_decay。','事实ADE/FDE未在此阶段评分；需单独冻结的detected-only原始位置恢复。响应持续不证明真实反事实行为、战术收益或Goal因果效应。']
 (out/'中文报告.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
 dump(out/'source_manifest.json',dict(prediction_completion_sha256=freeze_sha,analysis_code_sha256=sha(__file__),blocks=sources))
 dump(out/'completion.json',dict(status='COMPLETE_SAVED_RESPONSE_ANALYSIS',cells=list(cells),planned_rows_per_cell=4992,contrasts=CONTRASTS,max_sham_vector_identity_residual=decompmax,truth_read=False,training_updates=0,new_model_calls=0,performance_gate=False,source_completion_sha256=freeze_sha))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--prediction-freeze-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.run_dir,a.prediction_freeze_sha256,a.out)
