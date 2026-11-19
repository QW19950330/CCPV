"""Export only frozen prefix-generated arrays; no model, raw XML or scores read."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent
P=R.parent/'CCPV_native_event_chain_pilot_20261025'
EXPECTED_STUDY='8cb108c465cd668f45052965f8eff3da9f70c138bf7ecebbfad9c9050e0d6801'
EXPECTED_KEYS=('X0','X2_cv','Z_cv','match_id','anchor_id','period','source_frame_N','fixed_team_A','source_match_ids','eligible_prefix','eligible_recorded10','y_early2','y_late8','y_10')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,j):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 study=read(R/'study_plan.json');assert sha(R/'study_plan.json')==EXPECTED_STUDY
 assert study['fixed_data']['whitelist']==list(EXPECTED_KEYS)
 assert sha(P/'delivery_manifest.json')==study['parent_delivery_manifest_sha256']
 sealed=read(P/'delivery_manifest.json');pins={q['path']:q['sha256'] for q in sealed['files']}
 def parent_pin(relative):
  path=P/relative;assert sha(path)==pins[relative]
  return dict(path=str(path),sha256=pins[relative])
 dataset=parent_pin('prepared/native_chain.npz');assert dataset['sha256']==study['parent_data_sha256']
 parent_meta=parent_pin('prepared/manifest.json');priorQA=parent_pin('saved_data_QA.json')
 pure_math=parent_pin('audit_saved_native_data.py')
 with np.load(dataset['path'],allow_pickle=False) as z:a={k:z[k] for k in EXPECTED_KEYS}
 assert set(a)==set(EXPECTED_KEYS) and all(k not in a for k in ['X2_obs','Z_obs','X0_available','X2_cv_available','Z_cv_available'])
 N=len(a['anchor_id']);assert N==13372 and len(set(a['anchor_id']))==N
 assert a['X0'].shape==(N,10) and a['X2_cv'].shape==(N,10) and a['Z_cv'].shape==(N,18)
 assert len(a['source_match_ids'])==7 and set(a['source_match_ids'])==set(a['match_id'])
 original=np.arange(N,dtype=np.int64)
 cohort=np.flatnonzero(a['eligible_prefix']&a['eligible_recorded10']);risk=cohort[a['y_early2'][cohort]==0]
 assert len(cohort)==5344 and len(risk)==5290 and len(set(a['match_id'][cohort]))==6
 assert not np.any(a['eligible_recorded10'][a['match_id']=='DFL-MAT-J03WOH'])
 for key in ['y_early2','y_late8','y_10']:assert np.isin(a[key],[-1,0,1]).all() and np.isin(a[key][cohort],[0,1]).all()
 assert np.array_equal(a['y_10'][cohort],a['y_early2'][cohort]+(1-a['y_early2'][cohort])*a['y_late8'][cohort])
 assert all(str(a['anchor_id'][i])==f'{a["match_id"][i]}|{a["period"][i]}|N:{a["source_frame_N"][i]}' for i in original)
 receipt=dict(status='PASS_PRIOR_ACTUAL_READONLY_PREFIX_RECOMPUTATION_RECORDED',
  recorded_utc=datetime.now(timezone.utc).isoformat(),parent_delivery_manifest_sha256=sha(P/'delivery_manifest.json'),
  parent_data_sha256=dataset['sha256'],independent_feature_math_source=pure_math,
  execution_evidence='Earlier authorized read-only verification in this agent: exec session14111 completed exit0; seven PREFIX_DETERMINISM_PASS markers and PASS_READONLY_PREFIX_ONLY_GENERATED_FEATURE_DETERMINISM. Recorded here without rerunning.',
  source_coordinate_selection='Stream ZIP .npy members and materialize only each anchor N-5,N rows; skipped ZIP bytes may inflate but future coordinate arrays are not materialized.',
  actual_past_current_sparse_rows_materialized=26744,actual_future_coordinate_rows_materialized=0,
  recomputed_anchor_count=13372,features_recomputed=['X0','X2_cv','Z_cv'],
  actual_max_absolute_difference=5.684341886080802e-14,recomputation_reruns_by_this_export=0,
  new_raw_scans=0,new_model_forward=0,new_fits=0,
  interpretation='Generated features are deterministic functions of the full source-ID prefix coordinate pair, not necessarily of compressed X0 alone. LateG means a generated-context conditional risk, not actualH2 V8.')
 put(R/'prefix_recompute_receipt.json',receipt)
 target=R/'prefix_only_data.npz'
 with target.open('xb') as f:np.savez_compressed(f,**a)
 with np.load(target,allow_pickle=False) as z:
  assert z.files==list(EXPECTED_KEYS)
  for key,value in a.items():np.testing.assert_array_equal(z[key],value)
 assets={}
 for key,relative in [('final_weights_lock','final_weights_lock.json'),('implementation_lock','implementation_lock.json'),
                      ('heads_source','heads.py'),('train_chain_source','train_chain.py'),
                      ('prospective_predictions','evaluation/prospective_predictions.csv')]:assets[key]=parent_pin(relative)
 final=read(assets['final_weights_lock']['path']);records=[q for q in final['records'] if q['head'] in ['Direct10','EarlyDecoder2','Landmark8']]
 assert len(records)==90 and len({(q['held_match_id'],q['head'],q['training_seed']) for q in records})==90
 cp_records=[]
 for record in records:
  checkpoint=parent_pin(record['checkpoint'].replace('\\','/'))
  assert checkpoint['sha256']==record['sha256']
  cp_records.append(dict(held_match_id=record['held_match_id'],head=record['head'],training_seed=record['training_seed'],
    training_match_ids=record['training_match_ids'],path=checkpoint['path'],sha256=checkpoint['sha256']))
 binding=dict(status='PASS_FROZEN_PREFIX_ONLY_DTO_IDENTICAL_ALL_ROWS_LABELS_AND_ROLES',protocol_sha256=EXPECTED_STUDY,
  path=str(target),sha256=sha(target),data_path=str(target),data_sha256=sha(target),
  parent_delivery_manifest=dict(path=str(P/'delivery_manifest.json'),sha256=sha(P/'delivery_manifest.json')),
  parent_data=dataset,parent_prepared_manifest=parent_meta,parent_source_QA=priorQA,
  export_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
  prefix_recompute_receipt=dict(path=str(R/'prefix_recompute_receipt.json'),sha256=sha(R/'prefix_recompute_receipt.json')),
  whitelist=list(EXPECTED_KEYS),forbidden_arrays_absent=['X2_obs','Z_obs','actual_future_trajectory','future_possession'],
  availability='Not exported. Loader computes isfinite on each raw feature group; no availability array is an additional raw feature.',
  original_row_indices=original.tolist(),original_indices_sha256=ah(original),original_row_order_unchanged=True,
  cohort_indices=cohort.tolist(),cohort_indices_sha256=ah(cohort),risk_indices=risk.tolist(),risk_indices_sha256=ah(risk),
  planned_anchors=N,source_match_ids=a['source_match_ids'].tolist(),supported_match_ids=sorted(set(a['match_id'][cohort])),
  supported_match_count=6,initial_rows=5344,noearly_risk_rows=5290,
  parent_assets=assets,parent_checkpoint_records=cp_records,
  arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=ah(v)) for k,v in a.items()},
  source_identity='Allseven previously exposed sources; six supported and WOH zero retained. No new independent or blind matches.',
  no_new_labels=True,no_parent_edits=True,new_fits=0,new_model_forward=0,raw_XML_scans=0)
 put(R/'data_binding.json',binding)
 put(R/'data_binding_lock.json',dict(status='LOCKED_PREFIX_ONLY_DTO_BEFORE_NEW_FITS',protocol_sha256=EXPECTED_STUDY,
  source_sha256=sha(__file__),binding_sha256=sha(R/'data_binding.json'),data_sha256=sha(target),
  prefix_receipt_sha256=sha(R/'prefix_recompute_receipt.json'),parent_delivery_manifest_sha256=sha(P/'delivery_manifest.json')))
 print(json.dumps(dict(status=binding['status'],rows=N,cohort=len(cohort),risk=len(risk),data_sha256=sha(target),binding_sha256=sha(R/'data_binding.json'),lock_sha256=sha(R/'data_binding_lock.json')),ensure_ascii=False))
if __name__=='__main__':main()
