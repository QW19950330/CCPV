"""Compare two predeclared rows at batch=2; never compare against historical batch=32 outputs."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--adapter-root',required=True);p.add_argument('--original-workspace',required=True);p.add_argument('--weights-root',required=True);p.add_argument('--prefixes',required=True);p.add_argument('--clean-run',required=True);p.add_argument('--out',required=True);x=p.parse_args()
R=Path(x.adapter_root);spec=importlib.util.spec_from_file_location('bridge',R/'run_clean_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
protocol=m.read(R/'validation_protocol.json')
assert m.sha(R/'run_clean_diagnostic.py')==protocol['entry_sha256']
assert m.sha(__file__)==protocol['validation_source_sha256']
binding,runner,torch,api,op,clean=m.verified_runtime(protocol['binding_sha256'])
a=m.load_prefixes(x.prefixes,binding);rows=np.asarray(protocol['row_positions'],dtype=np.int64);assert rows.tolist()==[0,1]
original=Path(x.original_workspace);cleanrun=Path(x.clean_run);cr=m.read(cleanrun/'completion.json');selection=m.read(cleanrun/'selection.json')
assert cr['status']=='COMPLETE_BOUNDED_CLEAN_NPZ_DIAGNOSTIC' and cr['planned_source_windows']==2 and not cr['full_cohort']
assert cr['binding_sha256']==protocol['binding_sha256'] and selection['row_positions']==[0,1]
assert len(cr['cells'])==6 and cr['paths']==list(runner.PATHS)
source_before={r['source']:m.sha(original/r['source']) for r in binding['source_pins']}
assert all(source_before[r['source']]==r['source_sha256'] for r in binding['source_pins'])
records=[]
for cell in cr['cells']:
 seed=-1 if cell['cell']=='CV_s-1' else int(cell['cell'].rsplit('s',1)[1]);model=None;parameter_equality=None;origcp=None
 if seed>=0:
  rec=binding['models'][seed];entry=rec['clean_entry'];origcp=original/entry['original_source']
  assert m.sha(origcp)==rec['original_checkpoint_sha256']
  payload=torch.load(origcp,map_location='cpu',weights_only=False);assert payload['epoch']==4
  model=api.make_model('D_coupled_ar',seed);model.load_state_dict(payload['state_dict'],strict=True);model.eval()
  fresh=m.load_model(rec,x.weights_root,torch,api,clean)
  oldstate=model.state_dict();newstate=fresh.state_dict();assert oldstate.keys()==newstate.keys()
  assert all(runner.bits(oldstate[k].numpy(),newstate[k].numpy()) for k in oldstate)
  parameter_equality=dict(arrays=len(oldstate),all_tensor_bits_equal=True,state_hash=api.state_hash(oldstate),clean_state_hash=api.state_hash(newstate))
  assert cell['parameter_hash_before']==parameter_equality['state_hash']==cell['parameter_hash_after']
 before=None if model is None else api.state_hash(model.state_dict())
 arrays,reasons,checks,eligible=m.evaluate_rows(runner,torch,api,op,model,a,rows,runner.read(runner.PROTOCOL))
 assert eligible==2, 'Predeclared rows do not exercise two eligible real prefixes'
 assert len(cell['blocks'])==1
 br=cell['blocks'][0];bp=cleanrun/cell['cell']/br['path'];assert m.sha(bp)==br['sha256']
 saved_rec=m.read(bp.with_suffix('.json'));assert saved_rec['checks']==runner.clean(checks) and saved_rec['reasons']==runner.clean(reasons)
 compared=[]
 with np.load(bp,allow_pickle=False) as z:
  assert set(z.files)==set(arrays)
  for k,b in arrays.items():
   got=z[k];ok=runner.bits(b,got);assert ok,(cell['cell'],k)
   compared.append(dict(array=k,shape=list(got.shape),dtype=got.dtype.str,bitwise_equal=True))
 after=None if model is None else api.state_hash(model.state_dict());assert before==after
 if origcp is not None:assert m.sha(origcp)==rec['original_checkpoint_sha256']
 records.append(dict(cell=cell['cell'],rows=2,batch_size=2,eligible_rows=eligible,paths=list(runner.PATHS),parameter_equality=parameter_equality,outputs=compared,reasons_equal=True,checks_equal=True,frozen_parameter_hash_unchanged=True))
assert m.sha(x.prefixes)==binding['prefix_sha256']
assert all(m.sha(original/s)==h for s,h in source_before.items())
report=dict(status='PASS_FIVE_SEED_AND_CV_SAME_BATCH_SIX_PATH_BYTE_EQUIVALENCE',validation_protocol_sha256=m.sha(R/'validation_protocol.json'),entry_sha256=m.sha(R/'run_clean_diagnostic.py'),binding_sha256=protocol['binding_sha256'],prefix_sha256=binding['prefix_sha256'],source_row_positions=[0,1],source_indices=a['source_indices'][rows].tolist(),matches=a['match_ids'][rows].tolist(),batch_size=2,records=records,original_sources_rehashed_unchanged=len(source_before),original_checkpoints_rehashed_unchanged=5,real_original_checkpoint_loads=5,clean_checkpoint_loads_validation=5,clean_checkpoint_loads_portable_run=5,full_cohort_forward=False,historical_prediction_arrays_read=False,comparison_to_historical_batch32_claimed=False,future_truth_reads=0,new_training_updates=0,scope='Two predeclared original prefixes, same batch2, all5 fixed D seeds and CV, six original paths. Does not certify full4992 replay or cross-platform bitwise equivalence.')
runner.write(x.out,report);print(json.dumps({k:report[k] for k in ['status','source_row_positions','batch_size','full_cohort_forward','future_truth_reads','new_training_updates']}))
