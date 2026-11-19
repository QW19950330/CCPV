"""Portable explicit clean-NPZ bridge; the original six-path mathematics is unchanged."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BINDING=HERE/'clean_diagnostic_binding.json'
RUNNER='runtime_sources/CCPV_external_response_audit_20261119/run_skillcorner_response.py'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
def verified_runtime(expected):
 if not expected or sha(BINDING)!=expected:raise ValueError('Explicit new binding SHA required')
 binding=read(BINDING)
 for r in binding['source_pins']:
  f=(HERE/r['member']).resolve()
  if not f.is_relative_to(HERE) or sha(f)!=r['sha256']:raise ValueError('Runtime/evidence source pin changed')
 if binding['old_guard_satisfied'] is not False:raise ValueError('New binding must not claim the historical byte guard')
 if [r['seed'] for r in binding['models']]!=list(range(5)):raise ValueError('All five fixed seeds required')
 for r in binding['models']:
  ev=read(HERE/r['epoch_evidence']);entry=r['clean_entry']
  if (ev['arm'],ev['seed'],ev['epochs_completed'],ev['updates'],ev['checkpoint_sha256'])!=('D_coupled_ar',r['seed'],4,96,r['original_checkpoint_sha256']):raise ValueError('Epoch provenance mismatch')
  if entry['original_sha256']!=r['original_checkpoint_sha256'] or entry['arm']!='D_coupled_ar' or entry['seed']!=r['seed']:raise ValueError('Clean-to-original identity mismatch')
 runner=module('frozen_sixpath_runner',HERE/RUNNER)
 if sha(runner.PROTOCOL)!=binding['protocol_sha256'] or sha(HERE/RUNNER)!=binding['old_runner_sha256']:raise ValueError('Protocol or runner changed')
 torch,api,operator=runner.runtime()
 clean=module('clean_numeric_inference_adapter',HERE/'inference_adapter.py')
 return binding,runner,torch,api,operator,clean

def load_model(record,weights_root,torch,api,clean):
 entry=record['clean_entry'];arrays=clean.load_export(entry,Path(weights_root))
 for k,v in arrays.items():
  if hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()!=entry['arrays'][k]['sha256_numeric_bytes']:raise ValueError('Tensor numeric bytes changed')
 model=api.make_model('D_coupled_ar',record['seed'])
 model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in arrays.items()},strict=True)
 return model.cpu().eval()

def load_prefixes(path,binding):
 if sha(path)!=binding['prefix_sha256']:raise ValueError('Exact original prepared prefix SHA required')
 required=('history_states','history_valid','operator_candidate_slots','technical_unsupported','entity_ids','identity_status','teams','operator_candidate_ids','source_indices','match_ids')
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in required}
 if a['history_states'].shape!=(4992,10,23,4) or a['history_valid'].shape!=(4992,10,23) or a['teams'].shape!=(23,):raise ValueError('Full fixed prefix contract')
 if a['history_states'].dtype!=np.float32 or a['history_valid'].dtype!=np.bool_:raise ValueError('Original prefix dtypes required')
 return a

def evaluate_rows(runner,torch,api,operator,model,a,rows,protocol):
 rows=np.asarray(rows,dtype=np.int64);hs=a['history_states'][rows];vv=a['history_valid'][rows]
 slots=a['operator_candidate_slots'][rows,0]
 take=np.flatnonzero((slots>=0)&~a['technical_unsupported'][rows])
 if len(take):
  regs=operator.make_registries(a['entity_ids'][rows][take],a['identity_status'][rows][take],a['teams'])
  targets=a['operator_candidate_ids'][rows][take].tolist()
  h=torch.from_numpy(hs[take].copy());v=torch.from_numpy(vv[take].copy());tm=torch.from_numpy(a['teams'].copy())
  results,cmd,v0=runner.paths(torch,api,operator,model,h,v,tm,regs,targets,slots[take].tolist(),protocol)
  arrays,reasons,checks=runner.compact(results,take,len(rows),vv,slots,cmd,v0)
 else:
  arrays,reasons,checks=runner.compact({},take,len(rows),vv,slots,np.empty((0,2,2),np.float32),np.empty((0,2,2),np.float32))
 arrays.update(source_indices=a['source_indices'][rows],match_ids=a['match_ids'][rows],actor_slots=slots,technical_unsupported=a['technical_unsupported'][rows])
 return arrays,reasons,checks,len(take)

def preflight(binding_sha,weights_root):
 binding,runner,torch,api,operator,clean=verified_runtime(binding_sha);records=[]
 for r in binding['models']:
  model=load_model(r,weights_root,torch,api,clean)
  records.append(dict(seed=r['seed'],parameter_hash=api.state_hash(model.state_dict()),original_checkpoint_sha256=r['original_checkpoint_sha256'],clean_sha256=r['clean_entry']['export_sha256']))
 return dict(status='PASS_PORTABLE_CLEAN_LOADING_NO_FORWARD',binding_sha256=binding_sha,models=records,old_checkpoint_byte_guard_claimed=False,model_forwards=0)

def run(binding_sha,weights_root,prefixes,out,rows=None,all_planned=False):
 binding,runner,torch,api,operator,clean=verified_runtime(binding_sha);protocol=runner.read(runner.PROTOCOL)
 a=load_prefixes(prefixes,binding)
 if all_planned:selected=np.arange(4992,dtype=np.int64)
 else:
  selected=np.asarray(rows,dtype=np.int64)
  if not len(selected) or np.any(selected<0) or np.any(selected>=4992) or np.any(np.diff(selected)<=0):raise ValueError('Rows must be distinct ascending original row positions')
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 runner.write(out/'selection.json',dict(binding_sha256=binding_sha,entry_sha256=sha(__file__),prefix_sha256=sha(prefixes),row_positions=selected,selection='Explicit original planned order; no replacement or response-based selection',all_planned=bool(all_planned)))
 cells=[dict(arm='CV',seed=-1)]+[dict(arm='D_coupled_ar',seed=s) for s in range(5)];receipts=[]
 for c in cells:
  model=None if c['arm']=='CV' else load_model(binding['models'][c['seed']],weights_root,torch,api,clean)
  before=None if model is None else api.state_hash(model.state_dict());folder=out/(c['arm']+'_s'+str(c['seed']));folder.mkdir();blocks=[]
  cp=None if model is None else dict(format='clean_numeric_npz',seed=c['seed'],epoch_from_original_evidence=4,original_checkpoint_sha256=binding['models'][c['seed']]['original_checkpoint_sha256'],clean_sha256=binding['models'][c['seed']]['clean_entry']['export_sha256'],new_binding_sha256=binding_sha,old_byte_guard_satisfied=False)
  for start in range(0,len(selected),32):
   stop=min(start+32,len(selected));arrays,reasons,checks,eligible=evaluate_rows(runner,torch,api,operator,model,a,selected[start:stop],protocol)
   bp=folder/('batch_%05d.npz'%start);runner.npz(bp,**arrays)
   rec=dict(start=start,stop=stop,path=bp.name,sha256=sha(bp),implementation_sha256=binding_sha,implementation_binding_kind='explicit_clean_npz_bridge',input_sha256=binding['prefix_sha256'],eligible_initial_rows=eligible,status_counts=dict(collections.Counter(arrays['status'].ravel())),checks=checks,reasons=reasons,arm=c['arm'],seed=c['seed'])
   runner.write(bp.with_suffix('.json'),rec);blocks.append(dict(path=bp.name,sha256=rec['sha256']))
  after=None if model is None else api.state_hash(model.state_dict())
  if before!=after:raise AssertionError('Frozen model changed')
  cr=dict(cell=folder.name,planned_rows=len(selected),batches=len(blocks),parameter_hash_before=before,parameter_hash_after=after,checkpoint=cp,blocks=blocks)
  runner.write(folder/'completion.json',cr);receipts.append(cr)
 if sha(prefixes)!=binding['prefix_sha256'] or sha(BINDING)!=binding_sha:raise ValueError('Input or binding changed during execution')
 for r in binding['source_pins']:
  if sha(HERE/r['member'])!=r['sha256']:raise ValueError('Source changed during execution')
 final=dict(status='COMPLETE_FIXED4992_SIX_CELLS_SIX_PATHS' if all_planned else 'COMPLETE_BOUNDED_CLEAN_NPZ_DIAGNOSTIC',cells=receipts,paths=runner.PATHS,planned_source_windows=len(selected),new_training_updates=0,future_truth_read=False,original_sources_unchanged=True,execution_kind='new_explicit_clean_npz_binding',binding_sha256=binding_sha,old_checkpoint_byte_guard_claimed=False,full_cohort=bool(all_planned))
 runner.write(out/'completion.json',final);return final

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','run']);p.add_argument('--binding-sha256',required=True);p.add_argument('--weights-root',required=True);p.add_argument('--prefixes');p.add_argument('--out');g=p.add_mutually_exclusive_group();g.add_argument('--rows',nargs='+',type=int);g.add_argument('--all-planned',action='store_true');x=p.parse_args()
 if x.stage=='preflight':result=preflight(x.binding_sha256,x.weights_root)
 else:
  if not x.prefixes or not x.out or (not x.all_planned and not x.rows):p.error('run requires explicit prefixes, new out, and rows or all-planned')
  result=run(x.binding_sha256,x.weights_root,x.prefixes,x.out,x.rows,x.all_planned)
 print(json.dumps(dict(status=result['status'],old_checkpoint_byte_guard_claimed=False),ensure_ascii=False))
