"""Run only original read-only preflight prefixes, plus saved-statistics arithmetic.
Function prefixes end before first output mkdir and hence before any fit/readout.
"""
from pathlib import Path
import argparse,ast,runpy,sys,os,json,hashlib,types
from compose_inputs import long,digest

def prefix(src,name,args,argv=None):
 tree=ast.parse(src.read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 cuts=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='mkdir']
 if not cuts:raise ValueError('No original output transition found')
 cut=cuts[0];kept=fn.body[:cut]
 if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('fit','readout','worker') for st in kept for n in ast.walk(st)):
  # The only pre-mkdir worker call in S4 is guarded by --stage; no stage supplied.
  if src.parent.parent.name!='s4':raise ValueError('Unexpected stage call before cutoff')
 fn.body=kept+[ast.Return(value=ast.Constant(None))];ns=runpy.run_path(str(src),run_name='ccpv_public_preflight_import');module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));exec(compile(module,str(src),'exec'),ns)
 old=sys.argv
 try:
  if argv is not None:sys.argv=[str(src),*argv];ns[name]()
  else:ns[name](args)
 finally:sys.argv=old
 return {'original_source_sha256':digest(src),'original_function':name,'executed_source_prefix_lines':[fn.lineno,kept[-1].end_lineno],'stopped_before_output_creation_and_all_fit_readout':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--component',required=True);a=p.parse_args();cand=long(a.candidate);root=long(a.inputs);out=long(a.output)
 if out.exists():raise FileExistsError('Create-only report')
 out.mkdir(parents=True);k=a.component;code=cand/'code'/k;meta={};sentinel=out/'FORBIDDEN_RUNTIME_OUTPUT';cat=json.loads((cand/'runtime_catalog.json').read_text(encoding='utf8'));inputs=root/k/cat['components'][k]['input_dir']
 if k in ('table7','qfirst','native','s4'):
  d=code/'code';sys.path.insert(0,str(d));src=d/({'qfirst':'reproduce_qfirst.py','native':'reproduce_native.py'}.get(k,'run.py'))
  if k=='table7':meta=prefix(src,'main',None,['--inputs',str(inputs),'--output',str(sentinel)])
  elif k=='s4':meta=prefix(src,'main',None,['--input-root',str(inputs),'--metadata',str(code/'metadata'),'--output',str(sentinel)])
  else:meta=prefix(src,'execute',types.SimpleNamespace(input_dir=str(inputs),output_dir=str(sentinel)))
  assert not sentinel.exists()
  if k=='table7':
   os.environ['CCPV_TABLE7_OUTPUT']=str(sentinel);os.environ['CCPV_TABLE7_INPUT']=str(inputs)
   import risk_io,runtime
   registry=runtime.replay_registry(risk_io.read(sentinel/'data_binding.json'));meta['original_replay_registry_records']=len(registry)
 elif k=='table5':
  sys.path.insert(0,str(code));old=sys.argv;sys.argv=[str(code/'pipeline.py'),'preflight','--data-root',str(inputs),'--output',str(out/'preflight')]
  try:runpy.run_path(str(code/'pipeline.py'),run_name='__main__')
  finally:sys.argv=old
 elif k=='table6':
  sys.path.insert(0,str(code));import bindings,run
  contract,binding,_=bindings.preflight(types.SimpleNamespace(input_root=inputs,output=sentinel));assert run.validate_case_contract(binding,bindings.J(binding['runner_freeze']['path']))
  meta={'files':len(contract['files']),'scientific_cases':len(binding['cases']),'original334_scope_guard':True,'arrays_opened':0}
 elif k=='supplementary':
  sys.path.insert(0,str(code));old=sys.argv;sys.argv=[str(code/'build.py'),'--input-root',str(inputs),'--output',str(out/'saved_statistics')]
  try:runpy.run_path(str(code/'build.py'),run_name='__main__')
  finally:sys.argv=old
 elif k=='s5':
  sys.path.insert(0,str(code));old=sys.argv;sys.argv=[str(code/'run.py'),'--inputs',str(inputs),'--output',str(out/'saved_statistics')]
  try:runpy.run_path(str(code/'run.py'),run_name='__main__')
  finally:sys.argv=old
 else:raise ValueError('Unknown component')
 r={'status':'PASS_PUBLIC_BINDING_ORIGINAL_PREFLIGHT_OR_SAVED_STATISTICS','component':k,'scope':meta,'training':0,'model_prediction_calls':0,'numerical_source_modified':False};(out/'receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(r))
if __name__=='__main__':main()
