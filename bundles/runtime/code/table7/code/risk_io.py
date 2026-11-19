"""Portable I/O adapter only. Original numerical modules are byte-identical."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, os, sys
import numpy as np

P = Path(os.environ['CCPV_TABLE7_OUTPUT']).resolve()
INPUT = Path(os.environ['CCPV_TABLE7_INPUT']).resolve()
CODE = Path(__file__).resolve().parent
PLAN = 'experiment_plan.json'
BINDING = 'data_binding.json'
STATE_KEYS = ('hidden_weight','hidden_bias','output_weight','output_bias')

def utc(): return datetime.now(timezone.utc).isoformat()
def locate(path):
    path = Path(path)
    if path == P/PLAN: return INPUT/'snapshots/CCPV_risk_objective_factorial_20261030/experiment_plan.json'
    if path == P/BINDING: return INPUT/'portable_binding.json'
    return path
def sha(path):
    h = hashlib.sha256()
    with locate(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def read(path):
    original = Path(path)
    obj = json.loads(locate(path).read_text(encoding='utf-8-sig'))
    if original == P/BINDING:
        def resolve(v):
            if isinstance(v,dict):
                return {k:str((INPUT/x).resolve()) if k in ('path','history') and isinstance(x,str) else resolve(x) for k,x in v.items()}
            if isinstance(v,list): return [resolve(x) for x in v]
            return v
        obj = resolve(obj)
    return obj
def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(P): raise ValueError('Output escapes create-only run')
    path.parent.mkdir(parents=True,exist_ok=True)
    return path
def put(path,value):
    with output_path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def save(path,**arrays):
    with output_path(path).open('xb') as f: np.savez_compressed(f,**arrays)
def table(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with output_path(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader(); writer.writerows(rows)
def guard(expected,stage):
    lp=P/'implementation_lock.json'
    if sha(lp)!=expected: raise ValueError('Portable implementation lock changed')
    lock=read(lp)
    if sha(INPUT/'manifest.json')!=lock['input_manifest_sha256']: raise ValueError('Input manifest changed')
    if sha(P/PLAN)!=lock['experiment_plan_sha256'] or sha(P/BINDING)!=lock['data_binding_sha256']: raise ValueError('Frozen plan/binding changed')
    for r in lock['files']:
        if sha(CODE/r['path'])!=r['sha256']: raise ValueError('Portable code changed: '+r['path'])
    auth=read(P/'runtime_authorization.json')
    if auth['implementation_lock_sha256']!=expected or stage not in auth['allowed_stages']: raise ValueError('Stage not authorized by fixed portable run')
    return lock
def selected(path,keys):
    if not keys or len(set(keys))!=len(keys): raise ValueError('Explicit unique whitelist required')
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in keys}
def checkpoint(path):
    with np.load(path,allow_pickle=False) as z:
        if not set(STATE_KEYS).issubset(z.files): raise ValueError('Restricted to fitted head checkpoints')
        return {k:z[k] for k in z.files}
def process_identity(stage,expected):
    import torch
    return dict(utc=utc(),pid=os.getpid(),parent_pid=os.getppid(),python_executable=sys.executable,
        argv=sys.argv,cwd=str(Path.cwd()),stage=stage,implementation_lock_sha256=expected,
        numpy=np.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads(),
        input_root=str(INPUT),output_root=str(P),code_root=str(CODE))
