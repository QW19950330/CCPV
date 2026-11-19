"""Create-only artifacts and root-locked whitelist saved-state reads."""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
BASE=P.parent

def now():return datetime.now(timezone.utc).isoformat()
def J(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def put(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pin(path,digest=None):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=digest if digest else sha(path))
def load(record,keys,reads):
    path=Path(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('Pinned saved asset changed')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in keys}
    reads.append(dict(path=str(path),members=list(keys),decoded_bytes=sum(x.nbytes for x in a.values())))
    return a
def table(path,rows):
    if not rows:raise ValueError('No table rows')
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class Stream:
    def __init__(self,path):
        self.file=gzip.open(path,'xt',encoding='utf-8-sig',newline='',compresslevel=3)
        self.writer=None;self.count=0
    def add(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.count+=1
    def close(self):self.file.close()

_execution = None

def configure(output, contract_sha, code_root):
    global P, BASE, _execution
    if _execution is not None: raise RuntimeError('One execution per process')
    P=Path(output).resolve(); BASE=Path(code_root).resolve()
    _execution=(contract_sha,BASE)

def guard(expected):
    if _execution is None or expected != _execution[0]:
        raise PermissionError('Configure a verified portable execution first')
    if sha(P/'execution_contract.json') != expected:
        raise PermissionError('Portable execution contract changed')
    c=J(P/'execution_contract.json')
    if c['scope'] != 'FULL_FIXED_SAVED_OUTPUT_SCREEN_NO_TRAINING':
        raise PermissionError('Unexpected execution scope')
    for r in c['code_files']:
        if sha(BASE/r['path']) != r['sha256']: raise ValueError('Portable code changed')
    for key,name in [('plan_sha256','experiment_plan.json'),('binding_sha256','data_binding.json')]:
        if sha(P/name) != c[key]: raise ValueError('Runtime input contract changed')
    return J(P/'experiment_plan.json'),J(P/'data_binding.json')
