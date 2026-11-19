"""New create-only root gates. No payload access on import."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,time
import numpy as np
P=Path(__file__).resolve().parent;S=P.parent;B=S.parent
STAGES=('oof_logits','calibrate','readout','score')
class PauseBetweenShards(RuntimeError):pass
def now():return datetime.now(timezone.utc).isoformat()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def pathof(r):
    q=Path(r['path']);return q if q.is_absolute() else P/q
def check(r):
    if sha(pathof(r))!=r['sha256']:raise ValueError('Frozen SHA mismatch: '+r['path'])
def rec(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**arrays)
def table(p,rows):
    rows=list(rows)
    if not rows:raise ValueError('No empty table')
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(record,members):
    check(record)
    with np.load(pathof(record),allow_pickle=False) as z:
        if not set(members).issubset(z.files):raise ValueError('Missing declared NPZ member')
        out={k:z[k] for k in members}
    check(record);return out
def original_source():
    p=S/'source_contract/audit_receipt.json'
    if sha(p)!='c7fcd23887e15b0eb8d8ee7a1832ca2ed0d52fa51c8faa3cd460a27c2aacd6db':raise ValueError('Original11source receipt')
    for r in J(p)['files']:
        if sha(S/'source_contract'/r['path'])!=r['sha256']:raise ValueError('Immutable source contract changed')
def guard(stage,expected,auth_path=None,auth_sha=None):
    if stage not in STAGES or np.__version__!='1.26.4':raise ValueError('Pinned stage/NumPy1.26.4')
    lp=P/'implementation_lock.json'
    if sha(lp)!=expected:raise PermissionError('Exactrootlock required')
    lock=J(lp);a=J(auth_path or P/'runtime_authorization.json');attempt=a.get('attempt_id')
    if lock.get('status')!='LOCKED_E_OOF_INTERCEPT_PAYLOAD_V2' or a.get('status')!='AUTHORIZED_E_OOF_INTERCEPT_TRANSFER' or a.get('implementation_lock_sha256')!=expected or stage not in a.get('allowed_stages',[]):raise PermissionError('Rootstage authorization absent')
    if not isinstance(attempt,str) or not attempt or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in attempt):raise ValueError('Safeattempt ID')
    if auth_path is not None and (Path(auth_path).resolve()!=(P/'execution'/stage/attempt/'authorization_snapshot.json').resolve() or sha(auth_path)!=auth_sha):raise PermissionError('Exactauthority snapshot')
    original_source()
    for r in lock['files']:check(r)
    if sha(P/'payload_contract.json')!=lock['payload_contract_sha256']:raise ValueError('Payload contract changed')
    c=J(P/'payload_contract.json')
    if stage!='oof_logits':
        name={'calibrate':'oof_freeze.json','readout':'calibration_freeze.json','score':'readout_freeze.json'}[stage]
        r=a.get('stage_prerequisites',{}).get(stage)
        if not r or Path(r['path']).resolve()!=(P/name).resolve():raise PermissionError('Precedingfreeze required')
        check(r);prior=J(P/name)
        if prior['status']!=c['prerequisites'][stage] or prior['implementation_lock_sha256']!=expected:raise ValueError('Precedingfreeze lineage')
        for r in prior['files']:check(r)
    return a,c
def closed(p,expected):
    p=Path(p)
    if not p.exists():return None
    r=J(p)
    if r['implementation_lock_sha256']!=expected:raise ValueError('Closedshard lock')
    for x in r['files']:check(x)
    return r
def boundary(start,c):
    if (P/'PAUSE_REQUESTED').exists():raise PauseBetweenShards('Explicitpause atclosedunit;noautorestart')
    if time.monotonic()-start>c['softcap_seconds']:raise PauseBetweenShards('600s softcap atclosedunit;noactivewrite killed')
def identity(path,stage,expected,a,authority):
    import sys,psutil
    p=Path(path).resolve()
    if p!=(P/'execution'/stage/a['attempt_id']/'compute_identity.json').resolve():raise PermissionError('Exact exclusive computeidentity path')
    proc=psutil.Process();put(p,dict(pid=proc.pid,create_time=proc.create_time(),ppid=proc.ppid(),cmdline=proc.cmdline(),python=sys.version,executable=sys.executable,numpy=np.__version__,utc=now(),stage=stage,implementation_lock_sha256=expected,authorization_snapshot=authority,device='cpu',threads=1,new_NN=0))
