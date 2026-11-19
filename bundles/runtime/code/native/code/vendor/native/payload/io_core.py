"""Create-only stage I/O and root gates. No model or automatic payload access."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,time
import numpy as np

P=Path(__file__).resolve().parent;S=P.parent;BASE=S.parent
OLD_RECEIPT_SHA='fe675d672ed037f09048624757490e4721637f1a7ca27c4c5823403a86c12a03'
STAGES=('trajectory_features','readout','score')
class PauseBetweenShards(RuntimeError):pass

def now():return datetime.now(timezone.utc).isoformat()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def path_of(r):
    p=Path(r['path']);return p if p.is_absolute() else P/p
def check(r):
    if sha(path_of(r))!=r['sha256']:raise ValueError('SHA changed: '+str(r['path']))
def rec(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**arrays)
def table(p,rows):
    rows=list(rows)
    if not rows:raise ValueError('No silent empty output table')
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_members(record,members):
    """Only named NPZ members are decoded; hash covers the container."""
    check(record)
    with np.load(path_of(record),allow_pickle=False) as z:
        if not set(members).issubset(z.files):raise ValueError('Missing exact declared member')
        a={k:z[k] for k in members}
    check(record);return a
def old_source_integrity():
    r=S/'source_preflight_receipt.json'
    if sha(r)!=OLD_RECEIPT_SHA:raise ValueError('Immutable25-file source receipt changed')
    for x in J(r)['files']:
        if sha(S/x['path'])!=x['sha256']:raise ValueError('Immutable source changed: '+x['path'])
def guard(stage,expected,authorization_path=None,authorization_sha=None):
    if stage not in STAGES:raise ValueError('Declared stage only')
    if np.__version__!='1.26.4':raise RuntimeError('PinnedNumPy1.26.4 required')
    lp=P/'implementation_lock.json'
    if sha(lp)!=expected:raise ValueError('Root lock argument')
    lock=J(lp)
    if lock.get('status')!='LOCKED_NATIVE_OUTCOME_TRANSFER_PAYLOAD_V1':raise PermissionError('Wrong implementation lock')
    old_source_integrity()
    for r in lock['files']:check(r)
    if sha(P/'payload_contract.json')!=lock['payload_contract_sha256']:raise ValueError('Payload contract changed')
    authpath=Path(authorization_path) if authorization_path is not None else P/'runtime_authorization.json'
    auth=J(authpath)
    if auth.get('status')!='AUTHORIZED_FROZEN_WORLD_NATIVE_OUTCOME_TRANSFER' or auth.get('implementation_lock_sha256')!=expected or stage not in auth.get('allowed_stages',[]):
        raise PermissionError('Separate root stage authorization absent')
    attempt=auth.get('attempt_id')
    if not isinstance(attempt,str) or not attempt or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in attempt):raise ValueError('Explicit safe attempt ID')
    if authorization_path is not None:
        required=P/'execution'/stage/attempt/'authorization_snapshot.json'
        if authpath.resolve()!=required.resolve() or authorization_sha is None or sha(authpath)!=authorization_sha:
            raise PermissionError('Exact immutable per-attempt authorization snapshot required')
    if stage=='trajectory_features' and auth.get('whole_torch_checkpoint_container_deserialization') is not True:raise PermissionError('Whole trusted Torch container access not authorized')
    if stage!='trajectory_features':
        name={'readout':'trajectory_freeze.json','score':'readout_freeze.json'}[stage]
        prereq=auth.get('stage_prerequisites',{}).get(stage)
        if not prereq or Path(prereq['path']).resolve()!=(P/name).resolve():raise PermissionError('Exact prerequisite freeze required')
        check(prereq)
        frozen=J(P/name)
        status={'readout':'ALL11_TRAJECTORIES_AND_FEATURES_FROZEN_NO_Q_OR_LABELS','score':'ALL12_ARMS5SEEDS5344_READOUTS_FROZEN_BEFORE_LABELS'}[stage]
        if frozen['status']!=status or frozen['implementation_lock_sha256']!=expected:raise ValueError('Prerequisite lineage')
        for r in frozen['files']:check(r)
    return lock,auth,J(P/'payload_contract.json')
def closed(p,expected):
    p=Path(p)
    if not p.exists():return None
    j=J(p)
    if j['implementation_lock_sha256']!=expected:raise ValueError('Closed shard lock changed')
    for r in j['files']:check(r)
    return j
def boundary(start):
    if (P/'PAUSE_REQUESTED').exists():raise PauseBetweenShards('Explicit pause request at closed-shard boundary; no automatic restart')
    if time.monotonic()-start>3600:raise PauseBetweenShards('Predeclared60min soft cap at closed-shard boundary; preserve closed work')
def identity(path,stage,expected,attempt,authorization):
    import sys,psutil
    p=Path(path).resolve();allowed=P/'execution'/stage/attempt
    if not p.is_relative_to(allowed.resolve()):raise PermissionError('Identity confined to exact stage attempt')
    proc=psutil.Process()
    put(p,dict(pid=proc.pid,create_time=proc.create_time(),ppid=proc.ppid(),cmdline=proc.cmdline(),executable=sys.executable,python=sys.version,numpy=np.__version__,utc=now(),stage=stage,attempt=attempt,implementation_lock_sha256=expected,authorization_snapshot=authorization,device='cpu',threads=1,SDPA='MATH',TF32=False))
