"""Create-only run-owned I/O. No original study tree is mutated."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,uuid
import numpy as np
P=None
def now():return datetime.now(timezone.utc).isoformat()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(1<<20),b''):h.update(a)
    return h.hexdigest()
def array_hash(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def owned(p):
    p=Path(p).resolve()
    if P is None or not p.is_relative_to(Path(P).resolve()):raise ValueError('Output escapes explicit run root')
    return p
def record(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if hasattr(x,'detach'):return plain(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def put(p,x):
    p=owned(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(plain(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save(p,**x):
    with owned(p).open('xb') as f:np.savez_compressed(f,**x)
def torch_save(p,x):
    import torch
    p=owned(p)
    if p.exists():raise FileExistsError('Immutable checkpoint: '+str(p))
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:torch.save(x,f)
    deadline=time.monotonic()+30.
    while True:
        try:tmp.rename(p);return
        except OSError as e:
            if getattr(e,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:raise
            time.sleep(.1)
def check(r):
    if sha(r['path'])!=r['sha256']:raise ValueError('Frozen file changed: '+r['path'])
def closed(path,expected):
    p=owned(path)
    if not p.exists():return None
    x=J(p)
    if x['implementation_lock_sha256']!=expected:raise ValueError('Foreign closed unit')
    for r in x['files']:owned(r['path']);check(r)
    return x
def pause_requested():
    p=Path(P)/'user_pause.json'
    return p.exists() and J(p).get('status') in ('PAUSE_REQUESTED','PAUSING','PAUSED_USER')
