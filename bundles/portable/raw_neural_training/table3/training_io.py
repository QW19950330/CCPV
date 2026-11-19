"""Bounded, output-tree-only checkpoint writes and immutable epoch snapshots."""
import hashlib,json,os,shutil,time,uuid
from pathlib import Path
import numpy as np
import torch
ROOT=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()

def atomic(p,writer):
    p=Path(p).resolve()
    if ROOT is None or not p.is_relative_to(ROOT):raise ValueError('Write outside this run')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');writer(tmp);deadline=time.monotonic()+30
    while True:
        try:os.replace(tmp,p);return
        except OSError as e:
            if getattr(e,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:raise
            time.sleep(.2)

def dump(p,v):
    atomic(p,lambda q:q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'))

def save_torch(p,v):
    atomic(p,lambda q:torch.save(v,q))

def preserve_epoch(out,epoch):
    target=out/f'epoch{epoch}'
    target.mkdir(exist_ok=False)
    for name in ('world_last.pt','history.json'):
        shutil.copyfile(out/name,target/name)
    dump(target/'receipt.json',dict(epoch=epoch,checkpoint_sha256=sha(target/'world_last.pt'),history_sha256=sha(target/'history.json')))
