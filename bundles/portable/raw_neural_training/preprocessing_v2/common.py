"""Small explicit I/O helpers. The launcher owns a newly created output tree."""
from pathlib import Path
import hashlib
import json
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def ah(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()

def dump(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def save_npz(path,**arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)

def require_separate_new_output(output,inputs):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Output exists; preserve it and choose a new directory')
    if not output.parent.is_dir():raise FileNotFoundError('Output parent must already exist')
    code=Path(__file__).resolve().parent
    protected=[code]+[Path(p).resolve() if Path(p).is_dir() else Path(p).resolve().parent for p in inputs]
    for p in protected:
        if output==p or p in output.parents or output in p.parents:
            raise ValueError('Output must not overlap code or source trees')
    return output
