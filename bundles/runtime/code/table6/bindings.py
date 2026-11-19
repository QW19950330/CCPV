"""Relocate read-only historic references beneath an explicit input root."""
from pathlib import Path
import hashlib,json
ROOT=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def resolve_binding(x):
    if isinstance(x,dict):return {k:resolve_binding(v) for k,v in x.items()}
    if isinstance(x,list):return [resolve_binding(v) for v in x]
    if isinstance(x,str):
        s=x.replace('\\','/');parts=s.split('/')
        if (len(s)>2 and s[1]==':') or s.startswith('CCPV_'):
            first=next((i for i,v in enumerate(parts) if v.startswith('CCPV_')),None)
            if first is not None:
                p=(ROOT/Path(*parts[first:])).resolve()
                if not p.is_relative_to(ROOT):raise ValueError('Reference escapes explicit input root')
                return str(p)
    return x

def J(p):return resolve_binding(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def preflight(args):
    global ROOT
    ROOT=Path(args.input_root).resolve(strict=True)
    here=Path(__file__).resolve().parent
    contract=json.loads((here/'input_contract.json').read_text(encoding='utf-8'))
    if sha(here/'metrics.py')!=contract['metrics_sha256']:raise ValueError('Retained metric source changed')
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Output exists; preserve it')
    if not output.parent.is_dir():raise FileNotFoundError('Output parent must exist')
    protected={here}
    for rec in contract['files']:
        p=(ROOT/rec['path']).resolve(strict=True)
        if not p.is_relative_to(ROOT):raise ValueError('Input path escape')
        if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:raise ValueError('Pinned input differs: '+rec['path'])
        protected.add(p.parent)
    for p in protected:
        if output==p or p in output.parents or output in p.parents:raise ValueError('Output overlaps code/input files')
    return contract,resolve_binding(contract['binding']),output
