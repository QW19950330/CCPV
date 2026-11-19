"""Explicit portable inputs, create-only output, and strict saved-reference checks."""
from pathlib import Path
import csv, gzip, hashlib, json, os
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def rows(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def checked_csv(p, pin):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if fields != pin['fields'] or len(fields) != len(set(fields)):
            raise ValueError('CSV field order or duplicate-column mismatch: ' + Path(p).name)
        result = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in result):
            raise ValueError('CSV row width differs from its header: ' + Path(p).name)
    if len(result) != pin['rows'] or canonical(result) != pin['canonical_rows_sha256']:
        raise ValueError('Exact ordered CSV reference mismatch: ' + Path(p).name)
    return result

def resolve(root, rec, hash_now=True):
    root = Path(root).resolve()
    p = (root / rec['relative_path']).resolve()
    if not p.is_relative_to(root): raise ValueError('Input path escapes explicit root')
    if not p.is_file() or p.stat().st_size != rec['bytes']:
        raise ValueError('Missing or wrong-sized input: ' + rec['relative_path'])
    if hash_now and sha(p) != rec['sha256']:
        raise ValueError('Raw source SHA256 mismatch: ' + rec['relative_path'])
    return dict(rec, path=str(p))

def output_dir(p, raw_roots):
    p = Path(p).resolve()
    for r in raw_roots:
        if p.is_relative_to(Path(r).resolve()):
            raise ValueError('Output must not be inside a raw input root')
    p.mkdir(parents=True, exist_ok=False)
    return p

def put(p, obj):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def pin(p): return dict(file=Path(p).name, bytes=Path(p).stat().st_size, sha256=sha(p))

def identity(stage, out):
    obj = dict(stage=stage, pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
               source_files=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()],
               role='Existing-data reconstruction; not a new validation cohort or tuning run')
    put(out/'execution.json', obj)
    return obj

def check_arrays(p, expected):
    import numpy as np
    result = {}
    with np.load(p, allow_pickle=False) as z:
        if set(z.files) != set(expected): raise ValueError('Feature array names differ')
        for k, e in expected.items():
            a=z[k]
            got=dict(shape=list(a.shape),dtype=a.dtype.str,
                     sha256_c_order_bytes=hashlib.sha256(a.tobytes(order='C')).hexdigest())
            if got != e: raise ValueError('Exact feature array mismatch: '+k)
            result[k]=got
    return result
