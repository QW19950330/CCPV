"""Independent read-only comparison of all feature NPZ arrays and the original byte pin."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",required=True)
    p.add_argument("--rebuilt",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    expected="b5f94c95aa8d14d60a1b7d5600e2b6a6f090e81b4a64958ee0aa2f8dd8528d07"
    assert sha(a.original)==expected
    checks=[]
    with np.load(a.original,allow_pickle=False) as old, np.load(a.rebuilt,allow_pickle=False) as new:
        assert old.files==new.files
        for name in old.files:
            x,y=old[name],new[name]
            same_dtype=x.dtype==y.dtype
            same_shape=x.shape==y.shape
            exact=same_dtype and same_shape and x.tobytes(order="C")==y.tobytes(order="C")
            checks.append(dict(array=name,dtype=str(x.dtype),shape=list(x.shape),bit_exact=exact,
                original_c_bytes_sha256=hashlib.sha256(x.tobytes(order="C")).hexdigest(),
                rebuilt_c_bytes_sha256=hashlib.sha256(y.tobytes(order="C")).hexdigest()))
    passed=len(checks)==8 and all(r["bit_exact"] for r in checks)
    result=dict(status="PASS_INDEPENDENT_8_ARRAYS_BIT_EXACT" if passed else "FAIL",
        original_sha256=sha(a.original),rebuilt_sha256=sha(a.rebuilt),
        container_byte_exact=sha(a.original)==sha(a.rebuilt),checks=checks,
        does_not_validate_raw_events_or_training=True)
    with Path(a.output).open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","container_byte_exact","rebuilt_sha256")}))
    if not passed: sys.exit(1)

if __name__=="__main__":main()
