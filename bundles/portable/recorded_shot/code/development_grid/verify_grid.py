"""Strict string-cell comparison for the frozen 13,372-row candidate grid."""
import argparse,csv,gzip,hashlib,io,json,struct
from collections import Counter
from pathlib import Path

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def put_json(p,data):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def file_pin(p):p=Path(p);return dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
def gzip_parts(data):
    if data[:3]!=b"\x1f\x8b\x08":raise ValueError("Not gzip/deflate")
    flags=data[3];i=10
    if flags&4:
        n=int.from_bytes(data[i:i+2],"little");i+=2+n
    filename=None
    if flags&8:
        j=data.index(b"\0",i);filename=data[i:j].decode("latin1");i=j+1
    if flags&16:i=data.index(b"\0",i)+1
    if flags&2:i+=2
    return dict(mtime=int.from_bytes(data[4:8],"little"),flags=flags,filename=filename,
                header_sha256=sha_bytes(data[:i]),deflate_sha256=sha_bytes(data[i:-8]),footer_hex=data[-8:].hex())
def read_grid(path):
    data=Path(path).read_bytes();plain=gzip.decompress(data)
    reader=csv.DictReader(io.StringIO(plain.decode("utf-8"),newline=""))
    header=reader.fieldnames
    if not header or len(header)!=len(set(header)):raise ValueError("Missing/duplicate grid header")
    rows=list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):raise ValueError("Malformed CSV row")
    return header,rows,plain,data
def inspect_grid(path,contract,require_original_container=False):
    header,rows,plain,data=read_grid(path);expected=contract["reference_grid"]
    checks=dict(header_exact=header==expected["columns"],rows_exact=len(rows)==expected["rows"],
        plain_sha256_exact=sha_bytes(plain)==expected["uncompressed_sha256"],
        by_match_exact=dict(Counter(r["match_id"] for r in rows))==expected["by_match"],
        original_container_exact=sha_bytes(data)==expected["sha256"])
    if not all(checks[k] for k in checks if k!="original_container_exact"):
        raise ValueError("Frozen grid content mismatch: "+json.dumps(checks))
    if require_original_container and not checks["original_container_exact"]:
        raise ValueError("Reference gzip container is not the original hard-pinned file")
    return header,rows,dict(file=file_pin(path),uncompressed_bytes=len(plain),
        uncompressed_sha256=sha_bytes(plain),rows=len(rows),columns=len(header),
        by_match=dict(Counter(r["match_id"] for r in rows)),checks=checks,gzip_header=gzip_parts(data))
def compare_grid(reference,candidate,contract):
    h0,r0,p0,b0=read_grid(reference);h1,r1,p1,b1=read_grid(candidate)
    differences=[]
    if h0!=h1:differences.append(dict(kind="HEADER",reference=h0,candidate=h1))
    if len(r0)!=len(r1):differences.append(dict(kind="ROW_COUNT",reference=len(r0),candidate=len(r1)))
    count=0;different_cells=0
    for i,(a,b) in enumerate(zip(r0,r1)):
        for key in h0:
            count+=1
            if a.get(key)!=b.get(key):
                different_cells+=1
                if len(differences)<20:differences.append(dict(row=i,column=key,reference=a.get(key),candidate=b.get(key)))
    result=dict(rows=len(r1),columns=len(h1),cells_compared=count,different_cells=different_cells,
        exact_header=h0==h1,exact_row_count=len(r0)==len(r1),all_string_cells_equal=not differences,
        uncompressed_bytes_equal=p0==p1,gzip_container_bytes_equal=b0==b1,
        reference_sha256=sha_bytes(b0),candidate_sha256=sha_bytes(b1),
        reference_gzip=gzip_parts(b0),candidate_gzip=gzip_parts(b1),first_differences=differences)
    result["same_deflate_stream"]=result["reference_gzip"]["deflate_sha256"]==result["candidate_gzip"]["deflate_sha256"]
    result["same_gzip_footer"]=result["reference_gzip"]["footer_hex"]==result["candidate_gzip"]["footer_hex"]
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-grid",type=Path,required=True);p.add_argument("--candidate-grid",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True);a=p.parse_args()
    if a.receipt.exists():raise FileExistsError(a.receipt)
    c=read_json(Path(__file__).with_name("input_contract.json"))
    inspect_grid(a.reference_grid,c,True);inspect_grid(a.candidate_grid,c)
    r=compare_grid(a.reference_grid,a.candidate_grid,c);put_json(a.receipt,r)
    if not r["all_string_cells_equal"] or not r["uncompressed_bytes_equal"]:raise SystemExit(1)
    print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
