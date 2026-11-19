"""Independent full-content comparison. Does not import any producer or its contract."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

EVENT_TABLES=('all_event_versions.csv','shot_candidates.csv','event_clocks.csv','author_clock.csv','context_fields.csv')
PLAYER_TABLES=('all_anchor_ledger.csv','event_clock_summary.csv','event_links.csv')

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f);fields=r.fieldnames
        if not fields or len(fields)!=len(set(fields)):
            raise ValueError('Missing or duplicate CSV columns: '+path.name)
        a=list(r)
        if any(None in row or any(v is None for v in row.values()) for row in a):
            raise ValueError('CSV row width differs from its header: '+path.name)
        return fields,a

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def compare_tables(got, ref, names):
    result={}
    for n in names:
        gh,gr=read_csv(got/n);rh,rr=read_csv(ref/n)
        if gh!=rh or len(gr)!=len(rr): raise AssertionError('Table shape or header mismatch: '+n)
        for i,(g,r) in enumerate(zip(gr,rr)):
            for k in rh:
                if g[k]!=r[k]: raise AssertionError(f'Cell mismatch: {n} row {i} field {k}')
        result[n]=dict(rows=len(rr),columns=len(rh),cells=len(rr)*len(rh),all_cells_exact=True,
                       output_sha256=digest(got/n),reference_sha256=digest(ref/n),
                       file_bytes_exact=(got/n).read_bytes()==(ref/n).read_bytes())
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--aggregate',type=Path)
    ap.add_argument('--reference-event-dir',type=Path)
    ap.add_argument('--ledger',type=Path)
    ap.add_argument('--reference-player-dir',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if not a.aggregate and not a.ledger: raise ValueError('At least one stage required')
    if bool(a.aggregate)!=bool(a.reference_event_dir) or bool(a.ledger)!=bool(a.reference_player_dir):
        raise ValueError('Each stage requires its independent original reference directory')
    result=dict(status='PASS_INDEPENDENT_FULL_CONTENT_COMPARISON',producer_imports=False)
    if a.aggregate: result['event_tables']=compare_tables(a.aggregate,a.reference_event_dir,EVENT_TABLES)
    if a.ledger:
        result['player_tables']=compare_tables(a.ledger,a.reference_player_dir,PLAYER_TABLES)
        arrays={}
        with np.load(a.ledger/'features.npz',allow_pickle=False) as g, np.load(a.reference_player_dir/'features.npz',allow_pickle=False) as r:
            if g.files!=r.files: raise AssertionError('Feature array names or order differ')
            for k in r.files:
                if g[k].shape!=r[k].shape or g[k].dtype!=r[k].dtype or g[k].tobytes()!=r[k].tobytes():
                    raise AssertionError('Feature array bytes differ: '+k)
                arrays[k]=dict(shape=list(r[k].shape),dtype=r[k].dtype.str,bytes=r[k].nbytes,exact=True)
        result['feature_arrays']=arrays
        result['feature_container_bytes_exact']=(a.ledger/'features.npz').read_bytes()==(a.reference_player_dir/'features.npz').read_bytes()
    with a.output.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'],flush=True)

if __name__=='__main__':main()
