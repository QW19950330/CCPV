"""Independent complete-table comparison; never creates production inputs."""
from pathlib import Path
import argparse, csv, gzip, hashlib, itertools, json

def sha(path,decoded=False):
    h=hashlib.sha256(); opener=gzip.open if decoded else open
    with opener(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--replay',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();old=Path(a.original);new=Path(a.replay);out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    rows=[];total=0
    expected={'scene_path_status.csv.gz':110592,'scene_guard_metrics.csv.gz':442368,
        'scene_diagnostics.csv.gz':221184,'guard_by_match.csv':6912,'guard_by_seed.csv':576,'guard_summary.csv':144}
    sentinel=object()
    for name in ('scene_path_status.csv.gz','scene_guard_metrics.csv.gz','scene_diagnostics.csv.gz',
                 'guard_by_match.csv','guard_by_seed.csv','guard_summary.csv','first_failure_by_match.csv'):
        op=gzip.open if name.endswith('.gz') else open
        with op(old/name,'rt',encoding='utf-8-sig',newline='') as x,op(new/name,'rt',encoding='utf-8-sig',newline='') as y:
            xr,yr=csv.reader(x),csv.reader(y);xh,yh=next(xr),next(yr)
            if xh!=yh or len(xh)!=len(set(xh)):raise ValueError('Header mismatch or duplicate: '+name)
            n=0
            for u,v in itertools.zip_longest(xr,yr,fillvalue=sentinel):
                if u is sentinel or v is sentinel or len(u)!=len(xh) or len(v)!=len(xh) or u!=v:
                    raise ValueError('Exact cell/row mismatch: '+name+' row '+str(n))
                n+=1
        if name in expected and n!=expected[name]: raise ValueError('Planned rows changed: '+name)
        decoded=name.endswith('.gz');h1=sha(old/name,decoded);h2=sha(new/name,decoded)
        if h1!=h2:raise ValueError('Serialized content changed: '+name)
        total+=n*len(xh)
        rows.append(dict(file=name,rows=n,columns=len(xh),cells=n*len(xh),
            decoded_sha256=h2,original_container_sha256=sha(old/name),replay_container_sha256=sha(new/name),
            exact_cells_and_order=True,exact_uncompressed_bytes=True))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_ALL_SAVED_OUTPUT_METRICS_EXACT',
        files=rows,total_cells=total,compressed_mtime_ignored_only=True,
        scope='17 frozen array inputs to all saved-output engineering metrics, not upstream generation'),f,indent=2)
    print('ALL_FILES_EXACT cells='+str(total))

if __name__=='__main__':main()
