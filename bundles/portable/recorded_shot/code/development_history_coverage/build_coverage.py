"""Generate only the original history_coverage.csv from explicit frozen rebuilt inputs."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from frozen_table import coverage_row,table
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def pin(p):p=Path(p);return dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
def put(p,data):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def checked(p,expected):
    r=pin(p)
    if r["sha256"]!=expected["sha256"] or r["bytes"]!=expected["bytes"]:raise ValueError("Input bytes differ: "+Path(p).name)
    return r
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history-dir",type=Path,required=True)
    ap.add_argument("--motion-features",type=Path,required=True)
    ap.add_argument("--legacy-features",type=Path,required=True)
    ap.add_argument("--ledger",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();c=read(HERE/"input_contract.json");started=time.monotonic()
    if sys.version_info[:3]!=(3,11,5) or np.__version__!="1.26.4":raise RuntimeError("Use original Python 3.11.5 / NumPy 1.26.4")
    if a.output.exists():raise FileExistsError(a.output)
    out=a.output.resolve()
    if out.is_relative_to(a.history_dir.resolve()):raise ValueError("Output must be outside input history directory")
    codepins=read(HERE/"package_manifest.json")["files"]
    for r in codepins:
        if sha(HERE/r["file"])!=r["sha256"]:raise ValueError("Frozen code package differs: "+r["file"])
    inputs=dict(motion_features=checked(a.motion_features,c["motion_features"]),
        legacy_features=checked(a.legacy_features,c["legacy_features"]),ledger=checked(a.ledger,c["ledger"]),
        history=[checked(a.history_dir/r["file"],r) for r in c["history"]])
    with a.ledger.open(encoding="utf-8-sig",newline="") as f:ledger=list(csv.DictReader(f))
    if len(ledger)!=11542:raise ValueError("Candidate count changed")
    with np.load(a.legacy_features,allow_pickle=False) as z:
        oldX=z["X"];Y=z["y"];eligible=z["eligible"];matches=z["match"];grid=z["grid_index"]
    with np.load(a.motion_features,allow_pickle=False) as z:
        X=z["X"]
        for name,expected in [("y",Y),("eligible",eligible),("match",matches),("grid_index",grid)]:
            if not np.array_equal(z[name],expected):raise ValueError("Motion/legacy alignment differs: "+name)
        if z["conditions"].tolist()!=["native","grid"]:raise ValueError("Motion condition order differs")
    assert oldX.shape==(11542,16) and X.shape==(2,2,11542,16) and eligible.dtype==bool and eligible.sum()==9483
    assert np.array_equal(grid,np.array([int(r["grid_index"]) for r in ledger]))
    assert np.array_equal(matches,np.array([r["match_id"] for r in ledger]))
    assert np.array_equal(Y,np.array([int(r["y"]) for r in ledger]))
    assert np.array_equal(eligible,np.array([str(r["eligible"]).lower()=="true" for r in ledger]))
    assert set(matches)==set(c["matches"])
    coverage=[];loaded=[]
    for mid in c["matches"]:
        idx=np.flatnonzero(matches==mid);rr=[ledger[i] for i in idx]
        with np.load(a.history_dir/(mid+".npz"),allow_pickle=False) as h:
            assert np.array_equal(h["rows"],idx) and np.array_equal(h["grid_index"],grid[idx])
            assert np.array_equal(h["t0_ns"],np.array([int(r["t0_UTC_ns"]) for r in rr]))
            assert np.array_equal(h["offsets"],np.array([-25,-20,-15,-10,-5,0],dtype=np.int64))
            cm=h["current_mask"];hm=h["history_mask"]
            assert cm.dtype==hm.dtype==bool and cm.shape==hm.shape and cm.shape[0]==len(idx)
            assert np.all(~hm|cm)
        for j,(i,r) in enumerate(zip(idx,rr)):
            N=int(r["N"]);period=r["period"]
            coverage.append(coverage_row(i,grid,mid,period,N,eligible,oldX,cm,hm,j,X))
        loaded.append(dict(match_id=mid,rows=len(idx),stored_masks_used=True,coordinates_loaded=False))
    if len(coverage)!=11542:raise ValueError("Coverage row count differs")
    out.mkdir(parents=False,exist_ok=False)
    target=out/"history_coverage.csv";table(target,coverage)
    actual=pin(target);expected=c["coverage"]
    if actual["sha256"]!=expected["sha256"] or actual["bytes"]!=expected["bytes"]:
        put(out/"failure.json",dict(status="COVERAGE_BYTES_DIFFER_NO_RESTART",actual=actual,expected=expected))
        raise ValueError("Generated coverage bytes differ from original")
    receipt=dict(status="PASS_BYTE_EXACT_HISTORY_COVERAGE",pid=os.getpid(),python=sys.version,numpy=np.__version__,
        elapsed_seconds=time.monotonic()-started,inputs=inputs,source_code_pins=codepins,by_match=loaded,
        output=actual,rows=len(coverage),columns=list(coverage[0]),csv_container_and_content_exact=True,
        missing_values_preserved=True,original_row_order_preserved=True,
        output_scope="history_coverage.csv only; existing input masks and frozen feature counts",
        raw_xml_opened=False,coordinate_arrays_loaded=False,features_recomputed=False,fit=0,
        events_or_labels_rebuilt=False,legacy_bridge_generated=False,whole_goal_complete=False)
    put(out/"receipt.json",receipt)
    print(json.dumps(dict(status=receipt["status"],rows=len(coverage),columns=len(coverage[0]),
        sha256=actual["sha256"],bytes=actual["bytes"],elapsed_seconds=receipt["elapsed_seconds"])),flush=True)
if __name__=="__main__":main()
