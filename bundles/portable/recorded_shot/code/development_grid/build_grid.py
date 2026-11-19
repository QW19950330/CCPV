"""Rebuild the frozen seven-match metadata grid or adapt its original cache."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,os,sys,time,traceback
from collections import Counter
from pathlib import Path
import frozen_ball_metadata as frozen
from verify_grid import file_pin,inspect_grid,compare_grid,put_json,read_json,sha

ROOT=Path(__file__).resolve().parent
class CountingDiscardWriter:
    """Count original metadata writer calls without storing row/player data."""
    def __init__(self):self.rows=0
    def writerow(self,row):self.rows+=1

def check_package():
    m=read_json(ROOT/"package_manifest.json")
    for r in m["files"]:
        p=ROOT/r["file"]
        if p.stat().st_size!=r["bytes"] or sha(p)!=r["sha256"]:raise ValueError("Package source pin mismatch: "+r["file"])
    return dict(files=m["files"],manifest_sha256=sha(ROOT/"package_manifest.json"))

def write_six(rows,output,contract):
    cohort=frozen.six_cohort(rows,contract["six_scope"])
    if len(cohort)!=contract["six_candidates"]:raise ValueError("Six-match candidate count changed")
    path=output/"candidates_six.csv"
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["grid_index","match_id","period","N"]);w.writeheader();w.writerows(cohort)
    # Full seven-match index is retained. The six-match export is deliberately only these four fields.
    if sha(path)!=contract["six_export_sha256"]:raise ValueError("Candidate index/ordering/CSV differs from fixed source projection")
    return dict(file=file_pin(path),rows=len(cohort),columns=["grid_index","match_id","period","N"],
        global_index_preserved=True,selection="fixed six_scope membership only",by_match=dict(Counter(r["match_id"] for r in cohort)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    for mode in ("cache","raw"):
        q=sub.add_parser(mode);q.add_argument("--reference-grid",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
        if mode=="raw":
            q.add_argument("--raw-root",type=Path,required=True)
            q.add_argument("--authorize-raw-ball-metadata-scan",action="store_true")
    a=p.parse_args();c=read_json(ROOT/"input_contract.json");package=check_package()
    if a.output.exists():raise FileExistsError("Output must not exist: "+str(a.output))
    _,reference_rows,reference_info=inspect_grid(a.reference_grid,c,True)
    if a.mode=="raw" and not a.authorize_raw_ball_metadata_scan:
        raise PermissionError("Explicit raw BALL metadata scan authorization required")
    if a.mode=="raw":
        if sys.version_info[:3]!=(3,11,5):raise RuntimeError("Frozen validation runtime requires Python 3.11.5")
        import numpy as np
        from lxml import etree
        if np.__version__!="1.26.4":raise RuntimeError("Frozen validation runtime requires NumPy 1.26.4")
        runtime=dict(python=sys.version,numpy=np.__version__,lxml=etree.__version__)
        # Stat all files before any full positions read.
        for r in c["raw_files"]:
            f=a.raw_root/r["file"]
            if f.stat().st_size!=r["bytes"]:raise ValueError("Raw file byte-size pin differs: "+r["file"])
    else:runtime=dict(python=sys.version)
    a.output.mkdir(parents=False,exist_ok=False);start=time.monotonic()
    put_json(a.output/"identity.json",dict(pid=os.getpid(),started_epoch=time.time(),mode=a.mode,
        runtime=runtime,source_package=package,reference_grid=reference_info["file"]))
    try:
        source_records=[];period_records=[]
        target=a.output/"prefix_grid.csv.gz"
        if a.mode=="cache":
            with target.open("xb") as f:f.write(a.reference_grid.read_bytes())
        else:
            sink=CountingDiscardWriter()
            with gzip.open(frozen.xp(target),"xt",encoding="utf-8",newline="") as gf:
                writer=None
                for mid in frozen.MATCHES:
                    full="DFL-MAT-"+mid
                    ppin=next(r for r in c["raw_files"] if r["match_id"]==full and r["kind"].startswith("POSITION"))
                    mpin=next(r for r in c["raw_files"] if r["match_id"]==full and r["kind"].startswith("MATCH"))
                    meta=dict(mpin,path=str(a.raw_root/mpin["file"]));pin=dict(ppin,path=str(a.raw_root/ppin["file"]))
                    if sha(meta["path"])!=meta["sha256"]:raise ValueError("Match metadata source pin differs")
                    home,away,native=frozen.team_metadata(meta)
                    print("SCAN_BALL_METADATA "+full,flush=True)
                    before=sink.rows;z=frozen.scan(pin,sink)
                    for row in frozen.grid_rows(full,z,home,away):
                        if writer is None:writer=csv.DictWriter(gf,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
                    for period in z["rawcounts"]:
                        d=z["tables"][period];origin=z["origins"].get(period)
                        period_records.append(dict(match_id=full,period=period,
                            raw_ball_rows=z["rawcounts"][period],source_origin_N=origin,
                            max_N=max(d) if d else None,candidates=len(frozen.grid(d,origin))))
                    source_records.append(dict(match_id=full,
                        positions=dict(file=ppin["file"],bytes_read=z["bytes"],sha256=z["sha256"]),
                        matchinformation=file_pin(meta["path"]),ball_metadata_rows=sink.rows-before,
                        frame_attributes_read=["N","T","BallPossession","BallStatus"],
                        player_frame_attributes_read=False,ball_coordinates_accessed=False))
                    print("COMPLETE_BALL_METADATA "+full,flush=True)
                    del z
        _,rows,info=inspect_grid(target,c)
        comparison=compare_grid(a.reference_grid,target,c)
        if not comparison["all_string_cells_equal"] or not comparison["uncompressed_bytes_equal"]:
            put_json(a.output/"mismatch.json",comparison);raise ValueError("Grid differs from original")
        six=write_six(rows,a.output,c)
        result=dict(status="PASS_FROZEN_GRID_"+a.mode.upper(),mode=a.mode,runtime=runtime,
            elapsed_seconds=time.monotonic()-start,source_package=package,reference_grid=reference_info,
            rebuilt_grid=info,comparison=comparison,six_candidate_export=six,
            source_files=source_records,periods=period_records,
            positions_bytes_read=sum(r["positions"]["bytes_read"] for r in source_records),
            metadata_bytes=sum(r["matchinformation"]["bytes"] for r in source_records),
            ball_metadata_rows=sum(r["ball_metadata_rows"] for r in source_records),
            events_read=False,training_results_read=False,model_forward=0,fit=0,
            coordinates_accessed=False,labels_rebuilt=False,
            limits=["Candidate metadata only; source live is not certified non-restart open play.",
                    "No event validity/clock mapping, labels, player feature extraction, training or new holdout test.",
                    "Six candidate indices are the original full seven-match row indices, never renumbered."])
        put_json(a.output/"receipt.json",result)
        print(json.dumps(dict(status=result["status"],rows=len(rows),six_rows=six["rows"],
            cells_compared=comparison["cells_compared"],gzip_bytes_equal=comparison["gzip_container_bytes_equal"],
            elapsed_seconds=result["elapsed_seconds"])),flush=True)
    except BaseException as exc:
        put_json(a.output/"failure.json",dict(status="FAILED_NO_AUTOMATIC_RESTART",type=type(exc).__name__,
            message=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start))
        raise
if __name__=="__main__":main()
