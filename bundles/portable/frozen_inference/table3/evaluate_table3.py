"""Frozen Table 3 evaluation; no training, checkpoint selection or implicit GPU run."""
import os,sys
# Set before any torch import. CPU commands cannot allocate a CUDA device.
if "--device" not in sys.argv or ("--device" in sys.argv and sys.argv[sys.argv.index("--device")+1]=="cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[_key]="1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
import common as c
import runtime
from development_evaluate import compute as development_compute
from external_evaluate import compute as external_compute
from summaries import development_summary,external_summary

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("check-inputs","evaluate"))
    p.add_argument("--development-data",type=Path,required=True)
    p.add_argument("--development-truth",type=Path,required=True)
    p.add_argument("--external-data",type=Path,required=True)
    p.add_argument("--weights-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--authorize-full-evaluation",action="store_true")
    return p
def main():
    a=parser().parse_args()
    if a.stage=="evaluate" and not a.authorize_full_evaluation:raise PermissionError("Full frozen evaluation needs the explicit authorization flag")
    if a.stage=="check-inputs" and a.device!="cpu":raise ValueError("Input checks are CPU-only")
    code=c.verify_package();inputs=c.check_inputs(a);contract=c.contract()
    out=c.output_dir(a.output,[a.weights_root,a.development_data.parent,a.development_truth.parent,a.external_data.parent]);start=time.monotonic()
    c.put(out/"identity.json",dict(pid=os.getpid(),stage=a.stage,device=a.device,python=sys.version,
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,source_code_pins=code,inputs=inputs))
    if a.stage=="check-inputs":
        c.put(out/"completion.json",dict(status="PASS_INPUT_BYTE_PINS_NO_MODEL_FORWARD",inputs=inputs,
            expected_workload=contract["workload"],model_forward_calls=0,optimizer_updates=0))
        print("PASS_INPUT_BYTE_PINS_NO_MODEL_FORWARD",flush=True);return
    try:
        if a.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("Explicit CUDA device is unavailable")
        device=runtime.configure(a.device)
        dev=c.validation_rows(c.load_development(a.development_data,a.development_truth))
        ext=c.load_external(a.external_data)
        if len(dev["states"])!=256 or len(ext["history"])!=192:raise ValueError("Fixed cohort size differs")
        all_rows=[];cell_receipts=[];all_exact=True
        for cohort,arrays in [("development",dev),("external",ext)]:
            base=out/cohort;base.mkdir();evaluation=base/"evaluation";evaluation.mkdir();frames=[];signature=None
            for arm in contract["arms"]:
                for seed in ([-1] if arm=="cv" else contract["seeds"]):
                    folder=evaluation/(arm+"_seed"+str(seed));folder.mkdir()
                    model=c.load_model(arm,seed,a.weights_root,device) if arm!="cv" else None
                    if cohort=="development":
                        frame,values=development_compute(arm,seed,arrays,device,model,contract["acceleration_bound"])
                        arrays_out=dict(states=values,source_indices=arrays["source_indices"],conditions=np.array(["natural","tailgap2"]))
                    else:
                        frame,values,masks,sig=external_compute(arm,seed,arrays,device,model,contract["acceleration_bound"])
                        if signature is None:signature=sig
                        elif sig!=signature:raise ValueError("Cross-arm cohort/mask drift")
                        arrays_out=dict(states=values,source_indices=arrays["source_indices"],conditions=np.array(["natural","tailgap2"]),hidden_masks=masks)
                    c.save_csv(folder/"metrics.csv",frame);c.save_npz(folder/"predictions.npz",**arrays_out)
                    expected=contract["evaluation_reference"][cohort][arm+"_seed"+str(seed)]
                    array_checks={k:c.ah(v)==expected["arrays"][k] for k,v in arrays_out.items()}
                    exact=all(array_checks.values()) and c.sha(folder/"metrics.csv")==expected["metrics_sha256"]
                    all_exact=all_exact and exact
                    rec=dict(cohort=cohort,arm=arm,seed=seed,rows=len(frame),windows=len(values[0]),
                        metrics=c.pin(folder/"metrics.csv"),predictions=c.pin(folder/"predictions.npz"),
                        reference_prediction_arrays_exact=array_checks,
                        reference_metrics_csv_bytes_exact=c.sha(folder/"metrics.csv")==expected["metrics_sha256"],
                        reference_prediction_npz_bytes_exact=c.sha(folder/"predictions.npz")==expected["predictions_sha256"],
                        numerical_tolerance_used=False,optimizer_updates=0)
                    c.put(folder/"receipt.json",rec);cell_receipts.append(rec);frames.append(frame)
                    print("EVALUATED",cohort,arm,seed,"REFERENCE_EXACT",exact,flush=True)
                    del model
            # Read the emitted CSVs with the original parser before the original aggregation,
            # preserving the historical producer->CSV->pandas round trip.
            raw=pd.concat([pd.read_csv(evaluation/(r["arm"]+"_seed"+str(r["seed"]))/"metrics.csv") for r in cell_receipts if r["cohort"]==cohort],ignore_index=True)
            if cohort=="development":bymatch,byseed,summary=development_summary(raw)
            else:bymatch,byseed,summary=external_summary(raw,contract["external_summary_study"],{})
            analysis=base/"analysis";analysis.mkdir()
            for name,frame in [("by_seed_match.csv",bymatch),("by_seed.csv",byseed),("arm_summary.csv",summary)]:c.save_csv(analysis/name,frame)
            view=summary[(summary.condition=="natural")&(summary.horizon_seconds==2)&(summary.group=="players")]
            for arm in contract["table3_display_arms"][cohort]:
                r=view[view.arm==arm]
                if len(r)!=1:raise ValueError("Table3 row missing")
                r=r.iloc[0]
                all_rows.append(dict(population=cohort,arm=arm,ade_mean=float(r.ade_m_mean),ade_sd=float(r.ade_m_std),
                    fde_mean=float(r.fde_m_mean),fde_sd=float(r.fde_m_std),training_seeds=1 if arm=="cv" else 5))
        table=pd.DataFrame(all_rows);c.save_csv(out/"table3_numeric.csv",table)
        expected=contract["table3_expected"]
        table_exact=all(all(r[k]==q[k] for k in ("population","arm","ade_mean","ade_sd","fde_mean","fde_sd")) for r,q in zip(all_rows,expected)) and len(all_rows)==len(expected)
        status="PASS_FROZEN_TABLE3_REFERENCE_EXACT" if all_exact and table_exact else "EVALUATED_REFERENCE_DIFFERENCE_REQUIRES_REVIEW"
        c.put(out/"completion.json",dict(status=status,elapsed_seconds=time.monotonic()-start,device=device,
            inputs=inputs,cells=cell_receipts,table3_numeric=c.pin(out/"table3_numeric.csv"),
            all_prediction_arrays_and_metric_csvs_exact=all_exact,table3_full_precision_exact=table_exact,
            reference_differences_not_silently_tolerated=True,expected_workload=contract["workload"],
            optimizer_updates=0,training_performed=False,raw_data_reconstruction_performed=False,whole_goal_complete=False))
        print(status,flush=True)
        if status!="PASS_FROZEN_TABLE3_REFERENCE_EXACT":raise SystemExit(2)
    except BaseException as exc:
        if not (out/"completion.json").exists():
            c.put(out/"failure.json",dict(status="FAILED_OUTPUT_RETAINED_NO_RESTART",type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
if __name__=="__main__":main()
