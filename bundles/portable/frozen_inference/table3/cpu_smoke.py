"""Small CPU-only executable model/loss/scorer check; never a full evaluation or fit."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[_key]="1"
import argparse,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import default_collate
import common as c
import runtime
from losses import Windows,objective
from scoring_development import score as development_score
from scoring_external import score as external_score
from exact_time_adapter import exact_arrays

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("development-data","development-truth","external-data","weights-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    return p.parse_args()
def main():
    a=parse();code=c.verify_package();inputs=c.check_inputs(a);contract=c.contract()
    out=c.output_dir(a.output,[a.weights_root,a.development_data.parent,a.development_truth.parent,a.external_data.parent]);start=time.monotonic()
    device=runtime.configure("cpu")
    if torch.cuda.is_available():raise RuntimeError("CPU smoke must have CUDA hidden")
    full=c.load_development(a.development_data,a.development_truth);dev=c.validation_rows(full);ext=c.load_external(a.external_data)
    outputs={};scores={};counts=0;loss_records=[]
    for arm in ("B_free2d","C_coupled_tf","D_coupled_ar"):
        model=c.load_model(arm,0,a.weights_root,device)
        for cohort,arrays in [("development",dev),("external",ext)]:
            ix=np.arange(2)
            h=torch.tensor(arrays["states"][ix,:10] if cohort=="development" else arrays["history"][ix])
            v=torch.tensor(arrays["valid"][ix,:10] if cohort=="development" else arrays["history_valid"][ix],dtype=torch.bool)
            tm=arrays["teams"] if arrays["teams"].ndim==1 else arrays["teams"][ix]
            teams=torch.tensor(tm,dtype=torch.long)
            for condition in ("natural","tailgap2"):
                clean,usable,chosen=runtime.mask_history(h,v,teams,arrays["source_indices"][ix],condition=condition)
                anchor,_,_=runtime.current_anchor(clean,usable)
                with torch.no_grad():pred=runtime.predict(model,clean,usable,teams).cpu().numpy()
                counts+=20;key=cohort+"__"+arm+"__"+condition
                outputs[key+"__states"]=pred;outputs[key+"__chosen"]=chosen.numpy();outputs[key+"__anchor"]=anchor.numpy()
                if cohort=="development":
                    scored=development_score(pred,arrays["position_targets"][ix,10:30],arrays["position_valid"][ix,10:30],
                        arrays["states"][ix,10:30],arrays["valid"][ix,10:30],v.any(1).numpy(),tm,chosen.numpy(),anchor.numpy(),contract["acceleration_bound"])
                else:
                    scored=external_score(pred,arrays["position_targets"][ix],arrays["position_valid"][ix],
                        arrays["target_states"][ix],arrays["target_valid"][ix],v.any(1).numpy(),tm,chosen.numpy(),anchor.numpy(),contract["acceleration_bound"])
                labels=[dict(horizon=hor,group=group,metric=k) for hor,group,m in scored for k in m]
                outputs[key+"__scores"]=np.stack([val for hor,group,m in scored for val in m.values()])
                scores[key]=labels
        dataset=Windows(full);dataset.epoch=1
        batch=default_collate([dataset[0],dataset[1]]);batch.pop("item_index")
        model.train()
        with torch.no_grad():loss,diagnostics=objective(model,batch,arm,return_predictions=True)
        counts+=20
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite fixed tiny loss")
        outputs[arm+"__loss_predictions"]=diagnostics.pop("predictions").numpy()
        outputs[arm+"__loss"]=np.array(float(loss),dtype=np.float64)
        loss_records.append(dict(arm=arm,seed=0,training_batch_rows=2,**diagnostics))
    for i in range(2):
        derived=exact_arrays(ext["native_positions_with_pad"][i])
        for name,value in derived.items():
            if not np.array_equal(value,ext[name][i],equal_nan=True):raise AssertionError("Exact-time adapter mismatch: "+name)
            outputs["adapter_"+str(i)+"__"+name]=value
    # One fixed batch32 timing measurement; it does not emit a new scientific score.
    model=c.load_model("D_coupled_ar",0,a.weights_root,"cpu");h=torch.tensor(dev["states"][:32,:10]);v=torch.tensor(dev["valid"][:32,:10]);tm=torch.tensor(dev["teams"])
    bench_start=time.monotonic()
    with torch.no_grad():benchmark=runtime.predict(model,h,v,tm)
    batch_seconds=time.monotonic()-bench_start;counts+=20
    if not torch.isfinite(benchmark).all():raise AssertionError("Nonfinite benchmark")
    c.save_npz(out/"cpu_outputs.npz",**outputs)
    result=dict(status="PASS_CPU_SMALL_MODEL_LOSS_SCORE_CHECK",python=sys.version,torch=torch.__version__,
        numpy=np.__version__,pandas=pd.__version__,cuda_visible=False,elapsed_seconds=time.monotonic()-start,
        technical_seed=0,technical_windows_per_cohort=2,scientific_seed_selection=False,
        model_forward_calls=counts,optimizer_updates=0,backward_calls=0,full_evaluation_performed=False,
        inputs=inputs,source_code_pins=code,losses=loss_records,score_layout=scores,
        exact_time_adapter_first_two_windows_exact=True,
        batch32_20step_cpu_seconds=batch_seconds,
        projected_420_batch_cpu_compute_seconds=batch_seconds*420,
        compute_projection_excludes_loading_serialization_and_host_variability=True,
        outputs=c.pin(out/"cpu_outputs.npz"),arrays={k:c.ah(v) for k,v in outputs.items()})
    c.put(out/"receipt.json",result)
    print(result["status"],"elapsed",result["elapsed_seconds"],"batch32_seconds",batch_seconds,flush=True)
if __name__=="__main__":main()
