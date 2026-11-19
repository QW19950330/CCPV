"""Original 192-window computation and support signatures, no path-bound I/O."""
import numpy as np
import pandas as pd
import torch
import runtime as api
from common import ah
from scoring_external import score as exact_score

def compute(arm,seed,a,device,model,acceleration_bound):
    n=len(a['history']);rows=[];predictions=[];mask_records=[]
    def score(*args):return exact_score(*args,acceleration_bound)
    with torch.no_grad():
        for condition in ('natural','tailgap2'):
            values=[];chosen_all=[]
            for start in range(0,n,32):
                ix=np.arange(start,min(start+32,n));h=torch.tensor(a['history'][ix],device=device);v=torch.tensor(a['history_valid'][ix],device=device,dtype=torch.bool)
                teams=torch.tensor(a['teams'],device=device,dtype=torch.long)
                clean,usable,chosen=api.mask_history(h,v,teams,a['source_indices'][ix],condition=condition)
                anchor,_,_=api.current_anchor(clean,usable)
                pred=(api.predict(model,clean,usable,teams) if model is not None else api.analytic_forecast('cv',clean,usable)).cpu().numpy()
                values.append(pred);chosen_all.append(chosen.cpu().numpy())
                for horizon,group,metrics in score(pred,a['position_targets'][ix],a['position_valid'][ix],a['target_states'][ix],a['target_valid'][ix],v.any(1).cpu().numpy(),a['teams'],chosen.cpu().numpy(),anchor.cpu().numpy()):
                    for j,index in enumerate(ix):
                        row=dict(arm=arm,seed=int(seed),source_index=int(a['source_indices'][index]),match_id=str(a['match_ids'][index]),condition=condition,horizon_seconds=horizon/10,group=group)
                        for key,value in metrics.items():row[key]=int(value[j]) if key in ('observed_points','terminal_agents','velocity_targets','consistency_actor_steps') else float(value[j])
                        rows.append(row)
            predictions.append(np.concatenate(values));mask_records.append(np.concatenate(chosen_all))
    frame=pd.DataFrame(rows)
    assert len(frame)==n*40 and frame.source_index.nunique()==n
    signature={k:ah(v) for k,v in dict(original_history_seen=a['history_valid'].any(1),future_position_valid=a['position_valid'],future_velocity_valid=a['target_valid'],hidden_masks=np.stack(mask_records)).items()}
    return frame,np.stack(predictions),np.stack(mask_records),signature
