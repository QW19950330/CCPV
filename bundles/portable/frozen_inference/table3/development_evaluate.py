"""Original complete 256-window computation, no path-bound I/O."""
import numpy as np
import pandas as pd
import torch
from runtime import *
from scoring_development import score

def compute(arm,seed,arrays,device,model,acceleration_bound):
    learned=arm not in ('cv','smooth_cv');all_predictions=[];rows=[]
    with torch.no_grad():
        for condition in ('natural','tailgap2'):
            predictions=[]
            for start in range(0,256,32):
                ix=np.arange(start,min(start+32,256));h=torch.tensor(arrays['states'][ix,:10],device=device)
                v=torch.tensor(arrays['valid'][ix,:10],device=device,dtype=torch.bool)
                tm=arrays['teams'] if arrays['teams'].ndim==1 else arrays['teams'][ix];teams=torch.tensor(tm,device=device,dtype=torch.long)
                clean,usable,chosen=mask_history(h,v,teams,arrays['source_indices'][ix],condition=condition)
                anchor,_,_=current_anchor(clean,usable)
                pred=(predict(model,clean,usable,teams,integrate=arm=='A_integrate') if learned
                      else analytic_forecast(arm,clean,usable)).cpu().numpy()
                predictions.append(pred)
                scored=score(pred,arrays['position_targets'][ix,10:30],arrays['position_valid'][ix,10:30],
                    arrays['states'][ix,10:30],arrays['valid'][ix,10:30],v.any(1).cpu().numpy(),tm,
                    chosen.cpu().numpy(),anchor.cpu().numpy(),acceleration_bound)
                for horizon,group,metrics in scored:
                    for b,index in enumerate(ix):
                        row=dict(arm=arm,seed=int(seed),source_index=int(arrays['source_indices'][index]),
                            match_id=str(arrays['match_ids'][index]),condition=condition,horizon_seconds=horizon/10.,group=group)
                        for key,value in metrics.items():
                            row[key]=int(value[b]) if key in ('observed_points','terminal_agents','velocity_targets','consistency_actor_steps') else float(value[b])
                        rows.append(row)
            all_predictions.append(np.concatenate(predictions,axis=0))
    frame=pd.DataFrame(rows)
    if len(frame)!=8192:raise AssertionError('All256windows must be scored, including no-runner windows')
    return frame,np.stack(all_predictions)
