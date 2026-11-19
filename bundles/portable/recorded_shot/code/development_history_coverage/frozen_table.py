"""Original coverage expression and create-only CSV writer; no original entry point."""
from pathlib import Path
import csv
import numpy as np

def table(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def coverage_row(i,grid,mid,period,N,eligible,oldX,cm,hm,j,X):
    return dict(row=int(i),grid_index=int(grid[i]),match_id=mid,period=period,N=N,eligible=bool(eligible[i]),old_current_count=float(oldX[i,6]) if np.isfinite(oldX[i,6]) else None,old_200ms_count=float(oldX[i,7]) if np.isfinite(oldX[i,7]) else None,current_count=int(cm[j].sum()),six_point_history_count=int(hm[j].sum()),radial_native_count=float(X[0,0,i,8]) if np.isfinite(X[0,0,i,8]) else None,radial_grid_count=float(X[1,0,i,8]) if np.isfinite(X[1,0,i,8]) else None)
