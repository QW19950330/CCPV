"""Pure frozen-probability evaluation; no fit, clipping, filtering or I/O."""
import math
import numpy as np

HISTORICAL=('HIST_NeuralMatched','HIST_CalDirectUnion602','HIST_ObservedH2Oracle','HIST_FullCVChain')
PAIRS=(('E_Chain_OldAlpha','D_Chain_OldAlpha'),('E_Chain_Raw','D_Chain_Raw'),
 ('E_Direct_OldAlpha','D_Direct_OldAlpha'),('E_Direct_Raw','D_Direct_Raw'),
 ('E_Chain_OldAlpha','CV_Chain_OldAlpha'),('D_Chain_OldAlpha','CV_Chain_OldAlpha'),
 ('E_Chain_OldAlpha','E_Direct_OldAlpha'),('D_Chain_OldAlpha','D_Direct_OldAlpha'),
 ('CV_Chain_OldAlpha','CV_Direct_OldAlpha'),
 ('E_Chain_OldAlpha','HIST_FullCVChain'),('D_Chain_OldAlpha','HIST_FullCVChain'),('CV_Chain_OldAlpha','HIST_FullCVChain'))
METRICS=('logloss','Brier','predicted_rate','observed_rate','bias','macro_abs_bias')

def binary_scores(y,lp,ln):
    y=np.asarray(y);lp=np.asarray(lp,np.float64);ln=np.asarray(ln,np.float64)
    if y.ndim!=1 or y.shape!=lp.shape or y.shape!=ln.shape or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('No dropped/unknown labels')
    if not np.isfinite(lp).all() or not np.isfinite(ln).all() or np.max(np.abs(np.logaddexp(lp,ln)))>1e-12:raise ValueError('Invalid stored log pair; no clipping')
    p=np.exp(lp)
    return dict(logloss=float(-np.where(y==1,lp,ln).mean()),Brier=float(np.square(p-y).mean()),
                predicted_rate=float(p.mean()),observed_rate=float(y.mean()),bias=float((p-y).mean()))

def mean_sd(values):
    x=np.asarray(values,np.float64)
    if x.shape!=(5,) or not np.isfinite(x).all():raise ValueError('Exactly5pairedseeds')
    return float(x.mean()),float(x.std(ddof=1))

def evaluate(arms,lp,ln,y,match_ids):
    """All6matches in each5seed. Pooled rows are not the primary macro."""
    arms=tuple(arms);lp=np.asarray(lp);ln=np.asarray(ln);mids=np.asarray(match_ids).astype(str)
    if len(set(arms))!=len(arms) or lp.shape!=(5,len(arms),len(y)) or ln.shape!=lp.shape or mids.shape!=(len(y),):raise ValueError('Fixed readout axes')
    matches=sorted(set(mids.tolist()))
    if len(matches)!=6:raise ValueError('All6knownmatches must be retained')
    game=[];seedrows=[];lookup={}
    for a,arm in enumerate(arms):
        for s in range(5):
            rows=[]
            for mid in matches:
                mask=mids==mid;r=binary_scores(np.asarray(y)[mask],lp[s,a,mask],ln[s,a,mask]);r['macro_abs_bias']=abs(r['bias'])
                row=dict(arm=arm,seed=s,match_id=mid,rows=int(mask.sum()),**r);game.append(row);rows.append(row)
            macro={k:float(np.mean([r[k] for r in rows])) for k in METRICS}
            row=dict(arm=arm,seed=s,rows=len(y),matches=6,aggregation='equal6matchmacro',**macro);seedrows.append(row);lookup[(arm,s)]=row
    summary=[]
    for arm in arms:
        r=dict(arm=arm,seeds=5,rows=len(y),matches=6,SD_unit='paired_pipeline_seed_sampleSD_not_matchCI')
        for key in METRICS:r[key+'_mean'],r[key+'_seed_SD']=mean_sd([lookup[(arm,s)][key] for s in range(5)])
        summary.append(r)
    paired_seed=[];paired_summary=[]
    for candidate,reference in PAIRS:
        if candidate not in arms or reference not in arms:raise ValueError('Every predeclared comparison must exist')
        for metric in ('logloss','Brier','macro_abs_bias'):
            vals=[]
            for s in range(5):
                d=lookup[(candidate,s)][metric]-lookup[(reference,s)][metric];vals.append(d)
                paired_seed.append(dict(candidate=candidate,reference=reference,metric=metric,seed=s,difference=d,rows=len(y),matches=6))
            mu,sd=mean_sd(vals)
            paired_summary.append(dict(candidate=candidate,reference=reference,metric=metric,mean_difference=mu,paired_seed_SD=sd,
                improved_seeds=int(np.sum(np.asarray(vals)<0)),ties=int(np.sum(np.asarray(vals)==0)),seeds=5,rows=len(y),matches=6,
                primary=bool((candidate,reference)==PAIRS[0] and metric=='logloss')))
    return dict(match_metrics=game,seed_macro=seedrows,arm_summary=summary,paired_seed=paired_seed,paired_summary=paired_summary)
