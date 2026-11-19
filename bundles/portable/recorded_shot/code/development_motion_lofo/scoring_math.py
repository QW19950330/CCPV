"""Verbatim original motion-study scoring and aggregation definitions."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ARMS=('G_shared','GM200_shared','GM_OLS1s')

CONDITIONS=('native','grid')

METRICS=('logloss','brier','absolute_bias','mean_probability','observed_rate','signed_bias')

def table(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(y,p):
    q=np.clip(p,1e-15,1-1e-15)
    return dict(logloss=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean()),brier=float(np.mean((p-y)**2)),absolute_bias=float(abs(p.mean()-y.mean())),mean_probability=float(p.mean()),observed_rate=float(y.mean()),signed_bias=float(p.mean()-y.mean()))

def score_predictions(scope, predictions, out):
    groups=defaultdict(list)
    for r in predictions:groups[r['held'],r['condition'],r['arm'],r['seed']].append(r)
    fm=[]
    for (held,condition,arm,seed),rr in groups.items():
        y=np.array([r['y'] for r in rr]);p=np.array([r['p'] for r in rr])
        fm.append(dict(held=held,condition=condition,arm=arm,seed=seed,n=len(rr),positives=int(y.sum()),**metrics(y,p)))
    table(out/'fixture_metrics.csv',fm)
    summary=[];paired=[];interaction=[];fixture_contrasts=[]
    contrasts=[('GM_OLS1s','GM200_shared'),('GM200_shared','G_shared'),('GM_OLS1s','G_shared'),('G_shared','PREVALENCE'),('GM200_shared','PREVALENCE'),('GM_OLS1s','PREVALENCE')]
    index={(r['held'],r['condition'],r['arm'],r['seed']):r for r in fm}
    for seed in range(5):
        for metric in METRICS:
            means={}
            for condition in CONDITIONS:
                for arm in (*ARMS,'PREVALENCE'):
                    vals=[index[(h,condition,arm,-1 if arm=='PREVALENCE' else seed)][metric] for h in scope];value=float(np.mean(vals));means[condition,arm]=value
                    summary.append(dict(seed=seed,condition=condition,arm=arm,metric=metric,equal_fixture_mean=value,prevalence_repeated_for_paired_display_only=arm=='PREVALENCE'))
                for aa,bb in contrasts:
                    paired.append(dict(seed=seed,condition=condition,contrast=aa+'-'+bb,metric=metric,delta=means[condition,aa]-means[condition,bb]))
                    for h in scope:fixture_contrasts.append(dict(held=h,seed=seed,condition=condition,contrast=aa+'-'+bb,metric=metric,delta=index[h,condition,aa,seed][metric]-index[h,condition,bb,-1 if bb=='PREVALENCE' else seed][metric]))
            d_ols=means['grid','GM_OLS1s']-means['native','GM_OLS1s'];d_200=means['grid','GM200_shared']-means['native','GM200_shared']
            interaction.append(dict(seed=seed,metric=metric,OLS_grid_minus_native=d_ols,backward200_grid_minus_native=d_200,degradation_interaction=d_ols-d_200))
    table(out/'seed_summary.csv',summary);table(out/'paired_contrasts.csv',paired);table(out/'fixture_contrasts.csv',fixture_contrasts);table(out/'degradation_interaction.csv',interaction)
    return fm, summary, paired, fixture_contrasts, interaction
