"""Original held-score and macro aggregation block, with create-only CSV output."""
import csv
import numpy as np

def table(p, rows):
    with p.open("x", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def score_predictions(scope, held_predictions, out):
    metrics=[]
    for held in scope:
        for arm,seeds in [('PREVALENCE',[-1]),('G',range(5)),('GM',range(5))]:
            for seed in seeds:
                rr=[r for r in held_predictions if r['held']==held and r['arm']==arm and r['seed']==seed]
                if not rr:metrics.append(dict(held=held,arm=arm,seed=seed,n=0,positives=0,logloss=None,brier=None,absolute_bias=None,mean_probability=None,observed_rate=None,signed_bias=None));continue
                y=np.array([r['y'] for r in rr]);p=np.array([r['p'] for r in rr]);q=np.clip(p,1e-15,1-1e-15)
                metrics.append(dict(held=held,arm=arm,seed=seed,n=len(rr),positives=int(y.sum()),logloss=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean()),brier=float(((p-y)**2).mean()),absolute_bias=float(abs(p.mean()-y.mean())),mean_probability=float(p.mean()),observed_rate=float(y.mean()),signed_bias=float(p.mean()-y.mean())))
    table(out/'fixture_metrics.csv',metrics);summary=[];contrasts=[]
    for seed in range(5):
        for metric in ('logloss','brier','absolute_bias','mean_probability','observed_rate','signed_bias'):
            for arm in ('G','GM','PREVALENCE'):
                rr=[r for r in metrics if r['arm']==arm and r['seed']==(-1 if arm=='PREVALENCE' else seed)]
                value=float(np.mean([r[metric] for r in rr])) if len(rr)==6 and all(r[metric] is not None for r in rr) else None
                summary.append(dict(seed=seed,arm=arm,metric=metric,equal_fixture_mean=value,prevalence_repeated_for_paired_display_only=arm=='PREVALENCE'))
            vals={r['arm']:r['equal_fixture_mean'] for r in summary if r['seed']==seed and r['metric']==metric}
            for a,b in [('GM','G'),('GM','PREVALENCE'),('G','PREVALENCE')]:contrasts.append(dict(seed=seed,contrast=a+'-'+b,metric=metric,delta=vals[a]-vals[b] if vals[a] is not None and vals[b] is not None else None))
    table(out/'seed_summary.csv',summary);table(out/'paired_contrasts.csv',contrasts)
    return metrics, summary, contrasts
