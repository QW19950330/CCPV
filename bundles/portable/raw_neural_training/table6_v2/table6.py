"""Compute current Table 6 from freshly emitted kinematic tables."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import pandas as pd

def compute(results):
    pooled=pd.read_csv(results/'metric_pooled.csv')
    joint=pd.read_csv(results/'model_own_anchor_joint.csv')
    base=(pooled.coordinates=='native')&(pooled.fixed_group=='other_known_history_players')&(pooled.derivative=='head')
    original=pooled[base&(pooled.world=='RECORDED')]
    natural=pooled[base&(pooled.policy=='input_reset')&(pooled['mode']=='sample')&(pooled.path=='n_empty')]
    rows=[];display=[]
    for label,col in [('AB output complete (%)','complete_required_draws'),('Controlled actors strict pass (%)','strict_pass_draws')]:
        view=joint[(joint.policy=='runner_output')&(joint['mode']=='sample')&(joint.path=='AB_shared')&
            (joint.coordinates=='native')&(joint.fixed_group=='all_two_t0_valid_targets')]
        cells={}
        for world in ('D','E'):
            values=[]
            for seed in range(5):
                frame=view[(view.world==world)&(view.seed==seed)]
                if len(frame)!=24 or int(frame.planned_draws.sum())!=768:raise ValueError('24x32 AB denominator differs')
                values.append(float(frame[col].sum()/768*100))
            cells[world]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),seeds=values)
        rows.append(dict(diagnostic=label,recorded=None,recorded_count=None,denominator=768,**cells))
        display.append([label,'768 draws per seed, AB','Not an observed action',*[f"{cells[w]['mean']:.2f} ± {cells[w]['sd']:.2f}" for w in ('D','E')]])
    for metric,field,label,digits in [('acceleration','strict_rate','Background head acceleration exceedance (%)',3),
                                    ('turn','strict_rate','Background head turn exceedance (%)',3),
                                    ('speed','mean','Background head speed mean (m/s)',5),
                                    ('speed','p99','Background head speed p99 (m/s)',5)]:
        ref=original[original.metric==metric]
        if len(ref)!=1:raise ValueError('Unique recorded pooled reference required')
        r=ref.iloc[0];scale=100 if field=='strict_rate' else 1
        observed=float(r['factual_'+field])*scale;n=int(r.factual_defined_units)
        count=int(r.factual_strict_violations) if field=='strict_rate' else None
        cells={}
        for world in ('D','E'):
            rr=natural[(natural.world==world)&(natural.metric==metric)].sort_values('seed')
            if rr.seed.tolist()!=list(range(5)):raise ValueError('All five fixed seeds required')
            values=(rr['model_'+field].to_numpy()*scale).tolist()
            cells[world]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),seeds=values)
        rows.append(dict(diagnostic=label,recorded=observed,recorded_count=count,denominator=n,**cells))
        denom=f'Natural, {n:,} defined factual agent-steps' if field!='p99' else 'Per-seed pooled quantile, then seed mean'
        observed_text=f'{count}/{n:,} = {observed:.4f}' if count is not None else f'{observed:.5f}'
        display.append([label,denom,observed_text,*[f"{cells[w]['mean']:.{digits}f} ± {cells[w]['sd']:.{digits}f}" for w in ('D','E')]])
    return rows,display

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    a.output=a.output.resolve();a.results=a.results.resolve(strict=True)
    if a.output.exists():raise FileExistsError('Preserve existing output')
    if not a.output.parent.is_dir():raise FileNotFoundError('Output parent must exist')
    for root in (Path(__file__).resolve().parent,a.results):
        if a.output==root or root in a.output.parents or a.output in root.parents:raise ValueError('Output overlaps code/input tree')
    receipt=json.loads((a.results/'receipt.json').read_text())
    if receipt['status']!='EXACT_SAVED_WORLD_REFERENCE':raise ValueError('Full calculation receipt required')
    rows,display=compute(a.results)
    a.output.mkdir()
    with (a.output/'table6_numeric.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2)
    with (a.output/'table6.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['Diagnostic and unit','Fixed denominator / path','Recorded reference','D','E']);w.writerows(display)
    print(json.dumps(display,ensure_ascii=False))
if __name__=='__main__':main()
