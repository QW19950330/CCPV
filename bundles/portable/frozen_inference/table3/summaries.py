"""Original match-macro/seed aggregation prefixes; no contrast selection or new analysis."""
import itertools
import numpy as np
import pandas as pd

KEYS=['condition','horizon_seconds','group']
METRICS=['ade_m','fde_m','velocity_rmse_m_s','consistency_mean_m','consistency_max_m',
    'consistency_gt_1e4_fraction','head_speed_mean','head_speed_exceedance_fraction',
    'geometric_speed_mean','geometric_speed_exceedance_fraction','head_acceleration_mean',
    'head_acceleration_exceedance_fraction','geometric_acceleration_mean','geometric_acceleration_exceedance_fraction']
DENOMINATORS=['observed_points','terminal_agents','velocity_targets','consistency_actor_steps']
ERROR_METRICS=['ade_m','fde_m','velocity_rmse_m_s']

def strict_mean(value):
    a=np.asarray(value,dtype=float)
    return float(a.mean()) if a.size and np.isfinite(a).all() else np.nan

def expected_cells(study):
    f=study['forecast']
    return {(arm,int(seed)) for arm in [f['candidate']]+f['learned_controls'] for seed in f['seeds']}|{('cv',-1)}

def development_summary(raw):
    match_keys=['arm','seed','match_id']+KEYS
    grouped=raw.groupby(match_keys,dropna=False)
    bymatch=grouped[METRICS].mean().reset_index()
    support=grouped[['observed_points','terminal_agents','velocity_targets','consistency_actor_steps']].sum().reset_index()
    bymatch=bymatch.merge(support,on=match_keys,validate='one_to_one')
    byseed=bymatch.groupby(['arm','seed']+KEYS,dropna=False)[METRICS].mean().reset_index()
    summary=byseed.groupby(['arm']+KEYS,dropna=False)[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
    summary['deterministic_control']=summary.arm.isin(['cv','smooth_cv'])
    for metric in METRICS:summary.loc[summary.deterministic_control,metric+'_std']=0.
    return bymatch,byseed,summary

def external_summary(raw,study,analysis):
    """Pure summary for saved rows or synthetic fixtures; no filesystem access."""
    f=study['forecast'];matches=study['exposure']['external_matches'];seeds=f['seeds']
    arms=[f['candidate']]+f['learned_controls']+f['analytic_controls']
    metric_names=[c for c in raw.columns if c not in ['arm','seed','source_index','match_id']+KEYS+DENOMINATORS]
    metrics=[m for m in metric_names if pd.api.types.is_numeric_dtype(raw[m])]
    for m in ERROR_METRICS+['consistency_max_m']:
        if m not in metrics:raise ValueError('Missing required metric '+m)
    keys=['arm','seed','match_id']+KEYS
    if raw.duplicated(['arm','seed','source_index']+KEYS).any():raise AssertionError('Duplicated window readout')
    if set(raw.match_id.astype(str))!=set(matches):raise ValueError('All three expected matches must remain in raw data')
    if set(map(tuple,raw[['arm','seed']].drop_duplicates().to_numpy()))!=expected_cells(study):raise ValueError('Expected all15 frozen models plus one CV')
    groups=raw.groupby(keys,dropna=False)
    bymatch=groups[metrics].mean().reset_index()
    support=groups[DENOMINATORS].sum().reset_index()
    counts=groups.size().rename('retained_windows').reset_index()
    supported=groups[ERROR_METRICS].count().reset_index().rename(columns={m:m+'_supported_windows' for m in ERROR_METRICS})
    bymatch=bymatch.merge(support,on=keys,validate='one_to_one').merge(counts,on=keys,validate='one_to_one').merge(supported,on=keys,validate='one_to_one')
    expected=set()
    for arm,seed in expected_cells(study):
        for match,condition,horizon,group in itertools.product(matches,f['conditions'],f['reporting_horizons_s'],['players','hidden_players','visible_players','ball']):
            expected.add((arm,seed,match,condition,float(horizon),group))
    actual=set(map(tuple,bymatch[keys].to_numpy()))
    if actual!=expected:raise ValueError('Missing match/group/horizon row; never discard unsupported groups')
    byseed_rows=[]
    for key,g in bymatch.groupby(['arm','seed']+KEYS,dropna=False):
        if set(g.match_id)!=set(matches):raise AssertionError('A match disappeared before aggregation')
        row=dict(zip(['arm','seed']+KEYS,key));row['expected_matches']=len(matches)
        for metric in metrics:
            row[metric]=strict_mean(g[metric]);row[metric+'_supported_matches']=int(g[metric].notna().sum())
        byseed_rows.append(row)
    byseed=pd.DataFrame(byseed_rows);summary=[]
    for key,g in byseed.groupby(['arm']+KEYS,dropna=False):
        row=dict(zip(['arm']+KEYS,key));deterministic=row['arm']=='cv'
        row.update(deterministic_control=deterministic,expected_seeds=1 if deterministic else len(seeds))
        if set(g.seed)!=({-1} if deterministic else set(seeds)):raise AssertionError('Missing frozen seed')
        for metric in metrics:
            values=g[metric].to_numpy(float);all_supported=np.isfinite(values).all()
            row[metric+'_mean']=float(values.mean()) if all_supported else np.nan
            row[metric+'_std']=float(values.std(ddof=1)) if all_supported and not deterministic else (0. if all_supported else np.nan)
            row[metric+'_supported_seeds']=int(np.isfinite(values).sum())
        summary.append(row)
    summary=pd.DataFrame(summary)
    return bymatch,byseed,summary
