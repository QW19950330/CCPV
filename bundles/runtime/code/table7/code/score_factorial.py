"""Held-match scoring only after every fixed risk readout has been frozen."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
import argparse,json
import numpy as np
from risk_io import P,read,put,sha,selected,guard,table,utc,process_identity
from risk_data import dataset

ARMS=tuple(f'{f}_{p}_{o}' for f in ('D','CV') for p in ('Direct','Marginal') for o in ('NoEarlyBCE','WithEarlyBCE'))
METRICS=('logloss','Brier','predicted_rate','recorded_rate','signed_bias','absolute_bias')

def metrics(logp,logn,y):
    lp,ln,y=np.asarray(logp,float),np.asarray(logn,float),np.asarray(y,int)
    assert lp.shape==ln.shape==y.shape and lp.ndim==1 and len(y)>0
    assert np.isin(y,[0,1]).all() and np.isfinite(lp).all() and np.isfinite(ln).all()
    assert np.max(np.abs(np.logaddexp(lp,ln)))<=1e-12
    p=np.exp(lp);pred=float(p.mean());obs=float(y.mean())
    return dict(logloss=float(np.mean(-np.where(y==1,lp,ln))),Brier=float(np.mean((p-y)**2)),
        predicted_rate=pred,recorded_rate=obs,signed_bias=pred-obs,absolute_bias=abs(pred-obs),
        windows=len(y),positive_windows=int(y.sum()))
def paired_summary(values):
    v=np.asarray(values,np.float64);assert v.shape==(5,) and np.isfinite(v).all()
    return dict(paired_seeds=5,delta_mean=float(v.mean()),paired_delta_SD=float(v.std(ddof=1)),
        negative_seed_count=int((v<0).sum()),five_seed_deltas_json=json.dumps(v.tolist()),
        no_pvalue=True,not_independent_games=True)
def complete_gate(expected):
    guard(expected,'score');freeze=read(P/'readout_freeze.json')
    assert freeze['status']=='ALL240_RAW_OUTER_READOUTS_FROZEN_BEFORE_ANY_HELD_LABEL_SCORING'
    assert freeze['implementation_lock_sha256']==expected and freeze['seeds']==5 and freeze['eligible_rows']==5344 and tuple(freeze['arms'])==ARMS
    for r in freeze['files']:
        path=(P/r['path']).resolve();assert path.is_relative_to(P) and sha(path)==r['sha256']
    assert sha(P/'final_weights_lock.json')==freeze['weights_lock_sha256']
    weights=read(P/'final_weights_lock.json');assert len(weights['records'])==240 and len(weights['folds'])==6
    assert len(weights['historical_auxiliary_exact_replays'])==len(weights['exact_objective_pair_initialization'])==120
    coverage=np.zeros(13372,int)
    for fold in weights['folds']:
        assert fold['held_match'] not in fold['training_matches'] and len(fold['training_matches'])==5
        train,test=np.array(fold['training_rows'],int),np.array(fold['held_rows'],int)
        assert not np.intersect1d(train,test).size;coverage[test]+=1
    assert (coverage==1).sum()==5344 and coverage.max()==1
    for r in weights['records']:
        assert sha(P/r['path'])==r['sha256'] and sha(P/r['history'])==r['history_sha256'] and r['updates']==200 and r['parameters']==602
    pin=freeze['frozen_readouts'];assert sha(P/pin['path'])==pin['sha256']
    return freeze,weights
def score(expected):
    freeze,weights=complete_gate(expected)
    assert not (P/'evaluation').exists()
    a,bundles,supported,ix,matches,binding=dataset()
    keys=('arms','initial_indices','anchor_id','match_id','fixed_team_A','prefix_supported','fold_assignment','q_logp','q_log1mp','early_logp','early_log1mp')
    pred=selected(P/freeze['frozen_readouts']['path'],keys)
    assert tuple(pred['arms'])==ARMS and np.array_equal(pred['initial_indices'],ix)
    assert np.array_equal(pred['anchor_id'],a['anchor_id'][ix]) and np.array_equal(pred['match_id'],a['match_id'][ix])
    assert np.array_equal(pred['fixed_team_A'],a['fixed_team_A'][ix]) and np.array_equal(pred['prefix_supported'],supported[ix])
    assert supported[ix].all()
    for key in ('q_logp','q_log1mp','early_logp','early_log1mp'):assert pred[key].shape==(5,8,5344) and np.isfinite(pred[key]).all()
    for fold in weights['folds']:
        mask=np.isin(ix,fold['held_rows']);assert np.all(pred['fold_assignment'][mask]==fold['fold'])
        assert set(pred['match_id'][mask].astype(str))=={fold['held_match']}
    # This held-target consumption occurs only after complete_gate has verified
    # all 240 models, fold coverage and every readout freeze/hash.
    y=selected(binding['assets']['prefix_data']['path'],('y_10','y_early2','y_late8'))
    targets={key:np.asarray(value[ix],int) for key,value in y.items()}
    assert np.array_equal(targets['y_10'],np.maximum(targets['y_early2'],targets['y_late8']))
    assert all(np.isin(v,[0,1]).all() for v in targets.values())
    put(P/'logs/score_process_identity.json',process_identity('score',expected))
    per_match=[]
    for s in range(5):
        for ai,arm in enumerate(ARMS):
            for game in matches:
                use=pred['match_id'].astype(str)==game
                for target,yk,pk,nk in [('q10','y_10','q_logp','q_log1mp'),('Early2','y_early2','early_logp','early_log1mp')]:
                    role='TOTAL_RISK_PRIMARY_TARGET' if target=='q10' else ('SUPERVISED_EARLY_DIAGNOSTIC' if arm.endswith('WithEarlyBCE') else 'UNSUPERVISED_DIAGNOSTIC_NOT_VALID_COMPONENT')
                    per_match.append(dict(seed=s,arm=arm,match_id=game,target=target,target_semantics=role,
                        **metrics(pred[pk][s,ai,use],pred[nk][s,ai,use],targets[yk][use])))
    by_seed=[]
    for s in range(5):
        for arm in ARMS:
            for target in ('q10','Early2'):
                rr=[r for r in per_match if (r['seed'],r['arm'],r['target'])==(s,arm,target)];assert len(rr)==6
                by_seed.append(dict(seed=s,arm=arm,target=target,target_semantics=rr[0]['target_semantics'],
                    held_matches=6,windows=sum(r['windows'] for r in rr),positive_windows=sum(r['positive_windows'] for r in rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
    summary=[];lookup={(r['seed'],r['arm'],r['target']):r for r in by_seed}
    for arm in ARMS:
        for target in ('q10','Early2'):
            rr=[lookup[s,arm,target] for s in range(5)]
            out=dict(arm=arm,target=target,target_semantics=rr[0]['target_semantics'],seeds=5,held_matches=6)
            for k in METRICS:
                v=np.array([r[k] for r in rr]);out[k+'_mean']=float(v.mean());out[k+'_seed_SD']=float(v.std(ddof=1))
            summary.append(out)
    pairs=[];plan=read(P/'experiment_plan.json')
    for definition in plan['fixed_contrasts']:
        kind,candidate,reference,frontend=[definition[k] for k in ('kind','candidate','reference','frontend')]
        for target in ('q10','Early2'):
            for metric in ('logloss','Brier','absolute_bias'):
                if kind=='PAIR':
                    delta=[lookup[s,candidate,target][metric]-lookup[s,reference,target][metric] for s in range(5)]
                else:
                    assert kind=='DIFFERENCE_IN_DIFFERENCES' and frontend in ('D','CV')
                    delta=[(lookup[s,f'{frontend}_Marginal_NoEarlyBCE',target][metric]-lookup[s,f'{frontend}_Marginal_WithEarlyBCE',target][metric])
                        -(lookup[s,f'{frontend}_Direct_NoEarlyBCE',target][metric]-lookup[s,f'{frontend}_Direct_WithEarlyBCE',target][metric]) for s in range(5)]
                primary=(kind,candidate,reference,target,metric)==('PAIR','D_Marginal_NoEarlyBCE','D_Marginal_WithEarlyBCE','q10','logloss')
                pairs.append(dict(kind=kind,candidate=candidate,reference=reference,frontend=frontend,target=target,metric=metric,
                    role='PRIMARY_FIXED_MECHANISM_DESCRIPTOR' if primary else 'FIXED_DESCRIPTIVE_NO_SELECTION',
                    **paired_summary(delta)))
    assert len(per_match)==480 and len(by_seed)==80 and len(summary)==16 and len(pairs)==84
    counts=[]
    for game in matches:
        use=pred['match_id'].astype(str)==game
        counts.append(dict(match_id=game,eligible_windows=int(use.sum()),positive_q10=int(targets['y_10'][use].sum()),positive_Early2=int(targets['y_early2'][use].sum()),
            all8arms_all5seed_retained=True,new_independent_fixture=False))
    for name,rr in [('by_match.csv',per_match),('by_seed.csv',by_seed),('summary.csv',summary),('paired_comparisons.csv',pairs),('coverage.csv',counts)]:table(P/'evaluation'/name,rr)
    primary=next(r for r in pairs if r['role']=='PRIMARY_FIXED_MECHANISM_DESCRIPTOR')
    put(P/'evaluation/results.json',dict(status='COMPLETED_FIXED_RISK_OBJECTIVE_FACTORIAL_EXPOSED_DEVELOPMENT_ONLY',utc=utc(),
        implementation_lock_sha256=expected,weights_lock_sha256=sha(P/'final_weights_lock.json'),readout_freeze_sha256=sha(P/'readout_freeze.json'),
        old_auxiliary_exact_replay_models=120,new_NoEarlyBCE_models=120,all_trained_models=240,total_updates=48000,
        original_all_rows=13372,eligible_rows=5344,seeds=5,previously_exposed_matches=6,arms=list(ARMS),
        primary=primary,coverage=counts,summary=summary,
        NoEarlyBCE_still_has_training_Early_prior=True,NoEarly2_outputs_not_valid_partition_value=True,
        no_new_external_evaluation=True,no50=True,no_WC2018=True,no_new_trajectory_model_forward_or_training=True,
        statement='Local algorithmic objective contrast; no new blind validation, realGoal, or football-causal benefit claim'))
    files=[dict(path=p.relative_to(P).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((P/'evaluation').iterdir()) if p.is_file()]
    put(P/'evaluation/manifest.json',dict(status='ALL_FIXED_HELD_TARGET_METRICS_FROZEN',utc=utc(),files=files,implementation_lock_sha256=expected))
    print('ALL240_MODELS_FIXED_FACTORIAL_SCORING_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--implementation-lock-sha256',required=True);a=p.parse_args();score(a.implementation_lock_sha256)
