"""One new q-first candidate versus fixed first-stage references, all held rows."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
import argparse,json,csv
from pathlib import Path
import numpy as np
from qfirst_io import P,read,put,sha,selected,guard,table,utc,process_identity
from qfirst_data import dataset
NEW=('D_QFirstJoint','CV_QFirstJoint')
OLD=tuple(f'{f}_{p}_{o}' for f in ('D','CV') for p in ('Direct','Marginal') for o in ('NoEarlyBCE','WithEarlyBCE'))
ARMS=OLD+NEW
METRICS=('logloss','Brier','predicted_rate','recorded_rate','signed_bias','absolute_bias')
def metrics(lp,ln,y):
    lp,ln,y=np.asarray(lp,float),np.asarray(ln,float),np.asarray(y,int)
    assert lp.shape==ln.shape==y.shape and lp.ndim==1 and len(y)>0 and np.isin(y,[0,1]).all()
    assert np.isfinite(lp).all() and np.isfinite(ln).all() and np.max(np.abs(np.logaddexp(lp,ln)))<=1e-12
    prob=np.exp(lp);pm=float(prob.mean());om=float(y.mean())
    return dict(logloss=float(np.mean(-np.where(y==1,lp,ln))),Brier=float(np.mean((prob-y)**2)),predicted_rate=pm,recorded_rate=om,
        signed_bias=pm-om,absolute_bias=abs(pm-om),windows=len(y),positive_windows=int(y.sum()))
def paired_summary(v):
    v=np.asarray(v,float);assert v.shape==(5,) and np.isfinite(v).all()
    return dict(paired_seeds=5,delta_mean=float(v.mean()),paired_delta_SD=float(v.std(ddof=1)),negative_seed_count=int((v<0).sum()),five_seed_deltas_json=json.dumps(v.tolist()),no_pvalue=True,not_independent_games=True)
def complete_gate(expected):
    guard(expected,'score');g=read(P/'readout_freeze.json')
    assert g['status']=='ALL60_RAW_OUTER_READOUTS_FROZEN_BEFORE_ANY_HELD_LABEL_SCORING'
    assert g['implementation_lock_sha256']==expected and g['seeds']==5 and g['eligible_rows']==5344 and tuple(g['arms'])==NEW
    for r in g['files']:
        p=(P/r['path']).resolve();assert p.is_relative_to(P) and sha(p)==r['sha256']
    weights=read(P/'final_weights_lock.json');assert len(weights['records'])==60 and len(weights['folds'])==6
    assert sha(P/'final_weights_lock.json')==g['weights_lock_sha256']
    init=read(P/'initialization_freeze.json');assert len(init['records'])==60
    for r in init['records']:assert sha(P/r['path'])==r['sha256']
    cover=np.zeros(13372,int)
    for fold in weights['folds']:
        assert fold['held_match'] not in fold['training_matches'] and len(fold['training_matches'])==5
        train,test=np.array(fold['training_rows']),np.array(fold['held_rows']);assert not np.intersect1d(train,test).size;cover[test]+=1
    assert (cover==1).sum()==5344 and cover.max()==1
    for r in weights['records']:assert sha(P/r['path'])==r['sha256'] and sha(P/r['history'])==r['history_sha256'] and r['updates']==200
    return g,weights
def score(expected):
    freeze,weights=complete_gate(expected);assert not (P/'evaluation').exists()
    a,bundles,support,ix,matches,binding=dataset();ref=binding['reference_factorial']
    for k in ('weights_lock','readout_freeze','frozen_readouts','evaluation_manifest'):assert sha(ref[k]['path'])==ref[k]['sha256']
    parent=Path(ref['study_path']);oldman=read(ref['evaluation_manifest']['path'])
    for r in oldman['files']:assert sha(parent/r['path'])==r['sha256']
    ids=('arms','initial_indices','anchor_id','match_id','fixed_team_A','prefix_supported','fold_assignment')
    logs=('q_logp','q_log1mp','early_logp','early_log1mp')
    new=selected(P/freeze['frozen_readouts']['path'],ids+logs+('r_logp','r_log1mp','late_only_logp','late_logp','late_log1mp','category_logs'))
    old=selected(ref['frozen_readouts']['path'],ids+logs)
    assert tuple(new['arms'])==NEW and tuple(old['arms'])==OLD
    for k in ids[1:]:assert np.array_equal(new[k],old[k]),k
    assert np.array_equal(new['initial_indices'],ix) and np.array_equal(new['anchor_id'],a['anchor_id'][ix]) and np.array_equal(new['match_id'],a['match_id'][ix])
    assert new['prefix_supported'].all()
    pred={k:np.concatenate((old[k],new[k]),axis=1) for k in logs}
    for v in pred.values():assert v.shape==(5,10,5344) and np.isfinite(v).all()
    cat=new['category_logs'];assert cat.shape==(5,2,5344,3) and np.isfinite(cat).all()
    coherence=dict(category_normalization=float(np.max(np.abs(np.logaddexp.reduce(cat,axis=-1)))),
        Early_leq_q_max_log_violation=float(np.max(new['early_logp']-new['q_logp'])),
        risk_reconstruction=float(np.max(np.abs(np.logaddexp(new['early_logp'],new['late_only_logp'])-new['q_logp']))),
        noEarly_normalization=float(np.max(np.abs(np.logaddexp(new['early_logp'],new['early_log1mp'])))),
        latecond_normalization=float(np.max(np.abs(np.logaddexp(new['late_logp'],new['late_log1mp'])))),
        chain_q_reconstruction=float(np.max(np.abs(np.logaddexp(new['early_logp'],new['early_log1mp']+new['late_logp'])-new['q_logp']))))
    assert all(v<=1e-12 for v in coherence.values())
    # All prediction/hash/coherence checks precede actual held-target reads.
    raw=selected(binding['assets']['prefix_data']['path'],('y_10','y_early2','y_late8'))
    targets={k:np.asarray(v[ix],int) for k,v in raw.items()};q,e,late=targets['y_10'],targets['y_early2'],targets['y_late8']
    assert np.array_equal(q,np.maximum(e,late)) and all(np.isin(v,[0,1]).all() for v in targets.values())
    category=np.where(q==0,0,np.where(e==1,1,2));assert np.array_equal(category==1,e==1)
    put(P/'logs/score_process_identity.json',process_identity('score',expected))
    per_match=[];categorical=[]
    for seed in range(5):
        for ai,arm in enumerate(ARMS):
            for game in matches:
                use=new['match_id'].astype(str)==game
                for target,yy,pk,nk in [('q10',q,'q_logp','q_log1mp'),('Early2',e,'early_logp','early_log1mp')]:
                    role='TOTAL_RISK_PRIMARY_TARGET' if target=='q10' else ('UNSUPERVISED_DIAGNOSTIC_NOT_VALID_COMPONENT' if arm.endswith('NoEarlyBCE') else 'SUPERVISED_EARLY_DIAGNOSTIC')
                    per_match.append(dict(seed=seed,arm=arm,match_id=game,target=target,target_semantics=role,
                        provenance='NEW_SINGLE_CANDIDATE' if arm in NEW else 'REUSED_FROZEN_FIRST_STAGE_REFERENCE',
                        **metrics(pred[pk][seed,ai,use],pred[nk][seed,ai,use],yy[use])))
        for ai,arm in enumerate(NEW):
            for game in matches:
                use=new['match_id'].astype(str)==game;idx=np.flatnonzero(use);pos=use&(q==1)
                joint=float(-cat[seed,ai,idx,category[use]].mean())
                qll=metrics(new['q_logp'][seed,ai,use],new['q_log1mp'][seed,ai,use],q[use])['logloss']
                term=-np.where(e[pos]==1,new['r_logp'][seed,ai,pos],new['r_log1mp'][seed,ai,pos])
                contribution=float(term.sum()/len(idx));residual=joint-qll-contribution;assert abs(residual)<=1e-12
                categorical.append(dict(seed=seed,arm=arm,match_id=game,windows=len(idx),positive_q10_windows=int(pos.sum()),
                    joint_category_NLL=joint,q10_logloss=qll,original_mass_partition_NLL=contribution,
                    conditional_r_logloss=float(term.mean()) if pos.any() else None,
                    conditional_status='DEFINED' if pos.any() else 'UNDEFINED_NO_POSITIVE_Q10',identity_residual=residual,
                    descriptor_not_primary_endpoint=True))
    # Numerical recomputation cannot turn previous-stage references into new evidence.
    with (parent/'evaluation/by_match.csv').open(encoding='utf-8-sig',newline='') as f:literal=list(csv.DictReader(f))
    lookup_ref={(int(r['seed']),r['arm'],r['match_id'],r['target']):r for r in literal}
    max_reference_error=0.
    for r in per_match:
        if r['arm'] in OLD:
            oldr=lookup_ref[r['seed'],r['arm'],r['match_id'],r['target']]
            for k in METRICS:
                max_reference_error=max(max_reference_error,abs(r[k]-float(oldr[k])));assert r[k]==float(oldr[k])
                r[k]=float(oldr[k])
    by_seed=[]
    for seed in range(5):
        for arm in ARMS:
            for target in ('q10','Early2'):
                rr=[r for r in per_match if (r['seed'],r['arm'],r['target'])==(seed,arm,target)];assert len(rr)==6
                by_seed.append(dict(seed=seed,arm=arm,target=target,target_semantics=rr[0]['target_semantics'],provenance=rr[0]['provenance'],held_matches=6,
                    windows=sum(r['windows'] for r in rr),positive_windows=sum(r['positive_windows'] for r in rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
    lookup={(r['seed'],r['arm'],r['target']):r for r in by_seed};summary=[]
    for arm in ARMS:
        for target in ('q10','Early2'):
            rr=[lookup[s,arm,target] for s in range(5)];out=dict(arm=arm,target=target,target_semantics=rr[0]['target_semantics'],provenance=rr[0]['provenance'],seeds=5,held_matches=6)
            for k in METRICS:
                v=np.array([r[k] for r in rr]);out[k+'_mean']=float(v.mean());out[k+'_seed_SD']=float(v.std(ddof=1))
            summary.append(out)
    pairs=[];plan=read(P/'experiment_plan.json')
    for d in plan['fixed_contrasts']:
        for target in ('q10','Early2'):
            for metric in ('logloss','Brier','absolute_bias'):
                values=[lookup[s,d['candidate'],target][metric]-lookup[s,d['reference'],target][metric] for s in range(5)]
                primary=(d['candidate'],d['reference'],target,metric)==('D_QFirstJoint','D_Direct_WithEarlyBCE','q10','logloss')
                pairs.append(dict(**d,target=target,metric=metric,role='PRIMARY_FIXED_SINGLE_CANDIDATE_DESCRIPTOR' if primary else 'FIXED_DESCRIPTIVE_NO_SELECTION',**paired_summary(values)))
    assert len(per_match)==600 and len(categorical)==60 and len(by_seed)==100 and len(summary)==20 and len(pairs)==54
    for name,rr in [('by_match.csv',per_match),('by_seed.csv',by_seed),('summary.csv',summary),('paired_comparisons.csv',pairs),('categorical_metrics.csv',categorical)]:table(P/'evaluation'/name,rr)
    put(P/'evaluation/results.json',dict(status='COMPLETED_SINGLE_QFIRST_JOINT_CANDIDATE_EXPOSED_DEVELOPMENT_ONLY',utc=utc(),
        implementation_lock_sha256=expected,weights_lock_sha256=sha(P/'final_weights_lock.json'),readout_freeze_sha256=sha(P/'readout_freeze.json'),
        new_models=60,new_updates=12000,reused_reference_models=240,new_candidate_arms=list(NEW),reference_arms=list(OLD),
        eligible_rows=5344,previously_exposed_matches=6,seeds=5,primary=next(r for r in pairs if r['role']=='PRIMARY_FIXED_SINGLE_CANDIDATE_DESCRIPTOR'),
        coherence_errors=coherence,max_reused_reference_metric_error=max_reference_error,
        no_inference_target_gate=True,full_joint_likelihood_no_positive_mass_normalization=True,
        no_new_external_evaluation=True,no50=True,noWC2018=True,no_new_frontend_training_or_forward=True,
        role='One new post-first-stage-negative hypothesis; sameexposeddevelopmentdata notblindconfirmation;notrealGoalorfootballcausalpolicyvalue'))
    files=[dict(path=p.relative_to(P).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((P/'evaluation').iterdir()) if p.is_file()]
    put(P/'evaluation/manifest.json',dict(status='ALL_FIXED_QFIRST_CANDIDATE_AND_REFERENCE_SCORES_FROZEN',utc=utc(),files=files,implementation_lock_sha256=expected))
    print('ALL60_QFIRST_JOINT_CANDIDATE_SCORING_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--implementation-lock-sha256',required=True);a=p.parse_args();score(a.implementation_lock_sha256)
