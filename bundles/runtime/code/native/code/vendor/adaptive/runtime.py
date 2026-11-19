"""Four separately authorized NumPy stages. No Torch/world/head fitting."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    if os.environ.get(_k) not in (None,'1'):raise RuntimeError('CPU1thread environment required')
    os.environ[_k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse,sys,time,traceback
import numpy as np
from io_core import P,S,B,J,sha,check,rec,put,save,table,load,guard,closed,boundary,identity,now,PauseBetweenShards
import literal_math as M
from score_math import ARMS,REFERENCES,evaluate
from closure_validation import validate_unit,validate_records,validate_predecessor,validate_existing_units

KINDS=('EarlyD','LateD','DirectUnion')
SCALARS=('EarlyD','LateD','DirectQ','DirectEarly')

def getbinding(c):
    for key in ('source_receipt','source_binding','protocol'):check(c[key])
    b=c['binding']
    for r in b['sources']:check(r)
    if sha(P/'literal_math.py')!=c['math_sha256']:raise ValueError('Literalmath changed')
    return b
def relrecord(b,r):return dict(path=str(Path(b['old_value_root'])/r['path']),sha256=r['sha256'])
def maprows(rows,globalrows):
    if len(set(map(int,globalrows)))!=len(globalrows):raise ValueError('Duplicate source-global index')
    lookup={int(v):j for j,v in enumerate(globalrows)}
    if any(int(r) not in lookup for r in rows):raise ValueError('Unknown source-global index')
    return np.array([lookup[int(r)] for r in rows],np.int64)
def features(c,seed):
    b=getbinding(c);r=next(r for r in b['Efeatures'] if r['seed']==seed)
    a=load(r['array'],r['allowed_members'])
    if not np.array_equal(a['initial_indices'],c['selected_rows']) or a['Early26'].shape!=(5344,26) or a['Direct36'].shape!=(5344,36) or a['X2'].shape!=(5344,10):raise ValueError('Fixed5344 E dictionary')
    if len(set(a['anchor_id'].tolist()))!=5344:raise ValueError('Unique original anchors')
    for f in c['folds']:
        jj=maprows(f['held_rows'],a['initial_indices'])
        if not np.all(a['match_id'][jj]==f['held_match']) or not np.all(a['fold_assignment'][jj]==f['fold']):raise ValueError('SourceID/fold mapping')
    return a
def headpredict(cp,x,kind):
    d,k={'EarlyD':(26,1),'LateD':(10,1),'DirectUnion':(36,2)}[kind]
    if np.asarray(x).shape[1]!=d or int(cp['raw_dimension'])!=d or cp['hidden_weight'].shape!=(8,2*d) or cp['output_weight'].shape!=(k,8):raise ValueError('Exactfrozen433/177/602head')
    for key in ('hidden_weight','hidden_bias','output_weight','output_bias','scaler_mean','scaler_scale'):
        if not np.isfinite(cp[key]).all():raise ValueError('Nonfinitefixedweight/scaler')
    if not np.all(cp['scaler_scale']>0):raise ValueError('Positiveoldscale')
    return M.predict_logits(cp,x)
def assign_oof(destination,counts,positions,values):
    positions=np.asarray(positions,np.int64)
    if len(set(positions.tolist()))!=len(positions) or np.any(counts[positions]!=0):raise ValueError('Duplicate OOFassignment')
    if destination[positions].shape!=values.shape:raise ValueError('OOFoutputshape')
    destination[positions]=values;counts[positions]+=1
def assert_oof(destination,counts):
    if not np.all(counts==1) or not np.isfinite(destination).all():raise ValueError('Everytrainingrow musthaveonefiniteOOFassignment')
def start_stage(name):
    if (P/name).exists():raise FileExistsError('Alreadyfrozen stage,no rerun: '+name)
def finish_files(records,root):
    return [r for x in records for r in x['files']]+[rec(root/f'f{x["fold"]}_s{x["seed"]}'/'manifest.json') for x in records]

def oof_logits(expected,c):
    start_stage('oof_freeze.json');b=getbinding(c);validate_existing_units('oof_logits',c,expected,P);start=time.monotonic();done_all=[]
    for seed in range(5):
        a=features(c,seed)
        for f in c['folds']:
            boundary(start,c);fi=f['fold'];folder=P/'oof_shards'/f'f{fi}_s{seed}';done=closed(folder/'manifest.json',expected)
            if done is not None:
                validate_unit('oof_logits',done,c,expected,P,requested=(fi,seed));done_all.append(done);continue
            if folder.exists():raise RuntimeError('PartialOOFshard preserved,nooverwrite')
            folder.mkdir(parents=True,exist_ok=False)
            train=np.asarray(f['train_rows'],np.int64);loc=maprows(train,a['initial_indices']);mids=a['match_id'][loc].astype(str)
            if set(mids)!=set(f['training_matches']) or f['held_match'] in mids:raise ValueError('Outertrain5population')
            out={};headcalls=0;rowcalls=0
            for kind in KINDS:
                outputs=2 if kind=='DirectUnion' else 1
                z=np.full((len(train),outputs),np.nan);counts=np.zeros(len(train),np.int8)
                rr=[r for r in b['old_inner_heads'] if (r['fold'],r['seed'],r['head'])==(fi,seed,kind)]
                if len(rr)!=5:raise ValueError('Five oldinnerheads')
                for r in sorted(rr,key=lambda r:r['inner']):
                    inner=r['inner_held'];heldmask=mids==inner;heldglobal=train[heldmask];fitglobal=train[~heldmask]
                    if f['held_match'] in r['training_matches'] or inner in r['training_matches'] or set(r['training_matches'])!=set(f['training_matches'])-{inner}:raise ValueError('Doubleexclusion metadata')
                    cp=load(relrecord(b,r),b['old_inner_head_member_allowlist'])
                    if (str(cp['head']),int(cp['training_seed']),str(cp['outer_held_match']))!=(kind,seed,f['held_match']):raise ValueError('InnerCPidentity')
                    if not np.array_equal(cp['training_initial_rows'],fitglobal) or not np.array_equal(cp['held_rows'],heldglobal) or set(cp['training_match_ids'].astype(str))!=set(r['training_matches']):raise ValueError('ActualCP doubleexclusion/source-row metadata')
                    field={'EarlyD':'Early26','LateD':'X2','DirectUnion':'Direct36'}[kind]
                    pred=headpredict(cp,a[field][loc[heldmask]],kind)
                    assign_oof(z,counts,np.flatnonzero(heldmask),pred);headcalls+=1;rowcalls+=int(heldmask.sum())
                assert_oof(z,counts);out[kind]=z;out[kind+'_assignment_count']=counts
            if headcalls!=15 or rowcalls!=3*len(train):raise AssertionError('OOF15headcalls/count')
            save(folder/'oof.npz',**out,initial_global_rows=train,match_id=mids,anchor_id=a['anchor_id'][loc])
            done=dict(status='CLOSED_LABEL_FREE_E_OOF_LOGITS',implementation_lock_sha256=expected,fold=fi,seed=seed,outer_held=f['held_match'],train_rows=len(train),head_matrix_calls=headcalls,head_row_evaluations=rowcalls,array=rec(folder/'oof.npz'),authorization_snapshot=c['_authority'],files=[rec(folder/'oof.npz'),c['_authority']])
            put(folder/'manifest.json',done);validate_unit('oof_logits',done,c,expected,P,requested=(fi,seed));done_all.append(done);print('OOF_CLOSED',fi,seed,flush=True);boundary(start,c)
    validate_records('oof_logits',done_all,c,expected,P)
    for r in b['Efeatures']:check(r['array'])
    for r in b['old_inner_heads']:check(relrecord(b,r))
    put(P/'oof_freeze.json',dict(status='ALL_E_OOF_LOGITS_FROZEN_NO_LABELS',implementation_lock_sha256=expected,utc=now(),records=done_all,files=finish_files(done_all,P/'oof_shards'),head_matrix_calls=450,head_row_evaluations=400800,real_label_members_decoded=0,new_NN_or_MLP_updates=0,authorization_snapshot=c['_authority'],elapsed_s=time.monotonic()-start))

def capsule_targets(early,late,direct,train,match_ids,training_matches):
    train=np.asarray(train,np.int64)
    for a in (early,late,direct):
        if not np.array_equal(a['oof_initial_rows'],train):raise ValueError('TraincapsuleouterS mismatch')
    w=M.equal_match_weights(match_ids,training_matches)
    for a in (early,late,direct):
        if not np.array_equal(a['original_initial_weights'],w):raise ValueError('Originalequalmatchweight mismatch')
    ye=np.asarray(early['training_labels'],np.float64);yq=np.asarray(direct['training_labels'],np.float64);yl=np.asarray(late['training_labels'],np.float64)
    if ye.shape!=train.shape or yq.shape!=train.shape or not np.isin(ye,[0,1]).all() or not np.isin(yq,[0,1]).all() or np.any(ye>yq):raise ValueError('KnowntrainY2subsetY10')
    risk,wr,mass=M.restrict_risk(w,ye)
    if not np.array_equal(early['training_loss_rows'],train) or not np.array_equal(direct['training_loss_rows'],train) or not np.array_equal(late['training_loss_rows'],train[risk]):raise ValueError('LateR2capsule mapping')
    if yl.shape!=(int(risk.sum()),) or not np.isin(yl,[0,1]).all() or not np.array_equal(yl,yq[risk]):raise ValueError('KnownR2Late==Y10 sourceunion')
    return dict(Y2=ye,Y10=yq,Late=yl,w=w,risk=risk,wr=wr,riskmass=mass)

def calibration_closure(records):
    expected={(f,s) for f in range(6) for s in range(5)}
    if len(records)!=30 or {(r['fold'],r['seed']) for r in records}!=expected:raise AssertionError('30uniqueouterseed calibrationunits')
    if any(r['logical_scalar_intercepts']!=4 or r['bisections']!=400 for r in records):raise AssertionError('Fourliteral100-step roots perunit')
    scalars=sum(r['logical_scalar_intercepts'] for r in records);iterations=sum(r['bisections'] for r in records)
    if scalars!=120 or iterations!=12000:raise AssertionError('120scalar12000iterations')
    return dict(scalar_intercepts=scalars,bisections=iterations,closed_units=30)

def calibrate(expected,c):
    start_stage('calibration_freeze.json');b=getbinding(c);oof=validate_predecessor('oof_logits',J(P/'oof_freeze.json'),c,expected,P);validate_existing_units('calibrate',c,expected,P);oofmap={(r['fold'],r['seed']):r for r in oof['records']};start=time.monotonic();done_all=[]
    for f in c['folds']:
        fi=f['fold'];caps=None
        for seed in range(5):
            boundary(start,c);folder=P/'calibration_shards'/f'f{fi}_s{seed}';done=closed(folder/'manifest.json',expected)
            if done is not None:
                validate_unit('calibrate',done,c,expected,P,requested=(fi,seed));done_all.append(done);continue
            if folder.exists():raise RuntimeError('Partialalphashard preserved,nooverwrite')
            folder.mkdir(parents=True,exist_ok=False)
            if caps is None:
                caps={}
                for kind in KINDS:
                    r=next(r for r in b['train_label_capsules'] if r['fold']==fi and r['head']==kind)
                    a=load(relrecord(b,r),r['allowed_members'])
                    if (str(a['head']),int(a['seed']),int(a['output_index']),str(a['outer_held_match']))!=(kind,0,0,f['held_match']):raise ValueError('Canonicalseed0capsule identity')
                    caps[kind]=a
            z=load(oofmap[(fi,seed)]['array'],('EarlyD','LateD','DirectUnion','initial_global_rows','match_id','anchor_id'))
            t=capsule_targets(caps['EarlyD'],caps['LateD'],caps['DirectUnion'],z['initial_global_rows'],z['match_id'],f['training_matches'])
            if not np.array_equal(z['initial_global_rows'],f['train_rows']):raise ValueError('OOFsource trainrows')
            specs={'EarlyD':(z['EarlyD'][:,0],t['Y2'],t['w']),
                   'LateD':(z['LateD'][t['risk'],0],t['Late'],t['wr']),
                   'DirectQ':(z['DirectUnion'][:,0],t['Y10'],t['w']),
                   'DirectEarly':(z['DirectUnion'][:,1],t['Y2'],t['w'])}
            arrays={};hist={};diagnostics={}
            for name,(logits,y,w) in specs.items():
                cp,history=M.fit_intercept(logits,y,w)
                if int(cp['iterations'])!=100:raise AssertionError('Literal100bisections')
                arrays.update({name+'__'+key:val for key,val in cp.items()});hist[name]=history
                diagnostics[name]=dict(alpha=float(cp['alpha']),target_rate=float(cp['target_rate']),residual=float(cp['residual']),hessian=float(cp['hessian']),iterations=100,rows=len(y))
            save(folder/'intercepts.npz',**arrays);put(folder/'histories.json',hist)
            done=dict(status='CLOSED_FOUR_E_OOF_SCALAR_INTERCEPTS',implementation_lock_sha256=expected,fold=fi,seed=seed,outer_held=f['held_match'],train_rows=len(z['initial_global_rows']),risk_rows=int(t['risk'].sum()),logical_scalar_intercepts=4,bisections=400,riskmass=t['riskmass'],diagnostics=diagnostics,array=rec(folder/'intercepts.npz'),files=[rec(folder/'intercepts.npz'),rec(folder/'histories.json'),c['_authority']],authorization_snapshot=c['_authority'])
            put(folder/'manifest.json',done);validate_unit('calibrate',done,c,expected,P,requested=(fi,seed));done_all.append(done);print('ALPHA_CLOSED',fi,seed,flush=True);boundary(start,c)
    closure=calibration_closure(done_all)
    validate_records('calibrate',done_all,c,expected,P)
    for r in b['train_label_capsules']:check(relrecord(b,r))
    put(P/'calibration_freeze.json',dict(status='ALL120_E_OOF_INTERCEPTS_FROZEN_TRAIN_LABELS_ONLY',implementation_lock_sha256=expected,utc=now(),records=done_all,files=finish_files(done_all,P/'calibration_shards'),**closure,train_capsules=18,held_label_cache_decoded=False,head_or_NN_updates=0,authorization_snapshot=c['_authority'],elapsed_s=time.monotonic()-start))

def checkpair(lp,ln):
    if not np.isfinite(lp).all() or not np.isfinite(ln).all() or np.max(np.abs(np.logaddexp(lp,ln)))>1e-12:raise ValueError('Storedfinite logpair')
def shifted_predictions(raw,alpha):
    """raw is four finite logpair tuples; no trueY2 gate or head call."""
    z={name:lp-ln for name,(lp,ln) in raw.items()}
    ep,en=M.logpair(z['EarlyD']+alpha['EarlyD']);lp,ln=M.logpair(z['LateD']+alpha['LateD']);qp,qn,cat=M.compose(ep,en,lp,ln)
    dp,dn=M.logpair(z['DirectQ']+alpha['DirectQ']);de,dne=M.logpair(z['DirectEarly']+alpha['DirectEarly'])
    return dict(Chain=dict(q_logp=qp,q_log1mp=qn,early_logp=ep,early_log1mp=en,late_logp=lp,late_log1mp=ln,category_logs=cat),Direct=dict(q_logp=dp,q_log1mp=dn,early_logp=de,early_log1mp=dne))
def old_alphas(b,fold,seed,held):
    out={}
    for name,kind,index in (('EarlyD','EarlyD',0),('LateD','LateD',0),('DirectQ','DirectUnion',0),('DirectEarly','DirectUnion',1)):
        rs=[r for r in b['old_alpha'] if (r['fold'],r['seed'],r['head'],r['output_index'])==(fold,seed,kind,index)]
        if len(rs)!=1:raise ValueError('Uniqueoldalpha route')
        a=load(relrecord(b,rs[0]),b['old_alpha_readout_members'])
        if (str(a['head']),int(a['seed']),int(a['output_index']),str(a['outer_held_match']))!=(kind,seed,index,held):raise ValueError('Oldalphaheld identity')
        out[name]=float(a['alpha'])
    return out

def readout(expected,c):
    start_stage('readout_freeze.json');b=getbinding(c);calibration=validate_predecessor('calibrate',J(P/'calibration_freeze.json'),c,expected,P);validate_existing_units('readout',c,expected,P);q=load(b['current_frozen_readouts'],b['current_frozen_readouts']['allowed_members']);ix=q['initial_indices'];N=len(ix)
    if not np.array_equal(ix,c['selected_rows']) or N!=5344:raise ValueError('Original5344q rows')
    qi={str(name):i for i,name in enumerate(q['arms'])};ci={str(name):i for i,name in enumerate(q['chain_arms'])}
    required=('E_Chain_Raw','E_Chain_OldAlpha','E_Direct_Raw','E_Direct_OldAlpha','D_Chain_OldAlpha','D_Direct_OldAlpha','CV_Chain_OldAlpha','CV_Direct_OldAlpha')
    if not all(name in qi for name in required) or not all(name in ci for name in required[:2]):raise ValueError('Currentraw/oldalpha arm schema')
    am={(r['fold'],r['seed']):r for r in calibration['records']};start=time.monotonic();done_all=[]
    for f in c['folds']:
        loc=maprows(f['held_rows'],ix);fi=f['fold']
        if not np.all(q['match_id'][loc]==f['held_match']) or not np.all(q['fold_assignment'][loc]==fi):raise ValueError('Fixedouter routing')
        for seed in range(5):
            boundary(start,c);folder=P/'readout_shards'/f'f{fi}_s{seed}';done=closed(folder/'manifest.json',expected)
            if done is not None:
                validate_unit('readout',done,c,expected,P,requested=(fi,seed));done_all.append(done);continue
            if folder.exists():raise RuntimeError('Partialadaptiveq preserved,nooverwrite')
            folder.mkdir(parents=True,exist_ok=False)
            er=qi['E_Chain_Raw'];dr=qi['E_Direct_Raw'];cr=ci['E_Chain_Raw']
            raw=dict(EarlyD=(q['early_logp'][seed,er,loc],q['early_log1mp'][seed,er,loc]),LateD=(q['late_logp'][seed,cr,loc],q['late_log1mp'][seed,cr,loc]),DirectQ=(q['q_logp'][seed,dr,loc],q['q_log1mp'][seed,dr,loc]),DirectEarly=(q['early_logp'][seed,dr,loc],q['early_log1mp'][seed,dr,loc]))
            for lp,ln in raw.values():checkpair(lp,ln)
            oldpred=shifted_predictions(raw,old_alphas(b,fi,seed,f['held_match']));err=0.
            for name,current in (('Chain','E_Chain_OldAlpha'),('Direct','E_Direct_OldAlpha')):
                for key in ('q_logp','q_log1mp','early_logp','early_log1mp'):
                    err=max(err,float(np.max(np.abs(oldpred[name][key]-q[key][seed,qi[current],loc]))))
                if name=='Chain':
                    for key in ('late_logp','late_log1mp'):err=max(err,float(np.max(np.abs(oldpred[name][key]-q[key][seed,ci[current],loc]))))
            if err>1e-14:raise AssertionError('Predeclaredoldalpha reconstruction1e-14 failed;noloosening')
            alpha=load(am[(fi,seed)]['array'],tuple(name+'__alpha' for name in SCALARS));newpred=shifted_predictions(raw,{name:float(alpha[name+'__alpha']) for name in SCALARS})
            logs={key:np.empty((6,len(loc))) for key in ('q_logp','q_log1mp','early_logp','early_log1mp')};late={key:np.empty((3,len(loc))) for key in ('late_logp','late_log1mp')};cats=np.empty((3,len(loc),3))
            for kind,base in (('Chain',0),('Direct',3)):
                for variant,current in ((0,'E_'+kind+'_Raw'),(1,'E_'+kind+'_OldAlpha')):
                    for key in logs:logs[key][base+variant]=q[key][seed,qi[current],loc]
                    if kind=='Chain':
                        for key in late:late[key][variant]=q[key][seed,ci[current],loc]
                        _,_,cats[variant]=M.compose(logs['early_logp'][variant],logs['early_log1mp'][variant],late['late_logp'][variant],late['late_log1mp'][variant])
                for key in logs:logs[key][base+2]=newpred[kind][key]
                if kind=='Chain':
                    for key in late:late[key][2]=newpred[kind][key]
                    cats[2]=newpred[kind]['category_logs']
            save(folder/'q.npz',**logs,**late,category_logs=cats,initial_positions=loc)
            done=dict(status='CLOSED_IDENTITY_OLDD_ADAPTIVE_E_LOGPAIRS',implementation_lock_sha256=expected,fold=fi,seed=seed,outer_held=f['held_match'],rows=len(loc),head_matrix_calls=0,scalar_fits=0,oldalpha_interop_maxlog_error=err,array=rec(folder/'q.npz'),files=[rec(folder/'q.npz'),c['_authority']],authorization_snapshot=c['_authority'])
            put(folder/'manifest.json',done);validate_unit('readout',done,c,expected,P,requested=(fi,seed));done_all.append(done);print('ADAPTIVE_Q_CLOSED',fi,seed,flush=True);boundary(start,c)
    validate_records('readout',done_all,c,expected,P)
    arrays={key:np.full((5,6,N),np.nan) for key in ('q_logp','q_log1mp','early_logp','early_log1mp')};late={key:np.full((5,3,N),np.nan) for key in ('late_logp','late_log1mp')};cats=np.full((5,3,N,3),np.nan);count=np.zeros((5,N),np.int8)
    for r in done_all:
        a=load(r['array'],tuple(arrays)+tuple(late)+('category_logs','initial_positions'));seed=r['seed'];loc=a['initial_positions'];count[seed,loc]+=1
        for key in arrays:arrays[key][seed,:,loc]=a[key].T
        for key in late:late[key][seed,:,loc]=a[key].T
        cats[seed,:,loc,:]=a['category_logs'].transpose(1,0,2)
    if len(done_all)!=30 or not np.all(count==1) or not all(np.isfinite(a).all() for a in list(arrays.values())+list(late.values())+[cats]):raise ValueError('Adaptiveq all30shards/fullcoverage')
    for key in ('q','early'):checkpair(arrays[key+'_logp'],arrays[key+'_log1mp'])
    metadata={k:q[k] for k in ('initial_indices','anchor_id','match_id','fixed_team_A','fold_assignment')}
    save(P/'frozen_readouts.npz',arms=np.asarray(ARMS),**arrays,**late,category_logs=cats,**metadata)
    hist=load(c['historical_controls'],('arms','q_logp','q_log1mp','initial_indices','anchor_id','match_id'))
    for key in ('initial_indices','anchor_id','match_id'):
        if not np.array_equal(hist[key],q[key]):raise ValueError('FullCVliteral cohort')
    hi=list(hist['arms']).index('HIST_FullCVChain');names=('D_Chain_OldAlpha','D_Direct_OldAlpha','CV_Chain_OldAlpha','CV_Direct_OldAlpha')
    rp=np.stack([q['q_logp'][:,qi[name]] for name in names]+[hist['q_logp'][:,hi]],axis=1);rn=np.stack([q['q_log1mp'][:,qi[name]] for name in names]+[hist['q_log1mp'][:,hi]],axis=1);checkpair(rp,rn)
    save(P/'literal_references.npz',arms=np.asarray(REFERENCES),q_logp=rp,q_log1mp=rn,**metadata)
    check(b['current_frozen_readouts']);check(c['historical_controls'])
    put(P/'readout_freeze.json',dict(status='ALL_E_ADAPTIVE_READOUTS_FROZEN_BEFORE_HELD_LABELS',implementation_lock_sha256=expected,utc=now(),records=done_all,files=finish_files(done_all,P/'readout_shards')+[rec(P/'frozen_readouts.npz'),rec(P/'literal_references.npz')],arms=list(ARMS),rows=5344,seeds=5,oldalpha_interop_maxlog_error=max(r['oldalpha_interop_maxlog_error'] for r in done_all),held_labels_decoded=False,head_or_NN_calls=0,authorization_snapshot=c['_authority']))

def score(expected,c):
    start_stage('evaluation_freeze.json');getbinding(c);validate_predecessor('readout',J(P/'readout_freeze.json'),c,expected,P);folder=P/'evaluation'
    if folder.exists():raise RuntimeError('Partialscore preserved,nooverwrite')
    keys=('arms','q_logp','q_log1mp','initial_indices','anchor_id','match_id','fold_assignment');a=load(rec(P/'frozen_readouts.npz'),keys);r=load(rec(P/'literal_references.npz'),keys)
    if tuple(a['arms'])!=ARMS or tuple(r['arms'])!=REFERENCES:raise ValueError('Allfixed11arms')
    for k in ('initial_indices','anchor_id','match_id','fold_assignment'):
        if not np.array_equal(a[k],r[k]):raise ValueError('Referencecohort mismatch')
    y=load(c['score_only_outcomes'],('match_id','anchor_id','eligible_recorded10','y_early2','y_late8','y_10'));ix=a['initial_indices']
    if not np.array_equal(ix,c['selected_rows']) or not np.array_equal(ix,np.flatnonzero(y['eligible_recorded10'])) or not np.array_equal(y['anchor_id'][ix],a['anchor_id']) or not np.array_equal(y['match_id'][ix],a['match_id']):raise ValueError('Fixedheld5344source labels')
    if not all(np.isin(y[k][ix],[0,1]).all() for k in ('y_early2','y_late8','y_10')) or not np.array_equal(y['y_10'][ix],np.maximum(y['y_early2'][ix],y['y_late8'][ix])):raise ValueError('Unknownnot0/source2+8labels')
    result=evaluate(ARMS+REFERENCES,np.concatenate((a['q_logp'],r['q_logp']),axis=1),np.concatenate((a['q_log1mp'],r['q_log1mp']),axis=1),y['y_10'][ix],a['match_id']);folder.mkdir(exist_ok=False)
    for name,rows in result.items():table(folder/(name+'.csv'),rows)
    check(c['score_only_outcomes'])
    put(P/'evaluation_freeze.json',dict(status='ALL11_ARMS_HELD_SHOT10_SCORED_NO_MODEL_FIT',implementation_lock_sha256=expected,utc=now(),files=[rec(folder/(name+'.csv')) for name in result]+[c['_authority']],row_counts={k:len(v) for k,v in result.items()},rows=5344,matches=6,seeds=5,primary='E_Chain_EOOFAlpha minus E_Chain_OldDAlpha q10LL',new_NN_MLP_or_scalar_fits=0,authorization_snapshot=c['_authority'],claims='Exposeddevelopment,notindependent/Eoptimalhead/Goal/policybenefit'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('oof_logits','calibrate','readout','score'),required=True);p.add_argument('--lock',required=True);p.add_argument('--identity-path',required=True);p.add_argument('--authorization-snapshot',required=True);p.add_argument('--authorization-sha',required=True);args=p.parse_args()
    a,c=guard(args.stage,args.lock,args.authorization_snapshot,args.authorization_sha);authority=rec(args.authorization_snapshot);c['_authority']=authority;identity(args.identity_path,args.stage,args.lock,a,authority)
    try:
        {'oof_logits':oof_logits,'calibrate':calibrate,'readout':readout,'score':score}[args.stage](args.lock,c)
        guard(args.stage,args.lock,args.authorization_snapshot,args.authorization_sha)
    except BaseException as exc:
        put(P/'execution'/args.stage/a['attempt_id']/f'error_{time.time_ns()}.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc(),preserve_all_partials=True))
        if isinstance(exc,PauseBetweenShards):return 75
        raise
    return 0
if __name__=='__main__':raise SystemExit(main())
