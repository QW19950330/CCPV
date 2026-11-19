"""Three separately authorized payload stages. Import performs no real I/O.

Actual Torch/model import is inside trajectory_features after the root guard.
Readout and score are NumPy-only and cannot fit or change any coefficient.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    if os.environ.get(_key) not in (None,'1'):raise RuntimeError('Require explicitCPU1thread environment: '+_key)
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse,sys,time,traceback
import numpy as np
from io_core import P,S,BASE,J,sha,check,rec,put,save,table,load_members,guard,closed,boundary,identity,now,PauseBetweenShards
sys.path.insert(0,str(S/'implementation'))
from native_features import HistoryBatch,analytic_cv_native,features_from_native_paths,union_features
from world_rollout import rollout_D_native
from stage_contracts import HISTORY_MEMBERS,PREFIX_MEMBERS,HEAD_MEMBERS,ALPHA_MEMBERS,SCORE_MEMBERS
from readout_contract import WORLDS,READOUTS,ARMS,prospective_readouts,held_only_route
from score_math import HISTORICAL,evaluate

def binding(c):
    for key in ('source_receipt','original_binding','cohort','plan'):check(c[key])
    for r in c['model_source_files']:check(r)
    return c['bound_inputs'],J(c['cohort']['path'])

def causal_inputs(c):
    b,cohort=binding(c)
    z=load_members(b['history'],tuple(HISTORY_MEMBERS.values()))
    batch=HistoryBatch(**{k:z[v] for k,v in HISTORY_MEMBERS.items()}).validate()
    p=load_members(b['prefix'],PREFIX_MEMBERS)
    selected=np.asarray(cohort['selected_rows'],np.int64)
    if batch.states.shape!=(13372,10,23,4) or selected.shape!=(5344,):raise ValueError('Original13372/5344 population')
    if not np.array_equal(batch.anchor_id,p['anchor_id']) or p['X0'].shape!=(13372,10) or p['Z_cv'].shape!=(13372,18):raise ValueError('Original native/prefix alignment')
    if not np.array_equal(p['X0'],p['Z_cv'][:,:10],equal_nan=True):raise ValueError('FixedX0/CV context')
    if not np.array_equal(selected,np.flatnonzero(p['eligible_prefix']&p['eligible_recorded10'])):raise ValueError('No population reselection')
    if not batch.supported[selected].all():raise ValueError('Expected all5344prefix support; never silently discard')
    route=held_only_route(cohort['folds'],selected,p['match_id'][selected])
    # Source native conventions are fixed metadata, not a fitted normalization.
    physical=J(b['physical_review']['path']);check(b['physical_review'])
    expected=dict(native_position_unit='m',native_velocity_unit='m/s',native_origin='pitch_center',native_x_axis='longitudinal',native_y_axis='lateral',model_pitch=[105.,68.],attack_direction_flip=False)
    for k,v in expected.items():
        if physical.get(k)!=v:raise ValueError('Native physical contract: '+k)
    for r in physical['evidence']:check(r)
    return batch,p,selected,route

def already_frozen(name):
    if (P/name).exists():raise FileExistsError('Stage already frozen; never rerun '+name)

def output_metadata(batch,p,ix,route):
    return dict(initial_indices=ix,anchor_id=batch.anchor_id[ix],match_id=p['match_id'][ix],fixed_team_A=p['fixed_team_A'][ix],fold_assignment=route)

def load_model(c,reference,torch):
    check(reference['checkpoint'])
    cp=torch.load(reference['checkpoint']['path'],map_location='cpu',weights_only=False)
    if (cp['cell'],cp['seed'],cp['epoch'],cp['canonical_model'])!=(reference['cell'],reference['seed'],4,'coupled2D'):raise ValueError('Trusted checkpoint identity')
    model_path=Path(c['model_source_dir']).resolve();sys.path.insert(0,str(model_path))
    import api
    if Path(api.__file__).resolve()!=model_path/'api.py':raise ValueError('Exact frozen model API')
    api.configure('cpu')
    def forbidden(*args,**kwargs):raise PermissionError('Generic data/model loader forbidden')
    api.load_development=api.load_validation=api.load_model=forbidden
    model=api.make_model('D_coupled_ar',reference['seed']);model.load_state_dict(cp['state_dict'],strict=True);model.eval()
    for par in model.parameters():
        par.requires_grad_(False)
        if par.device.type!='cpu' or par.dtype!=torch.float32:raise ValueError('CPUF32 only')
    # optimizer/RNG/history have been deserialized as part of the trusted Torch
    # container but are not restored, inspected, used or copied to new outputs.
    del cp
    return api,model

def trajectory_features(expected,c):
    already_frozen('trajectory_freeze.json');batch,p,ix,route=causal_inputs(c);b=c['bound_inputs'];start=time.monotonic()
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if torch.__version__!='2.7.0+cu126':raise RuntimeError('PinnedTorch2.7.0+cu126 required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=[];actual_current=0
    worldrefs={(r['cell'],r['seed']):r for r in b['worlds']}
    if set(worldrefs)!={(w,s) for w in ('D','E') for s in range(5)}:raise ValueError('10frozenD/E checkpoints')
    with torch.inference_mode(),sdpa_kernel([SDPBackend.MATH]):
        for world,seed in [('CV',-1)]+[(w,s) for w in ('D','E') for s in range(5)]:
            boundary(start);folder=P/'trajectories'/f'{world}_s{seed}';done=closed(folder/'manifest.json',expected)
            if done is not None:
                if (done['world'],done['seed'])!=(world,seed):raise ValueError('Closedworld identity')
                records.append(done);continue
            if folder.exists():raise RuntimeError('Partial trajectory shard preserved; root review, no overwrite')
            folder.mkdir(parents=True,exist_ok=False)
            states=np.empty((5344,20,23,4),np.float32);support=np.empty((5344,20,23),bool);ages=np.empty((5344,20,23),np.float32)
            x2=np.full((5344,10),np.nan);z=np.full((5344,18),np.nan);model=None;api=None;counts={'calls':0};before=None
            if world!='CV':
                api,model=load_model(c,worldrefs[(world,seed)],torch);before=api.state_hash(model.state_dict())
                def hook(module,args,out):counts['calls']+=1
                handle=model.register_forward_hook(hook)
            for lo in range(0,5344,128):
                rows=ix[lo:lo+128]
                if world=='CV':
                    path,usable=analytic_cv_native(batch,rows)
                    age=batch.observation_age_s[rows,None]+np.arange(1,21,dtype=np.float32)[None,:,None]*np.float32(.1)
                else:path,usable,age,_=rollout_D_native(batch,rows,api,model,authorized=True,device='cpu')
                if not np.isfinite(path[usable]).all():raise FloatingPointError('Nonfinite supported natural state; not a score filter')
                states[lo:lo+len(rows)]=path;support[lo:lo+len(rows)]=usable;ages[lo:lo+len(rows)]=age
                xx,zz=features_from_native_paths(batch,rows,path,usable,p['X0']);x2[lo:lo+len(rows)]=xx;z[lo:lo+len(rows)]=zz
            if model is not None:
                handle.remove()
                if counts['calls']!=840 or before!=api.state_hash(model.state_dict()):raise AssertionError('840calls/frozenworld invariants')
                check(worldrefs[(world,seed)]['checkpoint']);actual_current+=counts['calls'];del model
            early,direct=union_features(p['X0'][ix],p['Z_cv'][ix],z,x2)
            metadata=output_metadata(batch,p,ix,route)
            save(folder/'states_features.npz',native_states=states,support=support,observation_age_s=ages,X2=x2,Z=z,Early26=early,Direct36=direct,**metadata)
            manifest=dict(status='COMPLETE_NATIVE_NATURAL_WORLD_SHARD',implementation_lock_sha256=expected,world=world,seed=seed,rows=5344,
                NNbatch_calls=counts['calls'],observations_generated_are_real=False,weight_state_before=before,weight_state_after=before,
                array=rec(folder/'states_features.npz'),files=[rec(folder/'states_features.npz'),c['_execution_authorization']],actual_labels_or_q_read=False,authorization_snapshot=c['_execution_authorization'])
            put(folder/'manifest.json',manifest);records.append(manifest);print('WORLD_SHARD_CLOSED',world,seed,counts['calls'],flush=True)
            boundary(start)
    if len(records)!=11 or sum(r['NNbatch_calls'] for r in records)!=8400:raise AssertionError('11shards8400total')
    ledger=P/'planned_ledger.csv'
    if ledger.exists():raise RuntimeError('Unfrozen ledger output preserved; root review rather than silent reuse')
    scheduled=set(ix.tolist())
    table(ledger,(dict(global_index=i,anchor_id=str(batch.anchor_id[i]),match_id=str(p['match_id'][i]),status='SCHEDULED' if i in scheduled else 'NOT_SCHEDULED',reason='ORIGINAL_FROZEN_5344_OUTCOME_ELIGIBILITY' if i in scheduled else 'OUTSIDE_ORIGINAL_FIXED_OUTCOME_ELIGIBILITY') for i in range(13372)))
    # Re-verify causal containers and source binding after all generation.
    for r in (b['history'],b['prefix'],b['physical_review']):check(r)
    for r in b['worlds']:check(r['checkpoint'])
    for r in c['model_source_files']:check(r)
    put(P/'trajectory_freeze.json',dict(status='ALL11_TRAJECTORIES_AND_FEATURES_FROZEN_NO_Q_OR_LABELS',implementation_lock_sha256=expected,utc=now(),
        records=records,files=[rec(ledger)]+[f for r in records for f in r['files']]+[rec(P/'trajectories'/f'{r["world"]}_s{r["seed"]}'/'manifest.json') for r in records],ledger_rows=13372,scheduled_rows=5344,NNbatch_calls_total=8400,NNbatch_calls_this_attempt=actual_current,
        new_fit_or_noise=0,actual_labels_or_q_read=False,authorization_snapshot=c['_execution_authorization'],runtime=dict(numpy=np.__version__,torch=torch.__version__,device='cpu',threads=torch.get_num_threads(),SDPA='MATH',TF32=False),elapsed_this_attempt_seconds=time.monotonic()-start))

def feature_data(record):
    return load_members(record,('X2','Z','Early26','Direct36','initial_indices','anchor_id','match_id','fixed_team_A','fold_assignment'))

def locate_npz(study,record):return dict(path=str(Path(study)/record['path']),sha256=record['sha256'])

def validate_pair(lp,ln):
    if not np.isfinite(lp).all() or not np.isfinite(ln).all() or np.max(np.abs(np.logaddexp(lp,ln)))>1e-12:raise ValueError('Frozen log pair normalization')

def readout(expected,c):
    already_frozen('readout_freeze.json');b,cohort=binding(c);tf=J(P/'trajectory_freeze.json');start=time.monotonic()
    index={(r['world'],r['seed']):r for r in tf['records']}
    cv=feature_data(index[('CV',-1)]['array']);ix=cv['initial_indices'];N=len(ix)
    if N!=5344 or not np.array_equal(ix,cohort['selected_rows']):raise ValueError('Frozen population')
    route=held_only_route(cohort['folds'],ix,cv['match_id'])
    if not np.array_equal(route,cv['fold_assignment']):raise ValueError('Frozenfold route')
    main_members=('initial_indices','anchor_id','match_id','fixed_team_A')+tuple(f'{arm}__{key}' for arm in ('NeuralMatched','CalDirectUnion602','ObservedH2Oracle') for key in ('q_logp','q_log1mp'))
    main=load_members(b['historical_main_readout'],main_members)
    oldcv=load_members(b['historical_fullCV_readout'],('initial_indices','anchor_id','match_id','q_logp','q_log1mp'))
    for old in (main,oldcv):
        for key in ('initial_indices','anchor_id','match_id'):
            if not np.array_equal(old[key],cv[key]):raise ValueError('Historical literal reference alignment')
    if not np.array_equal(main['fixed_team_A'],cv['fixed_team_A']):raise ValueError('Fixedt0A mismatch')
    refs={(r['domain'],r['seed']):r for r in c['interop']['old_features']}
    oldCV=load_members(refs[('CV',-1)],refs[('CV',-1)]['members'])
    if not np.array_equal(oldCV['anchor_id'][ix],cv['anchor_id']):raise ValueError('OldCV identity')
    for name,newname in (('X2','X2'),('Z','Z'),('Early26','Early26'),('DirectUnion36','Direct36')):
        if oldCV[name].dtype!=cv[newname].dtype or not np.array_equal(oldCV[name][ix],cv[newname],equal_nan=True):raise AssertionError('Exact numerical copiedCV feature interop: '+name)
    heads=b['heads'];alpha_records=b['alphas'];headroot=b['value_study_root'] if 'value_study_root' in b else str(BASE/'CCPV_native_neural_bridge_20261026/neural_bridge')
    completed=[];position={int(row):j for j,row in enumerate(ix)}
    for seed in range(5):
        fresh={'CV':cv,'D':feature_data(index[('D',seed)]['array']),'E':feature_data(index[('E',seed)]['array'])}
        for world,data in fresh.items():
            for key in ('initial_indices','anchor_id','match_id','fixed_team_A','fold_assignment'):
                if not np.array_equal(data[key],cv[key]):raise ValueError('Frozenworld alignment: '+world+'/'+key)
        oldD=load_members(refs[('D',seed)],refs[('D',seed)]['members'])
        if not np.array_equal(oldD['anchor_id'][ix],cv['anchor_id']):raise ValueError('OldDfeature identity')
        for fold in cohort['folds']:
            boundary(start);f=fold['fold'];held=fold['held_match'];globalrows=np.asarray(fold['held_global_rows'],np.int64);loc=np.asarray([position[int(row)] for row in globalrows])
            folder=P/'readout_shards'/f'f{f}_s{seed}';done=closed(folder/'manifest.json',expected)
            if done is not None:completed.append(done);continue
            if folder.exists():raise RuntimeError('Partialreadoutshard preserved; nooverwrite')
            folder.mkdir(parents=True,exist_ok=False)
            cps={};alphas={}
            for kind in ('EarlyD','LateD','DirectUnion'):
                selected=[r for r in heads if r['head']==kind and r['seed']==seed and r['outer_held']==held]
                if len(selected)!=1 or held in selected[0]['training_matches']:raise ValueError('Excluded-match unique head route')
                cp=load_members(locate_npz(headroot,selected[0]),HEAD_MEMBERS)
                if str(cp['head'])!=kind or int(cp['training_seed'])!=seed:raise ValueError('Savedhead identity')
                cps[kind]=cp
                aa=[]
                for output in range(2 if kind=='DirectUnion' else 1):
                    ar=[r for r in alpha_records if r['head']==kind and r['seed']==seed and r['fold']==f and r['output_index']==output]
                    if len(ar)!=1:raise ValueError('OriginalOOFalpha unique route')
                    a=load_members(locate_npz(headroot,ar[0]),ALPHA_MEMBERS)
                    if (str(a['head']),int(a['seed']),int(a['output_index']),str(a['outer_held_match']))!=(kind,seed,output,held):raise ValueError('Alpha identity/heldmatch')
                    aa.append(float(a['alpha']))
                alphas[kind]=np.array(aa)
            oldpred=prospective_readouts(cps,alphas,oldD['Early26'][globalrows],oldD['DirectUnion36'][globalrows],oldD['X2'][globalrows])
            maxerr=0.
            for newarm,oldarm in (('Chain_OldAlpha','NeuralMatched'),('Direct_OldAlpha','CalDirectUnion602')):
                for key in ('q_logp','q_log1mp'):
                    err=float(np.max(np.abs(oldpred[newarm][key]-main[oldarm+'__'+key][seed,loc])));maxerr=max(maxerr,err)
            if maxerr>c['interop']['old_D_logpair_max_absolute_difference']:raise AssertionError('Oldhead/logpair interoperability gate failed; preserve error, no tolerance change')
            logs={k:np.empty((12,len(loc)),np.float64) for k in ('q_logp','q_log1mp','early_logp','early_log1mp')}
            chainlogs={k:np.empty((6,len(loc)),np.float64) for k in ('late_logp','late_log1mp')};categories=np.empty((6,len(loc),3),np.float64)
            for wi,world in enumerate(WORLDS):
                x=fresh[world];pred=prospective_readouts(cps,alphas,x['Early26'][loc],x['Direct36'][loc],x['X2'][loc])
                for ri,name in enumerate(READOUTS):
                    ai=4*wi+ri
                    for key in logs:logs[key][ai]=pred[name][key]
                    if name.startswith('Chain'):
                        ci=2*wi+ri
                        for key in chainlogs:chainlogs[key][ci]=pred[name][key]
                        categories[ci]=pred[name]['category_logs']
            for key in ('q','early'):validate_pair(logs[key+'_logp'],logs[key+'_log1mp'])
            save(folder/'readout.npz',**logs,**chainlogs,category_logs=categories,initial_positions=loc)
            done=dict(status='COMPLETE_HELD_ONLY_FROZEN_READOUT_SHARD',implementation_lock_sha256=expected,fold=f,seed=seed,held_match=held,rows=len(loc),
                old_interop_maxlog_error=maxerr,primary_head_row_evaluations=9*len(loc),interop_head_row_evaluations=3*len(loc),array=rec(folder/'readout.npz'),files=[rec(folder/'readout.npz'),c['_execution_authorization']],authorization_snapshot=c['_execution_authorization'])
            put(folder/'manifest.json',done);completed.append(done);print('VALUE_SHARD_CLOSED',f,seed,'interop',maxerr,flush=True);boundary(start)
    arrays={key:np.full((5,12,N),np.nan) for key in ('q_logp','q_log1mp','early_logp','early_log1mp')}
    late={key:np.full((5,6,N),np.nan) for key in ('late_logp','late_log1mp')};cats=np.full((5,6,N,3),np.nan);coverage=np.zeros((5,N),np.int8)
    for r in completed:
        z=load_members(r['array'],tuple(arrays)+tuple(late)+('category_logs','initial_positions'));s=r['seed'];loc=z['initial_positions'];coverage[s,loc]+=1
        for key in arrays:arrays[key][s,:,loc]=z[key].T
        for key in late:late[key][s,:,loc]=z[key].T
        cats[s,:,loc,:]=z['category_logs'].transpose(1,0,2)
    if len(completed)!=30 or not np.all(coverage==1) or not all(np.isfinite(v).all() for v in list(arrays.values())+list(late.values())+[cats]):raise AssertionError('All30shards exact coverage')
    if sum(r['primary_head_row_evaluations'] for r in completed)!=240480 or sum(r['interop_head_row_evaluations'] for r in completed)!=80160:raise AssertionError('Primary240480 plusinterop80160 head-rowbudget')
    validate_pair(arrays['q_logp'],arrays['q_log1mp']);validate_pair(arrays['early_logp'],arrays['early_log1mp'])
    meta={k:cv[k] for k in ('initial_indices','anchor_id','match_id','fixed_team_A','fold_assignment')}
    save(P/'frozen_readouts.npz',arms=np.asarray(ARMS),chain_arms=np.asarray([a for a in ARMS if '_Chain_' in a]),**arrays,**late,category_logs=cats,**meta)
    controls_p=np.stack([main[a+'__q_logp'] for a in ('NeuralMatched','CalDirectUnion602','ObservedH2Oracle')]+[oldcv['q_logp']],axis=1)
    controls_n=np.stack([main[a+'__q_log1mp'] for a in ('NeuralMatched','CalDirectUnion602','ObservedH2Oracle')]+[oldcv['q_log1mp']],axis=1)
    validate_pair(controls_p,controls_n)
    save(P/'historical_controls.npz',arms=np.asarray(HISTORICAL),q_logp=controls_p,q_log1mp=controls_n,**meta)
    for ref in c['interop']['old_features']:check(ref)
    for ref in (b['historical_main_readout'],b['historical_fullCV_readout']):check(ref)
    for ref in heads+alpha_records:check(locate_npz(headroot,ref))
    files=[rec(P/'frozen_readouts.npz'),rec(P/'historical_controls.npz')]+[f for r in completed for f in r['files']]+[rec(P/'readout_shards'/f'f{r["fold"]}_s{r["seed"]}'/'manifest.json') for r in completed]
    put(P/'readout_freeze.json',dict(status='ALL12_ARMS5SEEDS5344_READOUTS_FROZEN_BEFORE_LABELS',implementation_lock_sha256=expected,utc=now(),files=files,arms=list(ARMS),rows=5344,seeds=5,
        records=completed,primary_head_row_evaluations=sum(r['primary_head_row_evaluations'] for r in completed),interop_head_row_evaluations=sum(r['interop_head_row_evaluations'] for r in completed),
        total_head_row_evaluations=320640,old_interop_maxlog_error=max(r['old_interop_maxlog_error'] for r in completed),CV_feature_interop='EXACT_NUMERICAL_NAN_EQUIVALENT_SAME_DTYPE',new_NN_or_fits=0,actual_score_labels_decoded=False,authorization_snapshot=c['_execution_authorization'],elapsed_this_attempt_seconds=time.monotonic()-start))

def score(expected,c):
    already_frozen('evaluation_freeze.json');b,cohort=binding(c);folder=P/'evaluation'
    if folder.exists():raise RuntimeError('Partialscore output preserved; no overwrite')
    new=load_members(rec(P/'frozen_readouts.npz'),('arms','q_logp','q_log1mp','initial_indices','anchor_id','match_id','fixed_team_A','fold_assignment'))
    old=load_members(rec(P/'historical_controls.npz'),('arms','q_logp','q_log1mp','initial_indices','anchor_id','match_id','fold_assignment'))
    if tuple(new['arms'])!=ARMS or tuple(old['arms'])!=HISTORICAL:raise ValueError('Fixedall16arms')
    for k in ('initial_indices','anchor_id','match_id','fold_assignment'):
        if not np.array_equal(new[k],old[k]):raise ValueError('Literalcontrol cohort mismatch')
    y=load_members(b['score_only_outcomes'],SCORE_MEMBERS);ix=new['initial_indices']
    if not np.array_equal(ix,cohort['selected_rows']) or not np.array_equal(y['anchor_id'][ix],new['anchor_id']) or not np.array_equal(y['match_id'][ix],new['match_id']):raise ValueError('Score-only label/anchor alignment')
    if not np.array_equal(ix,np.flatnonzero(y['eligible_recorded10'])):raise ValueError('Alloriginal5344 labels, no q/guard selection')
    for key in ('y_early2','y_late8','y_10'):
        if not np.isin(y[key][ix],[0,1]).all():raise ValueError('Unknowns never zero filled')
    if not np.array_equal(y['y_10'][ix],np.maximum(y['y_early2'][ix],y['y_late8'][ix])):raise ValueError('Original2+8 binarylabel identity')
    arms=ARMS+HISTORICAL;lp=np.concatenate((new['q_logp'],old['q_logp']),axis=1);ln=np.concatenate((new['q_log1mp'],old['q_log1mp']),axis=1)
    tables=evaluate(arms,lp,ln,y['y_10'][ix],new['match_id']);folder.mkdir(exist_ok=False)
    for name,rows in tables.items():table(folder/(name+'.csv'),rows)
    check(b['score_only_outcomes'])
    files=[rec(folder/(name+'.csv')) for name in tables]+[c['_execution_authorization']]
    put(P/'evaluation_freeze.json',dict(status='SAVED_ONLY_RECORDED_SHOT10_ALL16_ARMS_SCORED_NO_FIT',implementation_lock_sha256=expected,utc=now(),files=files,
        fixed_rows=5344,ledger_rows=13372,matches=6,seeds=5,arms=list(arms),y10_positive_windows=int(y['y_10'][ix].sum()),early_positive_windows=int(y['y_early2'][ix].sum()),
        row_counts={k:len(v) for k,v in tables.items()},claims='Exposed sixmatch per-pipelineheldout recordclock Shot10 transfer, not fresh/E-specific calibration/causal actionbenefit',new_model_or_fit_calls=0,authorization_snapshot=c['_execution_authorization']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('trajectory_features','readout','score'),required=True);p.add_argument('--lock',required=True);p.add_argument('--identity-path',required=True);p.add_argument('--authorization-snapshot',required=True);p.add_argument('--authorization-sha',required=True);a=p.parse_args()
    lock,auth,c=guard(a.stage,a.lock,a.authorization_snapshot,a.authorization_sha);authority=rec(a.authorization_snapshot);c['_execution_authorization']=authority;identity(a.identity_path,a.stage,a.lock,auth['attempt_id'],authority)
    try:
        {'trajectory_features':trajectory_features,'readout':readout,'score':score}[a.stage](a.lock,c)
        guard(a.stage,a.lock,a.authorization_snapshot,a.authorization_sha)
    except BaseException as exc:
        put(P/'execution'/a.stage/auth['attempt_id']/f'error_{time.time_ns()}.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc(),preserve_all_partials=True))
        if isinstance(exc,PauseBetweenShards):return 75
        raise
    return 0

if __name__=='__main__':raise SystemExit(main())
