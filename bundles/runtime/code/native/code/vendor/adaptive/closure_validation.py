"""Closed-unit ownership and complete predecessor validation, before payloads.

Only JSON metadata and declared integrity hashes are consulted. No NPZ member,
label, current q or alpha is decoded by this module.
"""
from pathlib import Path
import math
from io_core import J,check

RULES={
 'oof_logits':('oof_shards','oof.npz','CLOSED_LABEL_FREE_E_OOF_LOGITS','ALL_E_OOF_LOGITS_FROZEN_NO_LABELS'),
 'calibrate':('calibration_shards','intercepts.npz','CLOSED_FOUR_E_OOF_SCALAR_INTERCEPTS','ALL120_E_OOF_INTERCEPTS_FROZEN_TRAIN_LABELS_ONLY'),
 'readout':('readout_shards','q.npz','CLOSED_IDENTITY_OLDD_ADAPTIVE_E_LOGPAIRS','ALL_E_ADAPTIVE_READOUTS_FROZEN_BEFORE_HELD_LABELS')}
KEYS={(f,s) for f in range(6) for s in range(5)}
SCALARS={'EarlyD','LateD','DirectQ','DirectEarly'}

def require(condition,message):
    if not condition:raise ValueError(message)
def integer(value):return type(value) is int
def resolved(record):return Path(record['path']).resolve()
def numeric(value):return type(value) in (int,float) and math.isfinite(value)

def validate_unit(stage,r,c,expected,root,requested=None,verify_files=True):
    require(stage in RULES,'Known closurestage')
    root=Path(root).resolve();subdir,filename,status,_=RULES[stage]
    require(integer(r.get('fold')) and integer(r.get('seed')),'Exact integerfold/seed')
    key=(r['fold'],r['seed']);require(key in KEYS,'Fold/seed in fixed grid')
    if requested is not None:require(key==tuple(requested),'Closed unit differs from REQUESTED fold/seed')
    f=next((f for f in c['folds'] if f['fold']==key[0]),None);require(f is not None,'Knownfold')
    require(r.get('status')==status,'Exact closedstage status')
    require(r.get('implementation_lock_sha256')==expected,'Closedunit lock')
    require(r.get('outer_held')==f['held_match'],'Expected heldmatch')
    ns=len(f['train_rows']);nh=len(f['held_rows'])
    folder=root/subdir/f'f{key[0]}_s{key[1]}'
    array=folder/filename;manifest=folder/'manifest.json'
    require(isinstance(r.get('array'),dict) and resolved(r['array'])==array.resolve(),'Ownunit array path only')
    authority=r.get('authorization_snapshot');require(isinstance(authority,dict),'Authoritysnapshot record')
    authpath=resolved(authority)
    try:parts=authpath.relative_to(root/'execution'/stage).parts
    except ValueError:raise ValueError('Authority outside same stage/root')
    require(len(parts)==2 and parts[1]=='authorization_snapshot.json' and parts[0] and all(ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in parts[0]),'Ownedstage attemptauthority path')
    files=r.get('files');require(isinstance(files,list),'Unit files list')
    paths=[resolved(x) for x in files]
    wanted={array.resolve(),authpath}
    if stage=='calibrate':wanted.add((folder/'histories.json').resolve())
    require(len(paths)==len(set(paths)) and set(paths)==wanted,'Exact ownunit array/history/authority files')
    require(next(x for x in files if resolved(x)==array.resolve())==r['array'],'Array record matches files')
    require(next(x for x in files if resolved(x)==authpath)==authority,'Authority record matches files')
    if stage=='oof_logits':
        require(integer(r.get('train_rows')) and r['train_rows']==ns,'ExactOOF train rowcount')
        require(r.get('head_matrix_calls')==15 and integer(r.get('head_matrix_calls')),'15head calls perOOF unit')
        require(r.get('head_row_evaluations')==3*ns and integer(r.get('head_row_evaluations')),'ExactOOF rowbudget')
    elif stage=='calibrate':
        require(integer(r.get('train_rows')) and r['train_rows']==ns,'Exactcalibration train rowcount')
        nr=r.get('risk_rows');require(integer(nr) and 1<nr<ns,'StoredR2 rowcount')
        require(r.get('logical_scalar_intercepts')==4 and integer(r.get('logical_scalar_intercepts')) and r.get('bisections')==400 and integer(r.get('bisections')),'4roots x100iterations')
        require(numeric(r.get('riskmass')) and 0<r['riskmass']<1,'Finite weightedriskmass')
        diag=r.get('diagnostics');require(isinstance(diag,dict) and set(diag)==SCALARS,'Four fixed alphadiagnostics')
        for name,d in diag.items():
            require(d.get('iterations')==100 and integer(d.get('iterations')),'100iterations eachroot')
            require(d.get('rows')==(nr if name=='LateD' else ns) and integer(d.get('rows')),'Percomponent targetrow counts')
            require(all(numeric(d.get(k)) for k in ('alpha','target_rate','residual','hessian')),'Finite alphadiagnostics')
            require(-40<d['alpha']<40 and 0<d['target_rate']<1 and abs(d['residual'])<1e-10 and d['hessian']>0,'Unchanged root criteria')
    else:
        require(integer(r.get('rows')) and r['rows']==nh,'Exactq held rowcount')
        require(numeric(r.get('oldalpha_interop_maxlog_error')) and 0<=r['oldalpha_interop_maxlog_error']<=1e-14,'Unchanged oldalpha interoperability criterion')
        require(integer(r.get('head_matrix_calls')) and r['head_matrix_calls']==0 and integer(r.get('scalar_fits')) and r['scalar_fits']==0,'Readout0head0fit budget')
    # Metadata/path rejection happens BEFORE touching a foreign NPZ/hash.
    if verify_files:
        require(manifest.is_file(),'Ownunit manifest exists')
        require(J(manifest)==r,'Record equals own requested manifest JSON')
        for x in files:check(x)
        if stage=='calibrate':
            history=J(folder/'histories.json');require(set(history)==SCALARS and all(isinstance(h,list) and len(h)==100 for h in history.values()),'Exact ownunit four100step histories')
    return r

def validate_records(stage,records,c,expected,root,verify_files=True,freeze_files=None):
    require(isinstance(records,list) and len(records)==30,'Exactly30 closed units')
    keys=[(r.get('fold'),r.get('seed')) for r in records]
    require(all(integer(f) and integer(s) for f,s in keys),'Integer closurekeys')
    require(len(set(keys))==30 and set(keys)==KEYS,'ALL30 unique fixed fold/seed keys BEFORE nextpayload')
    for r in records:validate_unit(stage,r,c,expected,root,verify_files=verify_files)
    if stage=='oof_logits':require(sum(r['head_matrix_calls'] for r in records)==450 and sum(r['head_row_evaluations'] for r in records)==400800,'CompleteOOF budget')
    if stage=='calibrate':require(sum(r['logical_scalar_intercepts'] for r in records)==120 and sum(r['bisections'] for r in records)==12000,'Complete120alpha budget')
    if freeze_files is not None:
        require(isinstance(freeze_files,list),'Outerfreeze files')
        paths=[resolved(x) for x in freeze_files]
        for r in records:
            folder=Path(root).resolve()/RULES[stage][0]/f'f{r["fold"]}_s{r["seed"]}'
            for path in (folder/'manifest.json',resolved(r['array'])):
                require(paths.count(path.resolve())==1,'Each ownmanifest/array exactlyonce inouterfreeze')
        if verify_files:
            for x in freeze_files:check(x)
    return records

def validate_predecessor(stage,freeze,c,expected,root):
    require(freeze.get('status')==RULES[stage][3] and freeze.get('implementation_lock_sha256')==expected,'Exact precedingstage freeze/lock')
    validate_records(stage,freeze.get('records'),c,expected,root,freeze_files=freeze.get('files'))
    if stage=='oof_logits':require(freeze.get('head_matrix_calls')==450 and freeze.get('head_row_evaluations')==400800,'OOFfreeze totals')
    if stage=='calibrate':require(freeze.get('scalar_intercepts')==120 and freeze.get('bisections')==12000,'Calibrationfreeze totals')
    if stage=='readout':require(freeze.get('rows')==5344 and freeze.get('seeds')==5 and freeze.get('head_or_NN_calls')==0,'Readoutfreeze totals')
    return freeze

def validate_existing_units(stage,c,expected,root):
    """Inspect owned reused/partial units before this stage opens payloads."""
    root=Path(root).resolve();subdir=RULES[stage][0]
    for fold,seed in sorted(KEYS):
        folder=root/subdir/f'f{fold}_s{seed}'
        if not folder.exists():continue
        manifest=folder/'manifest.json'
        require(manifest.is_file(),'Partialunit preserved; no payload activation')
        validate_unit(stage,J(manifest),c,expected,root,requested=(fold,seed))
