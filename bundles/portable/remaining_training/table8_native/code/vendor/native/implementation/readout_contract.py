"""Label-free old D-domain functional. Pure NumPy, no I/O/model/fit."""
import numpy as np
from frozen_head import predict_logits
from chain_math import logpair, compose

READOUTS=('Chain_Raw','Chain_OldAlpha','Direct_Raw','Direct_OldAlpha')
WORLDS=('E','D','CV')
ARMS=tuple(f'{world}_{name}' for world in WORLDS for name in READOUTS)

def checked_logits(cp,X,raw_dimension,outputs):
    x=np.asarray(X,np.float64)
    if x.ndim!=2 or x.shape[1]!=raw_dimension or int(cp['raw_dimension'])!=raw_dimension:
        raise ValueError('Exact old26/10/36 dimension required, never13D Goal')
    shapes={'hidden_weight':(8,2*raw_dimension),'hidden_bias':(8,),
            'output_weight':(outputs,8),'output_bias':(outputs,),
            'scaler_mean':(raw_dimension,),'scaler_scale':(raw_dimension,),
            'scaler_completely_absent_training_feature':(raw_dimension,)}
    for name,shape in shapes.items():
        a=np.asarray(cp[name])
        if a.shape!=shape:raise ValueError('Frozen head/scaler shape: '+name)
        if name!='scaler_completely_absent_training_feature' and not np.isfinite(a).all():
            raise ValueError('Nonfinite frozen coefficient/scaler')
    if np.any(cp['scaler_scale']<=0):raise ValueError('Positive old training scale')
    return predict_logits(cp,x)

def prospective_readouts(checkpoints,alphas,Early26,Direct36,X2):
    """No labels, actualH2, risk-set gate, fitting or adjustable alpha inputs.

    Caller MUST bind alphas to the original OOF files; mathematical signature
    alone is not proof of provenance. Raw is literal zero intercept; oldalpha
    is the unmodified oldD-domain intercept, not recalibration on E or heldY.
    """
    if set(checkpoints)!=set(('EarlyD','LateD','DirectUnion')) or set(alphas)!=set(checkpoints):
        raise ValueError('Exactly three fixed D-domain heads/alpha families')
    e=checked_logits(checkpoints['EarlyD'],Early26,26,1)[:,0]
    late=checked_logits(checkpoints['LateD'],X2,10,1)[:,0]
    direct=checked_logits(checkpoints['DirectUnion'],Direct36,36,2)
    if not len(e)==len(late)==len(direct):raise ValueError('Aligned raw feature rows')
    a={k:np.asarray(v,np.float64) for k,v in alphas.items()}
    if a['EarlyD'].shape!=(1,) or a['LateD'].shape!=(1,) or a['DirectUnion'].shape!=(2,) or not all(np.isfinite(x).all() for x in a.values()):
        raise ValueError('Original scalar alpha shapes')
    result={}
    for suffix,use_alpha in (('Raw',False),('OldAlpha',True)):
        ep,en=logpair(e+(a['EarlyD'][0] if use_alpha else 0.))
        lp,ln=logpair(late+(a['LateD'][0] if use_alpha else 0.))
        qp,qn,categories=compose(ep,en,lp,ln)
        result['Chain_'+suffix]=dict(q_logp=qp,q_log1mp=qn,early_logp=ep,early_log1mp=en,
                                     late_logp=lp,late_log1mp=ln,category_logs=categories)
        z=direct+(a['DirectUnion'] if use_alpha else np.zeros(2))
        qp,qn=logpair(z[:,0]);ep,en=logpair(z[:,1])
        result['Direct_'+suffix]=dict(q_logp=qp,q_log1mp=qn,early_logp=ep,early_log1mp=en)
    return result

def held_only_route(folds,selected_global_rows,match_ids):
    """Each old DFL fixture maps only to the unique head that never trained it."""
    rows=np.asarray(selected_global_rows)
    mids=np.asarray(match_ids).astype(str)
    if rows.ndim!=1 or len(rows)!=len(mids) or len(set(rows.tolist()))!=len(rows):
        raise ValueError('Unique global rows aligned to match IDs')
    route=np.full(len(rows),-1,np.int64)
    lookup={int(row):j for j,row in enumerate(rows)}
    for fold in folds:
        held=fold['held_match']
        if held in fold['training_matches']:raise ValueError('Held match in training')
        for globalrow in fold['held_global_rows']:
            if globalrow not in lookup:raise ValueError('Route row outside fixed population')
            j=lookup[globalrow]
            if route[j]!=-1 or mids[j]!=held:raise ValueError('Duplicate/wrong-held route')
            route[j]=int(fold['fold'])
    if (route<0).any():raise ValueError('Unassigned initial row')
    return route
