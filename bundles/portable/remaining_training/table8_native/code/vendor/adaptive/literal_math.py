"""Exact legacy NumPy math. Only intercept optimizer can fit a scalar."""
import numpy as np
from types import SimpleNamespace


def transform(X,normalizer):
 X=np.asarray(X,dtype=np.float64);available=np.isfinite(X);z=np.where(available,(X-normalizer['mean'])/normalizer['scale'],0.)
 absent=np.asarray(normalizer['completely_absent_training_feature'],dtype=bool)
 if absent.shape!=(X.shape[1],):raise ValueError('training missing-feature mask shape')
 z[:,absent]=0.
 result=np.concatenate((z,(~available).astype(np.float64)),axis=1)
 if not np.isfinite(result).all():raise ValueError('normalization overflow')
 return result


H=SimpleNamespace(transform=transform)


def predict_logits(cp, X):
    norm = {key: cp['scaler_' + key] for key in ('mean', 'scale', 'completely_absent_training_feature')}
    z = H.transform(X, norm)
    logits = np.tanh(z @ cp['hidden_weight'].T + cp['hidden_bias']) @ cp['output_weight'].T + cp['output_bias']
    if not np.isfinite(logits).all():
        raise FloatingPointError('Nonfinite fixed head output')
    return logits


def equal_match_weights(ids, expected):
    ids=np.asarray(ids).astype(str); expected=list(map(str,expected))
    if not expected or len(set(expected))!=len(expected) or set(ids)!=set(expected):
        raise ValueError('every predeclared initial training match must be present')
    w=np.array([1/(len(expected)*np.count_nonzero(ids==m)) for m in ids],float)
    if not np.isclose(w.sum(),1,atol=1e-12,rtol=0):raise ValueError('initial weight mass')
    return w


def restrict_risk(initial_weights, early_y):
    w=np.asarray(initial_weights,float); y=np.asarray(early_y)
    if w.shape!=y.shape or not np.isin(y,[0,1]).all() or np.any(w<=0):raise ValueError('risk support')
    risk=y==0; mass=float(w[risk].sum())
    if mass<=0:raise ValueError('empty R2: no invented fallback')
    return risk,w[risk]/mass,mass


def logpair(logits):
    z=np.asarray(logits,float)
    if not np.isfinite(z).all():raise ValueError('nonfinite logits')
    return -np.logaddexp(0.,-z),-np.logaddexp(0.,z)


def compose(early_logp, early_log1mp, late_logp, late_log1mp):
    """ALL initial rows; no Y2 or observed-H2 arguments exist here.

    Category order none, early2, late8-without-early. Late is conditional on R2,
    not an unconditional late-event frequency. Fixed A belongs to data lineage.
    """
    le,lne,ll,lnl=map(lambda a:np.asarray(a,float),(early_logp,early_log1mp,late_logp,late_log1mp))
    if not (le.shape==lne.shape==ll.shape==lnl.shape):raise ValueError('unaligned probabilities')
    if not all(np.isfinite(a).all() for a in (le,lne,ll,lnl)):raise ValueError('finite log probabilities required')
    if max(float(np.max(np.abs(np.logaddexp(le,lne)))),float(np.max(np.abs(np.logaddexp(ll,lnl)))))>1e-12:raise ValueError('not normalized binary logs')
    category=np.stack((lne+lnl,le,lne+ll),axis=-1)
    logq=np.logaddexp(le,lne+ll); log1q=lne+lnl
    if np.max(np.abs(np.logaddexp(logq,log1q)))>1e-12:raise AssertionError('chain identity')
    return logq,log1q,category


def fit_intercept(logits,y,w):
    """Standard proper binary log-loss intercept on honest inner OOF R2 rows.

    Convex potential sum w softplus(z+alpha)-target*alpha, lambda=0.
    Fixed100 bisections in[-40,40]; no root fallback or outer score selection.
    """
    z=np.asarray(logits,float); y=np.asarray(y,float); w=np.asarray(w,float)
    if not (z.ndim==y.ndim==w.ndim==1 and z.shape==y.shape==w.shape and len(z)>0):raise ValueError('calibration shape')
    if not all(np.isfinite(a).all() for a in (z,y,w)) or not np.isin(y,[0,1]).all() or np.any(w<=0) or not np.isclose(w.sum(),1,atol=1e-12,rtol=0):raise ValueError('calibration support')
    target=float(w@y)
    if not 0<target<1:raise ValueError('intercept target lacks both classes')
    def g(alpha):return float(w@np.exp(-np.logaddexp(0.,-(z+alpha)))-target)
    low,high=-40.,40.
    if not g(low)<0<g(high):raise ValueError('fixed calibration bracket unsupported')
    history=[]
    for step in range(100):
        mid=(low+high)/2; grad=g(mid)
        history.append(dict(step=step,low=low,high=high,mid=mid,gradient=grad,potential=float(w@np.logaddexp(0,z+mid)-target*mid)))
        if grad>0:high=mid
        else:low=mid
    alpha=(low+high)/2; p=np.exp(-np.logaddexp(0,-(z+alpha))); residual=float(w@p-target); hessian=float(w@(p*(1-p)))
    if abs(residual)>=1e-10 or hessian<=0:raise ValueError('unresolved intercept')
    return dict(alpha=np.array(alpha),iterations=np.array(100),oof_logits=z,training_labels=y,training_weights=w,target_rate=np.array(target),residual=np.array(residual),hessian=np.array(hessian)),history
