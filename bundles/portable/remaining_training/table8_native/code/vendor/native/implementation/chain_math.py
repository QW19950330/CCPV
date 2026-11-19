"""Exact old coherent2+8 composition; no labels/I/O/fitting."""
import numpy as np

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
