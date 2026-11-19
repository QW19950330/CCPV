"""Pure fixed-protocol probability and match-weight functions; no data lookup."""
from __future__ import annotations
import numpy as np


def equal_match_weights(ids, expected):
    ids = np.asarray(ids).astype(str)
    expected = list(map(str, expected))
    if not expected or len(expected) != len(set(expected)) or set(ids) != set(expected):
        raise ValueError('Every fixed training match must be present')
    w = np.array([1 / (len(expected) * np.count_nonzero(ids == m)) for m in ids], np.float64)
    if not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('Invalid equal-match weight mass')
    return w


def logpair(logits):
    x = np.asarray(logits, np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite logits')
    return -np.logaddexp(0., -x), -np.logaddexp(0., x)


def compose(early_logp, early_log1mp, late_logp, late_log1mp):
    """All-prefix q=e+(1-e)l; no true-Y2 or future-state gate is accepted."""
    le, ne, ll, nl = [np.asarray(v, np.float64) for v in
                      (early_logp, early_log1mp, late_logp, late_log1mp)]
    if not (le.shape == ne.shape == ll.shape == nl.shape):
        raise ValueError('Unaligned logs')
    if not all(np.isfinite(v).all() for v in (le, ne, ll, nl)):
        raise ValueError('Nonfinite component logs')
    if max(np.max(np.abs(np.logaddexp(le, ne))), np.max(np.abs(np.logaddexp(ll, nl)))) > 1e-12:
        raise ValueError('Invalid normalized binary logs')
    q, nq = np.logaddexp(le, ne + ll), ne + nl
    if np.max(np.abs(np.logaddexp(q, nq))) > 1e-12:
        raise AssertionError('Chain accounting failed')
    return q, nq


def initial_q_discrepancy(direct_logits, marginal_logits, weights):
    """Train-only descriptive discrepancy; never an initialization selection rule."""
    d, m, w = map(np.asarray, (direct_logits, marginal_logits, weights))
    if d.shape != m.shape or d.ndim != 2 or d.shape[1] != 2 or w.shape != d.shape[:1]:
        raise ValueError('Initial-q discrepancy shapes')
    qd = np.exp(logpair(d[:, 0])[0])
    e, ne = logpair(m[:, 1])
    l, nl = logpair(m[:, 0])
    qm = np.exp(compose(e, ne, l, nl)[0])
    delta = qm - qd
    return dict(direct_q=qd, marginal_q=qm, marginal_minus_direct_q=delta), dict(
        weighted_direct_q=float(w @ qd), weighted_marginal_q=float(w @ qm),
        weighted_marginal_minus_direct=float(w @ delta),
        maximum_absolute_perrow_discrepancy=float(np.max(np.abs(delta))),
        weighted_mean_absolute_perrow_discrepancy=float(w @ np.abs(delta)),
        interpretation='DESCRIPTIVE_TRAIN_ONLY_NOT_A_FUNCTION_MATCH_OR_INITIALIZATION_SEARCH')


def ulp_distance(a, b):
    """Absolute representable-float distance, including negative float values."""
    def ordered(x):
        bits = np.asarray(x, dtype=np.float64).view(np.uint64).item()
        return ((~bits) & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)
    return abs(ordered(a) - ordered(b))
