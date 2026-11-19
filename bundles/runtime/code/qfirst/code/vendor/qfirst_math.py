"""Pure q-first categorical probabilities and fixed match weights."""
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


def qfirst_probabilities(logits):
    """No outcome/future inputs. Categories: none, Early, late-without-Early."""
    logits = np.asarray(logits, np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError('Expected row0 total-risk and row1 conditional-Early logits')
    qp, qn = logpair(logits[:, 0])
    rp, rn = logpair(logits[:, 1])
    ep = qp + rp
    late_only = qp + rn
    # log(1-q*r) = log[(1-q)+q*(1-r)], avoiding probability subtraction.
    en = np.logaddexp(qn, late_only)
    late_conditional, no_late_conditional = late_only - en, qn - en
    category = np.stack((qn, ep, late_only), axis=-1)
    if np.max(np.abs(np.logaddexp.reduce(category, axis=1))) > 1e-12:
        raise AssertionError('Three-category normalization failed')
    if np.max(np.abs(np.logaddexp(late_conditional, no_late_conditional))) > 1e-12:
        raise AssertionError('Late-given-no-Early normalization failed')
    return dict(q_logp=qp, q_log1mp=qn, early_logp=ep, early_log1mp=en,
                r_logp=rp, r_log1mp=rn, late_only_logp=late_only,
                late_logp=late_conditional, late_log1mp=no_late_conditional,
                category_logs=category, q_logits=logits[:, 0], r_logits=logits[:, 1])
