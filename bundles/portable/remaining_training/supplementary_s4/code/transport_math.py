"""Pure frozen DirectUnion readout arithmetic; no paths, labels or model fits."""
import numpy as np


def encoded(X, cp):
    X = np.asarray(X, np.float64)
    if X.ndim != 2 or X.shape[1] != 36:
        raise ValueError('Exactly the inherited 36 raw features')
    finite = np.isfinite(X)
    with np.errstate(invalid='ignore'):
        z = np.where(finite, (X-cp['scaler_mean'])/cp['scaler_scale'], 0.)
    z[:, np.asarray(cp['scaler_completely_absent_training_feature'], bool)] = 0.
    Z = np.concatenate((z, (~finite).astype(np.float64)), axis=1)
    if not np.isfinite(Z).all():
        raise FloatingPointError('Frozen preprocessing produced nonfinite inputs')
    return Z, z, ~finite


def head(cp, X):
    Z, z, missing = encoded(X, cp)
    eta = np.tanh(Z @ cp['hidden_weight'].T + cp['hidden_bias']) @ cp['output_weight'].T + cp['output_bias']
    if eta.shape != (len(Z), 2) or not np.isfinite(eta).all():
        raise FloatingPointError('Direct head shape or finite contract failed')
    # Direct row0 is Y10; row1 is Y2, NOT late/early composition.
    result = {'q_logp': -np.logaddexp(0., -eta[:, 0]),
              'q_log1mp': -np.logaddexp(0., eta[:, 0]),
              'early_logp': -np.logaddexp(0., -eta[:, 1]),
              'early_log1mp': -np.logaddexp(0., eta[:, 1])}
    return result, z, missing


def logmean(a, axis=1):
    a = np.asarray(a, np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a-m).mean(axis=axis))


def mixture(particle):
    if any(np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != 16 for x in particle.values()):
        raise ValueError('Exactly all 16 particles, no split/selection')
    result = {k: logmean(v) for k, v in particle.items()}
    for stem in ('q', 'early'):
        if np.max(np.abs(np.logaddexp(result[stem+'_logp'], result[stem+'_log1mp']))) > 1e-12:
            raise ArithmeticError('Probability mixture/complement identity')
    return result


def metrics(lp, ln, y):
    lp, ln, y = [np.asarray(a, np.float64) for a in (lp, ln, y)]
    if lp.shape != ln.shape or lp.shape != y.shape or lp.ndim != 1 or not len(y):
        raise ValueError('Nonempty paired window vectors required')
    if not np.isin(y, (0, 1)).all() or not np.isfinite(lp).all() or not np.isfinite(ln).all():
        raise ValueError('Eligible known outcomes and finite probabilities only')
    p = np.exp(lp)
    bias = float(np.mean(p-y))
    return dict(log_loss=float(-np.where(y == 1, lp, ln).mean()),
                brier=float(np.mean((p-y)**2)), mean_probability=float(p.mean()),
                observed_rate=float(y.mean()), signed_bias=bias,
                absolute_fixture_bias=abs(bias))
