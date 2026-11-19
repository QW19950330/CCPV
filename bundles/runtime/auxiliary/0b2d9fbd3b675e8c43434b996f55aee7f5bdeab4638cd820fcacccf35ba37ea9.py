"""Pure fold-only fixed-budget MC matched head; no filesystem or held inputs."""
import math
import numpy as np
import torch
from torch import nn

STATE = ('hidden_weight', 'hidden_bias', 'output_weight', 'output_bias')
SCALER = ('mean', 'scale', 'observed_weight_mass', 'completely_absent_training_feature')
SHAPES = ((8, 72), (8,), (2, 8), (2,))
UPDATES, LR, CLIP, RIDGE = 200, .003, 10., .005


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(72, 8, dtype=torch.float64)
        self.output = nn.Linear(8, 2, dtype=torch.float64)

    def forward(self, z):
        return self.output(torch.tanh(self.hidden(z)))


def make(initial):
    with torch.random.fork_rng(devices=[]):
        model = Head()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, shape in zip(STATE, SHAPES):
            a = np.asarray(initial[key])
            if a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Exact inherited float64 initial state required')
            params[key.replace('_', '.')].copy_(torch.from_numpy(a))
    assert sum(p.numel() for p in model.parameters()) == 602
    return model


def state(model):
    return {n.replace('.', '_'): p.detach().numpy().copy() for n, p in model.state_dict().items()}


def transform(X, norm):
    X = np.asarray(X, np.float64)
    if X.ndim != 2 or X.shape[1] != 36:
        raise ValueError('Fixed36 raw features')
    available = np.isfinite(X)
    z = np.where(available, (X-norm['mean'])/norm['scale'], 0.)
    z[:, np.asarray(norm['completely_absent_training_feature'], bool)] = 0.
    out = np.concatenate((z, (~available).astype(np.float64)), axis=1)
    if not np.isfinite(out).all():
        raise FloatingPointError('Inherited frozen transform overflow')
    return out


def terms(model, x, y, w):
    """x[N,K,72]; K configurable only for handmade parity tests, fit fixes16."""
    n, k, d = x.shape
    if d != 72 or k < 1 or y.shape != (n, 2) or w.shape != (n,):
        raise ValueError('Anchor/particle/target axes')
    eta = model(x.reshape(n*k, d)).reshape(n, k, 2)
    lp = torch.logsumexp(-torch.nn.functional.softplus(-eta), dim=1)-math.log(k)
    ln = torch.logsumexp(-torch.nn.functional.softplus(eta), dim=1)-math.log(k)
    bce = (w[:, None] * -torch.where(y == 1, lp, ln)).sum(0)
    ridge = RIDGE*sum(p.square().sum() for name, p in model.named_parameters() if name.endswith('weight'))
    return bce.sum()+ridge, bce, ridge, lp, ln


def fit(initial, norm, X_train, y_train, w_train, seed, *, tick=lambda: None):
    """Only explicitly whitelisted other-five-fixture arrays may reach this API."""
    X, y, w = map(lambda a: np.asarray(a, np.float64), (X_train, y_train, w_train))
    n = len(X)
    if seed not in range(5) or X.shape != (n, 16, 36) or y.shape != (n, 2) or w.shape != (n,):
        raise ValueError('Fixed16 particles/two outputs/five seeds')
    if not np.isin(y, (0, 1)).all() or not (y[:, 1] <= y[:, 0]).all():
        raise ValueError('Known nested training labels required')
    if any(set(np.unique(y[:, j])) != {0., 1.} for j in range(2)):
        raise ValueError('Both target classes required; no fallback')
    if not np.isfinite(w).all() or (w <= 0).any() or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('Original equal-fixture anchor weights required')
    z = transform(X.reshape(n*16, 36), norm).reshape(n, 16, 72)
    model = make(initial)
    assert all(np.array_equal(state(model)[key], initial[key]) for key in STATE)
    x, yy, ww = map(torch.from_numpy, (z, y, w))
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    assert len(opt.state) == 0
    history = []
    for step in range(UPDATES):
        tick(); opt.zero_grad(set_to_none=True)
        obj, bce, reg, lp, ln = terms(model, x, yy, ww)
        if not torch.isfinite(obj):
            raise FloatingPointError('Nonfinite training objective')
        obj.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        history.append(dict(step=step, objective=float(obj.detach()), BCE_by_target=bce.detach().tolist(),
                            ridge=float(reg.detach()), gradient_norm_before_clip=float(grad), clip_applied=bool(grad > CLIP)))
        opt.step()
    opt.zero_grad(set_to_none=True)
    obj, bce, reg, lp, ln = terms(model, x, yy, ww); obj.backward()
    grad = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
    if not np.isfinite([float(obj.detach()), grad]).all():
        raise FloatingPointError('Nonfinite final state')
    history.append(dict(step=200, objective=float(obj.detach()), BCE_by_target=bce.detach().tolist(),
                        ridge=float(reg.detach()), final_unclipped_gradient_norm=grad,
                        weighted_mean_predictions=(ww[:, None]*lp.exp()).sum(0).detach().tolist()))
    cp = {**state(model), **{'initial_'+k: np.asarray(v).copy() for k, v in initial.items()},
          **{'scaler_'+k: np.asarray(norm[k]).copy() for k in SCALER},
          'training_seed': np.array(seed), 'updates': np.array(200), 'parameter_count': np.array(602),
          'raw_dimension': np.array(36), 'output_targets': np.array(['Y10', 'Y2']),
          'training_weights': w.copy(), 'training_labels': y.copy(), 'K': np.array(16)}
    for name, p in model.named_parameters():
        key, slot = name.replace('.', '_'), opt.state[p]
        for item in ('step', 'exp_avg', 'exp_avg_sq'):
            cp['adam_'+item+'_'+key] = slot[item].detach().numpy().copy()
    return cp, dict(status='FINITE_FIXED200_NOT_GLOBAL_CONVERGENCE', history=history,
                    new_updates=200, anchor_labels=n, particle_head_inputs_per_update=n*16)
