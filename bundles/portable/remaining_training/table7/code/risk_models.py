"""Fixed 602-parameter 2x2 models; no file or held-label access.

NoEarlyBCE retains the training-fold Early-prior initialization. It removes
only the auxiliary Early BCE contribution, and is not Y10-only supervision.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

SEED_BASE = 2026102551
INIT_STD = .05
UPDATES = 200
LR = .003
CLIP = 10.
PENALTY = .005
STATE_KEYS = ('hidden_weight', 'hidden_bias', 'output_weight', 'output_bias')
SCALER_KEYS = ('mean', 'scale', 'observed_weight_mass', 'completely_absent_training_feature')
PARAMETERIZATIONS = ('Direct', 'Marginal')
OBJECTIVES = ('NoEarlyBCE', 'WithEarlyBCE')


class Full602(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(72, 8, dtype=torch.float64)
        self.output = nn.Linear(8, 2, dtype=torch.float64)

    def forward(self, x):
        return self.output(torch.tanh(self.hidden(x)))


def state(model):
    return {n.replace('.', '_'): p.detach().cpu().numpy().copy() for n, p in model.state_dict().items()}


def scaler_fit(X, w):
    # Exact inherited all-initial, train-only, finite-feature scaler.
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if (X.ndim != 2 or len(X) != len(w) or len(X) == 0 or not np.isfinite(w).all()
            or np.any(w <= 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Invalid feature/weight support')
    available = np.isfinite(X)
    mass = (w[:, None] * available).sum(0)
    safe = np.where(available, X, 0.)
    mu = np.divide((w[:, None] * safe).sum(0), mass, out=np.zeros(X.shape[1]), where=mass > 0)
    var = np.divide((w[:, None] * np.where(available, X - mu, 0.) ** 2).sum(0), mass,
                    out=np.zeros(X.shape[1]), where=mass > 0)
    scale = np.sqrt(var)
    scale = np.where(scale < 1e-12, 1., scale)
    return dict(mean=mu, scale=scale, observed_weight_mass=mass, completely_absent_training_feature=mass == 0)


def transform(X, normalizer):
    X = np.asarray(X, dtype=np.float64)
    available = np.isfinite(X)
    z = np.where(available, (X - normalizer['mean']) / normalizer['scale'], 0.)
    absent = np.asarray(normalizer['completely_absent_training_feature'], dtype=bool)
    if absent.shape != (X.shape[1],):
        raise ValueError('Training missing-feature mask shape')
    z[:, absent] = 0.
    result = np.concatenate((z, (~available).astype(np.float64)), axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Normalization overflow')
    return result


def make(seed, early_prior, q_prior, parameterization):
    if (seed not in range(5) or parameterization not in PARAMETERIZATIONS
            or not 0 < early_prior < q_prior < 1):
        raise ValueError('Fixed original train-only priors/seed/parameterization required')
    late_prior = (q_prior - early_prior) / (1 - early_prior)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED_BASE + seed)
        model = Full602()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith('weight'):
                    nn.init.normal_(p, 0., INIT_STD)
                else:
                    p.zero_()
            for j, rate in enumerate((q_prior if parameterization == 'Direct' else late_prior, early_prior)):
                model.output.bias[j].fill_(math.log(rate) - math.log1p(-rate))
            generator = torch.Generator(device='cpu')
            generator.manual_seed(SEED_BASE + 700 + seed)
            nn.init.normal_(model.output.weight[1], 0., INIT_STD, generator=generator)
    if sum(p.numel() for p in model.parameters()) != 602:
        raise AssertionError('Fixed capacity changed')
    return model


def chain_logs(logits):
    l, e = logits[:, 0], logits[:, 1]
    le = -torch.nn.functional.softplus(-e)
    ne = -torch.nn.functional.softplus(e)
    ll = -torch.nn.functional.softplus(-l)
    nl = -torch.nn.functional.softplus(l)
    return torch.logaddexp(le, ne + ll), ne + nl, le, ne, ll, nl


def terms(model, x, y10, y2, y8, w, parameterization, objective, frontend):
    if parameterization not in PARAMETERIZATIONS or objective not in OBJECTIVES or frontend not in ('D', 'CV'):
        raise ValueError('No implicit aliases or objective/frontend search')
    logits = model(x)
    if parameterization == 'Direct' and frontend == 'D':
        # P0 D-Direct numerical contract: retain Nx2 reduction for exact replay.
        y = torch.stack((y10, y2), dim=1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
        bce = (w[:, None] * losses).sum(0)
        loss_q, loss_e = bce[0], bce[1]
        core = bce.sum() if objective == 'WithEarlyBCE' else bce[0]
        rates = (w[:, None] * torch.sigmoid(logits)).sum(0)
    else:
        # Audit Marginal and CV-Direct numerical contract.
        e = logits[:, 1]
        loss_e = torch.sum(w * torch.nn.functional.binary_cross_entropy_with_logits(e, y2, reduction='none'))
        if parameterization == 'Direct':
            lq = -torch.nn.functional.softplus(-logits[:, 0])
            nq = -torch.nn.functional.softplus(logits[:, 0])
            rates = torch.stack(((w * torch.exp(lq)).sum(), (w * torch.sigmoid(e)).sum()))
        else:
            lq, nq, le, ne, ll, nl = chain_logs(logits)
            risk_weights = w * (y2 == 0)
            mass = risk_weights.sum()
            rates = torch.stack(((w * torch.exp(lq)).sum(), (w * torch.exp(le)).sum(),
                                 (risk_weights * torch.exp(ll)).sum() / mass))
        loss_q = torch.sum(w * -torch.where(y10 == 1, lq, nq))
        core = loss_q + loss_e if objective == 'WithEarlyBCE' else loss_q
    ridge = PENALTY * sum(p.square().sum() for n, p in model.named_parameters() if n.endswith('weight'))
    return core + ridge, dict(BCE_q=loss_q, BCE_E=loss_e, ridge=ridge), rates


def initialize(X, y10, y2, w, seed, parameterization, frontend):
    norm = scaler_fit(X, w)
    if parameterization == 'Direct' and frontend == 'D':
        # P0 target construction has strided Nx2 columns. NumPy dot may differ
        # from contiguous-vector dot by ULPs; preserve the original contract.
        targets = np.c_[y10, y2]
        q0, e0 = [float(w @ targets[:, k]) for k in range(2)]
    else:
        e0, q0 = float(w @ y2), float(w @ y10)
    model = make(seed, e0, q0, parameterization)
    initial = state(model)
    return model, norm, initial, e0, q0


def gradient_snapshot(model, objective_tensor, diagnostics, objective):
    """Separate autograd.grad calls do not populate .grad or change RNG/state."""
    params = list(model.parameters())
    before_rng = torch.random.get_rng_state().clone()
    before_grad = [None if p.grad is None else p.grad.detach().clone() for p in params]
    vectors, blocks, block_vectors = {}, {}, {}
    for name, loss in [('gq', diagnostics['BCE_q']), ('gE', diagnostics['BCE_E']),
                       ('gR', diagnostics['ridge']), ('actual_total', objective_tensor)]:
        grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        pieces = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grad)]
        vector = torch.cat([g.reshape(-1) for g in pieces]).detach().numpy().copy()
        vectors[name] = vector
        block_arrays = dict(hidden=np.r_[pieces[0].detach().numpy().ravel(), pieces[1].detach().numpy().ravel()],
                            row0=np.r_[pieces[2][0].detach().numpy().ravel(), pieces[3][0].detach().numpy().ravel()],
                            row1=np.r_[pieces[2][1].detach().numpy().ravel(), pieces[3][1].detach().numpy().ravel()])
        block_vectors[name] = block_arrays
        blocks[name] = dict(total_norm=float(np.linalg.norm(vector)),
                            hidden_weight_norm=float(torch.linalg.vector_norm(pieces[0])),
                            hidden_bias_norm=float(torch.linalg.vector_norm(pieces[1])),
                            row0_bias_gradient=float(pieces[3][0]), row1_bias_gradient=float(pieces[3][1]),
                            **{k + '_norm': float(np.linalg.norm(v)) for k, v in block_arrays.items()})
    coefficient = 1. if objective == 'WithEarlyBCE' else 0.
    reconstructed = vectors['gq'] + coefficient * vectors['gE'] + vectors['gR']
    vectors['reconstructed_total'] = reconstructed
    error = reconstructed - vectors['actual_total']
    pairs = {}
    for a, b in [('gq', 'gE'), ('gq', 'gR'), ('gE', 'gR')]:
        dot = float(vectors[a] @ vectors[b])
        denominator = np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b])
        group_pairs = {}
        for group in ('hidden', 'row0', 'row1'):
            ga, gb = block_vectors[a][group], block_vectors[b][group]
            gdot = float(ga @ gb)
            gdenom = np.linalg.norm(ga) * np.linalg.norm(gb)
            group_pairs[group] = dict(dot=gdot, cosine=float(gdot / gdenom) if gdenom > 0 else None)
        pairs[a + '_' + b] = dict(dot=dot, cosine=float(dot / denominator) if denominator > 0 else None,
                                 **group_pairs)
    norm = blocks['actual_total']['total_norm']
    parameter_layout, offset = [], 0
    for name, p in model.named_parameters():
        parameter_layout.append(dict(name=name, shape=list(p.shape), offset_start=offset,
                                     offset_stop_exclusive=offset + p.numel()))
        offset += p.numel()
    summary = dict(component_norms=blocks, component_pair_geometry=pairs, parameter_vector_layout=parameter_layout,
                   Early_BCE_coefficient=coefficient, actual_global_gradient_norm=norm,
                   actual_global_clip_scale=min(1., CLIP / (norm + 1e-6)),
                   reconstruction_max_absolute_error=float(np.max(np.abs(error))),
                   reconstruction_l2_error=float(np.linalg.norm(error)),
                   NoEarly_gE_diagnostic_only=objective == 'NoEarlyBCE')
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise AssertionError('Gradient diagnostics changed RNG')
    if any((a is None) != (p.grad is None) or (a is not None and not torch.equal(a, p.grad))
           for a, p in zip(before_grad, params)):
        raise AssertionError('Gradient diagnostics changed parameter .grad')
    return vectors, summary


def fit(X, y10, y2, y8, w, seed, parameterization, objective, frontend, *, _diagnostics_enabled=True):
    X = np.asarray(X, float)
    y10, y2, y8, w = [np.asarray(v, float) for v in (y10, y2, y8, w)]
    if X.shape != (len(w), 36) or not all(v.shape == w.shape for v in (y10, y2, y8)):
        raise ValueError('Fixed36/all-initial target shapes required')
    if not all(np.isin(v, [0, 1]).all() for v in (y10, y2, y8)) or not np.array_equal(y10, np.maximum(y2, y8)):
        raise ValueError('Native training-only label dictionary required')
    model, norm, initial, e0, q0 = initialize(X, y10, y2, w, seed, parameterization, frontend)
    z = transform(X, norm)
    x, yy10, yy2, yy8, ww = map(torch.from_numpy, (z, y10, y2, y8, w))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    history, gradient_arrays = [], {}
    for step in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        obj, diag, _ = terms(model, x, yy10, yy2, yy8, ww, parameterization, objective, frontend)
        if not torch.isfinite(obj):
            raise FloatingPointError('Nonfinite fixed objective')
        geometry = {}
        if step == 0 and _diagnostics_enabled:
            vectors, snapshot = gradient_snapshot(model, obj, diag, objective)
            gradient_arrays.update({'diagnostic_step0_' + k: v for k, v in vectors.items()})
            geometry['gradient_decomposition'] = snapshot
        obj.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        history.append(dict(step=step, objective=float(obj.detach()),
                            **{k: float(v.detach()) for k, v in diag.items()},
                            gradient_norm_before_clip=float(grad), clip_applied=bool(float(grad) > CLIP), **geometry))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    obj, diag, rates = terms(model, x, yy10, yy2, yy8, ww, parameterization, objective, frontend)
    final_geometry = {}
    if _diagnostics_enabled:
        vectors, snapshot = gradient_snapshot(model, obj, diag, objective)
        gradient_arrays.update({'diagnostic_step200_' + k: v for k, v in vectors.items()})
        final_geometry['gradient_decomposition'] = snapshot
    obj.backward()
    gn = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
    if not np.isfinite([float(obj.detach()), gn]).all() or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite final objective/weights/gradient')
    history.append(dict(step=200, objective=float(obj.detach()),
                        **{k: float(v.detach()) for k, v in diag.items()},
                        final_unclipped_gradient_norm=gn, actual_weighted_prediction_rates=rates.detach().numpy().tolist(),
                        **final_geometry))
    cp = {**state(model), **{'initial_' + k: v for k, v in initial.items()},
          **{'scaler_' + k: np.asarray(v) for k, v in norm.items()},
          'parameter_count': np.array(602), 'raw_dimension': np.array(36), 'updates': np.array(200),
          'training_seed': np.array(seed), 'actual_init_seed': np.array(SEED_BASE + seed),
          'Early_row_init_seed': np.array(SEED_BASE + 700 + seed),
          'parameterization': np.array(parameterization), 'objective': np.array(objective),
          'frontend': np.array(frontend), 'training_weights': w.copy(),
          'training_labels': np.c_[y10, y2, y8], 'label_order': np.array(['Y10', 'Y2', 'Ylate8']),
          'reference_priors_q_e_l': np.array([q0, e0, (q0 - e0) / (1 - e0)]),
          'Early_prior_supervision_retained_without_Early_BCE': np.array(True),
          'numerical_contract': np.array('P0_MATRIX' if parameterization == 'Direct' and frontend == 'D' else 'AUDIT_VECTOR'),
          **gradient_arrays}
    for name, parameter in model.named_parameters():
        s = optimizer.state[parameter]
        k = name.replace('.', '_')
        cp['adam_step_' + k] = np.array(float(s['step']))
        cp['adam_exp_avg_' + k] = s['exp_avg'].detach().numpy().copy()
        cp['adam_exp_avg_sq_' + k] = s['exp_avg_sq'].detach().numpy().copy()
    return cp, dict(status='FINITE_FIXED200_NOT_GLOBAL_CONVERGENCE', updates=200,
                    parameterization=parameterization, objective=objective, frontend=frontend, history=history)


def predict(cp, X, initial=False):
    norm = {k: cp['scaler_' + k] for k in SCALER_KEYS}
    z = transform(X, norm)
    prefix = 'initial_' if initial else ''
    logits = (np.tanh(z @ cp[prefix + 'hidden_weight'].T + cp[prefix + 'hidden_bias'])
              @ cp[prefix + 'output_weight'].T + cp[prefix + 'output_bias'])
    if logits.shape != (len(X), 2) or not np.isfinite(logits).all():
        raise FloatingPointError('Nonfinite fixed two-logit readout')
    return logits
