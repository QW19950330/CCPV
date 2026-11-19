"""One fixed 602-parameter q-first joint-category candidate; no file lookup.

q=sigmoid(row0), r=sigmoid(row1)=P(Early|any10), e=q*r. The loss is
Lq+sum(w*Y10*BCE(r,Y2))+.005*sum(all weight matrices squared), with no
positive-mass normalization. Shared hidden parameters see both data terms.
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


def reference_priors(y10, y2, w, frontend):
    if frontend == 'D':
        targets = np.c_[y10, y2]
        q0, e0 = [float(w @ targets[:, k]) for k in range(2)]
    elif frontend == 'CV':
        e0, q0 = float(w @ y2), float(w @ y10)
    else:
        raise ValueError('No frontend search')
    if not 0 < e0 < q0 < 1:
        raise ValueError('Fixed training category priors must be interior')
    return q0, e0, e0 / q0


def make(seed, q_prior, early_prior):
    if seed not in range(5) or not 0 < early_prior < q_prior < 1:
        raise ValueError('Fixed supported training-only priors/seed required')
    r_prior = early_prior / q_prior
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED_BASE + seed)
        model = Full602()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith('weight'):
                    nn.init.normal_(p, 0., INIT_STD)
                else:
                    p.zero_()
            for j, rate in enumerate((q_prior, r_prior)):
                model.output.bias[j].fill_(math.log(rate) - math.log1p(-rate))
            generator = torch.Generator(device='cpu')
            generator.manual_seed(SEED_BASE + 700 + seed)
            nn.init.normal_(model.output.weight[1], 0., INIT_STD, generator=generator)
    if sum(p.numel() for p in model.parameters()) != 602:
        raise AssertionError('Fixed capacity changed')
    return model


def initialize(X, y10, y2, w, seed, frontend):
    norm = scaler_fit(X, w)
    q0, e0, r0 = reference_priors(y10, y2, w, frontend)
    model = make(seed, q0, e0)
    return model, norm, state(model), (q0, e0, r0)


def terms(model, x, y10, y2, w, frontend):
    logits = model(x)
    if frontend == 'D':
        # Preserve same-frontend P0 Direct q reduction path exactly.
        target_matrix = torch.stack((y10, y2), dim=1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_matrix, reduction='none')
        loss_q = (w[:, None] * losses).sum(0)[0]
    elif frontend == 'CV':
        lq = -torch.nn.functional.softplus(-logits[:, 0])
        nq = -torch.nn.functional.softplus(logits[:, 0])
        loss_q = torch.sum(w * -torch.where(y10 == 1, lq, nq))
    else:
        raise ValueError('No frontend search')
    partition_losses = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 1], y2, reduction='none')
    partition_nll = torch.sum(w * y10 * partition_losses)
    ridge = PENALTY * sum(p.square().sum() for n, p in model.named_parameters() if n.endswith('weight'))
    joint_nll = loss_q + partition_nll
    return joint_nll + ridge, dict(BCE_q=loss_q, partition_NLL_original_mass=partition_nll,
                                  joint_NLL=joint_nll, ridge=ridge), logits


def gradient_snapshot(model, obj, diagnostics):
    params = list(model.parameters())
    before_rng = torch.random.get_rng_state().clone()
    before_grad = [None if p.grad is None else p.grad.detach().clone() for p in params]
    vectors, norms, block_vectors = {}, {}, {}
    for key, loss in [('gq', diagnostics['BCE_q']), ('gPartition', diagnostics['partition_NLL_original_mass']),
                      ('gR', diagnostics['ridge']), ('actual_total', obj)]:
        raw = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        pieces = [torch.zeros_like(p) if g is None else g for p, g in zip(params, raw)]
        vectors[key] = torch.cat([p.reshape(-1) for p in pieces]).detach().numpy().copy()
        block_vectors[key] = dict(hidden=np.r_[pieces[0].detach().numpy().ravel(), pieces[1].detach().numpy().ravel()],
                                  row0=np.r_[pieces[2][0].detach().numpy().ravel(), pieces[3][0].detach().numpy().ravel()],
                                  row1=np.r_[pieces[2][1].detach().numpy().ravel(), pieces[3][1].detach().numpy().ravel()])
        norms[key] = dict(total_norm=float(np.linalg.norm(vectors[key])),
                          hidden_weight_norm=float(torch.linalg.vector_norm(pieces[0])),
                          hidden_bias_norm=float(torch.linalg.vector_norm(pieces[1])),
                          row0_bias_gradient=float(pieces[3][0]), row1_bias_gradient=float(pieces[3][1]),
                          **{k + '_norm': float(np.linalg.norm(v)) for k, v in block_vectors[key].items()})
    pair_geometry = {}
    for block in ('full', 'hidden', 'row0', 'row1'):
        a = vectors['gq'] if block == 'full' else block_vectors['gq'][block]
        b = vectors['gPartition'] if block == 'full' else block_vectors['gPartition'][block]
        dot = float(a @ b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        pair_geometry[block] = dict(dot=dot, cosine=float(dot / denominator) if denominator > 0 else None)
    reconstructed = vectors['gq'] + vectors['gPartition'] + vectors['gR']
    vectors['reconstructed_total'] = reconstructed
    layout, offset = [], 0
    for name, p in model.named_parameters():
        layout.append(dict(name=name, shape=list(p.shape), offset_start=offset, offset_stop_exclusive=offset + p.numel()))
        offset += p.numel()
    summary = dict(component_norms=norms, gq_partition_geometry=pair_geometry, parameter_vector_layout=layout,
                   actual_global_gradient_norm=norms['actual_total']['total_norm'],
                   actual_global_clip_scale=min(1., CLIP / (norms['actual_total']['total_norm'] + 1e-6)),
                   reconstruction_max_absolute_error=float(np.max(np.abs(reconstructed - vectors['actual_total']))))
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise AssertionError('Gradient diagnostics changed RNG')
    if any((g is None) != (p.grad is None) or (g is not None and not torch.equal(g, p.grad)) for g, p in zip(before_grad, params)):
        raise AssertionError('Gradient diagnostics changed .grad')
    return vectors, summary


def fit(X, y10, y2, w, seed, frontend, *, _diagnostics_enabled=True):
    X = np.asarray(X, float)
    y10, y2, w = [np.asarray(v, float) for v in (y10, y2, w)]
    if X.shape != (len(w), 36) or not (y10.shape == y2.shape == w.shape):
        raise ValueError('Fixed Raw36/all-initial target shapes required')
    if not np.isin(y10, [0, 1]).all() or not np.isin(y2, [0, 1]).all() or (y2 > y10).any():
        raise ValueError('Training-only Early must be a subset of anyShot10')
    model, norm, initial, priors = initialize(X, y10, y2, w, seed, frontend)
    z = transform(X, norm)
    x, yy10, yy2, ww = map(torch.from_numpy, (z, y10, y2, w))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    history, gradient_arrays = [], {}
    for step in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        obj, diag, logits = terms(model, x, yy10, yy2, ww, frontend)
        if not torch.isfinite(obj):
            raise FloatingPointError('Nonfinite fixed joint objective')
        extra = {}
        if step == 0 and _diagnostics_enabled:
            vectors, geometry = gradient_snapshot(model, obj, diag)
            gradient_arrays.update({'diagnostic_step0_' + k: v for k, v in vectors.items()})
            extra['gradient_decomposition'] = geometry
        obj.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        history.append(dict(step=step, objective=float(obj.detach()), **{k: float(v.detach()) for k, v in diag.items()},
                            gradient_norm_before_clip=float(grad), clip_applied=bool(float(grad) > CLIP), **extra))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    obj, diag, logits = terms(model, x, yy10, yy2, ww, frontend)
    extra = {}
    if _diagnostics_enabled:
        vectors, geometry = gradient_snapshot(model, obj, diag)
        gradient_arrays.update({'diagnostic_step200_' + k: v for k, v in vectors.items()})
        extra['gradient_decomposition'] = geometry
    obj.backward()
    gradient_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
    if not np.isfinite([float(obj.detach()), gradient_norm]).all() or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite final joint model')
    qr = torch.sigmoid(logits)
    rates = torch.stack(((ww * qr[:, 0]).sum(), (ww * qr[:, 0] * qr[:, 1]).sum(),
                         (ww * yy10 * qr[:, 1]).sum() / (ww * yy10).sum()))
    history.append(dict(step=200, objective=float(obj.detach()), **{k: float(v.detach()) for k, v in diag.items()},
                        final_unclipped_gradient_norm=gradient_norm,
                        actual_weighted_rates_q_Early_r_given_Y10=rates.detach().numpy().tolist(), **extra))
    cp = {**state(model), **{'initial_' + k: v for k, v in initial.items()},
          **{'scaler_' + k: np.asarray(v) for k, v in norm.items()}, **gradient_arrays,
          'parameter_count': np.array(602), 'raw_dimension': np.array(36), 'updates': np.array(200),
          'training_seed': np.array(seed), 'actual_init_seed': np.array(SEED_BASE + seed),
          'conditional_Early_row_init_seed': np.array(SEED_BASE + 700 + seed),
          'frontend': np.array(frontend), 'training_weights': w.copy(), 'training_labels': np.c_[y10, y2],
          'label_order': np.array(['Y10', 'Y2']), 'reference_priors_q_e_r': np.array(priors),
          'row_semantics': np.array(['q10', 'Early_given_anyShot10']),
          'joint_partition_positive_mass': np.array(float(w @ y10)),
          'joint_partition_loss_divided_by_positive_mass': np.array(False)}
    for name, p in model.named_parameters():
        value = optimizer.state[p]
        key = name.replace('.', '_')
        cp['adam_step_' + key] = np.array(float(value['step']))
        cp['adam_exp_avg_' + key] = value['exp_avg'].detach().numpy().copy()
        cp['adam_exp_avg_sq_' + key] = value['exp_avg_sq'].detach().numpy().copy()
    return cp, dict(status='FINITE_FIXED200_NOT_GLOBAL_CONVERGENCE', updates=200, frontend=frontend,
                    parameter_count=602, partition_positive_mass_not_normalized=True, history=history)


def predict(cp, X, initial=False):
    norm = {k: cp['scaler_' + k] for k in SCALER_KEYS}
    z = transform(X, norm)
    prefix = 'initial_' if initial else ''
    logits = (np.tanh(z @ cp[prefix + 'hidden_weight'].T + cp[prefix + 'hidden_bias'])
              @ cp[prefix + 'output_weight'].T + cp[prefix + 'output_bias'])
    if logits.shape != (len(X), 2) or not np.isfinite(logits).all():
        raise FloatingPointError('Nonfinite fixed q-first two-logit readout')
    return logits
