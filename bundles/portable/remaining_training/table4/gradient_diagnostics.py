"""Read-only training-batch component gradients; no optimizer or RNG mutation."""
import hashlib
import itertools
import json
import random
import numpy as np
import torch
from factorized_objective import CELLS, rollout_components

def _hash_tree(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                h.update((type(key).__name__ + repr(key)).encode()); add(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x: add(item)
        else:
            h.update((type(x).__name__ + repr(x)).encode())
    add(value)
    return h.hexdigest()

def _rng_state():
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch_cpu=torch.get_rng_state())
    # Do not initialize a previously unused CUDA context for diagnostics.
    if torch.cuda.is_initialized(): state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state

def _group(name):
    if name.startswith('temporal.step_gap_projection.'): return 'gap_step'
    if name.startswith('temporal.trailing_gap_projection.'): return 'gap_trailing'
    if name.startswith('temporal.'): return 'temporal_non_gap'
    if name.startswith('spatial.'): return 'spatial'
    if name.startswith('logits_head.'): return 'logits'
    if name.startswith('velocity_loc_head.'): return 'velocity_location'
    if name.startswith('velocity_log_scale_head.'): return 'velocity_scale'
    raise ValueError('Unclassified parameter: ' + name)

def _snapshot(model, named, optimizer):
    return dict(model_state=_hash_tree(model.state_dict()),
                existing_grad=_hash_tree({name: p.grad for name, p in named}),
                optimizer=None if optimizer is None else _hash_tree(optimizer.state_dict()),
                rng=_hash_tree(_rng_state()),
                module_modes=_hash_tree([(name, m.training) for name, m in model.named_modules()]))

def gradient_diagnostic(model, batch, optimizer=None):
    """Four gradients at the SAME theta and fixed B24, returns metadata,vectors.

    Raw vectors are FP32 in registered parameter order. Summaries are FP64 and
    never substitute for the training FP32 backward/clip200/Adam operation.
    """
    named = tuple(model.named_parameters())
    if not named or not all(p.requires_grad and p.dtype == torch.float32 for _, p in named):
        raise ValueError('Expected all trainable FP32 coupled parameters')
    before = _snapshot(model, named, optimizer)
    layout, offset = [], 0
    for name, p in named:
        layout.append(dict(name=name, shape=list(p.shape), start=offset,
                           stop=offset + p.numel(), group=_group(name)))
        offset += p.numel()
    vectors, unused, losses = {}, {}, {}
    counts = None
    for feedback in ('TF', 'AR'):
        branch = rollout_components(model, batch, feedback)
        current = (branch['nll_count'], branch['position_count'])
        if counts is not None and counts != current:
            raise ValueError('Diagnostic source counts differ across contexts')
        counts = current
        for index, (loss_kind, key) in enumerate((('NLL', 'nll_loss'), ('position', 'position_loss'))):
            label = loss_kind + '_' + feedback
            loss = branch[key]
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diagnostic loss')
            grads = torch.autograd.grad(loss, [p for _, p in named],
                                        retain_graph=index == 0, create_graph=False,
                                        allow_unused=True)
            pieces, missing = [], []
            for (name, p), grad in zip(named, grads):
                if grad is None:
                    missing.append(name); grad = torch.zeros_like(p)
                if not torch.isfinite(grad).all(): raise FloatingPointError('Nonfinite component gradient')
                pieces.append(grad.detach().cpu().reshape(-1).numpy().copy())
            vectors[label] = np.concatenate(pieces).astype(np.float32, copy=False)
            unused[label] = missing; losses[label] = float(loss.detach())
        del branch
    after = _snapshot(model, named, optimizer)
    if before != after:
        raise RuntimeError('Gradient diagnostics mutated an invariant: ' + json.dumps(dict(before=before, after=after)))
    groups = sorted(set(row['group'] for row in layout))
    masks = {g: np.zeros(offset, bool) for g in groups}
    for row in layout: masks[row['group']][row['start']:row['stop']] = True
    masks['body'] = masks['temporal_non_gap'] | masks['spatial'] | masks['gap_step'] | masks['gap_trailing']
    masks['all'] = np.ones(offset, bool)
    for label in ('position_TF', 'position_AR'):
        if np.count_nonzero(vectors[label][masks['velocity_scale']]):
            raise RuntimeError('Mean-position loss unexpectedly depends directly on scale-head parameters')
    summaries = {}
    for group, mask in masks.items():
        v = {k: x[mask].astype(np.float64) for k, x in vectors.items()}
        norms = {k: float(np.linalg.norm(x)) for k, x in v.items()}
        pairs = {}
        for left, right in itertools.combinations(v, 2):
            dot = float(np.dot(v[left], v[right])); denom = norms[left] * norms[right]
            pairs[left + '__' + right] = dict(dot=dot, cosine=None if denom == 0 else dot / denom)
        combined = {}
        for cell, (nll_feedback, pos_feedback) in CELLS.items():
            combined[cell] = float(np.linalg.norm(v['NLL_' + nll_feedback] + v['position_' + pos_feedback]))
        summaries[group] = dict(parameters=int(mask.sum()), component_norms=norms,
                                pairs=pairs, combined_norms_FP64_descriptive=combined)
    metadata = dict(status='READ_ONLY_COMPONENT_GRADIENTS_COMPLETE',
                    counts=dict(nll=counts[0], position=counts[1]), losses=losses,
                    parameter_layout=layout, parameter_layout_sha256=_hash_tree(layout),
                    vector_order='registered named_parameters; FP32; unused flags saved separately',
                    unused_parameters=unused, groups=summaries,
                    position_scale_gradient_exact_zero=True,
                    state_before=before, state_after=after,
                    model_forwards=40, gradient_vector_reverse_calls=4, optimizer_steps=0,
                    summary_dtype='float64 only; original FP32 clip200 remains authoritative',
                    limitation='Local training gradients do not identify unique causal mechanisms or real-world effects.')
    return metadata, vectors

