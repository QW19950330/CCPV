"""Two loss contexts, one coupled model and one optimizer update.

TF is the historical C feedback with generated fallback, not rolling-observed
likelihood. AR is deterministic mixture-mean feedback, never sampled feedback.
"""
import torch
from torch.nn import functional as F
from model_source.api import initial_provenance, advance_provenance, output_mean

CELLS = {'C': ('TF', 'TF'), 'D': ('AR', 'AR'),
         'E': ('TF', 'AR'), 'F': ('AR', 'TF')}

def rollout_components(model, batch, feedback):
    """Original twenty-step count-weighted components, without feedback detach."""
    if feedback not in ('TF', 'AR'):
        raise ValueError('Feedback must explicitly be TF or AR')
    if getattr(model, 'position_residual_head', 'missing') is not None:
        raise ValueError('Only the frozen coupled VelocityResidualMDN is allowed')
    history, valid = batch['history'], batch['valid']
    seen, teams = batch['history_seen'], batch['teams']
    observed, age = initial_provenance(valid)
    nll_total = history.new_zeros(())
    position_total = history.new_zeros(())
    nll_count, position_count = 0, 0
    for t in range(20):
        output = model(history, valid, teams,
                       observation_valid=observed, observation_age=age)
        state = output_mean(model, output)
        full = batch['target_valid'][:, t] & seen
        pos = batch['position_valid'][:, t] & seen
        count, pc = int(full.sum()), int(pos.sum())
        loss = model.velocity_nll(output, batch['target'][:, t, :, 2:], full)
        truth = torch.where(pos[..., None], batch['position'][:, t], 0.)
        safe = torch.where(pos[..., None], state[..., :2], 0.)
        per_actor = F.smooth_l1_loss(safe / 1., truth / 1., reduction='none', beta=1.).mean(-1)
        position_total = position_total + torch.where(pos, per_actor, 0.).sum()
        position_count += pc
        nll_total = nll_total + loss * count
        nll_count += count
        real = full if feedback == 'TF' else torch.zeros_like(full)
        actual = torch.where(full[..., None], batch['target'][:, t], 0.)
        next_state = torch.where(real[..., None], actual, state)
        next_state = torch.where(seen[..., None], next_state, 0.)
        history = torch.cat((history[:, 1:], next_state[:, None]), 1)
        valid = torch.cat((valid[:, 1:], seen[:, None]), 1)
        observed, age = advance_provenance(observed, age, real)
    if nll_count <= 0 or position_count <= 0:
        raise ValueError('Both genuine target populations must be nonempty')
    return dict(nll_loss=nll_total / nll_count,
                position_loss=position_total / position_count,
                nll_count=nll_count, position_count=position_count,
                branch_forwards=20, feedback=feedback)

def factorized_loss(model, batch, cell):
    """Return total plus differentiable selected components and scalar metadata."""
    if cell not in CELLS:
        raise ValueError('Cell must be one of C,D,E,F; tuple order is NLL,position')
    nll_feedback, position_feedback = CELLS[cell]
    nll_branch = rollout_components(model, batch, nll_feedback)
    position_branch = (nll_branch if nll_feedback == position_feedback else
                       rollout_components(model, batch, position_feedback))
    for count in ('nll_count', 'position_count'):
        if nll_branch[count] != position_branch[count]:
            raise ValueError('Source target count changed with feedback context')
    nll_loss = nll_branch['nll_loss']
    position_loss = position_branch['position_loss']
    total = nll_loss + position_loss
    return total, dict(nll_loss=nll_loss, position_loss=position_loss,
                       nll_count=nll_branch['nll_count'],
                       position_count=position_branch['position_count'],
                       nll=float(nll_loss.detach()),
                       position_huber=float(position_loss.detach()),
                       loss=float(total.detach()),
                       branch_forwards=20 if nll_feedback == position_feedback else 40,
                       nll_feedback=nll_feedback, position_feedback=position_feedback)

