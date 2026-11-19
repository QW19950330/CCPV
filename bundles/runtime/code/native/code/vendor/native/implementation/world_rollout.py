"""Exact old mean rollout; no execution/import of Torch until guarded call."""
import numpy as np
from native_features import F,HistoryBatch,to_model_coordinates,to_native_coordinates

def rollout_D_native(batch: HistoryBatch, indices, api, model, *, authorized=False, device='cpu'):
    """Mean AR; preserve real provenance from native asof/history processing."""
    if not authorized:
        raise PermissionError('No real forward before locked root authorization')
    import torch
    indices = np.asarray(indices, np.int64)
    if not batch.supported[indices].all():
        raise ValueError('Unsupported rows retained as missing, never sent to D')
    history = torch.as_tensor(to_model_coordinates(batch, indices), device=device)
    valid = torch.as_tensor(batch.valid[indices].copy(), device=device)
    observed = torch.as_tensor(batch.real_observation_valid[indices].copy(), device=device)
    age = torch.as_tensor(batch.observation_age_s[indices].astype(np.float32), device=device)
    teams = torch.as_tensor(batch.teams.copy(), device=device)
    outputs, masks, ages = [], [], []
    old_state_hash = api.state_hash(model.state_dict())
    with torch.inference_mode():
        for _ in range(F):
            state = api.step(model, history, valid, teams, observed, age, mode='mean')
            if not torch.isfinite(state).all():
                raise FloatingPointError('Nonfinite D generation; retain failed attempt')
            usable = valid.any(1)
            outputs.append(state.cpu().numpy())
            masks.append(usable.cpu().numpy())
            history = torch.cat((history[:, 1:], state[:, None]), 1)
            valid = torch.cat((valid[:, 1:], usable[:, None]), 1)
            observed, age = api.advance_provenance(observed, age)
            ages.append(age.cpu().numpy())
    if old_state_hash != api.state_hash(model.state_dict()):
        raise AssertionError('Frozen D parameter mutated')
    model_paths = np.stack(outputs, 1)
    return (to_native_coordinates(model_paths, batch, indices), np.stack(masks, 1),
            np.stack(ages, 1), model_paths)
