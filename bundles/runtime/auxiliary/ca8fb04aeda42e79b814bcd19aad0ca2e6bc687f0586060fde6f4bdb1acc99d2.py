"""Locked label-free generation stage; no fitting or scoring in this script."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
import numpy as np

from core_io import P, sha, read, put, save, utc, guard
from bridge import (HistoryBatch, analytic_cv_native, array_sha,
                    features_from_native_paths, load_frozen_D, rollout_D_native,
                    source_pins, union_features, verify_physical_contract)

HISTORY_FIELDS = {
    'states': 'history_states', 'valid': 'history_fullstate_valid',
    'position_valid': 'position_valid',
    'real_observation_valid': 'history_real_observation_valid',
    'observation_age_s': 'last_source_observation_age_s', 'teams': 'teams',
    'entity_ids': 'entity_ids', 'anchor_id': 'anchor_id',
    'fixed_team_A_index': 'fixed_team_A_index', 'pitch_length_m': 'pitch_length_m',
    'pitch_width_m': 'pitch_width_m', 'native_to_model_sign': 'native_to_model_sign',
    'supported': 'supported', 'unsupported_reason': 'unsupported_reason'}
PREFIX_FIELDS = ('X0', 'X2_cv', 'Z_cv', 'match_id', 'anchor_id', 'period',
                 'source_frame_N', 'fixed_team_A', 'source_match_ids',
                 'eligible_prefix', 'eligible_recorded10', 'y_early2', 'y_late8', 'y_10')


def binding():
    b = read(P / 'data_binding.json')
    for record in b['assets'].values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Changed bound asset ' + record['path'])
    return b


def load_causal_inputs():
    b = binding()
    # Select only causal keys. No labels, actual X2 or future masks are decoded.
    with np.load(b['assets']['history']['path'], allow_pickle=False) as z:
        data = {name: z[key] for name, key in HISTORY_FIELDS.items()}
    batch = HistoryBatch(**data).validate()
    verify_physical_contract(read(b['assets']['physical_review']['path']))
    with np.load(b['assets']['prefix_data']['path'], allow_pickle=False) as z:
        if set(z.files) != set(PREFIX_FIELDS):
            raise ValueError('Only the exact old14-key prefix DTO may be bound')
        X0, old_Z_cv, anchor_id = (z[key] for key in ('X0', 'Z_cv', 'anchor_id'))
    if not np.array_equal(anchor_id, batch.anchor_id) or X0.shape != (len(anchor_id), 10):
        raise ValueError('Original native anchor mapping mismatch')
    if not np.array_equal(X0, old_Z_cv[:, :10], equal_nan=True):
        raise ValueError('Old CV/X0 dictionary changed')
    return batch, X0, old_Z_cv


def generate(expected, device):
    guard(expected, 'generate')
    batch, X0, old_Z_cv = load_causal_inputs()
    if (P / 'generated').exists() or (P / 'feature_bundle_lock.json').exists():
        raise FileExistsError('Never overwrite previous generation/feature lock')
    (P / 'generated').mkdir()
    n = len(batch.anchor_id)
    indices = np.flatnonzero(batch.supported)
    records = []
    source_records, checkpoints = source_pins()
    immutable_hist = {key: array_sha(value) for key, value in batch.__dict__.items()}
    for name, seed in [('CV', -1)] + [('D', s) for s in range(5)]:
        full = np.zeros((n, 20, 23, 4), np.float32)
        valid = np.zeros((n, 20, 23), bool)
        age = np.zeros((n, 20, 23), np.float32)
        x2, z = np.full((n, 10), np.nan), np.full((n, 18), np.nan)
        z[:, :10] = X0
        record = None
        if name == 'D':
            api, model, record = load_frozen_D(seed, authorized=True, device=device)
        calls, forward_batches = 0, []
        for start in range(0, len(indices), 128):
            rows = indices[start:start + 128]
            if name == 'CV':
                path, usable = analytic_cv_native(batch, rows)
                carried_age = batch.observation_age_s[rows, None] + np.arange(1, 21, dtype=np.float32)[None, :, None] * np.float32(.1)
            else:
                from bridge import to_model_coordinates
                before = api.state_hash(model.state_dict())
                path, usable, carried_age, model_path = rollout_D_native(batch, rows, api, model, authorized=True, device=device)
                after = api.state_hash(model.state_dict())
                calls += 20
                forward_batches.append(dict(
                    original_row_indices=rows.tolist(), actual_step_calls=20,
                    checkpoint_sha256=record['sha256'], model_class=type(model).__name__,
                    parameter_state_before=before, parameter_state_after=after,
                    input_model_history_sha256=array_sha(to_model_coordinates(batch, rows)),
                    input_observation_flags_sha256=array_sha(batch.real_observation_valid[rows]),
                    input_observation_age_sha256=array_sha(batch.observation_age_s[rows]),
                    model_path_sha256=array_sha(model_path), native_path_sha256=array_sha(path)))
            if not np.isfinite(path).all():
                raise FloatingPointError('No invalid generated value is silently converted to missing')
            full[rows], valid[rows], age[rows] = path, usable, carried_age
            x2[rows], z[rows] = features_from_native_paths(batch, rows, path, usable, X0)
        if name == 'D':
            del model
        early, union = union_features(X0, old_Z_cv, z, x2)
        path = P / 'generated' / f'{name}_s{seed}.npz'
        save(path, generated_native=full, generated_valid=valid, real_observation_age_s=age,
             X2=x2, Z=z, Early26=early, DirectUnion36=union,
             anchor_id=batch.anchor_id, fixed_team_A_index=batch.fixed_team_A_index,
             supported=batch.supported, unsupported_reason=batch.unsupported_reason,
             entity_ids=batch.entity_ids, teams=batch.teams,
             pitch_length_m=batch.pitch_length_m, pitch_width_m=batch.pitch_width_m,
             native_to_model_sign=batch.native_to_model_sign)
        records.append(dict(domain=name, seed=seed, path=str(path.relative_to(P)), sha256=sha(path),
                            rows=n, supported_rows=len(indices), unsupported_rows=n - len(indices),
                            batched_neural_forward_calls=calls,
                            forward_batches=forward_batches,
                            checkpoint=record if name == 'D' else None,
                            support_sha256=array_sha(batch.supported),
                            generated_usable_sha256=array_sha(valid),
                            X2_sha256=array_sha(x2), Z_sha256=array_sha(z)))
        print('FROZEN_GENERATED', name, seed, len(indices), '/', n, flush=True)
    for key, value in batch.__dict__.items():
        if array_sha(value) != immutable_hist[key]:
            raise AssertionError('Native input was mutated')
    source_pins()   # hash all old15 weights/results again, not just five reused D
    import torch
    put(P / 'feature_bundle_lock.json', dict(
        status='ALL_SIX_LABEL_FREE_FRONTEND_BUNDLES_FROZEN_BEFORE_ANY_VALUE_FIT', utc=utc(),
        implementation_lock_sha256=expected, data_binding_sha256=sha(P / 'data_binding.json'),
        records=records, prefix_input_array_hashes=immutable_hist,
        native_history_rows=n, unsupported_reason_counts=dict(Counter(batch.unsupported_reason[~batch.supported].tolist())),
        old_sources=source_records, old_checkpoint_records=checkpoints,
        hard_integrity=dict(native_history_identity_and_mask_validation=True,
                            physical_transform_evidence_verified=True,
                            all15_old_checkpoint_and_result_hashes_unchanged=True,
                            all_native_prefix_input_array_hashes_unchanged=True),
        runtime=dict(device=device, torch_version=torch.__version__,
                     cuda_version=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                     SDPA='MATH_ONLY', TF32=False),
        actual_Y2_or_future_XY_decoded=False, value_head_fits=0, new_scores=0,
        generated_real_observation_flags=False,
        common_support_definition='Original outcome eligibility intersect native-history supported; no future XY/label/model-quality filtering',
        limitation='Cross-source exposed development pipeline; physical fixture overlap not established from provider IDs'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--implementation-lock-sha256', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    generate(args.implementation_lock_sha256, args.device)
