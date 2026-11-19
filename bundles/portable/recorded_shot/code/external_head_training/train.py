"""Reconstruct the fixed 15 deployment heads from an authorized frozen cache.

This is a replay of the exposed development fit, not a new validation study.
No raw data, credentials, source-specific cache, or weights ship with this code.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from training_math import NAMES, fit, normfit, transform

FEATURE_SHA256 = 'b5f94c95aa8d14d60a1b7d5600e2b6a6f090e81b4a64958ee0aa2f8dd8528d07'
MATCHES = ('DFL-MAT-J03WMX', 'DFL-MAT-J03WN1', 'DFL-MAT-J03WOY',
           'DFL-MAT-J03WPY', 'DFL-MAT-J03WQQ', 'DFL-MAT-J03WR9')
ARMS = ('G_shared', 'GM200_shared', 'GM_OLS1s')
ROLE = 'FIXED_DEVELOPMENT_FIT_RECONSTRUCTION_NOT_NEW_EXTERNAL_VALIDATION'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def run(features_path, output_path):
    started = time.monotonic()
    features_path, out = Path(features_path).resolve(), Path(output_path).resolve()
    require(not out.exists(), 'Output must not exist; prior results are never overwritten.')
    require(features_path.is_file(), 'An authorized frozen feature cache is required.')
    require(features_path.stat().st_size == 2803791 and digest(features_path) == FEATURE_SHA256,
            'Feature cache differs from the fixed development input.')
    with np.load(features_path, allow_pickle=False) as z:
        require(z['conditions'].tolist() == ['native', 'grid'], 'Wrong coordinate conditions.')
        require(z['motion_variants'].tolist() == ['200ms_shared', 'OLS1s_shared'], 'Wrong motion variants.')
        require(z['X'].shape == (2, 2, 11542, 16), 'Wrong candidate grid.')
        require(z['feature_names'].tolist() == NAMES, 'Feature definitions changed.')
        X, y, eligible, matches = z['X'][0].copy(), z['y'].copy(), z['eligible'].copy(), z['match'].copy()
    require(eligible.dtype == bool and int(eligible.sum()) == 9483, 'Training support changed.')
    require(int(y[eligible].sum()) == 363 and set(y[eligible]) == {0, 1}, 'Training outcomes changed.')
    require(set(matches) == set(MATCHES), 'Training fixtures changed.')
    require(np.array_equal(X[0, :, :13], X[1, :, :13], equal_nan=True), 'Common geometry changed.')
    indices = np.flatnonzero(eligible)
    weights = np.array([1 / (6 * np.sum(matches[indices] == m)) for m in matches[indices]], dtype=np.float64)
    require(abs(weights.sum() - 1) < 1e-12, 'Match weights do not sum to one.')
    prior = float(weights @ y[indices])
    initial_bias = float(np.log(prior / (1 - prior)))
    out.mkdir(parents=True, exist_ok=False)
    heads = out / 'heads'
    heads.mkdir()
    save(out / 'run_identity.json', {
        'role': ROLE, 'utc': datetime.now(timezone.utc).isoformat(),
        'features_sha256': FEATURE_SHA256, 'candidate_windows': 11542,
        'eligible_training_windows': 9483, 'positive_training_windows': 363,
        'matches': list(MATCHES), 'seeds': list(range(5)), 'updates_per_head': 200,
        'external_data_accessed': False, 'formal50': False,
        'python': sys.version, 'numpy': np.__version__, 'cpu_threads': 1,
        'source_sha256': digest(__file__), 'math_sha256': digest(Path(__file__).with_name('training_math.py')),
    })
    records, normalizers = [], {}
    for arm in ARMS:
        variant, d = (1 if arm == 'GM_OLS1s' else 0), (13 if arm == 'G_shared' else 16)
        raw = X[variant, indices, :d]
        require(np.isfinite(raw[:, :13]).all() and not np.isinf(raw).any(), 'Invalid training geometry.')
        mu, scale = normfit(raw, weights)
        Z = transform(raw, mu, scale)
        require(np.isfinite(Z).all() and np.isfinite(mu).all() and np.all(np.isfinite(scale) & (scale > 0)),
                'Invalid standardization.')
        if normalizers:
            require(np.array_equal(mu[:13], normalizers['G_shared'][0]) and
                    np.array_equal(scale[:13], normalizers['G_shared'][1]), 'Shared normalizers differ.')
        normalizers[arm] = mu, scale
        for seed in range(5):
            require(not (out / 'PAUSE_REQUESTED').exists(), 'User pause requested; completed heads retained.')
            require(time.monotonic() - started < 600, 'Fixed replay exceeded 600 seconds; completed heads retained.')
            initial = np.random.default_rng(20261118 + seed).normal(0, .01, 16)[:d]
            w, bias, m, v, history, final = fit(Z, y[indices], weights, initial, initial_bias)
            require(history.shape == (200, 6) and np.array_equal(history[:, 0], np.arange(1, 201)),
                    'Update history incomplete.')
            require(all(np.isfinite(a).all() for a in (w, m, v, history, final)), 'Nonfinite fit.')
            path = heads / f'{arm}_s{seed}.npz'
            npz(path, w=w, b=np.array(bias), initial_w=initial, initial_b=np.array(initial_bias),
                adam_m=m, adam_v=v, adam_step=np.array(200, dtype=np.int32), history=history,
                final=final, mean=mu, scale=scale, train_indices=indices,
                native_variant=np.array(variant, dtype=np.int32))
            records.append({'arm': arm, 'seed': seed, 'parameters': d + 1, 'train_n': len(indices),
                            'status': 'FROZEN_200_NATIVE_UPDATES', 'path': path.name,
                            'bytes': path.stat().st_size, 'sha256': digest(path)})
            print(f'RECONSTRUCTED {arm} seed={seed}', flush=True)
    save(heads / 'head_freeze.json', {
        'status': 'FROZEN_ALL_15_BEFORE_READOUT', 'role': ROLE,
        'freeze_scope': 'This replay freezes all heads before any replay readout. Original external fixtures are already exposed.',
        'prior': prior, 'heads': records, 'features_sha256': FEATURE_SHA256,
    })
    save(out / 'completion.json', {
        'status': 'COMPLETED_FIXED_DEVELOPMENT_RECONSTRUCTION', 'role': ROLE,
        'heads': 15, 'updates_per_head': 200, 'prior': prior,
        'head_manifest_sha256': digest(heads / 'head_freeze.json'),
        'wall_seconds': time.monotonic() - started,
        'original_head_equality': 'Requires separate array comparison; not asserted by fitting.',
        'raw_data_extraction_reproduced': False, 'public_release': False,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.features, args.output)
