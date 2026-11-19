"""Replay the original fixed-cache LOFO60 development fit and held scoring.

This is a reconstruction of already-exposed development evidence, not a new
experiment. Only an explicitly supplied, hash-pinned feature cache is read.
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
from scoring_math import score_predictions, table

FEATURE_SHA256 = 'b0c9cca42cb64e140afe9f4071ff16cae6ce3b01e9b36ebc9c703e6faff43797'
FEATURE_BYTES = 784832
ROLE = 'FIXED_CACHE_LOFO60_DEVELOPMENT_RECONSTRUCTION_NOT_NEW_VALIDATION'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def freeze(folder, name):
    records = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
               for p in sorted(folder.iterdir()) if p.is_file() and p.name != name]
    save(folder / name, {'status': 'COMPLETE', 'role': ROLE, 'files': records})
    return {'path': name, 'sha256': digest(folder / name)}


def run(features_path, output_path):
    started = time.monotonic()
    features_path, out = Path(features_path).resolve(), Path(output_path).resolve()
    require(not out.exists(), 'Create-only output directory already exists.')
    require(features_path.is_file(), 'Supply the authorized original feature cache.')
    require(features_path.stat().st_size == FEATURE_BYTES and digest(features_path) == FEATURE_SHA256,
            'Input differs from the original LOFO60 feature cache.')
    with np.load(features_path, allow_pickle=False) as z:
        require(set(z.files) == {'X', 'y', 'eligible', 'match', 'grid_index', 'feature_names'},
                'Unexpected cache schema.')
        require(z['feature_names'].tolist() == NAMES, 'Feature definition mismatch.')
        X, Y, eligible = z['X'].copy(), z['y'].copy(), z['eligible'].copy()
        matches, grid_index = z['match'].copy(), z['grid_index'].copy()
    require(X.shape == (11542, 16) and X.dtype == np.float64, 'Feature shape or dtype changed.')
    require(Y.shape == eligible.shape == matches.shape == grid_index.shape == (11542,),
            'Candidate vector shape mismatch.')
    require(Y.dtype == np.int32 and eligible.dtype == bool and grid_index.dtype == np.int32,
            'Cached outcome, eligibility or grid dtype changed.')
    require(int(eligible.sum()) == 9483 and int(Y[eligible].sum()) == 363,
            'Frozen eligible/positive population changed.')
    require(int((Y < 0).sum()) == 2059 and set(Y.tolist()) == {-1, 0, 1},
            'Original unknown outcomes must remain unknown.')
    require(np.array_equal(eligible, np.isfinite(X[:, :13]).all(1) & (Y >= 0)),
            'Cache eligibility does not match the preserved rule.')
    # Original extraction concatenated rows in the original six-fold plan order.
    # Recover that order from the pinned input; no source fixture IDs ship here.
    scope = list(dict.fromkeys(matches.tolist()))
    require(len(scope) == 6, 'Exactly six original development fixtures are required.')
    require(np.array_equal(np.concatenate([np.flatnonzero(matches == h) for h in scope]),
                           np.arange(len(matches))), 'Original grouped fixture order changed.')
    out.mkdir(parents=True, exist_ok=False)
    heads = out / 'heads'
    heads.mkdir()

    def guard():
        require(not (out / 'PAUSE_REQUESTED').exists(), 'Pause requested; prior heads retained.')
        require(time.monotonic() - started < 1800, 'Fixed reconstruction exceeded 1800 seconds.')
        require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) <= 536870912,
                'Fixed 512 MiB output budget exceeded.')

    save(out / 'run_identity.json', {
        'role': ROLE, 'utc': datetime.now(timezone.utc).isoformat(),
        'features_sha256': FEATURE_SHA256, 'features_bytes': FEATURE_BYTES,
        'candidate_windows': len(X), 'eligible_windows': int(eligible.sum()),
        'eligible_positive_windows': int(Y[eligible].sum()),
        'unknown_labels_retained': int((Y < 0).sum()),
        'folds': 6, 'seeds': list(range(5)), 'arms': ['G', 'GM'],
        'updates_per_head': 200, 'cpu_threads': 1,
        'python': sys.version, 'numpy': np.__version__,
        'source_sha256': digest(__file__),
        'math_sha256': digest(Path(__file__).with_name('training_math.py')),
        'scoring_sha256': digest(Path(__file__).with_name('scoring_math.py')),
        'external_inputs_read': False, 'raw_source_extraction_reproduced': False,
        'formal50': False,
    })
    # Preserve all input qualification states; do not invent raw exclusion reasons.
    table(out / 'cached_candidate_status.csv', [
        {'row': i, 'grid_index': int(grid_index[i]), 'match': str(matches[i]),
         'eligible': bool(eligible[i]), 'y': int(Y[i])} for i in range(len(X))])
    head_records, held_predictions = [], []
    for held in scope:
        tr = np.flatnonzero(eligible & (matches != held))
        te = np.flatnonzero(eligible & (matches == held))
        train_matches = sorted(set(matches[tr]))
        require(held not in train_matches, 'Held fixture entered training.')
        supported = len(train_matches) == 5 and len(te) > 0 and len(set(Y[tr])) == 2
        if not supported:
            head_records.extend(dict(held=held, seed=s, arm=arm,
                                     status='UNSUPPORTED_FIXED_FOLD', train_n=len(tr),
                                     held_n=len(te), path='')
                                for s in range(5) for arm in ('G', 'GM'))
            continue
        a = np.array([1 / (5 * np.sum(matches[tr] == m)) for m in matches[tr]])
        require(abs(a.sum() - 1) < 1e-12, 'Fold weights do not sum to one.')
        mu, sd = normfit(X[tr], a)
        Z = transform(X[tr], mu, sd)
        prior = float(a @ Y[tr])
        bias = float(np.log(prior / (1 - prior)))
        npz(heads / (held + '_fold.npz'), train_indices=tr, held_indices=te,
            weights=a, mean=mu, scale=sd, prior=np.array(prior))
        for seed in range(5):
            init = np.random.default_rng(20261118 + seed).normal(0, .01, 16)
            for arm, d in (('G', 13), ('GM', 16)):
                guard()
                w, b, m, v, hist, final = fit(Z[:, :d], Y[tr], a, init[:d], bias)
                require(hist.shape == (200, 6) and np.array_equal(hist[:, 0], np.arange(1, 201)),
                        'Incomplete fixed optimizer history.')
                path = heads / f'{held}_s{seed}_{arm}.npz'
                npz(path, initial_w=init[:d], initial_b=np.array(bias), w=w, b=np.array(b),
                    adam_m=m, adam_v=v, adam_step=np.array(200, dtype=np.int32),
                    history=hist, final=final, train_indices=tr, mean=mu, scale=sd)
                head_records.append(dict(held=held, seed=seed, arm=arm,
                                         status='FROZEN_200_UPDATES', train_n=len(tr),
                                         held_n=len(te), path=path.name, sha256=digest(path)))
        print('FIT_FOLD_COMPLETE', len([r for r in head_records if r['status'] == 'FROZEN_200_UPDATES']),
              flush=True)
    require(len(head_records) == 60, 'Missing fixed head statuses.')
    save(heads / 'head_records.json', head_records)
    headfreeze = freeze(heads, 'head_freeze.json')
    # No held row is read out until all 60 planned head statuses are frozen.
    for held in scope:
        records = [r for r in head_records if r['held'] == held]
        te = np.flatnonzero(eligible & (matches == held))
        if any(r['status'] != 'FROZEN_200_UPDATES' for r in records):
            continue
        with np.load(heads / (held + '_fold.npz'), allow_pickle=False) as q:
            mu, sd, prior = q['mean'], q['scale'], float(q['prior'])
        for i in te:
            held_predictions.append(dict(row=int(i), grid_index=int(grid_index[i]), held=held,
                                         seed=-1, arm='PREVALENCE', y=int(Y[i]), p=prior))
        Z = transform(X[te], mu, sd)
        for r in records:
            path = heads / r['path']
            require(digest(path) == r['sha256'], 'Head changed after freeze.')
            with np.load(path, allow_pickle=False) as q:
                z = Z[:, :len(q['w'])] @ q['w'] + q['b']
                prob = np.exp(-np.logaddexp(0, -z))
            for i, p in zip(te, prob):
                held_predictions.append(dict(row=int(i), grid_index=int(grid_index[i]), held=held,
                                             seed=r['seed'], arm=r['arm'], y=int(Y[i]), p=float(p)))
    require(bool(held_predictions), 'No supported fixed folds; no automatic replacement or retry.')
    table(out / 'predictions.csv', held_predictions)
    save(out / 'prediction_freeze.json', {
        'head_freeze': headfreeze, 'predictions_sha256': digest(out / 'predictions.csv'),
        'role': ROLE, 'all_readouts_frozen_before_scoring': True,
    })
    metrics, summary, contrasts = score_predictions(scope, held_predictions, out)
    save(out / 'completion.json', {
        'status': 'COMPLETED_FIXED_CACHE_LOFO60_RECONSTRUCTION', 'role': ROLE,
        'candidate_windows': len(X), 'eligible_windows': int(eligible.sum()),
        'positive_windows': int(Y[eligible].sum()), 'unknown_labels_retained': int((Y < 0).sum()),
        'heads_planned': 60, 'heads_completed': sum(r['status'] == 'FROZEN_200_UPDATES' for r in head_records),
        'updates_per_head': 200, 'prediction_rows': len(held_predictions),
        'fixture_metric_rows': len(metrics), 'seed_summary_rows': len(summary),
        'paired_contrast_rows': len(contrasts), 'wall_seconds': time.monotonic() - started,
        'original_result_identity': 'Requires independent comparison; not asserted by this replay.',
        'raw_labels_or_extraction_verified': False, 'new_validation_claimed': False,
        'full_CCPV_reproduction': False, 'public_release': False,
    })
    freeze(out, 'result_freeze.json')
    print('COMPLETE_FIXED_CACHE_LOFO60', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.features, args.output)
