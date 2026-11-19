"""Reconstruct all original motion LOFO90 heads and both fixed held conditions.

Inputs are explicitly supplied, hash-pinned authorized features and the frozen
coverage ledger. No raw extraction, external-label input or new comparison.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from training_math import NAMES, fit, normfit, transform
from scoring_math import ARMS, CONDITIONS, score_predictions, table

FEATURE_SHA256 = 'b5f94c95aa8d14d60a1b7d5600e2b6a6f090e81b4a64958ee0aa2f8dd8528d07'
FEATURE_BYTES = 2803791
COVERAGE_SHA256 = '0781f1789082190a58dab27304a613a8d75d6f1e6746a42b18f25ba1e8651e37'
COVERAGE_BYTES = 859428
ROLE = 'FIXED_CACHE_MOTION_LOFO90_NATIVE_TRAINING_TWO_CONDITION_RECONSTRUCTION'


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


def validate_coverage(path, X, eligible, matches, grid):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(grid), 'Coverage population changed.')
    for i, r in enumerate(rows):
        require(int(r['row']) == i and int(r['grid_index']) == int(grid[i]) and r['match_id'] == matches[i]
                and (r['eligible'] == 'True') == bool(eligible[i]), 'Coverage identity/eligibility mismatch.')
        for ci, col, field in [(0, 6, 'current_count'), (0, 7, 'six_point_history_count'),
                               (0, 8, 'radial_native_count'), (1, 8, 'radial_grid_count')]:
            value = X[ci, 0, i, col]
            if np.isfinite(value):
                require(r[field] != '' and float(r[field]) == float(value), 'Coverage count/cache mismatch.')
            else:
                require(int(r['current_count']) == 0 and r[field] in ('', '0'), 'Unexpected missing coverage count.')
    e = [r for r in rows if r['eligible'] == 'True']
    return {
        'role': 'AGGREGATION_OF_FROZEN_COVERAGE_INPUT_NOT_RAW_HISTORY_REEXTRACTION',
        'candidates': len(rows), 'eligible': len(e),
        'old_motion_person_windows': int(sum(float(r['old_200ms_count']) for r in e)),
        'new_motion_person_windows': sum(int(r['six_point_history_count']) for r in e),
        'lost_person_windows': int(sum(float(r['old_200ms_count']) - int(r['six_point_history_count']) for r in e)),
        'windows_with_less_motion': sum(float(r['old_200ms_count']) > int(r['six_point_history_count']) for r in e),
        'windows_all_motion_missing': sum(int(r['six_point_history_count']) == 0 for r in e),
        'radial_changed_eligible': sum(r['radial_native_count'] != r['radial_grid_count'] for r in e),
        'radial_changed_all': sum(r['radial_native_count'] != r['radial_grid_count'] for r in rows),
    }


def run(features_path, coverage_path, output_path):
    started = time.monotonic()
    features_path, coverage_path, out = (Path(p).resolve() for p in (features_path, coverage_path, output_path))
    require(not out.exists(), 'Create-only output directory already exists.')
    for p, size, sha in [(features_path, FEATURE_BYTES, FEATURE_SHA256), (coverage_path, COVERAGE_BYTES, COVERAGE_SHA256)]:
        require(p.is_file() and p.stat().st_size == size and digest(p) == sha, 'Frozen input identity mismatch.')
    with np.load(features_path, allow_pickle=False) as z:
        require(set(z.files) == {'X', 'y', 'eligible', 'match', 'grid_index', 'conditions', 'motion_variants', 'feature_names'},
                'Unexpected cache schema.')
        require(z['feature_names'].tolist() == NAMES and z['conditions'].tolist() == list(CONDITIONS)
                and z['motion_variants'].tolist() == ['200ms_shared', 'OLS1s_shared'], 'Feature/condition definition changed.')
        X, Y, eligible = z['X'].copy(), z['y'].copy(), z['eligible'].copy()
        matches, grid = z['match'].copy(), z['grid_index'].copy()
    require(X.shape == (2, 2, 11542, 16) and X.dtype == np.float64, 'Cache shape or dtype changed.')
    require(Y.shape == eligible.shape == matches.shape == grid.shape == (11542,), 'Candidate vector shape mismatch.')
    require(Y.dtype == np.int32 and eligible.dtype == bool and grid.dtype == np.int32, 'Label/qualification/grid dtype changed.')
    require(int(eligible.sum()) == 9483 and int(Y[eligible].sum()) == 363 and int((Y < 0).sum()) == 2059
            and set(Y.tolist()) == {-1, 0, 1}, 'Original candidate population changed.')
    require(np.isfinite(X[:, :, :, :13][:, :, eligible]).all(), 'Original eligible shared features unavailable.')
    require(np.array_equal(X[:, 0, :, :13], X[:, 1, :, :13], equal_nan=True), 'Within-condition shared features changed.')
    scope = list(dict.fromkeys(matches.tolist()))
    require(len(scope) == 6 and np.array_equal(np.concatenate([np.flatnonzero(matches == h) for h in scope]),
                                             np.arange(len(matches))), 'Original six grouped folds changed.')
    coverage_summary = validate_coverage(coverage_path, X, eligible, matches, grid)
    out.mkdir(parents=True, exist_ok=False)
    heads = out / 'heads'
    heads.mkdir()

    def guard():
        require(not (out / 'PAUSE_REQUESTED').exists(), 'Pause requested; completed outputs retained.')
        require(time.monotonic() - started < 1800, 'Fixed reconstruction exceeded 1800 seconds.')
        require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < 536870912, '512 MiB output budget exceeded.')

    save(out / 'run_identity.json', {
        'role': ROLE, 'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'numpy': np.__version__,
        'features_sha256': FEATURE_SHA256, 'coverage_sha256': COVERAGE_SHA256,
        'folds': 6, 'seeds': list(range(5)), 'arms': list(ARMS), 'held_conditions': list(CONDITIONS),
        'training_condition': 'native only', 'updates_per_head': 200, 'CPU_threads': 1,
        'source_sha256': digest(__file__), 'math_sha256': digest(Path(__file__).with_name('training_math.py')),
        'scoring_sha256': digest(Path(__file__).with_name('scoring_math.py')),
        'raw_history_extracted': False, 'external_inputs_read': False, 'new_validation': False, 'formal50': False,
    })
    with (out / 'history_coverage.csv').open('xb') as stream:
        stream.write(coverage_path.read_bytes())
    save(out / 'coverage_summary.json', coverage_summary)
    table(out / 'cached_candidate_status.csv', [dict(row=i, grid_index=int(grid[i]), match=str(matches[i]),
                                                    eligible=bool(eligible[i]), y=int(Y[i])) for i in range(len(X[0, 0]))])
    records = []
    for held in scope:
        tr = np.flatnonzero(eligible & (matches != held))
        te = np.flatnonzero(eligible & (matches == held))
        require(len(set(matches[tr])) == 5 and len(te) > 0 and len(set(Y[tr])) == 2, 'Unsupported original fold; no replacement or retry.')
        a = np.array([1 / (5 * np.sum(matches[tr] == m)) for m in matches[tr]])
        require(abs(a.sum() - 1) < 1e-12, 'Fold weights do not sum to one.')
        prior = float(a @ Y[tr])
        bias = float(np.log(prior / (1 - prior)))
        foldstats = []
        for ai, arm in enumerate(ARMS):
            vi = 1 if arm == 'GM_OLS1s' else 0
            d = 13 if arm == 'G_shared' else 16
            mu, sd = normfit(X[0, vi, tr, :d], a)
            Z = transform(X[0, vi, tr, :d], mu, sd)
            foldstats.append((mu, sd))
            if ai:
                require(np.array_equal(mu[:13], foldstats[0][0]) and np.array_equal(sd[:13], foldstats[0][1]),
                        'Original first-thirteen-column normalizers differ.')
            npz(heads / f'{held}_{arm}_fold.npz', train_indices=tr, held_indices=te, weights=a,
                mean=mu, scale=sd, prior=np.array(prior), native_variant=np.array(vi, dtype=np.int32))
            for seed in range(5):
                guard()
                initial = np.random.default_rng(20261118 + seed).normal(0, .01, 16)[:d]
                w, b, m, v, history, final = fit(Z, Y[tr], a, initial, bias)
                require(history.shape == (200, 6) and np.array_equal(history[:, 0], np.arange(1, 201)),
                        'Incomplete fixed optimizer history.')
                path = heads / f'{held}_s{seed}_{arm}.npz'
                npz(path, initial_w=initial, initial_b=np.array(bias), w=w, b=np.array(b),
                    adam_m=m, adam_v=v, adam_step=np.array(200, dtype=np.int32), history=history,
                    final=final, train_indices=tr, mean=mu, scale=sd)
                records.append(dict(held=held, seed=seed, arm=arm, status='FROZEN_NATIVE_ONLY_200_UPDATES',
                                    train_n=len(tr), held_n=len(te), path=path.name, sha256=digest(path)))
        print('FIT_FOLD_COMPLETE', len(records), flush=True)
    require(len(records) == 90, 'Missing original head statuses.')
    save(heads / 'head_records.json', records)
    head_freeze = freeze(heads, 'head_freeze.json')
    predictions = []
    # Both held conditions are read only after all ninety native-only heads freeze.
    for held in scope:
        te = np.flatnonzero(eligible & (matches == held))
        with np.load(heads / f'{held}_G_shared_fold.npz', allow_pickle=False) as z:
            prior = float(z['prior'])
        for ci, condition in enumerate(CONDITIONS):
            predictions.extend(dict(row=int(i), grid_index=int(grid[i]), held=held, condition=condition,
                                    arm='PREVALENCE', seed=-1, y=int(Y[i]), p=float(prior)) for i in te)
            for rec in (r for r in records if r['held'] == held):
                path = heads / rec['path']
                require(digest(path) == rec['sha256'], 'Head changed after freeze.')
                vi = 1 if rec['arm'] == 'GM_OLS1s' else 0
                with np.load(path, allow_pickle=False) as z:
                    ZZ = transform(X[ci, vi, te, :len(z['w'])], z['mean'], z['scale'])
                    logits = ZZ @ z['w'] + z['b']
                    pp = np.exp(-np.logaddexp(0, -logits))
                require(np.isfinite(pp).all() and np.all((pp > 0) & (pp < 1)), 'Invalid held prediction.')
                predictions.extend(dict(row=int(i), grid_index=int(grid[i]), held=held, condition=condition,
                                        arm=rec['arm'], seed=rec['seed'], y=int(Y[i]), p=float(p)) for i, p in zip(te, pp))
    table(out / 'predictions.csv', predictions)
    save(out / 'prediction_freeze.json', {'head_freeze': head_freeze, 'predictions_sha256': digest(out / 'predictions.csv'),
                                         'role': ROLE, 'all_conditions_frozen_before_scoring': True})
    fm, summary, paired, fixture_contrasts, interaction = score_predictions(scope, predictions, out)
    guard()
    save(out / 'completion.json', {
        'status': 'COMPLETED_FIXED_CACHE_MOTION_LOFO90_RECONSTRUCTION', 'role': ROLE,
        'candidate_windows': 11542, 'eligible_windows': int(eligible.sum()), 'positive_windows': int(Y[eligible].sum()),
        'unknown_labels_retained': int((Y < 0).sum()), 'heads_completed': len(records), 'updates_per_head': 200,
        'training_conditions': ['native'], 'held_conditions': list(CONDITIONS), 'prediction_rows': len(predictions),
        'fixture_metric_rows': len(fm), 'seed_summary_rows': len(summary), 'paired_contrast_rows': len(paired),
        'fixture_contrast_rows': len(fixture_contrasts), 'degradation_interaction_rows': len(interaction),
        'wall_seconds': time.monotonic() - started, 'coverage_source': 'Pinned frozen input, copied and aggregated; not regenerated.',
        'original_result_identity': 'Requires independent comparison; not asserted by this producer.',
        'raw_history_or_labels_reproduced': False, 'new_validation': False, 'full_CCPV_reproduction': False, 'public_release': False,
    })
    freeze(out, 'result_freeze.json')
    print('COMPLETE_FIXED_CACHE_MOTION_LOFO90', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--coverage', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.features, arguments.coverage, arguments.output)
