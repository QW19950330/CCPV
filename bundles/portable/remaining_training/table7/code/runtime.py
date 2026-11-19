"""Locked init-audit -> all240fits -> weightsfreeze -> readoutfreeze.

The fit and prediction APIs never accept outer-held labels. This module never
imports a trajectory model, runs new frontend predictions, or scores outcomes.
"""
from __future__ import annotations
import os
for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_env] = '1'
import argparse
from pathlib import Path
import numpy as np
import torch
import risk_models as H
from risk_io import P, read, put, save, sha, selected, checkpoint, guard, utc, process_identity
from risk_data import dataset, features, training_targets
from risk_math import equal_match_weights, logpair, compose, initial_q_discrepancy, ulp_distance

CELLS = tuple(f'{front}_{parameterization}_{objective}' for front in ('D', 'CV')
              for parameterization in H.PARAMETERIZATIONS for objective in H.OBJECTIVES)
ARMS = CELLS


def setup(expected, stage):
    guard(expected, stage)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    put(P / 'logs' / (stage + '_process_identity.json'), process_identity(stage, expected))


def folds(a, ix, matches):
    for fold, held in enumerate(matches):
        train = ix[a['match_id'][ix].astype(str) != held]
        test = ix[a['match_id'][ix].astype(str) == held]
        mids = sorted(set(matches) - {held})
        if set(a['match_id'][train].astype(str)) & set(a['match_id'][test].astype(str)):
            raise AssertionError('Outer-held match entered training')
        yield fold, held, train, test, mids


def init_audit(expected):
    setup(expected, 'init_audit')
    a, bundles, supported, ix, matches, binding = dataset()
    records = []
    for f, held, train, test, mids in folds(a, ix, matches):
        y10, y2, y8 = training_targets(a, binding, train, mids, held)
        w = equal_match_weights(a['match_id'][train], mids)
        for seed in range(5):
            for frontend in ('D', 'CV'):
                X = features(bundles, frontend, seed, train)
                cps, priors, archive = {}, {}, dict(training_rows=train, training_weights=w)
                for parameterization in H.PARAMETERIZATIONS:
                    model, norm, initial, e0, q0 = H.initialize(X, y10, y2, w, seed, parameterization, frontend)
                    cps[parameterization] = {**initial, **{'scaler_' + k: v for k, v in norm.items()}}
                    priors[parameterization] = [q0, e0, (q0 - e0) / (1 - e0)]
                    archive[parameterization + '_reference_priors_q_e_l'] = np.array(priors[parameterization])
                    archive.update({parameterization + '_' + k: v for k, v in initial.items()})
                    if parameterization == 'Direct':
                        archive.update({'scaler_' + k: v for k, v in norm.items()})
                    else:
                        for key in H.SCALER_KEYS:
                            if not np.array_equal(norm[key], archive['scaler_' + key]):
                                raise AssertionError('Cross-parameterization scaler changed')
                d, m = cps['Direct'], cps['Marginal']
                for key in ('hidden_weight', 'hidden_bias', 'output_weight'):
                    if not np.array_equal(d[key], m[key]):
                        raise AssertionError('Cross-parameterization random arrays differ')
                arrays, description = initial_q_discrepancy(H.predict(d, X), H.predict(m, X), w)
                archive.update(arrays)
                path = P / 'initialization' / f'o{f}_s{seed}_{frontend}_training_only.npz'
                save(path, **archive)
                records.append(dict(fold=f, held_match=held, seed=seed, frontend=frontend,
                                    training_matches=mids, training_rows=len(train),
                                    path=str(path.relative_to(P)), sha256=sha(path),
                                    reference_priors_q_e_l=priors,
                                    prior_float_hex={k: [float(v).hex() for v in values] for k, values in priors.items()},
                                    cross_parameterization_Early_prior_ulp_distance=ulp_distance(priors['Direct'][1], priors['Marginal'][1]),
                                    cross_parameterization_q_prior_ulp_distance=ulp_distance(priors['Direct'][0], priors['Marginal'][0]),
                                    cross_parameterization_Early_bias_values=[float(d['output_bias'][1]), float(m['output_bias'][1])],
                                    cross_parameterization_Early_bias_ulp_distance=ulp_distance(d['output_bias'][1], m['output_bias'][1]),
                                    cross_parameterization_Early_bias_absolute_difference=float(abs(d['output_bias'][1] - m['output_bias'][1])),
                                    **description))
        print('INITIALIZATION_TRAIN_ONLY_FOLD', f + 1, '/6', flush=True)
    if len(records) != 60:
        raise AssertionError('Initialization audit matrix incomplete')
    put(P / 'initialization_freeze.json', dict(
        status='ALL60_TRAIN_ONLY_INITIALIZATION_AUDITS_FROZEN_BEFORE_ANY_UPDATE', utc=utc(),
        implementation_lock_sha256=expected, records=records, real_model_updates=0,
        held_predictions=0, held_labels_used=0, original_Early_training_prior_retained=True,
        discrepancy_not_used_to_change_initialization=True, no_initial_q_function_match_claim=True))


def replay_registry(binding):
    out = {}
    for record in binding['source_metadata'] + [binding['parent_binding']]:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Bound inherited source/metadata changed')
    for source in binding['replay_sources']:
        for key in ('final_weights_lock', 'readout_freeze'):
            r = source[key]
            if sha(r['path']) != r['sha256']:
                raise ValueError('Old bound replay metadata changed')
        old_lock = read(source['final_weights_lock']['path'])
        old_records = {str((Path(source['final_weights_lock']['path']).parent / r['path']).resolve()): r
                       for r in old_lock['records']}
        old_freeze = read(source['readout_freeze']['path'])
        if old_freeze['readouts_sha256'] != source['readout_bundle']['sha256']:
            raise ValueError('Historical readout binding differs from frozen metadata')
        if source['final_weights_lock']['sha256'] not in [old_freeze.get('final_weights_lock_sha256'), old_freeze.get('weights_lock_sha256')]:
            raise ValueError('Historical readout/weights locks do not refer to one another')
        for r in source['records']:
            old = old_records.get(str(Path(r['path']).resolve()))
            if old is None or old['sha256'] != r['sha256'] or old['history_sha256'] != r['history_sha256']:
                raise ValueError('Historical checkpoint binding differs from frozen registry')
            if old.get('arm', old.get('head')) != source['old_arm']:
                raise ValueError('Historical checkpoint belongs to a different arm')
            out[(source['frontend'], source['parameterization'], r['fold'], r['seed'])] = r
    if len(out) != 120:
        raise ValueError('Exactly 120 old auxiliary checkpoint records are required')
    return out


def exact_replay(cp, record, parameterization, frontend):
    if sha(record['path']) != record['sha256'] or sha(record['history']) != record['history_sha256']:
        raise ValueError('Frozen old checkpoint/history changed')
    old = checkpoint(record['path'])
    keys = list(H.STATE_KEYS) + ['initial_' + k for k in H.STATE_KEYS]
    keys += ['scaler_' + k for k in H.SCALER_KEYS]
    keys += [prefix + k for prefix in ('adam_step_', 'adam_exp_avg_', 'adam_exp_avg_sq_') for k in H.STATE_KEYS]
    keys += ['training_initial_rows', 'held_rows', 'training_initial_weights', 'training_weights', 'training_seed', 'updates']
    for key in keys:
        if not np.array_equal(cp[key], old[key]):
            difference = float(np.max(np.abs(np.asarray(cp[key], float) - np.asarray(old[key], float))))
            raise AssertionError('AUXILIARY_EXACT_REPLAY_FAILED ' + key + ' maxabs=' + str(difference))
    target_columns = 2 if parameterization == 'Direct' and frontend == 'D' else 3
    if not np.array_equal(cp['training_labels'][:, :target_columns], old['training_labels']):
        raise AssertionError('Auxiliary replay training labels differ')
    return dict(old_checkpoint_path=record['path'], old_checkpoint_sha256=record['sha256'],
                exact_fields=keys + ['training_labels'], bitwise_exact=True,
                interpretation='HISTORICAL_AUXILIARY_REPLAY_NOT_NEW_CONFIRMATION')


def fit_all(expected):
    setup(expected, 'fit')
    if (P / 'models').exists() or (P / 'final_weights_lock.json').exists():
        raise FileExistsError('No automatic restart/overwrite of a fixed study run')
    init = read(P / 'initialization_freeze.json')
    if init['implementation_lock_sha256'] != expected or len(init['records']) != 60:
        raise ValueError('All fixed initialization audits must freeze before fit')
    initial_records = {(r['fold'], r['seed'], r['frontend']): r for r in init['records']}
    for r in init['records']:
        if sha(P / r['path']) != r['sha256']:
            raise ValueError('Frozen training-only initialization audit changed')
    a, bundles, supported, ix, matches, binding = dataset()
    old_registry = replay_registry(binding)
    records, fold_records, pairs, replays = [], [], [], []
    for f, held, train, test, mids in folds(a, ix, matches):
        y10, y2, y8 = training_targets(a, binding, train, mids, held)
        w = equal_match_weights(a['match_id'][train], mids)
        fold_records.append(dict(fold=f, held_match=held, training_matches=mids,
                                 training_rows=train.tolist(), held_rows=test.tolist()))
        for seed in range(5):
            for frontend in ('D', 'CV'):
                X = features(bundles, frontend, seed, train)
                rinit = initial_records[(f, seed, frontend)]
                init_keys = tuple(p + '_' + k for p in H.PARAMETERIZATIONS for k in H.STATE_KEYS)
                ia = selected(P / rinit['path'], init_keys + tuple('scaler_' + k for k in H.SCALER_KEYS))
                for parameterization in H.PARAMETERIZATIONS:
                    previous = None
                    for objective in ('WithEarlyBCE', 'NoEarlyBCE'):
                        cp, hist = H.fit(X, y10, y2, y8, w, seed, parameterization, objective, frontend)
                        for key in H.STATE_KEYS:
                            if not np.array_equal(cp['initial_' + key], ia[parameterization + '_' + key]):
                                raise AssertionError('Original fixed initialization changed at fit')
                        for key in H.SCALER_KEYS:
                            if not np.array_equal(cp['scaler_' + key], ia['scaler_' + key]):
                                raise AssertionError('Original fixed scaler changed at fit')
                        cell = f'{frontend}_{parameterization}_{objective}'
                        cp.update(training_initial_rows=train, training_loss_rows=train,
                                  training_initial_weights=w, held_rows=test, outer_held_match=np.array(held),
                                  training_match_ids=np.array(mids), cell=np.array(cell))
                        replay = None
                        if objective == 'WithEarlyBCE':
                            replay = exact_replay(cp, old_registry[(frontend, parameterization, f, seed)], parameterization, frontend)
                            replays.append(dict(fold=f, seed=seed, cell=cell, **replay))
                            previous = cp
                        else:
                            pair_keys = ['initial_' + k for k in H.STATE_KEYS] + ['scaler_' + k for k in H.SCALER_KEYS]
                            if not all(np.array_equal(cp[k], previous[k]) for k in pair_keys):
                                raise AssertionError('No/With objective pair lost bitwise identical initialization/scaler')
                            pairs.append(dict(fold=f, seed=seed, frontend=frontend, parameterization=parameterization,
                                              exact_fields=pair_keys, bitwise_exact=True))
                        stem = f'o{f}_s{seed}_{cell}'
                        path, hp = P / 'models' / (stem + '.npz'), P / 'models' / (stem + '_history.json')
                        save(path, **cp)
                        put(hp, hist)
                        records.append(dict(fold=f, seed=seed, frontend=frontend, parameterization=parameterization,
                                            objective=objective, cell=cell, path=str(path.relative_to(P)), sha256=sha(path),
                                            history=str(hp.relative_to(P)), history_sha256=sha(hp),
                                            updates=200, parameters=602, training_matches=mids, held_match=held,
                                            auxiliary_historical_exact_replay=replay is not None))
                        print('FIXED_MODEL_FROZEN', len(records), '/240', stem, flush=True)
        print('FROZEN_RISK_OBJECTIVE_FOLD', f + 1, '/6', flush=True)
    if len(records) != 240 or len(pairs) != 120 or len(replays) != 120:
        raise AssertionError('The locked fixed factorial matrix is incomplete')
    put(P / 'final_weights_lock.json', dict(
        status='ALL240_FIXED200_WEIGHTS_FROZEN_BEFORE_ANY_OUTER_READOUT', utc=utc(),
        implementation_lock_sha256=expected, data_binding_sha256=sha(P / 'data_binding.json'),
        initialization_freeze_sha256=sha(P / 'initialization_freeze.json'), records=records,
        folds=fold_records, exact_objective_pair_initialization=pairs, historical_auxiliary_exact_replays=replays,
        model_updates=48000, new_frontend_training=0, new_frontend_forward_calls=0,
        held_predictions=0, held_labels_used=0, no_automatic_restart=True))


def prefix_readout(cp, X, parameterization):
    """Only fixed prefix features; no labels or true-future arguments exist."""
    z = H.predict(cp, X)
    ep, en = logpair(z[:, 1])
    out = dict(early_logp=ep, early_log1mp=en, early_logits=z[:, 1])
    if parameterization == 'Direct':
        qp, qn = logpair(z[:, 0])
        out.update(q_logp=qp, q_log1mp=qn, q_logits=z[:, 0])
    elif parameterization == 'Marginal':
        lp, ln = logpair(z[:, 0])
        qp, qn = compose(ep, en, lp, ln)
        out.update(q_logp=qp, q_log1mp=qn, late_logp=lp, late_log1mp=ln, late_logits=z[:, 0])
    else:
        raise ValueError('Unknown fixed parameterization')
    return out


def readout(expected):
    setup(expected, 'readout')
    lock = read(P / 'final_weights_lock.json')
    if lock['implementation_lock_sha256'] != expected or len(lock['records']) != 240:
        raise ValueError('All 240 models must freeze before any outer readout')
    for r in lock['records']:
        if sha(P / r['path']) != r['sha256'] or sha(P / r['history']) != r['history_sha256']:
            raise ValueError('Frozen model or complete training history changed')
    a, bundles, supported, ix, matches, binding = dataset()
    position = {int(r): j for j, r in enumerate(ix)}
    arrays, checks = {}, []
    for r in lock['records']:
        cp = checkpoint(P / r['path'])
        rows = cp['held_rows'].astype(int)
        jj = np.array([position[int(row)] for row in rows])
        X = features(bundles, r['frontend'], r['seed'], rows)
        pred = prefix_readout(cp, X, r['parameterization'])
        repeated = prefix_readout(cp, X.copy(), r['parameterization'])
        if not all(np.array_equal(pred[k], repeated[k]) for k in pred):
            raise AssertionError('Prediction-only API repeat differs')
        for key, value in pred.items():
            name = r['cell'] + '__' + key
            if name not in arrays:
                arrays[name] = np.full((5, len(ix)), np.nan)
            arrays[name][r['seed'], jj] = value
        checks.append(dict(fold=r['fold'], seed=r['seed'], cell=r['cell'],
                           prefix_only_API_bitwise_repeat=True, no_label_or_future_prediction_argument=True))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise AssertionError('All fixed initial rows must receive all eight cell predictions')
    replay_checks = []
    for source in binding['replay_sources']:
        r = source['readout_bundle']
        if sha(r['path']) != r['sha256']:
            raise ValueError('Frozen historical readout bundle changed')
        cell = f"{source['frontend']}_{source['parameterization']}_WithEarlyBCE"
        fields = [name.split('__', 1)[1] for name in arrays if name.startswith(cell + '__')]
        keys = ('initial_indices', 'anchor_id', 'match_id', 'fixed_team_A')
        old = selected(r['path'], keys + tuple(source['readout_arm'] + '__' + k for k in fields))
        for key, value in [('initial_indices', ix), ('anchor_id', a['anchor_id'][ix]),
                           ('match_id', a['match_id'][ix]), ('fixed_team_A', a['fixed_team_A'][ix])]:
            if not np.array_equal(old[key], value):
                raise AssertionError('Historical replay metadata alignment differs')
        for key in fields:
            if not np.array_equal(arrays[cell + '__' + key], old[source['readout_arm'] + '__' + key]):
                raise AssertionError('AUXILIARY_RAW_READOUT_EXACT_REPLAY_FAILED ' + cell + '__' + key)
        replay_checks.append(dict(cell=cell, bitwise_exact_fields=fields, old_readout_sha256=r['sha256'],
                                  interpretation='REPLAY_NOT_NEW_CONFIRMATION'))
    fold_assignment = np.full(len(ix), -1, dtype=int)
    for fold in lock['folds']:
        jj = np.array([position[int(row)] for row in fold['held_rows']])
        if np.any(fold_assignment[jj] != -1):
            raise AssertionError('Duplicate outer-fold coverage')
        fold_assignment[jj] = fold['fold']
    if (fold_assignment < 0).any():
        raise AssertionError('Incomplete outer-fold coverage')
    packed = {key: np.stack([arrays[cell + '__' + key] for cell in CELLS], axis=1)
              for key in ('q_logp', 'q_log1mp', 'early_logp', 'early_log1mp')}
    save(P / 'frozen_readouts.npz', **packed, arms=np.array(CELLS), initial_indices=ix,
         anchor_id=a['anchor_id'][ix], match_id=a['match_id'][ix], fold_assignment=fold_assignment,
         fixed_team_A=a['fixed_team_A'][ix], prefix_supported=supported[ix])
    frozen_file = dict(path='frozen_readouts.npz', sha256=sha(P / 'frozen_readouts.npz'))
    put(P / 'readout_freeze.json', dict(
        status='ALL240_RAW_OUTER_READOUTS_FROZEN_BEFORE_ANY_HELD_LABEL_SCORING', utc=utc(),
        implementation_lock_sha256=expected, final_weights_lock_sha256=sha(P / 'final_weights_lock.json'),
        weights_lock_sha256=sha(P / 'final_weights_lock.json'),
        readouts_sha256=frozen_file['sha256'], prediction_checks=checks,
        auxiliary_raw_readout_exact_replay=replay_checks, held_label_score_calls=0,
        all_initial_rows=5344, eligible_rows=5344, arms=list(CELLS), cells=list(CELLS), seeds=5,
        frozen_readouts=frozen_file,
        files=[frozen_file,
               dict(path='final_weights_lock.json', sha256=sha(P / 'final_weights_lock.json')),
               dict(path='initialization_freeze.json', sha256=sha(P / 'initialization_freeze.json'))],
        new_frontend_forward_calls=0,
        NoEarly_Direct_Early_output='UNSUPERVISED_BY_EARLY_BCE_DESCRIPTIVE_ONLY',
        NoEarly_Marginal_components='NOT_IDENTIFIED_AS_TRUE_EARLY_LATE_PARTITION_DESCRIPTIVE_ONLY'))
    print('ALL240_READOUTS_FROZEN_WITHOUT_HELD_LABEL_SCORING', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['init_audit', 'fit', 'readout'])
    parser.add_argument('--implementation-lock-sha256', required=True)
    args = parser.parse_args()
    {'init_audit': init_audit, 'fit': fit_all, 'readout': readout}[args.stage](args.implementation_lock_sha256)
