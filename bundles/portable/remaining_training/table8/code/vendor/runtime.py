"""Fixed60 q-first models: train-only init audit -> weights -> readout freeze."""
from __future__ import annotations
import os
for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_env] = '1'
import argparse
from pathlib import Path
import numpy as np
import torch
import qfirst_models as H
from qfirst_io import P, read, put, save, sha, selected, checkpoint, guard, utc, process_identity
from qfirst_data import dataset, features, training_targets
from qfirst_math import equal_match_weights, qfirst_probabilities

ARMS = ('D_QFirstJoint', 'CV_QFirstJoint')
CELLS = ARMS


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


def reference_registry(binding):
    ref = binding['reference_factorial']
    for name in ('implementation_lock', 'experiment_plan', 'weights_lock', 'readout_freeze', 'evaluation_manifest'):
        record = ref[name]
        if sha(record['path']) != record['sha256']:
            raise ValueError('First-stage reference metadata changed')
    lock = read(ref['weights_lock']['path'])
    registry = {str((Path(ref['study_path']) / r['path']).resolve()): r for r in lock['records']}
    out = {}
    for r in binding['initial_DirectWith']:
        old = registry.get(str(Path(r['path']).resolve()))
        if (old is None or old['sha256'] != r['sha256'] or old['parameterization'] != 'Direct'
                or old['objective'] != 'WithEarlyBCE' or old['frontend'] != r['frontend']
                or old['seed'] != r['seed'] or old['fold'] != r['fold']):
            raise ValueError('Direct initialization reference differs from first-stage frozen registry')
        out[(r['fold'], r['seed'], r['frontend'])] = r
    if len(out) != 60:
        raise ValueError('Expected exactly 60 fixed Direct initialization references')
    return out


def init_audit(expected):
    setup(expected, 'init_audit')
    a, bundles, support, ix, matches, binding = dataset()
    references = reference_registry(binding)
    records = []
    for fold, held, train, test, mids in folds(a, ix, matches):
        y10, y2, y8 = training_targets(a, binding, train, mids, held)
        w = equal_match_weights(a['match_id'][train], mids)
        for seed in range(5):
            for frontend in ('D', 'CV'):
                X = features(bundles, frontend, seed, train)
                model, norm, initial, priors = H.initialize(X, y10, y2, w, seed, frontend)
                reference = references[(fold, seed, frontend)]
                if sha(reference['path']) != reference['sha256']:
                    raise ValueError('Frozen same-frontend Direct checkpoint changed')
                keys = tuple('initial_' + k for k in H.STATE_KEYS) + tuple('scaler_' + k for k in H.SCALER_KEYS)
                old = selected(reference['path'], keys + ('training_initial_rows', 'held_rows', 'training_initial_weights', 'reference_priors_q_e_l'))
                for key, value in [('training_initial_rows', train), ('held_rows', test), ('training_initial_weights', w)]:
                    if not np.array_equal(old[key], value):
                        raise AssertionError('Same-frontend Direct training/fold population differs')
                for key in H.SCALER_KEYS:
                    if not np.array_equal(norm[key], old['scaler_' + key]):
                        raise AssertionError('Same-frontend Direct scaler differs')
                for key in ('hidden_weight', 'hidden_bias', 'output_weight'):
                    if not np.array_equal(initial[key], old['initial_' + key]):
                        raise AssertionError('Same-frontend inherited random array differs')
                if not np.array_equal(initial['output_bias'][0], old['initial_output_bias'][0]):
                    raise AssertionError('Same-frontend Direct q initial bias differs')
                if not np.array_equal(np.asarray(priors[:2]), old['reference_priors_q_e_l'][:2]):
                    raise AssertionError('Same-frontend Direct original prior numeric path differs')
                cp = {**initial, **{'scaler_' + k: v for k, v in norm.items()}}
                initial_logits = H.predict(cp, X)
                old_logits = H.predict(old, X, initial=True)
                if not np.array_equal(initial_logits[:, 0], old_logits[:, 0]):
                    raise AssertionError('Same-frontend Direct initial q function differs on training rows')
                pred = qfirst_probabilities(initial_logits)
                path = P / 'initialization' / f'o{fold}_s{seed}_{frontend}_training_only.npz'
                save(path, **initial, **{'scaler_' + k: v for k, v in norm.items()},
                     training_rows=train, training_weights=w, reference_priors_q_e_r=np.array(priors),
                     q_logits=initial_logits[:, 0], r_logits=initial_logits[:, 1],
                     q_logp=pred['q_logp'], early_logp=pred['early_logp'],
                     old_Direct_q_logits=old_logits[:, 0])
                records.append(dict(fold=fold, held_match=held, seed=seed, frontend=frontend,
                    path=str(path.relative_to(P)), sha256=sha(path), training_matches=mids,
                    training_rows=len(train), reference_priors_q_e_r=list(priors),
                    reference_priors_float_hex=[float(v).hex() for v in priors],
                    initial_weighted_q=float(w @ np.exp(pred['q_logp'])),
                    initial_weighted_Early=float(w @ np.exp(pred['early_logp'])),
                    reference_Direct_checkpoint=reference['path'], reference_Direct_checkpoint_sha256=reference['sha256'],
                    exact_Direct_q_initial_function=True, exact_Direct_random_arrays_and_scaler=True,
                    row1_bias_has_new_conditional_semantics=True, no_full_two_output_initial_function_match_claim=True))
        print('QFIRST_INITIALIZATION_TRAIN_ONLY_FOLD', fold + 1, '/6', flush=True)
    if len(records) != 60:
        raise AssertionError('Initialization audit matrix incomplete')
    put(P / 'initialization_freeze.json', dict(
        status='ALL60_QFIRST_TRAIN_ONLY_INITIALIZATION_AUDITS_FROZEN_BEFORE_ANY_UPDATE', utc=utc(),
        implementation_lock_sha256=expected, records=records, real_model_updates=0, held_predictions=0,
        held_labels_used=0, Direct_q_initial_function_exact=True, full_initial_output_match_claim=False))


def fit_all(expected):
    setup(expected, 'fit')
    if (P / 'models').exists() or (P / 'final_weights_lock.json').exists():
        raise FileExistsError('No automatic restart or overwrite of the fixed pilot')
    init = read(P / 'initialization_freeze.json')
    if init['implementation_lock_sha256'] != expected or len(init['records']) != 60:
        raise ValueError('All 60 initialization audits must freeze before fit')
    initial_records = {(r['fold'], r['seed'], r['frontend']): r for r in init['records']}
    for r in init['records']:
        if sha(P / r['path']) != r['sha256']:
            raise ValueError('Frozen initialization audit changed')
    a, bundles, support, ix, matches, binding = dataset()
    reference_registry(binding)
    records, fold_records = [], []
    for fold, held, train, test, mids in folds(a, ix, matches):
        y10, y2, y8 = training_targets(a, binding, train, mids, held)
        w = equal_match_weights(a['match_id'][train], mids)
        fold_records.append(dict(fold=fold, held_match=held, training_matches=mids,
                                 training_rows=train.tolist(), held_rows=test.tolist()))
        for seed in range(5):
            for frontend in ('D', 'CV'):
                cp, history = H.fit(features(bundles, frontend, seed, train), y10, y2, w, seed, frontend)
                rinit = initial_records[(fold, seed, frontend)]
                ia = selected(P / rinit['path'], H.STATE_KEYS + tuple('scaler_' + k for k in H.SCALER_KEYS))
                for key in H.STATE_KEYS:
                    if not np.array_equal(cp['initial_' + key], ia[key]):
                        raise AssertionError('Fixed initialization changed at fit')
                for key in H.SCALER_KEYS:
                    if not np.array_equal(cp['scaler_' + key], ia['scaler_' + key]):
                        raise AssertionError('Fixed scaler changed at fit')
                cell = frontend + '_QFirstJoint'
                cp.update(training_initial_rows=train, training_loss_rows=train, training_initial_weights=w,
                          held_rows=test, training_match_ids=np.array(mids), outer_held_match=np.array(held), cell=np.array(cell))
                stem = f'o{fold}_s{seed}_{cell}'
                path, hp = P / 'models' / (stem + '.npz'), P / 'models' / (stem + '_history.json')
                save(path, **cp)
                put(hp, history)
                records.append(dict(fold=fold, seed=seed, frontend=frontend, cell=cell,
                                    path=str(path.relative_to(P)), sha256=sha(path),
                                    history=str(hp.relative_to(P)), history_sha256=sha(hp),
                                    updates=200, parameters=602, training_matches=mids, held_match=held))
                print('QFIRST_FIXED_MODEL_FROZEN', len(records), '/60', stem, flush=True)
    if len(records) != 60:
        raise AssertionError('Fixed q-first matrix incomplete')
    put(P / 'final_weights_lock.json', dict(
        status='ALL60_FIXED200_WEIGHTS_FROZEN_BEFORE_ANY_OUTER_READOUT', utc=utc(),
        implementation_lock_sha256=expected, data_binding_sha256=sha(P / 'data_binding.json'),
        initialization_freeze_sha256=sha(P / 'initialization_freeze.json'), records=records, folds=fold_records,
        exact_Direct_q_initialization_records=init['records'], model_updates=12000,
        new_frontend_training=0, new_frontend_forward_calls=0, held_predictions=0, held_labels_used=0,
        no_automatic_restart=True, no_full_initial_output_match_claim=True))


def prefix_readout(cp, X):
    """Only fixed prefix features. No training/outcome/future arguments exist."""
    return qfirst_probabilities(H.predict(cp, X))


def readout(expected):
    setup(expected, 'readout')
    lock = read(P / 'final_weights_lock.json')
    if lock['implementation_lock_sha256'] != expected or len(lock['records']) != 60:
        raise ValueError('All 60 models must freeze before any outer readout')
    for r in lock['records']:
        if sha(P / r['path']) != r['sha256'] or sha(P / r['history']) != r['history_sha256']:
            raise ValueError('Frozen q-first model or complete history changed')
    a, bundles, support, ix, matches, binding = dataset()
    position = {int(row): j for j, row in enumerate(ix)}
    arrays, checks = {}, []
    fields = ('q_logp', 'q_log1mp', 'early_logp', 'early_log1mp', 'r_logp', 'r_log1mp',
              'late_only_logp', 'late_logp', 'late_log1mp', 'category_logs', 'q_logits', 'r_logits')
    for r in lock['records']:
        cp = checkpoint(P / r['path'])
        rows = cp['held_rows'].astype(int)
        jj = np.array([position[int(row)] for row in rows])
        arm_index = ARMS.index(r['cell'])
        X = features(bundles, r['frontend'], r['seed'], rows)
        pred, repeated = prefix_readout(cp, X), prefix_readout(cp, X.copy())
        if not all(np.array_equal(pred[k], repeated[k]) for k in fields):
            raise AssertionError('Prefix-only q-first readout repeat differs')
        for key in fields:
            if key not in arrays:
                arrays[key] = np.full((5, 2, len(ix)) + pred[key].shape[1:], np.nan)
            arrays[key][r['seed'], arm_index, jj] = pred[key]
        checks.append(dict(fold=r['fold'], seed=r['seed'], cell=r['cell'],
                           prefix_only_API_bitwise_repeat=True, no_label_or_future_prediction_argument=True))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise AssertionError('Every fixed initial row must receive both q-first cell predictions')
    fold_assignment = np.full(len(ix), -1, dtype=int)
    for fold in lock['folds']:
        jj = np.array([position[int(row)] for row in fold['held_rows']])
        if np.any(fold_assignment[jj] != -1):
            raise AssertionError('Duplicate outer-fold coverage')
        fold_assignment[jj] = fold['fold']
    if (fold_assignment < 0).any():
        raise AssertionError('Incomplete outer-fold coverage')
    save(P / 'frozen_readouts.npz', **arrays, arms=np.array(ARMS), initial_indices=ix,
         anchor_id=a['anchor_id'][ix], match_id=a['match_id'][ix], fixed_team_A=a['fixed_team_A'][ix],
         prefix_supported=support[ix], fold_assignment=fold_assignment,
         category_order=np.array(['none10', 'Early2_priority', 'late8_without_Early2']))
    pin = dict(path='frozen_readouts.npz', sha256=sha(P / 'frozen_readouts.npz'))
    put(P / 'readout_freeze.json', dict(
        status='ALL60_RAW_OUTER_READOUTS_FROZEN_BEFORE_ANY_HELD_LABEL_SCORING', utc=utc(),
        implementation_lock_sha256=expected, weights_lock_sha256=sha(P / 'final_weights_lock.json'),
        final_weights_lock_sha256=sha(P / 'final_weights_lock.json'),
        readouts_sha256=pin['sha256'], frozen_readouts=pin,
        files=[pin, dict(path='final_weights_lock.json', sha256=sha(P / 'final_weights_lock.json')),
               dict(path='initialization_freeze.json', sha256=sha(P / 'initialization_freeze.json'))],
        arms=list(ARMS), seeds=5, eligible_rows=5344, prediction_checks=checks,
        category_order=['none10', 'Early2_priority', 'late8_without_Early2'],
        held_label_score_calls=0, new_frontend_forward_calls=0, true_Y10_or_Y2_prediction_gate=False))
    print('ALL60_QFIRST_READOUTS_FROZEN_WITHOUT_HELD_LABEL_SCORING', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['init_audit', 'fit', 'readout'])
    parser.add_argument('--implementation-lock-sha256', required=True)
    args = parser.parse_args()
    {'init_audit': init_audit, 'fit': fit_all, 'readout': readout}[args.stage](args.implementation_lock_sha256)
