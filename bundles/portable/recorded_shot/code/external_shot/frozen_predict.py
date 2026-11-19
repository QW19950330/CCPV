"""Frozen player-only inference and keyed scoring; no fitting or data acquisition.

Only the manifest and its 15 adjacent, hash-pinned head NPZs are read. Paths in
the manifest are provenance: relocated copies use the same adjacent filenames.
Input X is (rows, 2, 16), variants [200ms_shared, OLS1s_shared]. Its first 13
columns must be finite and exactly equal between variants. Nonfinite motion
features are mean-imputed with each frozen head's training mean.

Row identity is the exact canonical string ``fixture|half|g``. Caller must retain
input-eligible rows even if future label-clock support is unknown; labels then
carry -1. The function does not determine eligibility or read event records.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Sequence

import numpy as np

ARMS = ('G_shared', 'GM200_shared', 'GM_OLS1s')
SEEDS = tuple(range(5))
METRICS = ('logloss', 'brier', 'signed_bias', 'absolute_bias')
MANIFEST_SHA256 = '38e92dfc400a851a4a2c7668b0f5fde7c745dcf2c374227b5cb6fefbd4cb2340'
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / 'CCPV_player_only_deployment_20261118/result01/heads/head_freeze.json'
LOGLOSS_CLIP = 1e-15
_KEY = re.compile(r'([^|\s]+)\|([12])\|([1-9][0-9]*)\Z')


class FrozenContractError(ValueError):
    """A frozen-file, input, identity, or scoring contract failed."""


def _require(condition, message):
    if not condition:
        raise FrozenContractError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _readonly(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


def make_row_key(fixture: str, half: int, g: int) -> str:
    _require(type(fixture) is str and fixture and not re.search(r'[|\s]', fixture), 'invalid fixture identifier')
    _require(type(half) is int and half in (1, 2), 'half must be integer 1 or 2')
    _require(type(g) is int and g >= 1, 'g must be a positive integer')
    return f'{fixture}|{half}|{g}'


def _keys(keys):
    _require(not isinstance(keys, (str, bytes)), 'row_keys must be a sequence')
    keys = tuple(keys)
    _require(all(type(k) is str and _KEY.fullmatch(k) for k in keys), 'noncanonical row key; expected fixture|half|g')
    _require(len(keys) == len(set(keys)), 'duplicate row keys')
    return keys


@dataclass(frozen=True)
class FrozenHead:
    arm: str
    seed: int
    variant: int
    w: np.ndarray
    b: float
    mean: np.ndarray
    scale: np.ndarray
    sha256: str


@dataclass(frozen=True)
class FrozenBundle:
    heads: tuple[FrozenHead, ...]
    prior: float
    manifest_sha256: str


@dataclass(frozen=True)
class PredictionBatch:
    row_keys: tuple[str, ...]
    probabilities: np.ndarray  # N x 3 arms x 5 seeds, exact ARMS/SEEDS order
    prior: np.ndarray          # N deterministic, not five independent fits
    manifest_sha256: str


def load_frozen_heads(manifest_path=DEFAULT_MANIFEST, *, expected_manifest_sha256=MANIFEST_SHA256) -> FrozenBundle:
    """Verify the supplied manifest pin and all 15 frozen files, then load heads.

    Changing expected_manifest_sha256 must be an explicit caller decision. No
    unpinned mode exists. The training cohort, plan, source features and saved
    predictions are never opened. History/state checks establish structure and
    200 saved updates, not an optimizer replay or new training certification.
    """
    path = Path(manifest_path)
    _require(type(expected_manifest_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256), 'a SHA-256 manifest pin is required')
    raw = path.read_bytes()
    _require(_sha(raw) == expected_manifest_sha256, 'manifest SHA-256 mismatch')
    manifest = json.loads(raw)
    _require(manifest.get('status') == 'FROZEN_ALL_15_BEFORE_READOUT', 'manifest is not frozen')
    prior = manifest.get('prior')
    _require(type(prior) in (int, float) and np.isfinite(prior) and 0 < prior < 1, 'invalid fixed prior')
    records = manifest.get('heads')
    _require(type(records) is list and len(records) == 15, 'exactly 15 heads required')
    expected = {(a, s) for a in ARMS for s in SEEDS}
    pairs = []
    for rec in records:
        _require(type(rec) is dict, 'head record must be an object')
        _require(type(rec.get('seed')) is int, 'seed must be an integer')
        pairs.append((rec.get('arm'), rec['seed']))
    _require(len(set(pairs)) == 15 and set(pairs) == expected, 'incomplete or duplicate arm-seed grid')
    loaded, scalers, reference_indices = {}, {}, None
    required_keys = {'w', 'b', 'initial_w', 'initial_b', 'adam_m', 'adam_v', 'adam_step', 'history', 'final', 'mean', 'scale', 'train_indices', 'native_variant'}
    for rec, pair in zip(records, pairs):
        arm, seed = pair
        d = 13 if arm == 'G_shared' else 16
        variant = 1 if arm == 'GM_OLS1s' else 0
        _require(rec.get('parameters') == d + 1 and rec.get('train_n') == 9483, 'parameter or training-row metadata mismatch')
        _require(rec.get('status') == 'FROZEN_200_NATIVE_UPDATES', 'head status is not 200 frozen updates')
        filename = f'{arm}_s{seed}.npz'
        _require(type(rec.get('path')) is str and PureWindowsPath(rec['path']).name == filename, 'unexpected head filename')
        head_raw = (path.parent / filename).read_bytes()
        _require(len(head_raw) == rec.get('bytes'), f'head size mismatch: {filename}')
        _require(_sha(head_raw) == rec.get('sha256'), f'head SHA-256 mismatch: {filename}')
        try:
            with np.load(io.BytesIO(head_raw), allow_pickle=False) as z:
                _require(set(z.files) == required_keys and len(z.files) == len(required_keys), 'unexpected NPZ arrays')
                a = {k: z[k] for k in z.files}
        except (OSError, TypeError, ValueError) as exc:
            raise FrozenContractError(f'invalid frozen NPZ: {filename}: {exc}') from exc
        shapes = {'w': (d,), 'b': (), 'initial_w': (d,), 'initial_b': (), 'adam_m': (d+1,), 'adam_v': (d+1,), 'history': (200, 6), 'final': (4,), 'mean': (d,), 'scale': (d,)}
        for k, shape in shapes.items():
            _require(a[k].shape == shape and a[k].dtype == np.dtype('float64'), f'{filename}: {k} shape/dtype mismatch')
            _require(np.isfinite(a[k]).all(), f'{filename}: nonfinite {k}')
        for k, val in [('adam_step', 200), ('native_variant', variant)]:
            _require(a[k].shape == () and a[k].dtype == np.dtype('int32') and a[k].item() == val, f'{filename}: {k} mismatch')
        idx = a['train_indices']
        _require(idx.shape == (9483,) and idx.dtype == np.dtype('int64'), 'training-index shape/dtype mismatch')
        _require(np.all(idx >= 0) and np.all(np.diff(idx) > 0), 'training indices must be strictly increasing')
        if reference_indices is None:
            reference_indices = idx.copy()
        _require(np.array_equal(idx, reference_indices), 'training cohort differs between frozen heads')
        _require(np.all(a['scale'] > 0), 'nonpositive frozen scale')
        _require(np.all(a['adam_v'] >= 0), 'negative Adam second moment')
        _require(np.array_equal(a['history'][:, 0], np.arange(1, 201, dtype=np.float64)), 'history does not contain steps 1 through 200')
        _require(abs(float(a['initial_b']) - float(np.log(prior) - np.log1p(-prior))) <= 1e-14, 'initial intercept differs from fixed prior')
        if arm not in scalers:
            scalers[arm] = (a['mean'].copy(), a['scale'].copy())
        _require(np.array_equal(a['mean'], scalers[arm][0]) and np.array_equal(a['scale'], scalers[arm][1]), 'scaler differs between seeds')
        loaded[pair] = FrozenHead(arm, seed, variant, _readonly(a['w']), float(a['b']), _readonly(a['mean']), _readonly(a['scale']), rec['sha256'])
    for arm in ARMS[1:]:
        _require(np.array_equal(scalers[arm][0][:13], scalers['G_shared'][0]) and np.array_equal(scalers[arm][1][:13], scalers['G_shared'][1]), 'common first-13 frozen scalers differ')
    return FrozenBundle(tuple(loaded[(a, s)] for a in ARMS for s in SEEDS), float(prior), expected_manifest_sha256)


def stable_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    _require(not np.isnan(z).any(), 'NaN logit')
    return np.exp(-np.logaddexp(0.0, -z))


def predict_frozen(bundle: FrozenBundle, X, row_keys: Sequence[str]) -> PredictionBatch:
    """Predict input-eligible rows only, without looking at labels or future data."""
    keys = _keys(row_keys)
    _require(isinstance(bundle, FrozenBundle), 'FrozenBundle required')
    _require(len(bundle.heads) == 15 and tuple((h.arm, h.seed) for h in bundle.heads) == tuple((a, s) for a in ARMS for s in SEEDS), 'bundle arm-seed grid changed')
    original = np.asarray(X)
    _require(original.dtype.kind in 'fiu', 'features must be real numeric values')
    X = np.asarray(original, dtype=np.float64)
    _require(X.shape == (len(keys), 2, 16), 'X must have shape (N, 2, 16) matching row_keys')
    _require(np.isfinite(X[:, :, :13]).all(), 'common first 13 features must be finite')
    _require(np.array_equal(X[:, 0, :13], X[:, 1, :13]), 'common first 13 features differ between variants')
    probabilities = np.empty((len(keys), 3, 5), dtype=np.float64)
    for h in bundle.heads:
        values = X[:, h.variant, :len(h.w)]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            transformed = (np.where(np.isfinite(values), values, h.mean) - h.mean) / h.scale
            logits = transformed @ h.w + h.b
        _require(np.isfinite(transformed).all() and np.isfinite(logits).all(), 'nonfinite transformed features or logits')
        probabilities[:, ARMS.index(h.arm), h.seed] = stable_sigmoid(logits)
    return PredictionBatch(keys, _readonly(probabilities), _readonly(np.full(len(keys), bundle.prior)), bundle.manifest_sha256)


def _metrics(p, y):
    clipped = np.clip(p, LOGLOSS_CLIP, 1.0 - LOGLOSS_CLIP)
    residual = p - y
    return {'logloss': float(np.mean(-(y*np.log(clipped) + (1-y)*np.log1p(-clipped)))),
            'brier': float(np.mean(residual**2)), 'signed_bias': float(np.mean(residual)),
            'absolute_bias': abs(float(np.mean(residual)))}


def _macro(fixtures):
    if not fixtures:
        return None
    models = {arm: [{m: float(np.mean([f['models'][arm][s][m] for f in fixtures])) for m in METRICS} for s in SEEDS] for arm in ARMS}
    return {'models': models, 'prior': {m: float(np.mean([f['prior'][m] for f in fixtures])) for m in METRICS}}


def _paired(macro):
    if macro is None:
        return None
    result = {}
    for name, a, b in [('primary_GM200_minus_G', 'GM200_shared', 'G_shared'), ('secondary_OLS_minus_GM200', 'GM_OLS1s', 'GM200_shared')]:
        result[name] = {}
        for metric in METRICS:
            differences = np.array([macro['models'][a][s][metric] - macro['models'][b][s][metric] for s in SEEDS], dtype=np.float64)
            result[name][metric] = {'seed_differences': differences.tolist(), 'mean': float(np.mean(differences)), 'sample_sd_ddof1': float(np.std(differences, ddof=1)), 'n_seeds': 5}
    return result


def score_frozen(predictions: PredictionBatch, labels: Sequence[dict], planned_fixtures: Sequence[str]):
    """Join exact row keys, score known labels, and retain missing-fixture NA.

    Each label contains {'row_key': canonical string, 'y': integer -1/0/1};
    extra audit fields are ignored. Key sets
    must match exactly; their order may differ. Unknown=-1 is excluded from
    metrics and explicitly counted. No pooled or class-weighted surrogate is
    substituted for equal-fixture macro metrics. If any planned fixture has no
    known labels, full_plan is None (JSON null/NA); an explicitly named available
    fixture diagnostic is reported separately. Bias is in probability units;
    absolute_bias is abs(mean(p-y)) within each fixture, then averaged by fixture.
    """
    _require(isinstance(predictions, PredictionBatch), 'PredictionBatch required')
    keys = _keys(predictions.row_keys)
    p = np.asarray(predictions.probabilities)
    prior = np.asarray(predictions.prior)
    _require(p.dtype == np.dtype('float64') and p.shape == (len(keys), 3, 5), 'prediction shape/dtype mismatch')
    _require(prior.dtype == np.dtype('float64') and prior.shape == (len(keys),), 'prior shape/dtype mismatch')
    _require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), 'invalid prediction probability')
    _require(np.isfinite(prior).all() and np.all((prior >= 0) & (prior <= 1)), 'invalid prior probability')
    _require(not len(prior) or np.all(prior == prior[0]), 'prior must be a single fixed constant')
    _require(not isinstance(planned_fixtures, (str, bytes)), 'planned_fixtures must be a sequence of identifiers')
    fixtures = tuple(planned_fixtures)
    _require(fixtures and len(set(fixtures)) == len(fixtures), 'planned fixtures must be nonempty and unique')
    _require(all(type(f) is str and f and not re.search(r'[|\s]', f) for f in fixtures), 'invalid planned fixture')
    label_keys, y_by_key = [], {}
    for row in labels:
        _require(type(row) is dict and {'row_key', 'y'} <= set(row), 'label requires row_key and y')
        k, y = row['row_key'], row['y']
        _require(type(y) is int and y in (-1, 0, 1), 'label y must be integer -1, 0, or 1')
        label_keys.append(k)
        y_by_key[k] = y
    label_keys = _keys(label_keys)
    _require(set(label_keys) == set(keys), 'prediction/label key sets differ; positional joining forbidden')
    y = np.array([y_by_key[k] for k in keys], dtype=np.int8)
    fixture_for_row = [k.split('|', 1)[0] for k in keys]
    _require(set(fixture_for_row) <= set(fixtures), 'prediction contains an unplanned fixture')
    per_fixture, available = [], []
    for fixture in fixtures:
        idx = np.array([i for i, f in enumerate(fixture_for_row) if f == fixture], dtype=np.int64)
        known = idx[y[idx] != -1]
        entry = {'fixture': fixture, 'n_input_eligible': len(idx), 'n_known': len(known), 'n_unknown_excluded': int(np.sum(y[idx] == -1)),
                 'n_positive': int(np.sum(y[known] == 1)), 'n_negative': int(np.sum(y[known] == 0)),
                 'metrics_status': 'KNOWN_LABEL_SUPPORT' if len(known) else 'NA_NO_KNOWN_LABEL_SUPPORT', 'models': None, 'prior': None}
        if len(known):
            truth = y[known].astype(np.float64)
            entry['models'] = {arm: [_metrics(p[known, ai, s], truth) for s in SEEDS] for ai, arm in enumerate(ARMS)}
            entry['prior'] = _metrics(prior[known], truth)
            available.append(entry)
        per_fixture.append(entry)
    diagnostic = _macro(available)
    complete = len(available) == len(fixtures)
    full = diagnostic if complete else None
    return {'status': 'COMPLETE_FOR_PLANNED_FIXTURES' if complete else 'INCOMPLETE_FIXTURE_SUPPORT',
            'row_join': 'EXACT_CANONICAL_KEYS_NOT_POSITION', 'arms': list(ARMS), 'seeds': list(SEEDS),
            'manifest_sha256': predictions.manifest_sha256, 'logloss_clip': LOGLOSS_CLIP,
            'bias_units': 'probability', 'absolute_bias_definition': 'per-fixture abs(mean(p-y)), then equal-fixture mean',
            'n_input_eligible': len(keys), 'n_known': int(np.sum(y != -1)), 'n_unknown_excluded': int(np.sum(y == -1)),
            'planned_fixture_count': len(fixtures), 'known_fixture_count': len(available),
            'coverage_status_definition': 'Complete means at least one known label in each planned fixture; caller must separately verify prediction keys equal the full two-half input-eligible ledger. This function cannot detect upstream omitted anchors.',
            'fixtures_without_known_support': [f['fixture'] for f in per_fixture if f['n_known'] == 0],
            'per_fixture': per_fixture,
            'macro_equal_fixture': {'full_plan': full, 'available_fixture_diagnostic': diagnostic},
            'paired_seed_differences': {'full_plan': _paired(full), 'available_fixture_diagnostic': _paired(diagnostic)},
            'interpretation_limit': 'These arithmetic metrics do not establish independent external validation, complete source coverage, or causal action value.'}
