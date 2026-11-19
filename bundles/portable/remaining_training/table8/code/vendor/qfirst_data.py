"""Fixed six-exposed-development-match, cached-prefix-only DTO boundary."""
from __future__ import annotations
import numpy as np
from qfirst_io import P, BINDING, read, selected, sha

# Labels are decoded for fold-restricted training; held labels are never an
# argument to fitting/prediction. Scoring receives them only after readout lock.
PREFIX_KEYS = ('X0', 'Z_cv', 'match_id', 'anchor_id', 'fixed_team_A', 'eligible_recorded10')
FEATURE_KEYS = ('anchor_id', 'supported', 'Early26', 'X2', 'DirectUnion36')


def dataset():
    b = read(P / BINDING)
    required = {'prefix_data', 'CV', *(f'D_s{s}' for s in range(5))}
    if set(b['assets']) != required:
        raise ValueError('Only the six fixed cached-prefix bundles and fixed label registry may be bound')
    for r in b['assets'].values():
        if sha(r['path']) != r['sha256']:
            raise ValueError('Bound fixed asset changed: ' + r['path'])
    a = selected(b['assets']['prefix_data']['path'], PREFIX_KEYS)
    bundles = {'D': [], 'CV': None}
    supported = None
    for name in ['CV'] + [f'D_s{s}' for s in range(5)]:
        f = selected(b['assets'][name]['path'], FEATURE_KEYS)
        if not np.array_equal(a['anchor_id'], f['anchor_id']):
            raise ValueError('Cached prefix anchor alignment changed')
        checks = ((f['Early26'][:, :10], a['X0']),
                  (f['Early26'][:, 10:18], a['Z_cv'][:, 10:]),
                  (f['DirectUnion36'][:, :26], f['Early26']),
                  (f['DirectUnion36'][:, 26:], f['X2']))
        if not all(np.array_equal(x, y, equal_nan=True) for x, y in checks):
            raise ValueError('Fixed cachedRaw36 composition changed')
        if supported is None:
            supported = f['supported']
        elif not np.array_equal(supported, f['supported']):
            raise ValueError('Fixed support mask differs across frontends')
        if name == 'CV':
            bundles['CV'] = f['DirectUnion36']
        else:
            bundles['D'].append(f['DirectUnion36'])
    bundles['D'] = np.stack(bundles['D'])
    ix = np.flatnonzero(a['eligible_recorded10'])
    matches = sorted(set(a['match_id'][ix].astype(str)))
    if len(a['anchor_id']) != 13372 or len(ix) != 5344 or len(matches) != 6:
        raise ValueError('Fixed exposed development cohort changed')
    return a, bundles, supported, ix, matches, b


def features(bundles, frontend, seed, rows):
    if frontend == 'D':
        return bundles['D'][seed, rows]
    if frontend == 'CV':
        return bundles['CV'][rows]
    raise ValueError('No frontend search')


def training_targets(a, b, rows, expected_training_matches, outer_held_match):
    """A fold whitelist is checked before the fixed target members are decoded.

    NPZ members store whole vectors; only whitelisted training slices are
    returned or inspected. Outer-held target slices are never returned here.
    """
    rows = np.asarray(rows, int)
    mids = set(a['match_id'][rows].astype(str))
    expected = set(map(str, expected_training_matches))
    if (mids != expected or str(outer_held_match) in mids or len(expected) != 5
            or not np.isin(rows, np.flatnonzero(a['eligible_recorded10'])).all()):
        raise ValueError('Targets requested outside the outer-training whitelist')
    y = selected(b['assets']['prefix_data']['path'], ('y_10', 'y_early2', 'y_late8'))
    out = tuple(np.asarray(y[k][rows], np.float64) for k in ('y_10', 'y_early2', 'y_late8'))
    if (not all(np.isin(v, [0, 1]).all() for v in out)
            or not np.array_equal(out[0], np.maximum(out[1], out[2]))):
        raise ValueError('Native training-only label dictionary changed')
    return out
