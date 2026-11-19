"""Train-prefix-only extraction. No model, future label, score or held-row decoding.

A compressed ZIP stream may inflate skipped bytes. Only explicitly selected
training rows and their first ten frames are converted from bytes to arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
OLD = BASE / 'CCPV_consistent_dynamics_outcomes_pilot_20261024/forecasting'
EXPANSION = BASE / 'CCPV_data_expansion_20261022'
ARCHIVE = OLD / 'data/development.npz'
ARCHIVE_SHA = '6c1df9e712e1928408ea3c4129589befd9e4d94213761a20aeb624e59d5f3ed2'
TRAIN_IDS = tuple('skillcorner:' + x for x in (
    '1874553', '1886347', '1899585', '1953632', '1959846', '1996435',
    '1996436', '2006229', '2007721', '2010085', '2013725', '2016236'))
PREFIX = 10
PER_MATCH = 64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError('Unsupported NPY header version')
    if fortran or dtype.hasobject or not shape:
        raise ValueError('Only nonobject C-order arrays with a row axis')
    return shape, dtype


def discard(stream, count):
    while count:
        block = stream.read(min(int(count), 1 << 20))
        if not block:
            raise EOFError('NPY stream ended while skipping unselected bytes')
        count -= len(block)


def selected(path, key, indices, prefix_frames=None, access_log=None):
    """Materialize selected row prefixes only; never full mixed NpzFile arrays."""
    ix = np.asarray(indices, dtype=np.int64)
    if ix.ndim != 1 or len(np.unique(ix)) != len(ix):
        raise ValueError('Indices must be one-dimensional and unique')
    order = np.argsort(ix)
    wanted = ix[order]
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        shape, dtype = header(stream)
        if len(ix) and (wanted[0] < 0 or wanted[-1] >= shape[0]):
            raise IndexError('Selected row outside source shape')
        row_shape = shape[1:]
        if prefix_frames is not None:
            if len(shape) < 2 or not 0 < prefix_frames <= shape[1]:
                raise ValueError('Invalid prefix frame count')
            out_shape = (prefix_frames,) + shape[2:]
        else:
            out_shape = row_shape
        row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
        take_bytes = int(np.prod(out_shape, dtype=np.int64)) * dtype.itemsize
        result = np.empty((len(ix),) + out_shape, dtype=dtype)
        cursor = 0
        for j, row in enumerate(wanted):
            discard(stream, (int(row) - cursor) * row_bytes)
            block = stream.read(take_bytes)
            if len(block) != take_bytes:
                raise EOFError('Truncated selected prefix')
            result[order[j]] = np.frombuffer(block, dtype=dtype).reshape(out_shape)
            discard(stream, row_bytes - take_bytes)
            cursor = int(row) + 1
    if access_log is not None:
        access_log.append(dict(key=key, source_shape=list(shape),
                               rows_materialized=len(ix),
                               prefix_frames=prefix_frames,
                               array_bytes_materialized=int(result.nbytes),
                               skipped_suffix_bytes_per_selected_row=row_bytes - take_bytes))
    return result


def global_metadata(path, key, expected_shape):
    if key not in ('train_indices', 'teams'):
        raise ValueError('Global reads restricted to nontrajectory metadata')
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        shape, dtype = header(stream)
        if shape != expected_shape:
            raise ValueError('Unexpected global metadata shape')
        size = int(np.prod(shape)) * dtype.itemsize
        block = stream.read(size)
        if len(block) != size:
            raise EOFError('Truncated global metadata')
    return np.frombuffer(block, dtype=dtype).reshape(shape).copy()


def choose_training_rows(match_ids, source_indices, bundle_rows):
    if len(match_ids) != 768 or set(match_ids) != set(TRAIN_IDS):
        raise ValueError('Only all 768 rows from the fixed 12 training matches')
    if len(set(map(int, source_indices))) != 768:
        raise ValueError('Source window indices are not unique')
    chosen = []
    for mid in sorted(TRAIN_IDS):
        positions = np.flatnonzero(match_ids == mid)
        if len(positions) != 64:
            raise ValueError('Each fixed training match must have 64 candidate rows')
        order = positions[np.argsort(source_indices[positions], kind='stable')]
        chosen.extend(bundle_rows[order[:PER_MATCH]].tolist())
    return np.asarray(chosen, dtype=np.int64)


def plan():
    manifest_path = OLD / 'data/manifest.json'
    manifest = read(manifest_path)
    if manifest['development_sha256'] != ARCHIVE_SHA or set(manifest['train_matches']) != set(TRAIN_IDS):
        raise ValueError('Unexpected frozen source roles')
    pin_path = EXPANSION / 'artifacts/skillcorner_download_manifest.json'
    pins = {r['path']: r for r in read(pin_path)['files']}
    metadata = []
    for mid in TRAIN_IDS:
        native = mid.split(':')[1]
        rel = f'data/matches/{native}/{native}_match.json'
        pin = pins[rel]
        path = EXPANSION / 'raw/skillcorner' / rel
        metadata.append(dict(match_id=mid, path=str(path), sha256=pin['sha256'],
                             purpose='home/away team IDs and player ID-to-team roster only'))
    inputs = dict(archive=dict(path=str(ARCHIVE), sha256=ARCHIVE_SHA),
                  source_manifest=dict(path=str(manifest_path), sha256=sha(manifest_path)),
                  download_manifest=dict(path=str(pin_path), sha256=sha(pin_path)), metadata=metadata)
    write(ROOT / 'data_input_manifest.json', inputs)
    write(ROOT / 'data_plan.json', dict(
        status='FIXED_BEFORE_TRAINING_PREFIX_EXTRACTION', created_utc=now(),
        source_training_matches=list(TRAIN_IDS), training_source_rows=768,
        selection='Within each training match sort source_indices ascending; all sixty-four; no quality/outcome selection',
        windows=768, windows_per_match=PER_MATCH, history_frames=PREFIX, fps=10.,
        source_input='Original same-budget A/C/D training states+valid, not newer common-support replacement',
        allowed_global_metadata=['train_indices', 'teams'],
        allowed_row_metadata=['match_ids', 'source_indices', 'entity_ids', 'source_period'],
        allowed_prefix_fields=['states', 'valid', 'position_valid', 'source_frame_ids', 'frame_times_s'],
        future_values_materialized=False, development_rows_materialized=False,
        compression_caveat='ZIP skipping can inflate bytes; skipped future/held bytes are never converted to arrays',
        missing_identity='Keep row/slot with explicit UNKNOWN; no replacement window',
        operator_case_selection='Per retained window: roster-verified home/model-team0 players with original t0 full-state valid; lexical sort of exact source-ID strings; first two. Fewer than two is UNSUPPORTED, never replace window.',
        team_semantics='Provider roster home/away identity only, not attacking/defending/current possession',
        attack_direction='UNKNOWN_NOT_INFERRED', t0_possession='UNKNOWN_NOT_INFERRED',
        excluded_reads=['future state/target arrays', 'validation/test rows', 'raw tracking', 'event labels', 'prediction/score payloads'],
        outputs=['prepared/train_prefixes.npz', 'prepared/metadata.json', 'prepared/manifest.json'],
        fits=0, model_calls=0))


def synthetic():
    access = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'toy.npz'
        values = np.arange(8 * 30 * 3 * 4, dtype=np.float32).reshape(8, 30, 3, 4)
        with path.open('wb') as stream:
            np.savez_compressed(stream, states=values,
                                metadata=np.arange(8, dtype=np.int64),
                                train_indices=np.array([0, 1, 2, 3]))
        ix = np.array([3, 0, 2])
        observed = selected(path, 'states', ix, PREFIX, access)
        np.testing.assert_array_equal(observed, values[ix, :PREFIX])
        np.testing.assert_array_equal(selected(path, 'metadata', ix), ix)
        np.testing.assert_array_equal(global_metadata(path, 'train_indices', (4,)), np.arange(4))
        rejected = 0
        for bad in ([0, 0], [-1], [8]):
            try:
                selected(path, 'states', bad, PREFIX)
            except (ValueError, IndexError):
                rejected += 1
        assert rejected == 3
        assert observed.shape[1] == 10 and access[0]['skipped_suffix_bytes_per_selected_row'] == 20 * 3 * 4 * 4
    mids = np.repeat(np.array(sorted(TRAIN_IDS)), 64)
    source = np.tile(np.arange(64)[::-1], 12) + np.repeat(np.arange(12) * 1000, 64)
    rows = choose_training_rows(mids, source, np.arange(768))
    assert len(rows) == 768 and all(np.array_equal(source[rows[i * 64:(i + 1) * 64]], np.arange(64) + i * 1000) for i in range(12))
    write(ROOT / 'data_synthetic.json', dict(status='PASS_PREFIX_LOADER_AND_FIXED_SELECTION',
          source_sha256=sha(__file__), checks=8, access_log=access,
          real_training_rows_materialized=0, real_data_model_calls=0, fits=0))


def lock(protocol, protocol_sha256):
    protocol = Path(protocol).resolve()
    if protocol.parent != ROOT or sha(protocol) != protocol_sha256:
        raise ValueError('Explicit frozen protocol in this new root required')
    toy = read(ROOT / 'data_synthetic.json')
    assert toy['status'].startswith('PASS') and toy['source_sha256'] == sha(__file__)
    write(ROOT / 'data_source_lock.json', dict(status='LOCKED_BEFORE_768_TRAIN_PREFIX_READ',
          source_sha256=sha(__file__), plan_sha256=sha(ROOT / 'data_plan.json'),
          inputs_sha256=sha(ROOT / 'data_input_manifest.json'),
          synthetic_sha256=sha(ROOT / 'data_synthetic.json'),
          protocol_path=str(protocol), protocol_sha256=protocol_sha256))


def prepare(authorized):
    if not authorized:
        raise PermissionError('Explicit training-prefix extraction authorization required')
    guard = read(ROOT / 'data_source_lock.json')
    for path, key in [(Path(__file__), 'source_sha256'), (ROOT / 'data_plan.json', 'plan_sha256'),
                      (ROOT / 'data_input_manifest.json', 'inputs_sha256'),
                      (ROOT / 'data_synthetic.json', 'synthetic_sha256'),
                      (Path(guard['protocol_path']), 'protocol_sha256')]:
        if sha(path) != guard[key]:
            raise ValueError('Frozen source/protocol changed: ' + str(path))
    if (ROOT / 'prepared').exists():
        raise FileExistsError('Do not overwrite prepared evidence')
    pins = read(ROOT / 'data_input_manifest.json')
    for record in [pins['archive'], pins['source_manifest'], pins['download_manifest'], *pins['metadata']]:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Pinned input changed')
    ix = global_metadata(ARCHIVE, 'train_indices', (768,))
    if not np.array_equal(ix, np.arange(768)):
        raise ValueError('Frozen training row map changed')
    access = []
    mids = selected(ARCHIVE, 'match_ids', ix, access_log=access)
    source = selected(ARCHIVE, 'source_indices', ix, access_log=access)
    rows = choose_training_rows(mids, source, ix)
    a = {k: selected(ARCHIVE, k, rows, access_log=access)
         for k in ('match_ids', 'source_indices', 'entity_ids', 'source_period')}
    for old, new in [('states', 'history_states'), ('valid', 'history_valid'),
                     ('position_valid', 'history_position_valid'),
                     ('source_frame_ids', 'source_frame_ids'), ('frame_times_s', 'frame_times_s')]:
        a[new] = selected(ARCHIVE, old, rows, PREFIX, access)
    a['teams'] = global_metadata(ARCHIVE, 'teams', (23,))
    if not np.array_equal(a['teams'], np.array([0] * 11 + [1] * 11 + [2])):
        raise ValueError('Model team encoding changed; do not guess a replacement')
    if a['history_valid'].dtype != np.bool_ or a['history_states'].shape != (768, 10, 23, 4):
        raise ValueError('Unexpected prefix contract')
    assert np.all(a['history_valid'] <= a['history_position_valid'])
    assert np.isfinite(a['history_states'][a['history_valid']]).all()
    assert np.all(np.diff(a['source_frame_ids'], axis=1) == 1)
    np.testing.assert_allclose(np.diff(a['frame_times_s'], axis=1), .1, atol=1e-8, rtol=0)
    a['bundle_row_indices'] = rows
    a['history_seen'] = a['history_valid'].any(1)
    a['provider_team_ids'] = np.full((768, 23), -1, dtype=np.int64)
    a['identity_status'] = np.full((768, 23), 'UNKNOWN_EMPTY_SOURCE_ID', dtype='<U40')
    a['home_team_id'] = np.full(768, -1, dtype=np.int64)
    a['away_team_id'] = np.full(768, -1, dtype=np.int64)
    registry_summary = []
    for record in pins['metadata']:
        # JSON source is decoded; only the following identity fields are accessed.
        meta = read(record['path'])
        home, away = int(meta['home_team']['id']), int(meta['away_team']['id'])
        if home == away:
            raise ValueError('Ambiguous fixture team identity')
        registry = {}
        for player in meta['players']:
            if player.get('id') is None or player.get('team_id') is None:
                continue
            key, team = str(player['id']), int(player['team_id'])
            if key in registry or team not in (home, away):
                raise ValueError('Nonunique or foreign player roster identity')
            registry[key] = team
        taken = np.flatnonzero(a['match_ids'] == record['match_id'])
        for w in taken:
            a['home_team_id'][w], a['away_team_id'][w] = home, away
            present_ids = [str(x) for x in a['entity_ids'][w, :22] if str(x)]
            if len(present_ids) != len(set(present_ids)):
                raise ValueError('Nonunique source player entity ID within window')
            for slot, value in enumerate(a['entity_ids'][w, :22]):
                eid = str(value)
                if not eid:
                    continue
                if eid not in registry:
                    a['identity_status'][w, slot] = 'UNKNOWN_NOT_IN_SOURCE_ROSTER'
                    continue
                expected = home if int(a['teams'][slot]) == 0 else away
                if registry[eid] != expected:
                    raise ValueError('Source roster contradicts model home/away encoding')
                a['provider_team_ids'][w, slot] = registry[eid]
                a['identity_status'][w, slot] = 'VERIFIED_SOURCE_PLAYER_TEAM'
            ball_id = str(a['entity_ids'][w, 22])
            if ball_id and ball_id not in present_ids:
                a['identity_status'][w, 22] = 'EXPLICIT_MODEL_BALL_SOURCE_ID'
            else:
                a['identity_status'][w, 22] = 'UNKNOWN_BALL_SOURCE_ID'
        registry_summary.append(dict(match_id=record['match_id'], metadata_sha256=record['sha256'],
                                     home_team_id=home, away_team_id=away,
                                     player_roster_entries=len(registry), selected_windows=len(taken)))
    a['operator_candidate_slots'] = np.full((768, 2), -1, dtype=np.int64)
    a['operator_candidate_ids'] = np.full((768, 2), '', dtype=a['entity_ids'].dtype)
    a['operator_eligible_home_count'] = np.zeros(768, dtype=np.int64)
    a['operator_case_status'] = np.full(768, 'UNSUPPORTED_FEWER_THAN_TWO', dtype='<U48')
    for w in range(768):
        eligible = [j for j in range(22) if a['teams'][j] == 0
                    and a['history_valid'][w, -1, j]
                    and a['identity_status'][w, j] == 'VERIFIED_SOURCE_PLAYER_TEAM']
        eligible.sort(key=lambda j: str(a['entity_ids'][w, j]))
        a['operator_eligible_home_count'][w] = len(eligible)
        if len(eligible) >= 2:
            chosen = eligible[:2]
            a['operator_candidate_slots'][w] = chosen
            a['operator_candidate_ids'][w] = a['entity_ids'][w, chosen]
            a['operator_case_status'][w] = 'SUPPORTED_TWO_SOURCE_HOME_T0_VALID'
    output = ROOT / 'prepared'
    output.mkdir()
    target = output / 'train_prefixes.npz'
    with target.open('xb') as stream:
        np.savez_compressed(stream, **a)
    windows = []
    for w in range(768):
        windows.append(dict(match_id=str(a['match_ids'][w]), source_index=int(a['source_indices'][w]),
                            bundle_row_index=int(rows[w]), period=int(a['source_period'][w]),
                            first_history_source_frame=int(a['source_frame_ids'][w, 0]),
                            t0_source_frame=int(a['source_frame_ids'][w, -1]),
                            usable_actor_count=int(a['history_seen'][w].sum()),
                            verified_source_players=int((a['identity_status'][w] == 'VERIFIED_SOURCE_PLAYER_TEAM').sum()),
                            operator_eligible_home_count=int(a['operator_eligible_home_count'][w]),
                            operator_case_status=str(a['operator_case_status'][w]),
                            operator_candidate_source_ids=a['operator_candidate_ids'][w].tolist(),
                            full_history_tensor_finite=bool(np.isfinite(a['history_states'][w]).all()),
                            retained_without_quality_selection=True))
    write(output / 'metadata.json', dict(status='PREPARED_768_TRAIN_ONLY_PREFIXES_NO_MODEL_CALLS',
          windows=windows, roster=registry_summary, access_log=access,
          source_model_teams='0=home;1=away;2=ball; actual team IDs independently checked against roster',
          t0_possession='UNKNOWN_NOT_INFERRED', attack_direction='UNKNOWN_NOT_INFERRED',
          observed_provenance='Original prepared valid flags; no new online/detection guarantee',
          future_values_materialized=False, development_rows_materialized=False,
          fits=0, model_calls=0))
    write(output / 'manifest.json', dict(status='COMPLETED_768_TRAIN_PREFIX_EXTRACTION', created_utc=now(),
          source_lock_sha256=sha(ROOT / 'data_source_lock.json'),
          protocol_sha256=guard['protocol_sha256'], input_archive_sha256=ARCHIVE_SHA,
          path=str(target), sha256=sha(target), metadata_sha256=sha(output / 'metadata.json'),
          rows=768, history_frames=10, training_matches=12,
          arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha(v)) for k, v in a.items()},
          no_future_labels=True, no_heldout_rows=True, fit_calls=0, model_calls=0))
    print(json.dumps(dict(status='COMPLETED_768_TRAIN_PREFIX_EXTRACTION', manifest=str(output / 'manifest.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'synthetic', 'lock', 'prepare'])
    parser.add_argument('--protocol')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--authorize-training-prefix-extraction', action='store_true')
    args = parser.parse_args()
    if args.stage == 'lock':
        lock(args.protocol, args.protocol_sha256)
    elif args.stage == 'prepare':
        prepare(args.authorize_training_prefix_extraction)
    else:
        globals()[args.stage]()
