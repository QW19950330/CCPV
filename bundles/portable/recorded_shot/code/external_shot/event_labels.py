"""Literal BAS event labels for a fixed GSR ledger; no fitting or file I/O.

The endpoint is a released canonical Shot record, not a verified physical shot,
goal or causal effect. Main time is provider position (integer milliseconds).
Semantics extend the pinned exposed-QC event contract to explicit fixture/half.
"""
from dataclasses import dataclass
import json
import re

TAXONOMY = ('Pass', 'Drive', 'Header', 'High Pass', 'Out', 'Cross',
            'Throw In', 'Shot', 'Ball Player Block', 'Player Successful Tackle',
            'Free Kick', 'Goal')
ALIASES = {raw: canonical for canonical in TAXONOMY
           for raw in (canonical, canonical.upper())}
SOURCE_CONTRACT_SHA256 = '92e4022bc91d472538afd2e068b781568dfa2825f5c6e98d01a7428548e4e25d'


def exact_int(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not an exact integer')
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r'-?\d+', value):
        return int(value)
    raise ValueError('not an exact integer')


def _fixture(value):
    if not isinstance(value, str) or not value or '|' in value:
        raise ValueError('fixture must be a nonempty string without |')
    return value


def _raw(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _half_fields(event):
    game_time = event.get('gameTime')
    gh = (exact_int(game_time.split(' - ', 1)[0])
          if isinstance(game_time, str) and ' - ' in game_time else None)
    eh = exact_int(event['half']) if event.get('half') is not None else None
    return gh, eh


def _clock(event):
    gh, eh = _half_fields(event)
    if gh is not None and eh is not None and gh != eh:
        raise ValueError('half/gameTime conflict')
    half = eh if eh is not None else gh
    if half not in (1, 2):
        raise ValueError('half unavailable')
    frame, position = exact_int(event.get('frame')), exact_int(event.get('position'))
    if frame < 1 or position < 0:
        raise ValueError('negative position or nonpositive frame')
    return half, frame, position


@dataclass(frozen=True)
class EventTable:
    fixture: str
    source_sha256: str
    event_container: str
    rows: tuple


def parse_event_payload(payload, source_sha256, fixture):
    """Parse an already-loaded payload, retaining original ordinal identity.

    No actor/team/ball/player/id values are inspected. The caller must verify
    file bytes and authorization before parsing; a supplied SHA alone is not
    proof that a payload belongs to a match or that annotations are complete.
    """
    fixture = _fixture(fixture)
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('source_sha256 must be a lowercase SHA-256')
    if not isinstance(payload, dict):
        raise ValueError('BAS payload must be an object')
    containers = [key for key in ('actions', 'annotations') if isinstance(payload.get(key), list)]
    if len(containers) != 1:
        raise ValueError('exactly one actions/annotations array required')
    rows = []
    for ordinal, event in enumerate(payload[containers[0]]):
        if not isinstance(event, dict):
            raise ValueError('non-object event at ordinal ' + str(ordinal))
        label = event.get('label')
        row = dict(source_row_ordinal=ordinal, source_row_key=f'{source_sha256}:{ordinal}',
                   label_literal=label if isinstance(label, str) else '',
                   label_raw_json=_raw(label),
                   label_canonical=ALIASES.get(label) if isinstance(label, str) else None,
                   label_known_exact=isinstance(label, str) and label in ALIASES,
                   half=None, frame=None, position_ms=None, frame_time_ms=None,
                   gameTime_raw=event.get('gameTime') if isinstance(event.get('gameTime'), str) else '',
                   half_raw_json=_raw(event.get('half')),
                   frame_raw_json=_raw(event.get('frame')),
                   position_raw_json=_raw(event.get('position')),
                   clock_valid=False, clock_error='')
        try:
            row['half'], row['frame'], row['position_ms'] = _clock(event)
            row['frame_time_ms'] = 40 * (row['frame'] - 1)
            row['clock_valid'] = True
        except (TypeError, ValueError) as error:
            row['clock_error'] = str(error)
            # A bad clock localized unambiguously to the other half does not
            # invalidate this half. Conflicting/unknown halves invalidate both.
            try:
                gh, eh = _half_fields(event)
                if gh in (1, 2) and (eh is None or gh == eh):
                    row['half'] = gh
                elif eh in (1, 2) and gh is None:
                    row['half'] = eh
            except (TypeError, ValueError):
                pass
        rows.append(row)
    return EventTable(fixture, source_sha256, containers[0], tuple(rows))


def _validate_window(window, fixture):
    if window.get('fixture') != fixture:
        raise ValueError('fixture does not match event table')
    half, g = window.get('half'), window.get('g')
    if type(half) is not int or half not in (1, 2) or type(g) is not int or g < 1:
        raise ValueError('half/g must be exact positive integer ledger fields')
    expected_key = f'{fixture}|{half}|{g}'
    if window.get('row_key') != expected_key:
        raise ValueError('row_key must exactly encode fixture|half|g')
    for key, expected in [('t0_ms', 40 * (g - 1)), ('end_ms', 40 * (g - 1) + 10000)]:
        if type(window.get(key)) is not int or window[key] != expected:
            raise ValueError(key + ' does not follow the fixed half-local 25Hz clock')
    for key in ('input_eligible', 'future_record_clock_complete'):
        if type(window.get(key)) is not bool:
            raise ValueError(key + ' must be a boolean')


def _endpoint(table, window, time_key):
    if not window['future_record_clock_complete']:
        return -1, [], 'INCOMPLETE_CLOCK_10S'
    half = window['half']
    if any(not row['clock_valid'] and row['half'] in (None, half) for row in table.rows):
        return -1, [], 'UNLOCALIZABLE_EVENT_TIME_OR_HALF'
    inside = [row for row in table.rows if row['clock_valid'] and row['half'] == half
              and window['t0_ms'] < row[time_key] <= window['end_ms']]
    shot_keys = [row['source_row_key'] for row in inside if row['label_canonical'] == 'Shot']
    if any(not row['label_known_exact'] for row in inside):
        return -1, shot_keys, 'UNKNOWN_LITERAL_LABEL_IN_WINDOW'
    return int(bool(shot_keys)), shot_keys, ''


def label_windows(table, ledger, fixture=None):
    """Return every ledger row, with score y=-1 for any exclusion.

    endpoint_y retains the published-record endpoint independently of current
    input eligibility. Only y is scored. Goal is not unioned into Shot; unknown
    labels are not rescued by a co-recorded Shot. Both halves use local clocks.
    """
    if not isinstance(table, EventTable):
        raise ValueError('use parse_event_payload to construct an EventTable')
    fixture = table.fixture if fixture is None else _fixture(fixture)
    if fixture != table.fixture:
        raise ValueError('fixture does not match event table')
    result, seen = [], set()
    for window in ledger:
        _validate_window(window, fixture)
        if window['row_key'] in seen:
            raise ValueError('duplicate ledger row_key')
        seen.add(window['row_key'])
        endpoint_y, shot_keys, endpoint_reason = _endpoint(table, window, 'position_ms')
        reasons = ([] if window['input_eligible'] else ['INPUT_INELIGIBLE'])
        if endpoint_reason:
            reasons.append(endpoint_reason)
        result.append(dict(fixture=fixture, half=window['half'], g=window['g'],
                           row_key=window['row_key'], y=endpoint_y if not reasons else -1,
                           endpoint_y=endpoint_y, endpoint_reason=endpoint_reason,
                           exclusion_reason=';'.join(reasons), source_row_keys=shot_keys,
                           endpoint='canonical_Shot', main_clock='position_ms',
                           source_sha256=table.source_sha256))
    return result


def frame_clock_diagnostic(table, ledger):
    """Descriptive alternative clock only; never replaces the main y field."""
    result = []
    for window in ledger:
        _validate_window(window, table.fixture)
        value, keys, reason = _endpoint(table, window, 'frame_time_ms')
        result.append(dict(row_key=window['row_key'], diagnostic_frame_endpoint_y=value,
                           diagnostic_frame_source_row_keys=keys, diagnostic_frame_reason=reason,
                           purpose='CLOCK_DESCRIPTION_ONLY_NOT_PRIMARY_SCORING'))
    return result
