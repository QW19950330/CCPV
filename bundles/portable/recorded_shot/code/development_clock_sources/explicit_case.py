"""Read-only observation of the already specified conflict; never changes a mapping."""
import hashlib
import json
from pathlib import Path
from lxml import etree
from author_logic import utc_ns


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def tag(el):
    return el.tag.rsplit('}', 1)[-1]


def canonical_selector(value):
    if set(value) != {'match_id', 'event_id', 'calculated_frame'}:
        raise ValueError('Only the fixed match/event/frame selector is accepted')
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def prepare_observer(package, selector_path, paths, config):
    pin = read(package / 'explicit_inputs.json')
    selector = read(selector_path)
    if hashlib.sha256(canonical_selector(selector)).hexdigest() != pin['selector_canonical_sha256']:
        raise ValueError('The pre-existing explicit-case selector has changed')
    mid = selector['match_id'].removeprefix('DFL-MAT-')
    root = etree.parse(str(paths[(mid, 'events_raw')])).getroot()
    versions = [e for e in root if tag(e) == 'Event' and e.get('EventId') == selector['event_id']]
    if not versions:
        raise ValueError('The fixed event does not occur in its pinned raw event source')
    final = versions[-1]
    shots = [e for e in final.iter() if tag(e) == 'ShotAtGoal']
    if len(shots) != 1 or final.get('MatchId') != selector['match_id']:
        raise ValueError('Fixed event identity changed')
    n = int(final.get('CalculatedFrame'))
    if n != selector['calculated_frame']:
        raise ValueError('Original calculated frame changed')
    row = {'match_id': selector['match_id'], 'event_id': selector['event_id'],
           'root_attributes': dict(final.attrib), 'shot_actor': shots[0].get('Player'),
           'shot_team': shots[0].get('Team')}
    entry = next(r for r in config['files'] if r['match_id'] == mid and r['kind'] == 'positions')
    return Observer(row, n, paths[(mid, 'positions')], entry)


class Observer:
    def __init__(self, row, n, path, entry):
        self.row, self.n, self.path, self.entry = row, n, path.resolve(), entry
        self.records, self.actor_ranges, self.scans = None, None, 0

    def facade(self, original):
        owner = self

        class ParserFacade:
            parse = staticmethod(original.parse)

            @staticmethod
            def iterparse(reader, *args, **kwargs):
                if Path(reader.f.name).resolve() != owner.path:
                    return original.iterparse(reader, *args, **kwargs)
                return owner.observe(original.iterparse(reader, *args, **kwargs), reader)

        return ParserFacade

    def observe(self, parser, reader):
        # Same record calculations as investigate_unmapped.main. XML clear/delete
        # remain exclusively in the untouched direct loop after each yielded event.
        row = self.row
        n = self.n
        records = []
        context = None
        actor_ranges = []
        current_range = None
        self.scans += 1
        if self.scans != 1:
            raise ValueError('Explicit source was scanned more than once by this observer')
        for ev, el in parser:
            name = tag(el)
            if ev == 'start' and name == 'FrameSet':
                context = dict(el.attrib)
                current_range = {'frameset': context.copy(), 'min_n': None, 'max_n': None,
                                 'samples': 0, 'explicit_frame_present': False} if el.get('PersonId') == row['shot_actor'] else None
            elif ev == 'end' and name == 'Frame' and context is not None:
                raw_n = int(el.get('N'))
                if current_range is not None:
                    current_range['samples'] += 1
                    current_range['min_n'] = raw_n if current_range['min_n'] is None else min(raw_n, current_range['min_n'])
                    current_range['max_n'] = raw_n if current_range['max_n'] is None else max(raw_n, current_range['max_n'])
                    current_range['explicit_frame_present'] |= raw_n == n
                if raw_n in {n - 2, n - 1, n, n + 1, n + 2}:
                    records.append({'frameset': context.copy(), 'frame': dict(el.attrib)})
            elif ev == 'end' and name == 'FrameSet':
                if current_range is not None:
                    actor_ranges.append(current_range)
                context = None
                current_range = None
            yield ev, el
        if reader.h.hexdigest() != self.entry['sha256']:
            raise ValueError('Full explicit-case raw stream digest mismatch')
        self.records, self.actor_ranges = records, actor_ranges

    def result(self):
        if self.scans != 1 or self.records is None:
            raise RuntimeError('Fixed conflict source was not completely observed')
        row, n = self.row, self.n
        case = {'match_id': row['match_id'], 'event_id': row['event_id'],
                'calculated_frame': n, 'calculated_timestamp': row['root_attributes']['CalculatedTimestamp'],
                'shot_actor': row['shot_actor'], 'shot_team': row['shot_team'],
                'source_pin': {'path': self.entry['relative_path'], 'sha256': self.entry['sha256'], 'bytes': self.entry['bytes']},
                'actor_framesets': self.actor_ranges, 'records_at_explicit_and_two_adjacent_frames': self.records,
                'adjacent_samples_are_context_not_fallback': True}
        same = [r for r in self.records if int(r['frame']['N']) == n
                and (r['frameset'].get('TeamId') == 'BALL' or r['frameset'].get('PersonId') == row['shot_actor'])]
        times = {utc_ns(r['frame']['T']) for r in same}
        if len(times) != 1:
            raise ValueError('Same-frame ball/actor times are not unique')
        residual = (next(iter(times)) - utc_ns(case['calculated_timestamp'])) / 1e9
        return case, {'signed_residual_s': residual, 'formula': 'exact raw frame T minus literal CalculatedTimestamp',
                      'same_frame_records': len(same), 'calculated_frame_used_without_shift': True,
                      'adjacent_samples_used_for_acceptance': False, 'mapping_was_not_changed': True}
