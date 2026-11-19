"""Original author-clock functions and per-match body; path/output adapter only. See source_fragments.json."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib, json, math
import numpy as np
import pandas as pd
from lxml import etree

MIDS = ['J03WMX', 'J03WN1', 'J03WPY']
SHOT_TYPES = {'shot', 'shot_freekick', 'shot_penalty'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def tag(el):
    return el.tag.rsplit('}', 1)[-1]

def utc_ns(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError('Missing timestamp')
    return int((t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')).value)

def seconds(value):
    mm, ss = str(value).split(':')
    return int(mm) * 60 + float(ss)

def strict_key(match, period, actor, subtype, stamp):
    return match, int(period), str(actor), str(subtype), utc_ns(stamp)

def unique_link(a_key, raw_keys, author_keys):
    rr = [i for i, k in enumerate(raw_keys) if k == a_key]
    aa = [i for i, k in enumerate(author_keys) if k == a_key]
    return (rr[0] if len(rr) == len(aa) == 1 else None), rr, aa

def period_origin(gt, period):
    sub = gt[gt.period_id == period]
    frame = sub.frame_id.to_numpy(dtype=float)
    sec = np.array([seconds(x) for x in sub.synced_ts], dtype=float)
    if not len(frame) or not np.isfinite(frame).all() or not np.equal(frame, np.round(frame)).all():
        return None
    lo = int(np.ceil(np.max(frame - 25 * sec - .500001)))
    hi = int(np.floor(np.min(frame - 25 * sec + .500001)))
    if lo != hi or not np.equal(np.round(lo + 25 * sec), frame).all():
        return None
    return lo

class HashReader:
    def __init__(self, path):
        self.f = Path(path).open('rb')
        self.h = hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b

def rebuild(files,source,DATA,old):
    raw_ledger, author_ledger, inventory, matches = [], [], [], []
    for mid in MIDS:
        gt = pd.read_parquet(files[(mid, 'gt')])
        u = pd.read_parquet(files[(mid, 'unsynced')])
        assert gt.index.equals(u.index) and gt.index.is_unique
        for c in ['period_id', 'player_id', 'spadl_type']:
            assert gt[c].equals(u[c])
        for idx in u.index:
            inventory.append({'match_id': mid, 'author_index': int(idx),
                              'period_id': int(u.at[idx, 'period_id']), 'player_id': str(u.at[idx, 'player_id']),
                              'spadl_type': str(u.at[idx, 'spadl_type']),
                              'scope': 'SHOT_CROSSWALK' if u.at[idx, 'spadl_type'] in SHOT_TYPES else 'NONSHOT_INVENTORY_NOT_RAW_CROSSWALKED'})
        entries = {k: next(x for x in source['downloaded_files'] if mid in x['path'] and k in x['path'])
                   for k in ['events_raw', 'matchinformation', 'positions']}
        for k in ['events_raw', 'matchinformation']:
            assert sha(DATA / entries[k]['path']) == entries[k]['sha256']
        mr = etree.parse(str(DATA / entries['matchinformation']['path'])).getroot()
        general = next(e for e in mr.iter() if tag(e) == 'General')
        teams = {general.get('HomeTeamId'): 'home', general.get('GuestTeamId'): 'away'}
        roster, object_ids = {}, {}
        for team in mr.iter():
            if tag(team) == 'Team':
                tid = team.get('TeamId'); assert tid in teams
                for p in team.iter():
                    if tag(p) == 'Player':
                        obj = teams[tid] + '_' + str(int(p.get('ShirtNumber')))
                        pid = p.get('PersonId')
                        assert obj not in object_ids or object_ids[obj] == pid
                        object_ids[obj] = pid; roster[pid] = (obj, tid)
        eroot = etree.parse(str(DATA / entries['events_raw']['path'])).getroot()
        terminal = {}
        for e in eroot:
            if tag(e) == 'Event':
                assert e.get('MatchId') == 'DFL-MAT-' + mid
                terminal[e.get('EventId')] = e
        active = [e for e in terminal.values() if not any(tag(x) == 'Delete' for x in e.iter())]
        bounds = defaultdict(dict)
        for e in active:
            for child in e:
                if tag(child) in ['KickOff', 'FinalWhistle'] and child.get('GameSection') in ['firstHalf', 'secondHalf']:
                    period = 1 if child.get('GameSection') == 'firstHalf' else 2
                    assert tag(child) not in bounds[period]
                    bounds[period][tag(child)] = utc_ns(e.get('EventTime'))
        raw = []
        for e in active:
            shots = [c for c in e.iter() if tag(c) == 'ShotAtGoal']
            if not shots:
                continue
            assert len(shots) == 1
            shot = shots[0]; pid = shot.get('Player'); team = shot.get('Team')
            mapped_roster = roster.get(pid)
            stamp = utc_ns(e.get('EventTime'))
            pp = [p for p, b in bounds.items() if b['KickOff'] <= stamp <= b['FinalWhistle']]
            period = pp[0] if len(pp) == 1 else 0
            parents = {tag(a) for a in shot.iterancestors()}
            subtype = 'shot_penalty' if 'Penalty' in parents else 'shot_freekick' if 'FreeKick' in parents else 'shot'
            valid = mapped_roster is not None and mapped_roster[1] == team and period in [1, 2]
            raw.append({'match_id': mid, 'event_id': e.get('EventId'), 'period_id': period,
                        'raw_actor': pid, 'raw_team': team, 'object_id': mapped_roster[0] if mapped_roster else None,
                        'spadl_type': subtype, 'EventTime': e.get('EventTime'),
                        'key': strict_key(mid, period, mapped_roster[0], subtype, e.get('EventTime')) if valid else None,
                        'raw_goal': any(tag(c) == 'SuccessfulShot' for c in shot.iter()),
                        'original_mapping': bool(old.loc[old.event_id == e.get('EventId'), 'mapped'].iloc[0])})
        us = u[u.spadl_type.isin(SHOT_TYPES)]
        raw_keys = [r['key'] for r in raw]
        author_keys = [strict_key(mid, a.period_id, a.player_id, a.spadl_type, a.utc_timestamp) for a in us.itertuples()]
        origins = {p: period_origin(gt, p) for p in [1, 2]}
        joined = []
        for pos, (idx, a) in enumerate(us.iterrows()):
            hit, rr, aa = unique_link(author_keys[pos], raw_keys, author_keys)
            g = gt.loc[idx]
            row = {'match_id': mid, 'author_index': int(idx), 'period_id': int(a.period_id),
                   'object_id': str(a.player_id), 'raw_actor': object_ids.get(str(a.player_id)),
                   'spadl_type': str(a.spadl_type), 'unsynced_UTC_ns': utc_ns(a.utc_timestamp),
                   'raw_candidate_ids': '|'.join(raw[z]['event_id'] for z in rr),
                   'raw_candidate_count': len(rr), 'author_signature_count': len(aa),
                   'raw_event_id': raw[hit]['event_id'] if hit is not None else None,
                   'identity_status': 'UNIQUE_EXACT_BIDIRECTIONAL' if hit is not None else 'UNRESOLVED_NO_GUESS',
                   'annotated_frame': int(g.frame_id), 'annotated_relative_ts': str(g.synced_ts),
                   'author_clock_accepted': False, 'clock_reason': 'identity_not_established',
                   'source_period_zero_frame': origins.get(int(a.period_id)),
                   'raw_goal': raw[hit]['raw_goal'] if hit is not None else None,
                   'author_success': bool(a.success), 'outcome_used_for_identity_or_clock': False}
            joined.append(row)
        wanted = {int(a['annotated_frame']) for a in joined}
        ball, actors = {}, {}
        reader = HashReader(DATA / entries['positions']['path']); context = None
        try:
            for ev, el in etree.iterparse(reader, events=['start', 'end'], huge_tree=True):
                name = tag(el)
                if ev == 'start' and name == 'FrameSet':
                    assert el.get('MatchId') == 'DFL-MAT-' + mid
                    context = (el.get('GameSection'), el.get('TeamId'), el.get('PersonId'))
                elif ev == 'end' and name == 'Frame' and context:
                    n = int(el.get('N'))
                    if n in wanted:
                        section, team, actor = context
                        p = 1 if section == 'firstHalf' else 2 if section == 'secondHalf' else 0
                        val = {'utc_ns': utc_ns(el.get('T')), 'finite': all(math.isfinite(float(el.get(c))) for c in ['X', 'Y'])}
                        target, key = (ball, (p, n)) if team == 'BALL' else (actors, (p, n, team, actor))
                        if key in target:
                            target[key]['count'] += 1
                        else:
                            target[key] = dict(val, count=1)
                    el.clear()
                    while el.getprevious() is not None:
                        del el.getparent()[0]
                elif ev == 'end' and name == 'FrameSet':
                    context = None; el.clear()
        finally:
            reader.f.close()
        assert reader.h.hexdigest() == entries['positions']['sha256']
        for row in joined:
            reasons = []
            if row['raw_event_id'] is None:
                reasons.append('identity_not_established')
            p, n = row['period_id'], row['annotated_frame']
            b = ball.get((p, n)); rec = roster.get(row['raw_actor']); a = actors.get((p, n, rec[1], row['raw_actor'])) if rec else None
            if b is None or b['count'] != 1 or not b['finite']:
                reasons.append('missing_ambiguous_or_nonfinite_exact_raw_ball')
            if a is None or a['count'] != 1 or not a['finite']:
                reasons.append('missing_ambiguous_or_nonfinite_exact_raw_actor')
            if a and b and abs(a['utc_ns'] - b['utc_ns']) > 1000:
                reasons.append('ball_actor_clock_inconsistent')
            zero = row['source_period_zero_frame']
            if zero is None or abs((n - zero) / 25 - seconds(row['annotated_relative_ts'])) > .021:
                reasons.append('author_relative_clock_origin_not_uniquely_verified')
            row['author_clock_accepted'] = not reasons
            row['clock_reason'] = '|'.join(reasons) if reasons else 'AUTHOR_ANNOTATED_TRACKING_CLOCK_NOT_VIDEO_TRUTH'
            row['exact_raw_UTC_ns'] = b['utc_ns'] if b else None
            row['success_agreement_after_identity'] = (row['raw_goal'] == row['author_success']) if row['raw_event_id'] is not None else None
            author_ledger.append(row)
        for row in raw:
            hits = [a for a in joined if a['raw_event_id'] == row['event_id']]
            raw_ledger.append({k: v for k, v in row.items() if k != 'key'} | {
                'author_indices': '|'.join(str(a['author_index']) for a in hits),
                'unique_author_identity': len(hits) == 1,
                'accepted_author_clock': len(hits) == 1 and hits[0]['author_clock_accepted'],
                'status': 'PRESERVED_RAW_WITH_AUTHOR_CLOCK_SEPARATE' if len(hits) == 1 else 'RAW_WITHOUT_UNIQUE_AUTHOR_LINK_NOT_DELETED'})
        matches.append({'match_id': mid, 'raw_shots': len(raw), 'author_shots': len(joined),
                        'author_all_events': len(u), 'unique_raw_author_shots': sum(a['raw_event_id'] is not None for a in joined),
                        'accepted_author_shot_clocks': sum(a['author_clock_accepted'] for a in joined),
                        'author_shots_without_unique_raw_ID': sum(a['raw_event_id'] is None for a in joined),
                        'raw_shots_without_unique_author_ID': sum(not any(a['raw_event_id'] == e['event_id'] for a in joined) for e in raw),
                        'period_zero_frames': origins, 'tracking_sha256': entries['positions']['sha256']})
        print(json.dumps(matches[-1]), flush=True)
    old13 = old[~old.mapped].copy()
    extra = []
    for a in old13.itertuples():
        linked = [z for z in author_ledger if z['raw_event_id'] == a.event_id]
        extra.append({'match_id': a.match_id, 'event_id': a.event_id,
                      'original_strict_mapped': False, 'in_author_match_scope': a.match_id.removeprefix('DFL-MAT-') in MIDS,
                      'unique_author_links': len(linked), 'new_separate_author_clock': len(linked) == 1 and linked[0]['author_clock_accepted'],
                      'author_index': linked[0]['author_index'] if len(linked) == 1 else None,
                      'old_UNKNOWN_overwritten': False})
    return {'author_shots.csv':author_ledger,'raw_shots.csv':raw_ledger,'all_author_rows_inventory.csv':inventory,'all13_original_unmapped.csv':extra},matches
