"""Recover position-only truth without replacing model inputs or masks."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
from common import sha,dump
FRAME=re.compile(rb'^\s*\{\s*"frame"\s*:\s*(\d+)')

def finite_xy(item):
    x,y=item.get('x'),item.get('y')
    if x is None or y is None:return None
    xy=np.asarray([x,y],dtype=np.float64)
    return xy if np.isfinite(xy).all() else None

def recover(raw: Path, development: Path, contract: dict, out: Path):
    with np.load(development,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    ids=a['match_ids'].astype(str);selected_matches=sorted(set(ids))
    if len(selected_matches)!=16:raise AssertionError('Expected 16 development matches')
    pinned={x['path']:x for x in contract['skillcorner']['files']}
    positions=np.full(a['states'].shape[:-1]+(2,),np.nan,dtype=np.float32)
    recovered=np.zeros_like(a['position_valid'])
    source_files=[];match_records=[]
    for match in selected_matches:
        mid=int(match.split(':')[1]);folder=raw/str(mid)
        metadata_path=folder/f'{mid}_match.json';raw_path=folder/f'{mid}_tracking_extrapolated.jsonl'
        meta_hash=sha(metadata_path)
        meta_key=f'data/matches/{mid}/{mid}_match.json';track_key=f'data/matches/{mid}/{mid}_tracking_extrapolated.jsonl'
        if meta_hash!=pinned[meta_key]['sha256']:raise ValueError('Pinned metadata mismatch')
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        length,width=float(metadata['pitch_length']),float(metadata['pitch_width'])
        if length<=0 or width<=0:raise ValueError('Invalid pitch units')
        player_teams={int(x['id']):int(x['team_id']) for x in metadata['players'] if x.get('id') is not None and x.get('team_id') is not None}
        home,away=int(metadata['home_team']['id']),int(metadata['away_team']['id'])
        windows=np.flatnonzero(ids==match);wanted={}
        for w in windows:
            for slot,eid in enumerate(a['entity_ids'][w].astype(str)):
                if not eid:continue
                if slot==22:
                    if eid!='ball' or a['teams'][slot]!=2:raise AssertionError('Ball slot mismatch')
                else:
                    team_id=player_teams[int(eid)]
                    expected=0 if team_id==home else 1 if team_id==away else -1
                    if expected!=int(a['teams'][slot]):raise AssertionError('Entity/team mapping mismatch')
            for t,frame in enumerate(a['source_frame_ids'][w]):
                wanted.setdefault(int(frame),[]).append((int(w),int(t)))
        observed_frames=set();track_hash=hashlib.sha256()
        with raw_path.open('rb') as stream:
            for line in stream:
                track_hash.update(line)
                header=FRAME.match(line)
                if header is None:raise ValueError('Unexpected raw frame header')
                frame=int(header.group(1))
                if frame not in wanted:continue
                if frame in observed_frames:raise ValueError('Duplicate selected raw frame')
                observed_frames.add(frame)
                # Only requested development frames are deserialized. Other raw
                # bytes are streamed solely to verify the pinned file hash.
                row=json.loads(line)
                players={}
                for item in row.get('player_data') or []:
                    pid=int(item['player_id'])
                    if pid in players:raise ValueError('Duplicate raw player identity')
                    players[pid]=item
                for w,t in wanted[frame]:
                    if int(row['period'])!=int(a['source_period'][w]):raise AssertionError('Period/frame mismatch')
                    for slot,eid in enumerate(a['entity_ids'][w].astype(str)):
                        if not eid:continue
                        item=(row.get('ball_data') or {}) if slot==22 else players.get(int(eid),{})
                        xy=finite_xy(item)
                        detected=item.get('is_detected') is True and xy is not None
                        if detected:
                            positions[w,t,slot]=xy*np.asarray([105./length,68./width],dtype=np.float64)
                            recovered[w,t,slot]=True
        actual_track_hash=track_hash.hexdigest()
        if actual_track_hash!=pinned[track_key]['sha256']:raise ValueError('Pinned tracking mismatch')
        if observed_frames!=set(wanted):raise ValueError('Selected raw frames missing')
        if not np.array_equal(recovered[windows],a['position_valid'][windows]):
            mismatch=np.argwhere(recovered[windows]!=a['position_valid'][windows])
            raise AssertionError('Original real position mask does not match raw observations: '+str(mismatch[:5]))
        support=a['valid'][windows]
        delta=np.abs(positions[windows][support]-a['states'][windows,:,:,:2][support])
        max_error=float(delta.max()) if len(delta) else 0.
        if max_error>1e-5:raise AssertionError('Recovered full-valid coordinates disagree with original state')
        source_files.extend([dict(path=str(metadata_path),sha256=meta_hash),dict(path=str(raw_path),sha256=actual_track_hash)])
        match_records.append(dict(match_id=match,windows=len(windows),selected_raw_frames=len(wanted),
            pitch_length_m=length,pitch_width_m=width,max_fullvalid_coordinate_error_m=max_error,
            position_only_points=int((a['position_valid'][windows]&~a['valid'][windows]).sum())))
        print(f'RECOVERED {match} windows={len(windows)} frames={len(wanted)}',flush=True)
    if not np.isfinite(positions[a['position_valid']]).all():raise ValueError('Observed position could not be recovered')
    if np.isfinite(positions[~a['position_valid']]).any():raise AssertionError('A nonexistent observed position was filled')
    def writer(temporary):
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,positions=positions,position_valid=a['position_valid'],
                source_indices=a['source_indices'],match_ids=a['match_ids'],source_frame_ids=a['source_frame_ids'],entity_ids=a['entity_ids'])
    with (out/'position_truth_v2.npz').open('xb') as stream:
        np.savez_compressed(stream,positions=positions,position_valid=a['position_valid'],
            source_indices=a['source_indices'],match_ids=a['match_ids'],source_frame_ids=a['source_frame_ids'],entity_ids=a['entity_ids'])
    only=a['position_valid']&~a['valid']
    dump(out/'truth_manifest.json',dict(status='RECOVERED_FROM_PINNED_RAW',
        source_files=source_files,match_records=match_records,total_position_only_points=int(only.sum()),
        truth_sha256=sha(out/'position_truth_v2.npz'),model_forward_calls=0,optimizer_updates=0,
        observed_position_mask_changed=False,unobserved_positions_imputed=False))
