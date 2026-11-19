"""Exact retained 25 Hz native frames on the nominal 10 Hz model grid."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import metrica_reader
from common import sha, ah, dump, save_npz
HISTORY_SOURCE=np.array([4,9,14,19,24])+1 # indexes in prefix+75 main block

HISTORY_SLOTS=np.array([1,3,5,7,9])

TARGET_SOURCE=np.arange(29,75,5)+1

TARGET_STEPS=np.arange(2,21,2)

def exact_arrays(native_positions):
    """native[0] is one causal pad frame; native[1:76] is main75 block."""
    p=np.asarray(native_positions,dtype=np.float32)
    assert p.shape==(76,23,2)
    seen=np.isfinite(p).all(-1)
    h=np.zeros((10,23,4),np.float32);v=np.zeros((10,23),bool)
    for slot,index in zip(HISTORY_SLOTS,HISTORY_SOURCE):
        support=seen[index]&seen[index-5]
        h[slot,:,:2]=np.where(support[:,None],p[index],0.)
        h[slot,:,2:]=np.where(support[:,None],(p[index]-p[index-5])/.2,0.)
        v[slot]=support
    truth=np.full((20,23,2),np.nan,np.float32);pv=np.zeros((20,23),bool)
    target=np.zeros((20,23,4),np.float32);tv=np.zeros((20,23),bool)
    for step,index in zip(TARGET_STEPS,TARGET_SOURCE):
        point=seen[index];support=point&seen[index-5]
        truth[step-1]=p[index];pv[step-1]=point;tv[step-1]=support
        target[step-1,:,:2]=np.where(support[:,None],p[index],0.)
        target[step-1,:,2:]=np.where(support[:,None],(p[index]-p[index-5])/.2,0.)
    return dict(history=h,history_valid=v,position_targets=truth,position_valid=pv,target_states=target,target_valid=tv)

def history_slot_positions(game,first):
    """History-only identities; no target visibility enters slot admission."""
    observed=np.isfinite(game.positions[first:first+25]).all(-1).any(0)
    players=[np.flatnonzero(observed&(game.teams==tm)).tolist() for tm in (0,1)]
    if any(len(x)>11 for x in players):return None
    ball=np.flatnonzero(game.teams==2)
    if len(ball)!=1:raise ValueError('Source must contain one physical ball identity')
    p=np.full((76,23,2),np.nan,np.float32);ids=['']*23
    for tm in (0,1):
        for j,entity in enumerate(players[tm]):
            slot=11*tm+j;p[:,slot]=game.positions[first-1:first+75,entity];ids[slot]=game.identities[entity]
    p[:,22]=game.positions[first-1:first+75,ball[0]];ids[22]=game.identities[ball[0]]
    return p,ids

def prepare(raw: Path, out: Path):
    m=metrica_reader;sources=[]
    relevant=[k for k in m.FILE_SPECS if 'RawTrackingData' in k or k.endswith('_tracking.txt') or k.endswith('_metadata.xml') or k=='README.md']
    for key in relevant:
        p=raw/key;actual=m.fingerprints(p)
        size,git=m.FILE_SPECS[key]
        if actual['bytes']!=size or actual['git_blob_sha1']!=git:raise ValueError('Pinned raw byte mismatch: '+key)
        sources.append(dict(file=key,path=str(p),**actual))
    records=[];match_qc=[]
    for number in (1,2,3):
        game=m.read_csv_game(raw,number) if number<3 else m.read_epts_game(raw)
        # Only period/clock boundaries define the original75-frame block grid.
        breaks=[0]+(np.flatnonzero((np.diff(game.periods)!=0)|(np.diff(game.frames)!=1))+1).tolist()+[len(game.frames)]
        available=[];excluded={'unsupported_history':0,'history_roster_gt11':0,'format_boundary':0}
        for a,b in zip(breaks,breaks[1:]):
            if int(game.periods[a]) not in (1,2):continue
            for first in range(a+1,b-74,75):
                # A fixed identity format change cannot be crossed or rescued.
                if any(int(game.frames[first-1])<f<=int(game.frames[first+74]) for f in game.forced_break_frames):
                    excluded['format_boundary']+=1;continue
                clocks=game.times[first-1:first+75];frames=game.frames[first-1:first+75]
                if not np.allclose(np.diff(clocks),.04,atol=1e-7,rtol=0) or not np.all(np.diff(frames)==1):raise ValueError('Noncontiguous raw source clock')
                slotted=history_slot_positions(game,first)
                if slotted is None:excluded['history_roster_gt11']+=1;continue
                p,ids=slotted;arrays=exact_arrays(p);seen=arrays['history_valid'].any(0)
                if not (seen[:11].any() and seen[11:22].any() and seen[22]):
                    excluded['unsupported_history']+=1;continue
                available.append(dict(first=first,period=int(game.periods[first]),p=p,entity_ids=ids,
                    clocks=clocks.copy(),frames=frames.copy(),arrays=arrays))
        available.sort(key=lambda x:(x['period'],int(x['frames'][1])))
        if not available:raise ValueError('No source-qualified history windows: '+game.match_id)
        count=min(64,len(available));selection=np.linspace(0,len(available)-1,count,dtype=np.int64)
        if len(set(selection.tolist()))!=count:raise AssertionError('Duplicate chosen windows')
        chosen=[available[int(i)] for i in selection]
        for row in chosen:
            identity=f'{m.COMMIT}|{game.match_id}|{row["period"]}|{int(row["frames"][1])}'
            source_index=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big')&((1<<63)-1)
            records.append(dict(**row,match_id=game.match_id,source_index=source_index,
                clip_id=f'{game.match_id}:p{row["period"]}:f{int(row["frames"][1])}-{int(row["frames"][-1])}'))
        match_qc.append(dict(match_id=game.match_id,source_qualified_blocks=len(available),chosen_windows=count,
            selection_positions=selection.tolist(),exclusions=excluded,source_provenance=game.provenance,
            history_only_selection=True,future_visibility_selection=False))
        print(f'PREPARED {game.match_id}: {count} selected/{len(available)} source-qualified blocks',flush=True)
        del game,available
    data={k:np.stack([r['arrays'][k] for r in records]) for k in records[0]['arrays']}
    data.update(native_positions_with_pad=np.stack([r['p'] for r in records]),
        source_frame_ids_with_pad=np.stack([r['frames'] for r in records]),
        source_times_s_with_pad=np.stack([r['clocks'] for r in records]),
        source_period=np.array([r['period'] for r in records],np.int64),
        source_indices=np.array([r['source_index'] for r in records],np.int64),
        match_ids=np.array([r['match_id'] for r in records]),clip_ids=np.array([r['clip_id'] for r in records]),
        entity_ids=np.array([r['entity_ids'] for r in records]),teams=np.array([0]*11+[1]*11+[2],np.int64))
    if len(set(data['source_indices'].tolist()))!=len(records):raise ValueError('Source identity hash collision')
    save_npz(out/'external_exact.npz',**data)
    report=dict(status='REBUILT_FROM_PINNED_RAW',windows=len(records),match_qc=match_qc,
        source_commit=m.COMMIT,sources=sources,array_hashes={k:ah(v) for k,v in data.items()},
        data_sha256=sha(out/'external_exact.npz'),model_forward_calls=0,optimizer_updates=0,
        source_index_rule='sha256(commit|match|period|firstmainframe), first8bytes, masked63bits',
        history_only_eligibility=True,future_truth_interpolation=False)
    dump(out/'manifest.json',report)
    return report
