"""Frozen source-test prefix extraction; no model, forward, labels or future numeric decoding.
Default stage is synthetic. Real extraction requires explicit protocol and self-source SHA arguments.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,tempfile,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent
ROOT=AUDIT.parent
LOADER_REL='CCPV_reviewer_validation_program_20261026/full_intervention/data_prepare.py'
LOADER_SHA='d2678d3bea5f8745504ebe97ec8d6866089fd90db95a3a9b10eaa2b41a36bea8'
PROTOCOL=AUDIT/'skillcorner_response_protocol.json'
OUT=AUDIT/'skillcorner_prepared'
PREFIX=chr(92)*2+'?'+chr(92)
def lp(p):
    p=Path(p).resolve();return Path(PREFIX+str(p)) if os.name=='nt' and not str(p).startswith(PREFIX) else p
def sha(p):
    h=hashlib.sha256()
    with lp(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def dump(p,value):
    lp(p.parent).mkdir(parents=True,exist_ok=True)
    with lp(p).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def load_upstream():
    src=ROOT/LOADER_REL
    if sha(src)!=LOADER_SHA:raise ValueError('Original streaming reader changed')
    spec=importlib.util.spec_from_file_location('frozen_train_prefix_reader',src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def slot_audit(entity_ids,valid,position_valid):
    """Source-code consequence only: no raw tracking/previous-position comparisons."""
    ids=[str(x) for x in entity_ids]
    home=[x for x in ids[:11] if x];away=[x for x in ids[11:22] if x]
    H,A=len(home),len(away)
    if [i for i in range(11) if ids[i]]!=list(range(H)) or [i for i in range(11,22) if ids[i]]!=list(range(11,11+A)):
        raise ValueError('Source packed-ID slot contract changed')
    # prepare_skillcorner enumerates H home IDs then A away IDs rather than away offset11.
    loop=home+away
    mismatch=[j for j in range(11,11+A) if j<len(loop) and loop[j]!=ids[j]]
    omitted=[j for j in range(11,11+A) if j>=len(loop)]
    mismatched_valid=[j for j in mismatch if bool(valid[0,j])]
    omitted_observed=[j for j in omitted if bool(position_valid[:,j].any())]
    return dict(home_id_count=H,away_id_count=A,home_id_count_under11=H<11,
        first_history_wrong_previous_id_candidate_slots=mismatch,
        first_history_wrong_previous_id_fullvalid_slots=mismatched_valid,
        loop_unvisited_away_slots=omitted,
        loop_unvisited_away_slots_with_history_position_observation=omitted_observed,
        first_history_wrong_previous_id_fullvalid_count=len(mismatched_valid),
        unvisited_history_position_valid_points=int(position_valid[:,omitted].sum()) if omitted else 0,
        technical_supported_no_known_wrong_fullvalid_previous_id=not mismatched_valid,
        interpretation='Potential mapping consequence from original ID layout. No new values are corrected; not a claim that every H<11 valid velocity differs numerically. Omitted slots alter missingness and remain separately reported.')

def synthetic():
    U=load_upstream();access=[]
    with tempfile.TemporaryDirectory(prefix='synthetic_prefix_',dir=HERE) as td:
        p=Path(td)/'toy.npz';values=np.arange(8*30*23*4,dtype=np.float32).reshape(8,30,23,4)
        np.savez_compressed(p,states=values,teams=np.array([0]*11+[1]*11+[2]),test_indices=np.array([1,4,7],np.int64))
        ix=U.selected(p,'test_indices',np.arange(3),access_log=access)
        got=U.selected(p,'states',ix,10,access)
        assert np.array_equal(ix,np.array([1,4,7])) and got.tobytes()==values[ix,:10].tobytes()
        shuffled=U.selected(p,'states',np.array([7,1,4]),10);assert shuffled.tobytes()==values[[7,1,4],:10].tobytes()
        assert U.global_metadata(p,'teams',(23,)).tolist()==[0]*11+[1]*11+[2]
        bad=0
        for ix0 in ([1,1],[-1],[8]):
            try:U.selected(p,'states',ix0,10)
            except (ValueError,IndexError):bad+=1
        assert bad==3
        ids=np.array([str(100+i) for i in range(11)]+[str(200+i) for i in range(11)]+['ball'])
        valid=np.ones((10,23),bool);pos=np.ones((10,23),bool)
        regular=slot_audit(ids,valid,pos);assert not regular['home_id_count_under11'] and regular['technical_supported_no_known_wrong_fullvalid_previous_id']
        ids[10]='';r=slot_audit(ids,valid,pos)
        assert r['home_id_count']==10 and r['first_history_wrong_previous_id_candidate_slots']==list(range(11,21)) and r['loop_unvisited_away_slots']==[21]
        assert len(r['first_history_wrong_previous_id_fullvalid_slots'])==10 and not r['technical_supported_no_known_wrong_fullvalid_previous_id']
        valid[0,11:21]=False;r=slot_audit(ids,valid,pos)
        assert r['technical_supported_no_known_wrong_fullvalid_previous_id'] and r['unvisited_history_position_valid_points']==10
    result=dict(status='PASS_SYNTHETIC_PREFIX_AND_SLOT_AUDIT',source_sha256=sha(__file__),upstream_loader_sha256=LOADER_SHA,
        checks=['fixed_indices_metadata','selected_prefix_exact','nonmonotonic_requested_order_preserved','teams_exact','duplicate_and_bounds_rejected','full_home_no_mismatch','home10_potential_slots','known_wrong_previous_fullvalid_separate_from_omitted_missingness'],access_log=access,scientific_arrays_decoded=0,checkpoints_loaded=0,model_calls=0,training_updates=0)
    dump(HERE/'extractor_synthetic_receipt_v2.json',result);print(json.dumps(result))

def prepare(args):
    if not args.execute or not args.protocol_sha256 or not args.source_sha256:raise PermissionError('Parent-sealed protocol/source SHA and explicit execute flag required')
    if sha(__file__)!=args.source_sha256 or sha(PROTOCOL)!=args.protocol_sha256:raise ValueError('Sealed protocol/source SHA mismatch')
    toy=read(HERE/'extractor_synthetic_receipt_v2.json')
    if toy['source_sha256']!=args.source_sha256 or toy['status']!='PASS_SYNTHETIC_PREFIX_AND_SLOT_AUDIT':raise ValueError('Synthetic evidence not for this source')
    P=read(PROTOCOL);C=P['cohort'];expected=dict(zip(C['matches'],C['expected_windows_per_match']));n=C['expected_total_windows']
    assert n==4992 and list(expected.values())==[1308,1261,1312,1111]
    archive=ROOT/C['source_archive'];splitpath=ROOT/C['split']
    if sha(archive)!=C['source_archive_sha256'] or sha(splitpath)!=C['split_sha256']:raise ValueError('Pinned archive/split changed')
    split=read(splitpath);assert split['groups']['test']==C['matches']
    if OUT.exists():raise FileExistsError('Create-only output already exists')
    U=load_upstream();access=[]
    # test_indices is row metadata; no states/future member is loaded globally.
    ix=U.selected(lp(archive),'test_indices',np.arange(n,dtype=np.int64),access_log=access)
    assert ix.dtype==np.int64 and ix.shape==(n,) and len(np.unique(ix))==n and np.all(np.diff(ix)>0)
    a={k:U.selected(lp(archive),k,ix,access_log=access) for k in ('match_ids','entity_ids','source_period')}
    a['source_indices']=ix.copy();a['bundle_row_indices']=ix.copy()
    assert {mid:int((a['match_ids']==mid).sum()) for mid in expected}==expected
    assert set(a['match_ids'])==set(expected)
    for old,new in [('states','history_states'),('valid','history_valid'),('position_valid','history_position_valid'),('source_frame_ids','source_frame_ids'),('frame_times_s','frame_times_s')]:
        a[new]=U.selected(lp(archive),old,ix,10,access)
    a['teams']=U.global_metadata(lp(archive),'teams',(23,))
    assert np.array_equal(a['teams'],np.array([0]*11+[1]*11+[2]))
    assert a['history_states'].shape==(n,10,23,4) and a['history_states'].dtype==np.float32
    assert a['history_valid'].shape==(n,10,23) and a['history_valid'].dtype==np.bool_
    assert a['history_position_valid'].shape==(n,10,23) and a['history_position_valid'].dtype==np.bool_
    assert np.all(a['history_valid']<=a['history_position_valid']) and np.isfinite(a['history_states'][a['history_valid']]).all()
    assert np.all(np.diff(a['source_frame_ids'],axis=1)==1)
    np.testing.assert_allclose(np.diff(a['frame_times_s'],axis=1),.1,atol=1e-8,rtol=0)
    a['history_seen']=a['history_valid'].any(1)
    a['provider_team_ids']=np.full((n,23),-1,np.int64)
    a['identity_status']=np.full((n,23),'UNKNOWN_EMPTY_SOURCE_ID',dtype='<U40')
    a['home_team_id']=np.full(n,-1,np.int64);a['away_team_id']=np.full(n,-1,np.int64)
    pins_path=ROOT/'CCPV_data_expansion_20261022/artifacts/skillcorner_download_manifest.json'
    pins={r['path']:r for r in read(pins_path)['files']};registry_summary=[]
    for mid in C['matches']:
        native=mid.split(':')[1];relative=f'data/matches/{native}/{native}_match.json';pin=pins[relative]
        p=ROOT/'CCPV_data_expansion_20261022/raw/skillcorner'/relative
        if sha(p)!=pin['sha256']:raise ValueError('Pinned roster metadata changed')
        # Decode source JSON, access home/away IDs and players id/team_id fields only.
        meta=read(p);home,away=int(meta['home_team']['id']),int(meta['away_team']['id']);assert home!=away
        registry={}
        for player in meta['players']:
            if player.get('id') is None or player.get('team_id') is None:continue
            key,team=str(player['id']),int(player['team_id'])
            if key in registry or team not in (home,away):raise ValueError('Nonunique/foreign roster identity')
            registry[key]=team
        taken=np.flatnonzero(a['match_ids']==mid)
        for w in taken:
            a['home_team_id'][w],a['away_team_id'][w]=home,away
            present=[str(x) for x in a['entity_ids'][w,:22] if str(x)]
            if len(present)!=len(set(present)):raise ValueError('Nonunique window source player ID')
            for slot,value in enumerate(a['entity_ids'][w,:22]):
                eid=str(value)
                if not eid:continue
                if eid not in registry:a['identity_status'][w,slot]='UNKNOWN_NOT_IN_SOURCE_ROSTER';continue
                if registry[eid]!=(home if int(a['teams'][slot])==0 else away):raise ValueError('Roster contradicts original home/away encoding')
                a['provider_team_ids'][w,slot]=registry[eid];a['identity_status'][w,slot]='VERIFIED_SOURCE_PLAYER_TEAM'
            if str(a['entity_ids'][w,22])!='ball':raise ValueError('Original explicit ball semantic key changed')
            a['identity_status'][w,22]='EXPLICIT_MODEL_BALL_SOURCE_ID'
        registry_summary.append(dict(match_id=mid,metadata_path=str(p.relative_to(ROOT)),metadata_sha256=pin['sha256'],home_team_id=home,away_team_id=away,player_roster_entries=len(registry),selected_windows=len(taken)))
    a['operator_candidate_slots']=np.full((n,2),-1,np.int64)
    a['operator_candidate_ids']=np.full((n,2),'',dtype=a['entity_ids'].dtype)
    a['operator_eligible_home_count']=np.zeros(n,np.int64)
    a['operator_case_status']=np.full(n,'UNSUPPORTED_FEWER_THAN_TWO',dtype='<U48')
    window_meta=[]
    for w in range(n):
        eligible=[j for j in range(22) if a['teams'][j]==0 and a['history_valid'][w,-1,j] and a['identity_status'][w,j]=='VERIFIED_SOURCE_PLAYER_TEAM']
        eligible.sort(key=lambda j:str(a['entity_ids'][w,j]));a['operator_eligible_home_count'][w]=len(eligible)
        if len(eligible)>=2:
            chosen=eligible[:2];a['operator_candidate_slots'][w]=chosen;a['operator_candidate_ids'][w]=a['entity_ids'][w,chosen];a['operator_case_status'][w]='SUPPORTED_TWO_SOURCE_HOME_T0_VALID'
        issue=slot_audit(a['entity_ids'][w],a['history_valid'][w],a['history_position_valid'][w])
        window_meta.append(dict(match_id=str(a['match_ids'][w]),source_index=int(ix[w]),bundle_row_index=int(ix[w]),period=int(a['source_period'][w]),first_history_source_frame=int(a['source_frame_ids'][w,0]),t0_source_frame=int(a['source_frame_ids'][w,-1]),operator_eligible_home_count=int(a['operator_eligible_home_count'][w]),operator_case_status=str(a['operator_case_status'][w]),operator_candidate_source_ids=a['operator_candidate_ids'][w].tolist(),retained_without_quality_selection=True,slot_mapping_audit=issue))
    # Additional flags are separate from the unchanged legacy actor selector.
    a['input_technical_supported']=np.array([r['slot_mapping_audit']['technical_supported_no_known_wrong_fullvalid_previous_id'] for r in window_meta],bool)
    a['technical_unsupported']=~a['input_technical_supported']
    a['input_technical_status']=np.where(a['input_technical_supported'],'NO_KNOWN_WRONG_FULLVALID_PREVIOUS_ID','UNSUPPORTED_WRONG_FULLVALID_PREVIOUS_ID')
    lp(OUT).mkdir(parents=True);target=OUT/'test_prefixes.npz'
    with lp(target).open('xb') as f:np.savez_compressed(f,**a)
    summary={mid:dict(rows=sum(r['match_id']==mid for r in window_meta),home_id_count_under11=sum(r['match_id']==mid and r['slot_mapping_audit']['home_id_count_under11'] for r in window_meta),known_wrong_first_history_previous_id_fullvalid_rows=sum(r['match_id']==mid and not r['slot_mapping_audit']['technical_supported_no_known_wrong_fullvalid_previous_id'] for r in window_meta),omitted_away_history_position_observed_rows=sum(r['match_id']==mid and r['slot_mapping_audit']['unvisited_history_position_valid_points']>0 for r in window_meta)) for mid in expected}
    dump(OUT/'metadata.json',dict(status='PREPARED_ALL4992_SOURCE_TEST_PREFIXES_NO_MODEL_CALLS',windows=window_meta,roster=registry_summary,access_log=access,slot_mapping_summary=summary,attack_direction='UNKNOWN_NOT_INFERRED',t0_possession='UNKNOWN_NOT_INFERRED',ball_identity='Explicit semantic ball_data key ball, not external unique numeric ID',population='Existing retrospective cache;30-frame identity union and history+future admission unchanged',future_values_materialized=False,raw_tracking_read=False,event_labels_read=False,model_calls=0,training_updates=0,compression_caveat='Skipping ZIP stream can inflate future/unselected bytes but only selected prefix bytes are converted to arrays'))
    manifest=dict(status='COMPLETED4992_TEST_PREFIX_EXTRACTION',source_sha256=sha(__file__),protocol_sha256=args.protocol_sha256,upstream_reader_sha256=LOADER_SHA,input_archive_sha256=C['source_archive_sha256'],split_sha256=C['split_sha256'],download_manifest_sha256=sha(pins_path),path=str(target.relative_to(ROOT)),sha256=sha(target),metadata_sha256=sha(OUT/'metadata.json'),rows=n,matches=C['matches'],row_indices='All original source test_indices; original order; source_indices=bundle_row_indices',history_frames=10,slot_mapping_summary=summary,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in a.items()},no_future_values_materialized=True,no_new_event_labels=True,model_calls=0,training_updates=0)
    dump(OUT/'manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],rows=n,slot_mapping_summary=summary,output=str(target.relative_to(ROOT)))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['synthetic','prepare']);p.add_argument('--execute',action='store_true');p.add_argument('--protocol-sha256');p.add_argument('--source-sha256');args=p.parse_args()
    if args.stage=='synthetic':synthetic()
    else:prepare(args)
