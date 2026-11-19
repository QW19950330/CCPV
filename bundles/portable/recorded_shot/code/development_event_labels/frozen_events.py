"""Original parser/identity/clock functions; exact source excerpts, no original entry point."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import csv, hashlib, json, math, os, re
from lxml import etree
import numpy as np
import pandas as pd

MATCHES=('J03WMX','J03WN1','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9')

AUTHOR=('J03WMX','J03WN1','J03WPY')

SECTIONS={1:'firstHalf',2:'secondHalf'}

def xp(p):
    q=str(Path(p).resolve())
    return ('\\\\?\\'+q) if os.name=='nt' and not q.startswith('\\\\?\\') else q

def read(p):
    with open(xp(p),encoding='utf-8-sig') as f:return json.load(f)

def sha(p):
    h=hashlib.sha256()
    with open(xp(p),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    with open(xp(p),'x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def csvrows(p):
    with open(xp(p),encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['no_rows']
    with open(xp(p),'x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def tag(x):return x.tag.rsplit('}',1)[-1]

def yn(x):return str(x).strip().lower()=='true'

def jint(x):
    if x is None or isinstance(x,bool) or not re.fullmatch(r'[+-]?\d+',str(x).strip()):return None
    return int(str(x).strip())

def saved_int(x):
    if x is None or isinstance(x,bool):return None
    try:
        q=Decimal(str(x));return int(q) if q.is_finite() and q==q.to_integral_value() else None
    except (InvalidOperation,ValueError):return None

def ns(x,naive_utc=False):
    if x is None or not str(x).strip():return None
    try:
        t=pd.Timestamp(x)
        if pd.isna(t) or (t.tzinfo is None and not naive_utc):return None
        return int((t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')).value)
    except (TypeError,ValueError,OverflowError):return None

def jf(x):
    try:v=float(x);return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None

def sec(x):
    a,b=str(x).split(':');return int(a)*60+float(b)

def period_origins(g):
    out={}
    for p in (1,2):
        s=g[g.period_id==p];f=s.frame_id.to_numpy(float);t=np.array([sec(x) for x in s.synced_ts])
        if not len(f) or not np.isfinite(f).all() or not np.equal(f,np.round(f)).all():out[p]=None;continue
        lo=int(np.ceil(np.max(f-25*t-.500001)));hi=int(np.floor(np.min(f-25*t+.500001)))
        out[p]=lo if lo==hi and np.equal(np.round(lo+25*t),f).all() else None
    return out

def validity_hints(e):
    return [{'tag':tag(n),'attribute':k,'value':v} for n in e.iter() for k,v in ([('__node_tag__',tag(n))]+list(n.attrib.items())) if re.search(r'cancel|disallow|invalid',str(k)+' '+str(v),re.I)]

def event_state(versions):
    final=versions[-1][1]
    if any(tag(x)=='Delete' for x in final.iter()):return 'CONFIRMED_FINAL_DELETE'
    shots=[x for x in final.iter() if tag(x)=='ShotAtGoal']
    if len(shots)==0:return 'CONFIRMED_FINAL_NONSHOT_REVISION'
    if len(shots)!=1:return 'UNKNOWN_MULTIPLE_SHOT_NODES'
    if validity_hints(final):return 'UNKNOWN_CANCELLATION_OR_VALIDITY_SEMANTICS'
    return 'ACTIVE_NATIVE_SHOT'

def period_for(stamp,bounds):
    if stamp is None:return None
    hits=[p for p,b in bounds.items() if len(b['KickOff'])==len(b['FinalWhistle'])==1 and b['KickOff'][0]<=stamp<=b['FinalWhistle'][0]]
    return hits[0] if len(hits)==1 else None

def subtype(shot):
    parents={tag(x) for x in shot.iterancestors()}
    return 'shot_penalty' if 'Penalty' in parents else 'shot_freekick' if 'FreeKick' in parents else 'shot'

def discover_match(mid,pins):
    mr=next(r for r in pins['raw_event_metadata_files'] if mid in Path(r['path']).name and 'matchinformation' in Path(r['path']).name)
    er=next(r for r in pins['raw_event_metadata_files'] if mid in Path(r['path']).name and 'events_raw' in Path(r['path']).name)
    meta=etree.parse(xp(mr['path'])).getroot();general=next(x for x in meta.iter() if tag(x)=='General')
    home,away=general.get('HomeTeamId'),general.get('GuestTeamId');assert home and away and home!=away
    roster={};objects={}
    for team in meta.iter():
        if tag(team)!='Team':continue
        tid=team.get('TeamId');assert tid in (home,away)
        for p in team.iter():
            if tag(p)!='Player':continue
            pid=p.get('PersonId');shirt=jint(p.get('ShirtNumber'));obj=('home_' if tid==home else 'away_')+str(shirt) if shirt is not None else None
            assert pid and (pid not in roster or roster[pid]==(obj,tid))
            if obj is not None:assert obj not in objects or objects[obj]==pid;objects[obj]=pid
            roster[pid]=(obj,tid)
    eroot=etree.parse(xp(er['path'])).getroot();groups=defaultdict(list);versionrows=[];context=[]
    for i,e in enumerate(eroot):
        if tag(e)!='Event':continue
        assert e.get('MatchId')=='DFL-MAT-'+mid
        eid=e.get('EventId');key=eid if eid else '__MISSING_NATIVE_ID_DOC_'+str(i)
        groups[key].append((i,e))
        versionrows.append(dict(match_id='DFL-MAT-'+mid,audit_version_key=mid+'|doc|'+str(i),native_event_id=eid,document_index=i,raw_EventTime=e.get('EventTime'),raw_CalculatedFrame=e.get('CalculatedFrame'),raw_CalculatedTimestamp=e.get('CalculatedTimestamp'),direct_child_tags='|'.join(tag(x) for x in e),shot_node_count=sum(tag(x)=='ShotAtGoal' for x in e.iter()),has_Delete=any(tag(x)=='Delete' for x in e.iter()),event_attributes_json=json.dumps(dict(e.attrib),ensure_ascii=False)))
        for node in e.iter():
            for k,v in node.attrib.items():
                if re.search(r'possession|ballstatus|playstate|gamestate|phase|control|inplay|deadball',k,re.I):context.append(dict(match_id='DFL-MAT-'+mid,native_event_id=eid,document_index=i,node_tag=tag(node),field=k,literal=v,semantics='UNVERIFIED_CONTEXT_CANDIDATE_NOT_POSSESSION_OR_OPENPLAY_CERTIFICATION'))
    bounds=defaultdict(lambda:defaultdict(list))
    for vv in groups.values():
        e=vv[-1][1]
        if any(tag(x)=='Delete' for x in e.iter()):continue
        for c in e:
            if tag(c) in ('KickOff','FinalWhistle') and c.get('GameSection') in ('firstHalf','secondHalf'):
                p=1 if c.get('GameSection')=='firstHalf' else 2;t=ns(e.get('EventTime'))
                if t is not None:bounds[p][tag(c)].append(t)
    candidates=[];active={}
    for key,vv in groups.items():
        if not any(any(tag(x)=='ShotAtGoal' for x in e.iter()) for _,e in vv):continue
        e=vv[-1][1];state=event_state(vv);ss=[x for x in e.iter() if tag(x)=='ShotAtGoal'];s=ss[0] if len(ss)==1 else None
        team=s.get('Team') if s is not None else None;actor=s.get('Player') if s is not None else None;rec=roster.get(actor);p=period_for(ns(e.get('EventTime')),bounds)
        r=dict(match_id='DFL-MAT-'+mid,audit_event_id=mid+'|native|'+key,native_event_id=e.get('EventId'),native_id_namespace='IDSSE_Sportec_raw_EventId' if e.get('EventId') else 'MISSING_NOT_INVENTED',version_count=len(vv),version_document_indices='|'.join(str(i) for i,_ in vv),final_document_index=vv[-1][0],final_state=state,final_child_tags='|'.join(tag(x) for x in e),team=team,actor=actor,roster_team=rec[1] if rec else None,actor_object_id=rec[0] if rec else None,team_roster_consistent=bool(rec and rec[1]==team),period_via_whistle_EventTime=p,period_evidence='UNIQUE_NATIVE_EVENTTIME_WHISTLE_BRACKET' if p else 'UNKNOWN',EventTime=e.get('EventTime'),eventtime_UTC_ns=ns(e.get('EventTime')),CalculatedFrame_literal=e.get('CalculatedFrame'),CalculatedFrame=jint(e.get('CalculatedFrame')),CalculatedTimestamp_literal=e.get('CalculatedTimestamp'),calculated_UTC_ns=ns(e.get('CalculatedTimestamp')),shot_subtype=subtype(s) if s is not None else None,calculated_nodes_json=json.dumps([{'tag':tag(n),'attributes':{k:v for k,v in n.attrib.items() if k in ('CalculatedFrame','CalculatedTimestamp')}} for n in e.iter() if any(k in n.attrib for k in ('CalculatedFrame','CalculatedTimestamp'))],ensure_ascii=False),validity_hints_json=json.dumps(validity_hints(e),ensure_ascii=False),physical_time_uncertainty_s=None,physical_synchronization='NOT_VALIDATED',human_review='NOT_CONDUCTED')
        candidates.append(r)
        if len(ss)==1 and state not in ('CONFIRMED_FINAL_DELETE','CONFIRMED_FINAL_NONSHOT_REVISION') and r['native_event_id']:active[r['native_event_id']]=r
    return home,away,versionrows,candidates,active,context

def authors(pins,active,saved):
    rows=[];saved_by={(r['match_id'],r['author_index']):r for r in saved}
    for mid in AUTHOR:
        gpath=next(r['path'] for r in pins['author_parquet_files'] if '/gt/' in r['repository_path'] and mid in Path(r['path']).name);upath=next(r['path'] for r in pins['author_parquet_files'] if '/unsynced/' in r['repository_path'] and mid in Path(r['path']).name)
        g=pd.read_parquet(xp(gpath));u=pd.read_parquet(xp(upath));assert g.index.equals(u.index)
        for k in ('period_id','player_id','spadl_type'):assert g[k].equals(u[k])
        origins=period_origins(g)
        rawkeys=defaultdict(list);author_keys=defaultdict(list)
        for eid,r in active[mid].items():
            if r['period_via_whistle_EventTime'] and r['team_roster_consistent'] and r['actor_object_id']:
                rawkeys[(r['period_via_whistle_EventTime'],r['actor_object_id'],r['shot_subtype'],r['eventtime_UTC_ns'])].append(eid)
        for idx,a in u.iterrows():
            if a.spadl_type not in ('shot','shot_penalty','shot_freekick'):continue
            key=(int(a.period_id),str(a.player_id),str(a.spadl_type),ns(a.utc_timestamp,naive_utc=True));author_keys[key].append(int(idx))
        for key,indices in author_keys.items():
            for idx in indices:
                cand=rawkeys[key];hit=cand[0] if len(cand)==len(indices)==1 else None;a=g.loc[idx];old=saved_by.get((mid,str(idx)));assert old is not None
                same=hit==old['raw_event_id'];frame=saved_int(a.frame_id);p=int(a.period_id);origin=origins.get(p)
                frame_ok=frame is not None and origin is not None and abs((frame-origin)/25-sec(a.synced_ts))<=.021
                accepted=bool(hit and same and frame_ok and yn(old['author_clock_accepted']) and frame==saved_int(old['annotated_frame']) and ns(u.loc[idx].utc_timestamp,True)==jint(old['unsynced_UTC_ns']))
                rows.append(dict(match_id='DFL-MAT-'+mid,author_index=idx,native_event_id=hit,raw_candidate_ids='|'.join(cand),raw_candidate_count=len(cand),author_signature_count=len(indices),identity_matches_prior_crosswalk=same,period=p,actor_object_id=key[1],shot_subtype=key[2],unsynced_UTC_ns=key[3],annotated_frame=frame,annotated_relative_ts=str(a.synced_ts),period_zero_frame=origin,clock_accepted=accepted,recorded_UTC_ns=jint(old['exact_raw_UTC_ns']) if accepted else None,clock_status='SAVED_EXACT_FRAME_T_AUTHOR_ANNOTATION_REUSED_NOT_PHYSICAL_TRUTH' if accepted else 'UNKNOWN_NO_GUESS',prior_source_clock_reason=old['clock_reason'],outcome_used_for_identity=False,physical_time_uncertainty_s=None))
    return rows

def assemble(pins, prior, saved, conflict):
    direct={(r['match_id'],r['event_id']):r for r in prior};assert len(direct)==len(prior)
    allv=[];candidates=[];context=[];active={};teams={}
    for mid in MATCHES:
        h,a,v,c,b,x=discover_match(mid,pins);teams[mid]=(h,a);allv+=v;candidates+=c;active[mid]=b;context+=x
    assert {('DFL-MAT-'+m,e) for m,rs in active.items() for e in rs}==set(direct),'Active native IDs differ from pinned direct ledger; inspect source parsing without forcing counts.'
    author=authors(pins,active,saved);ac={(r['match_id'],r['native_event_id']):r for r in author if r['clock_accepted']};assert len(ac)==len([r for r in author if r['clock_accepted']])
    conflict_residual=None
    cc=[r for r in conflict['records_at_explicit_and_two_adjacent_frames'] if jint(r['frame'].get('N'))==conflict['calculated_frame'] and (r['frameset'].get('TeamId')=='BALL' or r['frameset'].get('PersonId')==conflict['shot_actor'])]
    ctimes={ns(r['frame']['T']) for r in cc};assert len(ctimes)==1;conflict_residual=(next(iter(ctimes))-ns(conflict['calculated_timestamp']))/1e9
    clocks=[]
    for mid,events in active.items():
        for eid,r in events.items():
            k=('DFL-MAT-'+mid,eid);d=direct[k]
            assert (r['team'] or '')==d['team'] and (r['actor'] or '')==d['actor'] and r['CalculatedFrame']==saved_int(d['frame'])
            if r['calculated_UTC_ns'] is not None:assert abs(r['calculated_UTC_ns']/1e9-float(d['time']))<=1e-6
            valid=yn(d['mapped']) and r['team_roster_consistent'] and bool(r['team']) and bool(r['actor']);reason='DIRECT_EXACT_CALCULATED_CLOCK_PRIOR_PROOF' if valid else ('MISSING_LITERAL_CALCULATED_CLOCK' if r['CalculatedFrame'] is None or r['calculated_UTC_ns'] is None else 'EXACT_CALCULATED_CLOCK_CONFLICT_OR_UNSUPPORTED')
            base=dict(match_id=k[0],native_event_id=eid,team=r['team'],actor=r['actor'],raw_period_via_EventTime=r['period_via_whistle_EventTime'],physical_synchronization='NOT_VALIDATED',physical_time_interval_lo_s=None,physical_time_interval_hi_s=None,outcome_used_for_mapping=False,source_shot_validity_accepted=r['final_state']=='ACTIVE_NATIVE_SHOT',source_final_state=r['final_state'])
            row=base|dict(scenario='DIRECT_LITERAL',clock_accepted=valid,section=d['section'] if d['section'] else SECTIONS.get(r['period_via_whistle_EventTime']),recorded_frame=r['CalculatedFrame'],recorded_UTC_ns=r['calculated_UTC_ns'] if valid else None,clock_status=reason,prior_ball_abs_residual_s=jf(d['ball_clock_error_s']),prior_actor_abs_residual_s=jf(d['actor_clock_error_s']),explicit_same_frame_signed_residual_s=conflict_residual if k==(conflict['match_id'],conflict['event_id']) else None)
            clocks.append(row)
            alt=dict(row);alt['scenario']='AUTHOR3_PLUS_DIRECT_OTHER4'
            if mid in AUTHOR:
                a=ac.get(k);alt.update(clock_accepted=bool(a),section=SECTIONS.get(a['period']) if a else SECTIONS.get(r['period_via_whistle_EventTime']),recorded_frame=a['annotated_frame'] if a else None,recorded_UTC_ns=a['recorded_UTC_ns'] if a else None,clock_status='AUTHOR3_SAME_EVENT_ANNOTATED_CLOCK_PRIOR_EXACT_FRAME_PROOF' if a else 'UNKNOWN_AUTHOR_IDENTITY_OR_CLOCK',author_index=a['author_index'] if a else None)
            clocks.append(alt)
    represented={(r['match_id'],r['native_event_id']) for r in clocks if r['native_event_id']}
    for r in candidates:
        if r['final_state'] not in ('CONFIRMED_FINAL_DELETE','CONFIRMED_FINAL_NONSHOT_REVISION') and (not r['native_event_id'] or (r['match_id'],r['native_event_id']) not in represented):
            for scenario in ('DIRECT_LITERAL','AUTHOR3_PLUS_DIRECT_OTHER4'):
                clocks.append(dict(match_id=r['match_id'],native_event_id=r['native_event_id'],audit_event_id=r['audit_event_id'],team=r['team'],actor=r['actor'],scenario=scenario,clock_accepted=False,source_shot_validity_accepted=False,source_final_state=r['final_state'],section=SECTIONS.get(r['period_via_whistle_EventTime']),recorded_frame=None,recorded_UTC_ns=None,clock_status='UNKNOWN_NATIVE_ID_OR_SHOT_NODE_IDENTITY_NO_GUESS',physical_synchronization='NOT_VALIDATED',outcome_used_for_mapping=False))
    return {'all_event_versions.csv':allv,'shot_candidates.csv':candidates,'event_clocks.csv':clocks,'author_clock.csv':author,'context_fields.csv':context}, conflict_residual
