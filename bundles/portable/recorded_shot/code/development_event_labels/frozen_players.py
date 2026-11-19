"""Original two-point PLAYER features and conservative labels; no training or old entry point."""
from __future__ import annotations
import os, csv, gzip, hashlib, json, math, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from lxml import etree

NAMES=['mean_abs_x','mean_abs_y','mean_x_squared','mean_y_squared','mean_distance_nearer_goal','min_distance_nearer_goal','n_current_players','n_backward_velocity_players','n_radial_velocity_players','fraction_backward_velocity','fraction_radial_velocity','speed_missing','radial_missing','mean_speed','rms_speed','mean_toward_nearer_goal_velocity']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def put(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def pin(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)

def table(p,rows):
    with open(p,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rows(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def ns(s):
    m=re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})',str(s))
    if m is None:return -1
    d=datetime.fromisoformat(m[1]+m[3].replace('Z','+00:00')).astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (d.days*86400+d.seconds)*10**9+int((m[2] or '').ljust(9,'0'))

def yes(x):return str(x).lower()=='true'

def tag(e):return e.tag.rsplit('}',1)[-1]

def geometry(x,v,L):
    """Finite current players only. Velocity rows may be NaN; every count is shared."""
    x=np.asarray(x,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    assert x.ndim==2 and x.shape==v.shape and x.shape[1]==2 and len(x)>0 and np.isfinite(x).all()
    d=np.hypot(L/2-np.abs(x[:,0]),x[:,1]);vm=np.isfinite(v).all(1);rm=vm&(d>0)
    speed=np.linalg.norm(v[vm],axis=1)
    radial=((L/2-np.abs(x[rm,0]))*np.sign(x[rm,0])*v[rm,0]-x[rm,1]*v[rm,1])/d[rm]
    return np.array([np.abs(x[:,0]).mean(),np.abs(x[:,1]).mean(),(x[:,0]**2).mean(),(x[:,1]**2).mean(),d.mean(),d.min(),len(x),vm.sum(),rm.sum(),vm.mean(),rm.mean(),int(not vm.any()),int(not rm.any()),speed.mean() if len(speed) else np.nan,np.sqrt((speed**2).mean()) if len(speed) else np.nan,radial.mean() if len(radial) else np.nan],float)

def anyshot(events,period,t0,complete):
    relevant=[e for e in events if not e['section'] or e['section']==period]
    unknown=[e.get('native_event_id','') for e in relevant if not yes(e['clock_accepted']) or not yes(e['source_shot_validity_accepted']) or not e['recorded_UTC_ns']]
    links=[e['native_event_id'] for e in relevant if yes(e['clock_accepted']) and yes(e['source_shot_validity_accepted']) and e['section']==period and e['recorded_UTC_ns'] and t0<int(e['recorded_UTC_ns'])<=t0+10_000_000_000]
    return (int(bool(links)) if complete and not unknown else -1),unknown,links

class HashReader:
    def __init__(self,p):self.f=open(p,'rb');self.h=hashlib.sha256();self.n=0
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);self.n+=len(b);return b

def metadata(record,mid):
    assert sha(record['path'])==record['sha256'];tree=etree.parse(record['path']);g=next(e for e in tree.iter() if tag(e)=='General');assert g.get('MatchId')==mid
    teams=(g.get('HomeTeamId'),g.get('GuestTeamId'));assert all(teams) and teams[0]!=teams[1];roster={}
    for team in tree.iter():
        if tag(team)!='Team':continue
        tid=team.get('TeamId');assert tid in teams
        for e in team.iter():
            if tag(e)=='Player':
                pid=e.get('PersonId');assert pid and (pid not in roster or roster[pid]==tid);roster[pid]=tid
    env=next(e for e in tree.iter() if tag(e)=='Environment');L,W=float(env.get('PitchX')),float(env.get('PitchY'));assert (L,W)==(105.,68.)
    return roster,L

def scan(record,mid,wanted,roster,budget_check):
    """Read PLAYER N/T for clock coverage, XY only requested N and N−5. BALL values never fetched."""
    points={};clock={};context=None;reader=HashReader(record['path']);counts=Counter()
    for ev,e in etree.iterparse(reader,events=('start','end'),tag=('FrameSet','Frame'),huge_tree=True):
        if ev=='start' and tag(e)=='FrameSet':context=(e.get('GameSection'),e.get('TeamId'),e.get('PersonId'))
        elif ev=='end' and tag(e)=='Frame':
            counts['all_frame_bytes_streamed']+=1
            period,team,pid=context or (None,None,None)
            if pid in roster and roster[pid]==team:
                try:N=int(e.get('N'))
                except (TypeError,ValueError):N=None
                if N is not None:
                    T=ns(e.get('T'));key=(period,N)
                    if key not in clock:clock[key]=T
                    elif clock[key]!=T:clock[key]=-1
                    if key in wanted:
                        k=(period,N,pid)
                        if k in points:points[k]=(-1,np.full(2,np.nan));counts['duplicate_id_N']+=1
                        else:
                            try:x=np.array([float(e.get('X')),float(e.get('Y'))])
                            except (ValueError,TypeError):x=np.full(2,np.nan)
                            points[k]=(T,x)
            e.clear()
            while e.getprevious() is not None:del e.getparent()[0]
            if counts['all_frame_bytes_streamed']%100000==0:budget_check()
        elif ev=='end' and tag(e)=='FrameSet':context=None;e.clear()
    while reader.read(1<<20):budget_check()
    reader.f.close();assert reader.h.hexdigest()==record['sha256'] and reader.n==record['bytes']
    return points,clock,dict(match_id=mid,sha256=reader.h.hexdigest(),bytes=reader.n,counts=dict(counts))

def make_features(points,clock,period,N,roster,L):
    t=clock.get((period,N),-1);xs=[];vs=[]
    if t<0:return np.full(16,np.nan),t
    for pid in sorted(roster):
        T,x=points.get((period,N,pid),(-1,np.full(2,np.nan)))
        if T!=t or not np.isfinite(x).all():continue
        past,p=points.get((period,N-5,pid),(-1,np.full(2,np.nan)))
        good=past==t-200_000_000 and clock.get((period,N-5),-1)==past and np.isfinite(p).all()
        xs.append(x);vs.append((x-p)/.2 if good else np.full(2,np.nan))
    return (geometry(xs,vs,L) if xs else np.full(16,np.nan)),t

def freeze(folder,name):
    records=[pin(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!=name]
    put(folder/name,dict(status='COMPLETE',files=records));return pin(folder/name)

def build_data(out, plan, guard):
    grid=rows(next(q['path'] for q in plan['cached_inputs'] if 'prefix_grid' in Path(q['path']).name));assert len(grid)==13372
    # The whitelist deliberately discards all old eligibility, team, live, ball-T and label fields.
    counts=Counter(g['match_id'] for g in grid);scope=plan['scope']['matches'];assert set(scope)<=set(counts)
    cohort=[dict(grid_index=i,match_id=g['match_id'],period=g['period'],N=int(g['anchor_source_N'])) for i,g in enumerate(grid) if g['match_id'] in scope]
    put(out/'candidate_population_before_eligibility.json',dict(all_grid=13372,by_match=dict(counts),six_scope=len(cohort),fixed_matches=scope,excluded_match_ids=sorted(set(counts)-set(scope))))
    clocks=rows(next(q['path'] for q in plan['cached_inputs'] if 'event_clocks' in Path(q['path']).name));events=[e for e in clocks if e['scenario']=='AUTHOR3_PLUS_DIRECT_OTHER4' and e['match_id'] in scope]
    assert len({(e['match_id'],e['native_event_id']) for e in events})==len(events)
    feature_rows=[];ledger=[];source_records=[];links=[];event_summary=[]
    for mid in scope:
        guard();subset=[g for g in cohort if g['match_id']==mid];wanted={(g['period'],g['N']+d) for g in subset for d in (0,-5)}
        meta=next(q for q in plan['raw_sources'] if mid in Path(q['path']).name and 'matchinformation' in Path(q['path']).name);pos=next(q for q in plan['raw_sources'] if mid in Path(q['path']).name and 'positions' in Path(q['path']).name)
        roster,L=metadata(meta,mid);print('SCAN',mid,len(subset),flush=True);points,clock,record=scan(pos,mid,wanted,roster,guard);source_records.append(record)
        es=[e for e in events if e['match_id']==mid]
        event_summary.append(dict(match_id=mid,scenario='AUTHOR3_PLUS_DIRECT_OTHER4',event_rows=len(es),accepted=sum(yes(e['clock_accepted']) and yes(e['source_shot_validity_accepted']) for e in es),unknown=sum(not yes(e['clock_accepted']) or not yes(e['source_shot_validity_accepted']) for e in es),clock_statuses='|'.join(sorted(set(e['clock_status'] for e in es)))))
        for g in subset:
            N,period=g['N'],g['period'];x,t0=make_features(points,clock,period,N,roster,L)
            coverage=t0>=0 and all(clock.get((period,N+k),-1)==t0+k*40_000_000 for k in range(251))
            y,unknown,assoc=anyshot(es,period,t0,coverage);support=bool(np.isfinite(x[:13]).all());eligible=support and y>=0
            feature_rows.append(x);ledger.append(dict(**g,t0_UTC_ns=t0,player_support=support,complete_recorded_clock10=coverage,unknown_shot_ids='|'.join(unknown),eligible=eligible,y=y,reason='|'.join(s for s,b in [('NO_PLAYER_SUPPORT',not support),('INCOMPLETE_PLAYER_CLOCK10',not coverage),('UNRESOLVED_EVENT_CLOCK_OR_VALIDITY',bool(unknown))] if b)))
            links.extend(dict(grid_index=g['grid_index'],match_id=mid,native_event_id=eid,eligible=eligible) for eid in assoc)
        del points,clock
    X=np.array(feature_rows);matches=np.array([g['match_id'] for g in ledger]);Y=np.array([g['y'] for g in ledger]);eligible=np.array([g['eligible'] for g in ledger]);assert len(X)==len(cohort)
    np.savez_compressed(out/'features.npz',X=X,y=Y,eligible=eligible,match=matches,grid_index=np.array([g['grid_index'] for g in ledger]),feature_names=np.array(NAMES))
    table(out/'all_anchor_ledger.csv',ledger);table(out/'event_clock_summary.csv',event_summary)
    if links:table(out/'event_links.csv',links)
    put(out/'source_scan_receipt.json',source_records);freeze(out,'data_freeze.json')
    return {'candidates':len(ledger),'eligible':int(eligible.sum()),'positive_eligible':int(Y[eligible].sum()),'unknown':int((Y<0).sum()),'source_records':source_records}
