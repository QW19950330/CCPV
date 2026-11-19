"""Original direct-clock functions/fragments; path/output adapter only. See source_fragments.json."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib, json, math
import numpy as np
import pandas as pd
from lxml import etree

HALF_FRAME=.021

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def tag(x):return x.tag.rsplit('}',1)[-1]

def iso(x):
    try:return datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
    except (AttributeError,TypeError,ValueError):return None

def finite(x):
    try:q=float(x);return q if math.isfinite(q) else None
    except (ValueError,TypeError):return None

def frame_id(x):
    v=finite(x);return int(v) if v is not None and v.is_integer() else None

def shot_mapping(event,balls,actors,roster):
    """Original calculated frame+time only; EventTime deliberately unused."""
    n=frame_id(event.get('CalculatedFrame'));t=iso(event.get('CalculatedTimestamp'));shot=event['shot'];pid=shot.get('Player');team=shot.get('Team')
    candidates=[]
    if n is not None and t is not None:
        for section,clock in balls.items():
            b=clock.get(n)
            if b is None or b['count']!=1 or not b['finite'] or abs(b['time']-t)>HALF_FRAME:continue
            a=actors.get((section,n,team,pid))
            if a is not None and a['count']==1 and a['finite'] and abs(a['time']-t)<=HALF_FRAME and roster.get(pid)==team:candidates.append((section,abs(b['time']-t),abs(a['time']-t)))
    if len(candidates)==1:return dict(mapped=True,section=candidates[0][0],frame=n,time=t,ball_clock_error_s=candidates[0][1],actor_clock_error_s=candidates[0][2],reason=None)
    sections=[]
    if t is not None:
        for section,clock in balls.items():
            times=[q['time'] for q in clock.values()]
            if times and min(times)<=t<=max(times):sections.append(section)
    return dict(mapped=False,section=sections[0] if len(sections)==1 else None,frame=n,time=t,ball_clock_error_s=None,actor_clock_error_s=None,reason='missing_or_ambiguous_exact_calculated_frame_clock_actor_ball')

class HashingReader:
    def __init__(self,p):self.f=Path(p).open('rb');self.h=hashlib.sha256()
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);return b
    def close(self):self.f.close()

def rebuild(source,DATA):
    shotrows=[];matches=[]
    for mid in source['selected_match_ids']:
        entries={kind:next(x for x in source['downloaded_files'] if mid in x['path'] and kind in x['path']) for kind in ['matchinformation','events_raw','positions']}
        for kind in ['matchinformation','events_raw']:
            assert sha(DATA/entries[kind]['path'])==entries[kind]['sha256']
        tree=etree.parse(str(DATA/entries['matchinformation']['path']));root=tree.getroot();general=next(x for x in root.iter() if tag(x)=='General');home,away=general.get('HomeTeamId'),general.get('GuestTeamId');assert home and away
        roster={}
        for team in root.iter():
            if tag(team)=='Team':
                for person in team.iter():
                    if tag(person)=='Player':
                        pid=person.get('PersonId');old=roster.get(pid);assert old in [None,team.get('TeamId')];roster[pid]=team.get('TeamId')
        eroot=etree.parse(str(DATA/entries['events_raw']['path'])).getroot();versions=Counter();final={};attrs=Counter()
        for event in eroot:
            if tag(event)=='Event':versions[event.get('EventId')]+=1;final[event.get('EventId')]=event;attrs.update(event.attrib.keys())
        active=[x for x in final.values() if not any(tag(n)=='Delete' for n in x.iter())];shots=[]
        for event in active:
            shot=next((x for x in event.iter() if tag(x)=='ShotAtGoal'),None)
            if shot is not None:shots.append(dict(event.attrib,shot=dict(shot.attrib),event_id=event.get('EventId'),goal=any(tag(x)=='SuccessfulShot' for x in shot.iter())))
        matchname='DFL-MAT-'+mid
        wanted={frame_id(e.get('CalculatedFrame')) for e in shots}-{None}
        balls=defaultdict(dict);actors={};context=None;reader=HashingReader(DATA/entries['positions']['path']);posrootids=set();framesetcounts=Counter()
        try:
            parser=etree.iterparse(reader,events=['start','end'],huge_tree=True,recover=False)
            for ev,el in parser:
                name=tag(el)
                if ev=='start':
                    for key,value in el.attrib.items():
                        if key.lower() in ['matchid','match_id']:posrootids.add(value)
                    if name=='FrameSet':context=(el.get('GameSection'),el.get('TeamId'),el.get('PersonId'));framesetcounts[context[0]]+=1
                elif name=='Frame' and context:
                    section,team,pid=context;n=frame_id(el.get('N'));t=iso(el.get('T'));good=finite(el.get('X')) is not None and finite(el.get('Y')) is not None
                    if n is not None and t is not None:
                        key=(section,n,team,pid)
                        if team=='BALL':
                            old=balls[section].get(n);balls[section][n]=dict(time=t,count=1 if old is None else old['count']+1,finite=good and (old is None or old['finite']))
                        elif n in wanted:
                            old=actors.get(key);actors[key]=dict(time=t,count=1 if old is None else old['count']+1,finite=good and (old is None or old['finite']))
                    el.clear()
                    while el.getprevious() is not None:del el.getparent()[0]
                elif name=='FrameSet':context=None;el.clear()
        finally:reader.close()
        digest=reader.h.hexdigest();assert digest==entries['positions']['sha256'];assert not posrootids or posrootids=={matchname}
        events=[];invalid=set()
        for event in shots:
            mapping=shot_mapping(event,balls,actors,roster);row=dict(match_id=matchname,event_id=event['event_id'],team=event['shot'].get('Team'),actor=event['shot'].get('Player'),goal=event['goal'],**mapping)
            if row['team'] not in [home,away]:row['mapped']=False;row['reason']='shot_team_not_original_match_team'
            if not row['mapped']:invalid.add(row['section'])
            events.append(row);shotrows.append(row)
        per=dict(match_id=matchname,shots=len(shots),mapped=sum(e['mapped'] for e in events),unmapped=sum(not e['mapped'] for e in events),positions_sha256=digest,positions_bytes=entries['positions']['bytes'])
        matches.append(per)
        print(json.dumps(dict(stage='direct',**per)),flush=True)
    return shotrows,matches
