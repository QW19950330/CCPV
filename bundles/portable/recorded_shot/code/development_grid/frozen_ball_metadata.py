"""Verbatim source functions and exact grid/cohort expressions; no legacy run entry point."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import argparse,csv,gzip,hashlib,json,os,re
from lxml import etree

STEP=40_000_000
MATCHES=('J03WMX','J03WN1','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9')
PERIODS={'firstHalf','secondHalf','firstHalfExtra','secondHalfExtra'}

def xp(p):
    s=str(Path(p).resolve());return '\\\\?\\'+s if os.name=='nt' and not s.startswith('\\\\?\\') else s

def sha(p):
    h=hashlib.sha256()
    with open(xp(p),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def tag(x):return x.tag.rsplit('}',1)[-1]

def integer(x):
    if x is None or isinstance(x,bool) or not re.fullmatch(r'\d+',str(x).strip()):return None
    return int(str(x).strip())

def utcns(x):
    if not x:return None
    try:
        m=re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})',str(x))
        if m is None:return None
        # Parse the integral second separately: datetime must never truncate submicrosecond source precision.
        t=datetime.fromisoformat(m.group(1)+m.group(3).replace('Z','+00:00'))
        d=t.astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
        return (d.days*86400+d.seconds)*1_000_000_000+int((m.group(2) or '').ljust(9,'0'))
    except (TypeError,ValueError,OverflowError):return None

def literal_code(x,allowed):
    z=integer(x);return z if z in allowed else None

def extract(el):
    # These are the ONLY accessed Frame attributes. Poison tests reject all other keys.
    n,t,p,s=(el.get(k) for k in ('N','T','BallPossession','BallStatus'))
    return dict(N_literal=n,T_literal=t,possession_literal=p,status_literal=s,N=integer(n),T_ns=utcns(t),possession=literal_code(p,(1,2)),live=literal_code(s,(0,1)))

def add(table,row):
    n=row['N']
    if n is None:return
    a=table.get(n);v=(row['T_ns'],row['possession'],row['live'])
    if a is None:table[n]=(1,*v,False,False,False)
    else:table[n]=(a[0]+1,a[1],a[2],a[3],a[4] or a[1]!=v[0],a[5] or a[2]!=v[1],a[6] or a[3]!=v[2])

def at(table,n,home,away):
    a=table.get(n)
    if a is None:return dict(frame_present=False,unique_frame=False,T_ns=None,possession=None,live=None,team=None)
    unique=a[0]==1;valid=unique and a[1] is not None
    p=a[2] if valid else None;s=a[3] if valid else None
    return dict(frame_present=True,unique_frame=unique,T_ns=a[1] if unique else None,possession=p,live=s,team=home if p==1 else away if p==2 else None)

def clock(table,n,t,lo,hi):
    if t is None:return False,'ANCHOR_CLOCK_UNKNOWN'
    reasons=set()
    for k in range(lo,hi+1):
        a=table.get(n+k)
        if a is None:reasons.add('MISSING_N');continue
        if a[0]!=1:reasons.add('DUPLICATE_N');continue
        if a[1] is None:reasons.add('INVALID_T');continue
        if a[1]!=t+k*STEP:reasons.add('CADENCE_OR_N_T_CONFLICT')
    return not reasons,'|'.join(sorted(reasons))

def grid(table,origin):
    if origin is None or not table:return []
    end=max(table)
    if end-origin>1_000_000:raise ValueError('Resource guard: source period span exceeds1000000; abort, not filter')
    return range(origin+24,end+1,75)

class HashReader:
    def __init__(self,p):self.f=open(xp(p),'rb');self.h=hashlib.sha256();self.bytes=0
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);self.bytes+=len(b);return b
    def close(self):self.f.close()

def team_metadata(pin):
    root=etree.parse(xp(pin['path'])).getroot();g=next(x for x in root.iter() if tag(x)=='General')
    home,away=g.get('HomeTeamId'),g.get('GuestTeamId');native=g.get('MatchId')
    if not home or not away or home==away:raise ValueError('Unique home/guest metadata required')
    if native and native!=pin['match_id']:raise ValueError('Metadata MatchId mismatch')
    return home,away,native

def scan(pin,cache_writer):
    tables=defaultdict(dict);origins={};rawcounts=Counter();validn=Counter();invalidn=Counter();invalidt=Counter();nonincreasing=Counter();previous={};values=Counter();ballsets=Counter();seq=0
    r=HashReader(pin['path']);ctx=None
    try:
        for ev,el in etree.iterparse(r,events=('start','end'),tag=('FrameSet','Frame'),huge_tree=True):
            if ev=='start' and tag(el)=='FrameSet':
                tid=el.get('TeamId');isball=isinstance(tid,str) and tid.casefold()=='ball';period=el.get('GameSection')
                if isball:
                    native=el.get('MatchId')
                    if native!=pin['match_id']:raise ValueError('BALL FrameSet native MatchId mismatch')
                    ctx=period if period is not None else '__MISSING_PERIOD__';ballsets[ctx]+=1
                else:ctx=None
            elif ev=='end' and tag(el)=='Frame':
                if ctx is not None:
                    row=extract(el);seq+=1;rawcounts[ctx]+=1
                    n=row['N'];t=row['T_ns']
                    if n is None:invalidn[ctx]+=1
                    else:
                        validn[ctx]+=1
                        if ctx not in origins:origins[ctx]=n
                        if ctx in previous and n<=previous[ctx]:nonincreasing[ctx]+=1
                        previous[ctx]=n
                    if t is None:invalidt[ctx]+=1
                    for field,key in [('BallPossession','possession_literal'),('BallStatus','status_literal')]:values[(ctx,field,'__MISSING__' if row[key] is None else row[key])]+=1
                    add(tables[ctx],row)
                    cache_writer.writerow(dict(file_match_id=pin['match_id'],period=ctx,source_row=seq,**row))
                el.clear()
                while el.getprevious() is not None:del el.getparent()[0]
            elif ev=='end' and tag(el)=='FrameSet':
                ctx=None;el.clear()
                while el.getprevious() is not None:del el.getparent()[0]
    finally:r.close()
    if r.h.hexdigest()!=pin['sha256'] or r.bytes!=pin['bytes']:raise ValueError('Full original source byte pin differs')
    if not rawcounts:raise ValueError('No BALL source rows; stop rather than invent')
    return dict(tables=tables,origins=origins,rawcounts=rawcounts,validn=validn,invalidn=invalidn,invalidt=invalidt,nonincreasing=nonincreasing,values=values,ballsets=ballsets,bytes=r.bytes,sha256=r.h.hexdigest())

def grid_rows(full,z,home,away):
    for period in z['rawcounts']:
        d=z['tables'][period];origin=z['origins'].get(period)
        for n in grid(d,origin):
            a=at(d,n,home,away);hc,hr=clock(d,n,a['T_ns'],-24,0);fc,fr=clock(d,n,a['T_ns'],0,250)
            yield dict(match_id=full,period=period,period_definition_known=period in PERIODS,anchor_source_N=n,t0_UTC_ns=a['T_ns'],prefix_origin_N=origin,history_source_frames=25,history_intervals=24,history_nominal_seconds=.96,stride_source_frames=75,source_frame_present=a['frame_present'],source_frame_unique=a['unique_frame'],t0_BallPossession=a['possession'],source_team_A=a['team'],t0_BallStatus=a['live'],source_live='ALIVE' if a['live']==1 else 'DEAD' if a['live']==0 else 'UNKNOWN',metadata_known_team_and_live=bool(a['team'] and a['live']==1),history_clock_complete=hc,history_clock_reason=hr,future250_clock_complete=fc,future_clock_censor_reason=fr,future_codes_or_XY_used=False,openplay='UNKNOWN_ALIVE_NOT_NON_RESTART_CERTIFICATION',online_availability='NOT_INDEPENDENTLY_VALIDATED',primary_eligible=False)

def six_cohort(grid,scope):
    return [dict(grid_index=i,match_id=g['match_id'],period=g['period'],N=int(g['anchor_source_N'])) for i,g in enumerate(grid) if g['match_id'] in scope]
