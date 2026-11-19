"""Pure NumPy reference math. No IO, models, fitting or effect selection."""
import math
import numpy as np
DT=.1
LIMITS=dict(speed=10.5,acceleration=6.026811599731445,turn=3.837622240781784,screen_acceleration_extra=.02,screen_moving_speed=.1)
GROUPS=('all_two_t0_valid_targets','other_known_history_players','all_known_history_players')


def anchor_from_history(history,valid):
    """Exact frozen current-anchor F32 arithmetic, followed by later F64 metrics.

    history[10,23,4]. Targets with t0 fullvalid use actual t0; other players
    may be causally extrapolated from their last fullvalid original sample.
    """
    h=np.asarray(history);v=np.asarray(valid)
    if h.shape!=(10,23,4) or h.dtype!=np.float32 or v.shape!=(10,23) or v.dtype!=bool:
        raise ValueError('Original F32 history/bool fullstate mask required')
    present=v.any(0)
    last=np.max(np.where(v,np.arange(10)[:,None],-1),axis=0)
    a=h[last.clip(0),np.arange(23)].copy()
    age=(9-last.clip(0)).astype(np.float32)/np.float32(10.)
    a[:,:2]+=age[:,None]*a[:,2:]
    a=np.where(present[:,None],a,np.float32(0.)).astype(np.float32)
    if not np.isfinite(a[present]).all():raise ValueError('Nonfinite supported source anchor')
    return a,present,v[-1].copy(),age

def vector_turn(previous,current,moving_threshold):
    """Signed direction ignored; circular smallest turn, zero-speed = NaN/NA."""
    u=np.asarray(previous,np.float64);v=np.asarray(current,np.float64)
    nu=np.linalg.norm(u,axis=-1);nv=np.linalg.norm(v,axis=-1)
    moving=(nu>0)&(nv>0) if moving_threshold==0 else (nu>=moving_threshold)&(nv>=moving_threshold)
    finite=np.isfinite(u).all(-1)&np.isfinite(v).all(-1)
    angle=np.abs(np.arctan2(u[...,0]*v[...,1]-u[...,1]*v[...,0],(u*v).sum(-1)))/DT
    return np.where(moving&finite,angle,np.nan)

def kinematics(states,support,steps,anchor,anchor_present,scale):
    """Vectorized all draws×20×23. F32 inputs -> F64 inverse/difference.

    Position-derived first velocity is (x1-x_anchor)/.1; its first acceleration
    and turn compare to source-anchor velocity. Thus true-t0 and stale-extrap
    bridges share arithmetic but remain separately identified in output counts.
    """
    s=np.asarray(states);sp=np.asarray(support);st=np.asarray(steps)
    if s.ndim!=4 or s.shape[1:]!=(20,23,4) or s.dtype!=np.float32:
        raise ValueError('Saved states F32[M,20,23,4]')
    if sp.shape!=s.shape[:-1] or sp.dtype!=bool or st.shape!=(len(s),):raise ValueError('Support/step axes')
    if not np.issubdtype(st.dtype,np.integer) or not ((st>=0)&(st<=20)).all():raise ValueError('Completed steps0..20')
    factor=np.asarray(scale,np.float64)
    if factor.shape!=(4,) or not np.isfinite(factor).all() or np.any(factor<=0):raise ValueError('Explicit finite native transform')
    x=s.astype(np.float64)*factor
    a=np.asarray(anchor,np.float64)*factor
    active=np.arange(1,21)[None,:,None]<=st[:,None,None]
    good=active&sp&np.isfinite(x).all(-1)&np.asarray(anchor_present,bool)[None,None,:]
    previous_xy=np.concatenate((np.broadcast_to(a[None,None,:,:2],(len(s),1,23,2)),x[:,:-1,:,:2]),axis=1)
    previous_good=np.concatenate((np.broadcast_to(anchor_present,(len(s),1,23)),good[:,:-1]),axis=1)
    xy_valid=good&previous_good
    xy_v=np.where(xy_valid[...,None],(x[...,:2]-previous_xy)/DT,np.nan)
    head_v=np.where(good[...,None],x[...,2:],np.nan)
    out=dict(good=good,xy_valid=xy_valid,physical_states=x,physical_anchor=a)
    for name,vel,ok in (('xy',xy_v,xy_valid),('head',head_v,good)):
        prev=np.concatenate((np.broadcast_to(a[None,None,:,2:],(len(s),1,23,2)),vel[:,:-1]),axis=1)
        prevok=np.concatenate((np.broadcast_to(anchor_present,(len(s),1,23)),ok[:,:-1]),axis=1)
        pair=ok&prevok&np.isfinite(prev).all(-1)
        out[name]=dict(speed=np.where(ok,np.linalg.norm(vel,axis=-1),np.nan),
          acceleration=np.where(pair,np.linalg.norm(vel-prev,axis=-1)/DT,np.nan),
          turn=np.where(pair,vector_turn(prev,vel,0.),np.nan),
          turn_screen=np.where(pair,vector_turn(prev,vel,LIMITS['screen_moving_speed']),np.nan),
          valid=ok, pair_valid=pair, velocity=vel)
    return out


def recorded_metrics(states, fullvalid, positions, position_valid, scale):
    """One recorded realization. XY ALWAYS comes from independent truth positions.

    Future indices10..29. Observed first XY speed uses positions9&10 even when
    fullstate9 is absent. First XY a/turn uses cachedv9 only when fullvalid9;
    subsequent XY a/turn uses genuine consecutive position triples. Head means
    the cached backward-difference velocity here, not an observed neural head.
    """
    s=np.asarray(states);v=np.asarray(fullvalid);p=np.asarray(positions);pv=np.asarray(position_valid)
    f=np.asarray(scale,np.float64)
    if s.shape!=(30,23,4) or s.dtype!=np.float32 or p.shape!=(30,23,2) or p.dtype!=np.float32:
        raise ValueError('Recorded F32[30,23,4] and independent F32positions[30,23,2]')
    if v.shape!=(30,23) or pv.shape!=v.shape or v.dtype!=bool or pv.dtype!=bool:
        raise ValueError('Recorded fullstate and position masks')
    if f.shape!=(4,) or not np.isfinite(f).all() or np.any(f<=0) or not np.array_equal(f[:2],f[2:]):
        raise ValueError('Same positive F64 position/velocity inverse scales')
    if np.any(v&~pv) or not np.isfinite(s[v]).all() or not np.isfinite(p[pv]).all():
        raise ValueError('Supported recorded values must be finite')
    # Only the independent positions supply XY; erased s[~v,:2] is never used.
    xy=p.astype(np.float64)*f[:2]
    hv=s[...,2:].astype(np.float64)*f[2:]
    xyv=np.full((30,23,2),np.nan,np.float64)
    xyok=np.zeros((30,23),bool)
    xyok[1:]=pv[1:]&pv[:-1]
    xyv[1:]=np.where(xyok[1:,:,None],(xy[1:]-xy[:-1])/DT,np.nan)
    out={'t0_fullvalid':v[9].copy(),'t0_position_valid':pv[9].copy(),
         'position_only_first_speed':pv[9]&pv[10]&~v[9]}
    for name,velocity,valid in [('xy',xyv,xyok),('head',hv,v)]:
        cur=velocity[10:30].copy();prev=velocity[9:29].copy()
        ok=valid[10:30].copy();pair=ok&valid[9:29]
        if name=='xy':
            # Keep the exact old runner first-bridge metric definition, not a
            # post-result no-bridge alternative. Later triples retain pos-only.
            prev[0]=hv[9];pair[0]=ok[0]&v[9]
        cur=np.where(ok[...,None],cur,np.nan)
        out[name]=dict(speed=np.where(ok,np.linalg.norm(cur,axis=-1),np.nan),
            acceleration=np.where(pair,np.linalg.norm(cur-prev,axis=-1)/DT,np.nan),
            turn=np.where(pair,vector_turn(prev,cur,0.),np.nan),
            turn_screen=np.where(pair,vector_turn(prev,cur,LIMITS['screen_moving_speed']),np.nan),
            velocity=cur,valid=ok,pair_valid=pair)
    return out


def paired_observed_mask(recorded, deriv, metric, screen=False):
    name='turn_screen' if screen and metric=='turn' else metric
    mask=np.isfinite(recorded[deriv][name]).copy()
    if deriv=='xy' and metric=='speed':
        # Old generated first XY velocity uses its fullstate-based anchor.
        # Do not silently reanchor it to a newly read position-only t0.
        mask[0]&=recorded['t0_fullvalid']
    return mask


def fixed_groups(history_valid,identity,teams,slots):
    v=np.asarray(history_valid);ids=np.asarray(identity);tm=np.asarray(teams);sl=np.asarray(slots)
    if v.shape!=(10,23) or ids.shape!=(23,) or tm.shape!=(23,) or sl.shape!=(2,):raise ValueError('Fixed group shapes')
    if len(set(sl.tolist()))!=2 or not np.isin(sl,np.arange(22)).all():raise ValueError('Distinct player target slots')
    known=(ids=='VERIFIED_SOURCE_PLAYER_TEAM')&np.isin(tm,[0,1])&v.any(0)
    both=np.zeros(23,bool);both[sl]=True
    targets=both&known&v[-1]
    if int(targets.sum())!=2:raise ValueError('Original targets need genuine t0 fullvalid')
    return dict(zip(GROUPS,(targets,known&~both,known)))


def metric_summary(values,screen_values,metric):
    z=np.asarray(values,np.float64);z=z[np.isfinite(z)]
    sc=np.asarray(screen_values,np.float64);sc=sc[np.isfinite(sc)]
    limit=LIMITS[metric];sclimit=limit+(LIMITS['screen_acceleration_extra'] if metric=='acceleration' else 0.)
    n=len(z);sn=len(sc)
    d=dict(defined_units=n,strict_violations=int((z>limit).sum()),
        strict_rate=float((z>limit).mean()) if n else None,
        screen_defined_units=sn,screen_violations=int((sc>sclimit).sum()),
        screen_rate=float((sc>sclimit).mean()) if sn else None,
        mean=float(z.mean()) if n else None,
        SD_distribution=float(z.std(ddof=1)) if n>1 else None,
        minimum=float(z.min()) if n else None,maximum=float(z.max()) if n else None)
    for prob,name in [(.05,'p05'),(.5,'p50'),(.9,'p90'),(.95,'p95'),(.99,'p99')]:
        d[name]=float(np.quantile(z,prob)) if n else None
    return d


def matched_metric(model,recorded,group,deriv,metric,mode,world):
    """Project onto observed support. Controlled cases are NOT factual errors."""
    g=np.asarray(group,bool)
    name=metric;screen_name='turn_screen' if metric=='turn' else metric
    obs=paired_observed_mask(recorded,deriv,metric)&g[None]
    osc=paired_observed_mask(recorded,deriv,metric,True)&g[None]
    val=np.asarray(model[deriv][name],np.float64)
    sv=np.asarray(model[deriv][screen_name],np.float64)
    good=obs[None]&np.isfinite(val);sgood=osc[None]&np.isfinite(sv)
    mv=np.where(good,val,np.nan);ms=np.where(sgood,sv,np.nan)
    rv=np.where(obs,recorded[deriv][name],np.nan)
    rs=np.where(osc,recorded[deriv][screen_name],np.nan)
    M=len(val);N=int(obs.sum());ns=int(osc.sum())
    d={'factual_physical_agent_steps':N,'factual_screen_agent_steps':ns,
       'model_planned_draws':M,'model_planned_agent_draw_steps':M*N,
       'model_defined_agent_draw_steps':int(good.sum()),
       'model_draws_covering_all_factual_units':int((good.sum((1,2))==N).sum()) if N else 0,
       'factual_reused_not_new_real_N':True}
    for prefix,summary in [('factual_',metric_summary(rv,rs,metric)),('model_',metric_summary(mv,ms,metric))]:
        d.update({prefix+k:v for k,v in summary.items()})
    # Numerical integration precision only when all M draws share the complete
    # fixed factual mask. Partial-prefix or zero-turn selection is not silently
    # reinterpreted as a fixed-population MC estimand.
    fixed=bool(N and np.all(good.sum((1,2))==N))
    d['fixed_factual_mask_MCSE_applicable']=fixed and mode=='sample' and M>1
    d['model_drawmean_mean_MCSE']=None;d['model_draw_rate_MCSE']=None
    if fixed:
        means=np.where(good,val,0.).sum((1,2))/N
        rates=(good&(val>LIMITS[metric])).sum((1,2))/N
        if mode=='sample' and M>1:
            d['model_drawmean_mean_MCSE']=float(means.std(ddof=1)/math.sqrt(M))
            d['model_draw_rate_MCSE']=float(rates.std(ddof=1)/math.sqrt(M))
        elif world=='CV':
            d['model_drawmean_mean_MCSE']=0.;d['model_draw_rate_MCSE']=0.
    return d,(rv,rs,mv,ms)


def joint_summary(kin,group):
    """Three-way fullpath status: missing never converted to a physical fail."""
    g=np.asarray(group,bool);n=int(g.sum())
    sample=np.asarray(kin['head']['speed'])
    M=sample.shape[0] if sample.ndim==3 else 1
    if not n:return dict(planned_draws=M,fixed_players=0,complete_required_draws=0,
        strict_pass_draws=0,strict_known_violation_draws=0,strict_unknown_draws=M,
        screen_pass_draws=0,screen_known_violation_draws=0,screen_unknown_draws=M,
        strict_pass_rate_complete=None,screen_pass_rate_complete=None,
        status='NOT_APPLICABLE_EMPTY_GROUP')
    complete=np.ones(M,bool);bad={k:np.zeros(M,bool) for k in ['strict','screen']}
    for deriv in ('xy','head'):
        v=kin[deriv]
        vv=np.asarray(v['valid']);pp=np.asarray(v['pair_valid'])
        if vv.ndim==2:vv=vv[None];pp=pp[None]
        complete&=(vv[:,:,g]&pp[:,:,g]).all((1,2))
        for metric in ['speed','acceleration','turn']:
            for regime in ['strict','screen']:
                z=np.asarray(v['turn_screen' if regime=='screen' and metric=='turn' else metric])
                if z.ndim==2:z=z[None]
                lim=LIMITS[metric]+(LIMITS['screen_acceleration_extra'] if regime=='screen' and metric=='acceleration' else 0.)
                bad[regime]|=(np.isfinite(z[:,:,g])&(z[:,:,g]>lim)).any((1,2))
    row=dict(planned_draws=M,fixed_players=n,complete_required_draws=int(complete.sum()),status='TRISTATE_COMPLETE_PASS_KNOWN_VIOLATION_OR_UNKNOWN')
    for regime,b in bad.items():
        passed=complete&~b;unknown=~complete&~b
        row[regime+'_pass_draws']=int(passed.sum())
        row[regime+'_known_violation_draws']=int(b.sum())
        row[regime+'_unknown_draws']=int(unknown.sum())
        row[regime+'_pass_rate_complete']=float(passed.sum()/complete.sum()) if complete.any() else None
    return row
