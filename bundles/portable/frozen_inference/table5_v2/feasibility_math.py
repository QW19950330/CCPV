"""Pure saved-output engineering screens; no I/O, labels, model, fit or q.

An engineering-reference screen is not a physiological feasibility certificate.
Field, centre proximity, residuals and BALL remain diagnostic, never reject gates.
"""
from __future__ import annotations
import numpy as np

DT=.1
HORIZON=20
SPEED=10.5
ACCELERATION=6.026811599731445
ACCELERATION_SCREEN_EXTRA=.02
TURN=3.837622240781784
MOVING_SPEED=.1
NEAR_CENTRE_M=.25
PASSED='ENGINEERING_REFERENCE_SCREEN_PASSED'
EXCEEDS='EXCEEDS_ENGINEERING_REFERENCE_SCREEN'


def current_anchor(history,valid):
    h=np.asarray(history);v=np.asarray(valid,bool)
    if h.dtype!=np.float32 or h.shape[-3:]!=(10,23,4) or v.shape!=h.shape[:-1]:
        raise ValueError('Need original float32 fullstate-valid history')
    ix=np.where(v,np.arange(10)[None,:,None],-1).max(1)
    present=ix>=0
    last=np.take_along_axis(h,ix.clip(0)[:,None,:,None],axis=1)[:,0].copy()
    age=(9-ix.clip(0)).astype(np.float32)/np.float32(10.)
    last[...,:2]=last[...,:2]+age[...,None]*last[...,2:]
    last=np.where(present[...,None],last,0.)
    return last,present,age,ix


def first(mask,reason):
    where=np.argwhere(mask)
    if not len(where):return None
    t,j=where[0]
    return (int(t)+1,int(j),reason)


def series(anchor,states,support,steps,kind):
    a=np.asarray(anchor,float);q=np.asarray(states,float);s=np.asarray(support,bool).copy()
    if a.shape!=(23,4) or q.shape!=(20,23,4) or s.shape!=(20,23) or not 0<=int(steps)<=20:
        raise ValueError('Fixed20step23entity output required')
    s[int(steps):]=False
    xy_ok=s&np.isfinite(q[...,:2]).all(-1)
    v_ok=s&np.isfinite(q[...,2:]).all(-1)
    previous_xy=np.concatenate((a[None,:,:2],q[:-1,:,:2]),axis=0)
    previous_xy_ok=np.concatenate((np.isfinite(a[None,:,:2]).all(-1),xy_ok[:-1]),axis=0)
    if kind=='position':
        velocity=(q[...,:2]-previous_xy)/DT
        velocity_ok=xy_ok&previous_xy_ok
    elif kind=='head':
        velocity=q[...,2:]
        velocity_ok=v_ok
    else:raise ValueError('position or head only')
    previous_v=np.concatenate((a[None,:,2:],velocity[:-1]),axis=0)
    previous_v_ok=np.concatenate((np.isfinite(a[None,:,2:]).all(-1),velocity_ok[:-1]),axis=0)
    paired=velocity_ok&previous_v_ok
    speed=np.linalg.norm(velocity,axis=-1)
    previous_speed=np.linalg.norm(previous_v,axis=-1)
    acc=np.linalg.norm(velocity-previous_v,axis=-1)/DT
    defined=paired&(speed>0)&(previous_speed>0)
    moving=defined&(speed>=MOVING_SPEED)&(previous_speed>=MOVING_SPEED)
    cross=previous_v[...,0]*velocity[...,1]-previous_v[...,1]*velocity[...,0]
    dot=np.sum(previous_v*velocity,axis=-1)
    turn=np.full((20,23),np.nan)
    turn[defined]=np.arctan2(np.abs(cross[defined]),dot[defined])/DT
    return dict(velocity=velocity,velocity_ok=velocity_ok,paired=paired,speed=speed,acceleration=acc,
                turn=turn,turn_defined=defined,turn_moving=moving,zero_speed_turn_NA=paired&~defined,
                xy_ok=xy_ok,v_ok=v_ok,support=s)


def value(x,mask,operation):
    a=np.asarray(x)[mask]
    return float(operation(a)) if len(a) else None


def motion_summary(anchor,states,support,steps,group,group_supported,age,kind,data=None):
    group=np.asarray(group,bool)
    if group.shape!=(23,):raise ValueError('Fixed original group required')
    data=series(anchor,states,support,steps,kind) if data is None else data
    select=group[None]
    speed_ok=data['velocity_ok']&select
    paired=data['paired']&select
    defined=data['turn_defined']&select
    moving=data['turn_moving']&select
    speed_exceed=speed_ok&(data['speed']>SPEED)
    a_strict=paired&(data['acceleration']>ACCELERATION)
    a_screen=paired&(data['acceleration']>ACCELERATION+ACCELERATION_SCREEN_EXTRA)
    t_strict=defined&(data['turn']>TURN)
    t_screen=moving&(data['turn']>TURN)
    entities=int(group.sum())
    # Prefix v0 belongs to the starting condition; speed gate applies t1..t20.
    anchor_ok=np.isfinite(np.asarray(anchor)[group,2:]).all()
    if kind=='position':anchor_ok &= np.isfinite(np.asarray(anchor)[group,:2]).all()
    complete=bool(group_supported and entities>0 and int(steps)==20 and anchor_ok and data['velocity_ok'][:,group].all())
    if not group_supported or entities==0:structural='UNSUPPORTED_FIXED_GROUP_BINDING'
    elif not anchor_ok:structural='NONFINITE_CAUSAL_ANCHOR'
    elif int(steps)==0:structural='NO_GENERATED_OUTPUT'
    elif int(steps)<20:structural='PARTIAL_GENERATED_OUTPUT'
    elif not data['velocity_ok'][:,group].all():structural='MISSING_OR_NONFINITE_GENERATED_SUPPORT'
    else:structural='COMPLETE_FINITE_FIXED_GROUP_OUTPUT'
    strict_events=[first(speed_exceed,'SPEED_REFERENCE_EXCEED'),first(a_strict,'ACCELERATION_STRICT_REFERENCE_EXCEED'),first(t_strict,'TURN_STRICT_REFERENCE_EXCEED')]
    screen_events=[first(speed_exceed,'SPEED_REFERENCE_EXCEED'),first(a_screen,'ACCELERATION_NUMERICAL_SCREEN_EXCEED'),first(t_screen,'MOVING_TURN_REFERENCE_EXCEED')]
    structural_events=[]
    if not group_supported or entities==0:structural_events.append((0,-1,'UNSUPPORTED_FIXED_GROUP_BINDING'))
    elif not anchor_ok:structural_events.append((0,-1,'NONFINITE_CAUSAL_ANCHOR'))
    else:
        required=np.broadcast_to(group,(20,23))
        absent=required&~data['velocity_ok']
        event=first(absent,'MISSING_OR_NONFINITE_GENERATED_OUTPUT')
        if event:structural_events.append(event)
        if int(steps)<20:structural_events.append((int(steps)+1,-1,'GENERATED_PATH_INCOMPLETE'))
    def earliest(events):
        ee=[x for x in events if x is not None]
        # Deterministic ties: time, source slot, listed event order. No effect-based selection.
        return min(enumerate(ee),key=lambda x:(x[1][0],x[1][1],x[0]))[1] if ee else None
    sf=earliest(strict_events+structural_events);nf=earliest(screen_events+structural_events)
    strict_failed=bool(np.any(speed_exceed|a_strict|t_strict))
    screen_failed=bool(np.any(speed_exceed|a_screen|t_screen))
    result=dict(kind=kind,structural_status=structural,complete_finite_support=complete,
                strict_status=('ENGINEERING_STRICT_REFERENCE_PASSED' if not strict_failed else 'EXCEEDS_ENGINEERING_STRICT_REFERENCE') if complete else 'NOT_ASSESSABLE_COMPLETE_PATH',
                screen_status=(PASSED if not screen_failed else EXCEEDS) if complete else 'NOT_ASSESSABLE_COMPLETE_PATH',
                fixed_group_entities=entities,expected_full_agent_steps=entities*20,generated_window_steps=int(steps),
                anchor_t0_valid_entities=int(np.sum(group&(np.asarray(age)==0))),
                anchor_causally_extrapolated_entities=int(np.sum(group&(np.asarray(age)>0))),
                anchor_age_max_s=value(age,group,np.max),anchor_age_mean_s=value(age,group,np.mean),
                speed_agent_steps=int(speed_ok.sum()),acceleration_agent_steps=int(paired.sum()),
                turn_defined_agent_steps=int(defined.sum()),moving_turn_agent_steps=int(moving.sum()),
                turn_zero_speed_NOT_APPLICABLE_agent_steps=int((data['zero_speed_turn_NA']&select).sum()),
                defined_below_moving_screen_agent_steps=int((defined&~moving).sum()),
                speed_exceed_count=int(speed_exceed.sum()),acceleration_strict_exceed_count=int(a_strict.sum()),
                acceleration_screen_exceed_count=int(a_screen.sum()),turn_strict_exceed_count=int(t_strict.sum()),
                moving_turn_screen_exceed_count=int(t_screen.sum()),
                speed_mean=value(data['speed'],speed_ok,np.mean),speed_max=value(data['speed'],speed_ok,np.max),
                acceleration_mean=value(data['acceleration'],paired,np.mean),acceleration_max=value(data['acceleration'],paired,np.max),
                turn_defined_mean=value(data['turn'],defined,np.mean),turn_defined_max=value(data['turn'],defined,np.max),
                moving_turn_max=value(data['turn'],moving,np.max))
    for prefix,event in [('first_strict_failure',sf),('first_screen_failure',nf)]:
        result[prefix+'_step']=event[0] if event else None
        result[prefix+'_slot']=event[1] if event else None
        result[prefix+'_reason']=event[2] if event else None
    return result


def joint_summary(position,head):
    complete=position['complete_finite_support'] and head['complete_finite_support']
    passed=complete and position['screen_status']==head['screen_status']==PASSED
    candidates=[]
    for kind,r in [('position',position),('head',head)]:
        if r['first_screen_failure_step'] is not None:
            candidates.append((r['first_screen_failure_step'],r['first_screen_failure_slot'],kind,r['first_screen_failure_reason']))
    event=min(candidates,key=lambda x:(x[0],x[1],x[2]!='position')) if candidates else None
    return dict(joint_screen_status=(PASSED if passed else EXCEEDS) if complete else 'NOT_ASSESSABLE_COMPLETE_PATH',
                joint_complete_finite_support=bool(complete),joint_engineering_screen_passed=bool(passed),
                first_joint_failure_step=event[0] if event else None,first_joint_failure_slot=event[1] if event else None,
                first_joint_failure_representation=event[2] if event else None,first_joint_failure_reason=event[3] if event else None)


def diagnostics(anchor,states,support,steps,players,pitch_length,pitch_width,ball_present,motion=None):
    q=np.asarray(states,float);s=np.asarray(support,bool).copy();s[int(steps):]=False
    players=np.asarray(players,bool);xyok=s&np.isfinite(q[...,:2]).all(-1)
    previous=np.concatenate((np.asarray(anchor,float)[None,:,:2],q[:-1,:,:2]),0)
    previous_ok=np.concatenate((np.isfinite(np.asarray(anchor,float)[None,:,:2]).all(-1),xyok[:-1]),0)
    residual_ok=xyok&previous_ok&np.isfinite(q[...,2:]).all(-1)&players[None]
    residual=np.linalg.norm(q[...,:2]-previous-DT*q[...,2:],axis=-1)
    outside=(np.abs(q[...,0])>pitch_length/2)|(np.abs(q[...,1])>pitch_width/2)
    pm=xyok&players[None]
    minima=[];near_count=0;pair_denominator=0;complete_frames=0
    ix=np.flatnonzero(players)
    if len(ix)>1:
        upper=np.triu_indices(len(ix),1)
        for t in range(int(steps)):
            # Unknown members never disappear to improve nearest distance.
            if not xyok[t,ix].all():continue
            points=q[t,ix,:2];d=np.linalg.norm(points[:,None]-points[None,:],axis=-1)[upper]
            minima.append(d.min());near_count+=int(np.sum(d<NEAR_CENTRE_M));pair_denominator+=len(d);complete_frames+=1
    result=dict(player_position_velocity_residual_mean=value(residual,residual_ok,np.mean),
                player_position_velocity_residual_max=value(residual,residual_ok,np.max),residual_agent_steps=int(residual_ok.sum()),
                field_player_xy_agent_steps=int(pm.sum()),field_player_outside_rectangle_agent_steps=int((outside&pm).sum()),
                player_centre_min_distance=float(min(minima)) if minima else None,
                near_centre_lt_0p25_pair_steps=near_count,centre_pair_steps=pair_denominator,
                centre_distance_complete_fixed_group_frames=complete_frames,
                field_proximity_or_residual_used_as_reject=False,ball_present_in_prefix=bool(ball_present))
    ball=np.zeros(23,bool);ball[22]=bool(ball_present)
    for kind in ('position','head'):
        d=series(anchor,q,s,steps,kind) if motion is None else motion[kind]
        sel=ball[None]
        for metric,mask in [('speed',d['velocity_ok']&sel),('acceleration',d['paired']&sel),('turn',d['turn_defined']&sel)]:
            result[f'ball_{kind}_{metric}_max']=value(d[metric],mask,np.max)
            result[f'ball_{kind}_{metric}_steps']=int(mask.sum())
    result['ball_field_xy_steps']=int((xyok&ball[None]).sum())
    result['ball_field_outside_rectangle_steps']=int((outside&xyok&ball[None]).sum())
    result['ball_player_limits_applied']=False
    return result
