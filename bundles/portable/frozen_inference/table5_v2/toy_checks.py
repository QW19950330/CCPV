"""Pure handmade full-output tests. Never opens a saved trajectory or CP."""
from datetime import datetime,timezone
from pathlib import Path
import ast,argparse
import numpy as np
import feasibility_math as M
from feasibility_io import P,put,guard
from run_feasibility import Bucket

def fixture(speed=1.):
    anchor=np.zeros((23,4));anchor[:,0]=np.arange(23)*2.;anchor[:,2]=speed
    q=np.broadcast_to(anchor[None],(20,23,4)).copy()
    q[...,:2]+=np.arange(1,21)[:,None,None]*.1*anchor[None,:,2:]
    support=np.ones((20,23),bool);group=np.zeros(23,bool);group[:2]=True
    return anchor,q,support,group,np.zeros(23)

def both(a,q,s,g,age,n=20,supported=True):
    p=M.motion_summary(a,q,s,n,g,supported,age,'position')
    h=M.motion_summary(a,q,s,n,g,supported,age,'head')
    return p,h,M.joint_summary(p,h)

def main(output_name='toy_QA.json'):
    checks=[]
    a,q,s,g,age=fixture();p,h,j=both(a,q,s,g,age)
    assert p['screen_status']==h['screen_status']==j['joint_screen_status']==M.PASSED
    assert p['speed_agent_steps']==p['acceleration_agent_steps']==40
    checks.append('constant_motion_all20steps_both_representations_pass_scoped_engineering_screen')
    jump=q.copy();jump[:,0,0]+=10
    p,h,j=both(a,jump,s,g,age)
    assert p['screen_status']==M.EXCEEDS and h['screen_status']==M.PASSED and not j['joint_engineering_screen_passed']
    assert j['first_joint_failure_step']==1 and j['first_joint_failure_representation']=='position'
    checks.append('first_position_jump_not_hidden_by_quiet_velocity_head')
    jump=q.copy();jump[9:,0,0]+=10
    p,h,j=both(a,jump,s,g,age)
    assert j['first_joint_failure_step']==10 and p['acceleration_screen_exceed_count']>0
    checks.append('later_emitted_step_failure_detected_not_just_first_bridge')
    head=q.copy();head[9:,0,2]=10
    p,h,j=both(a,head,s,g,age)
    assert p['screen_status']==M.PASSED and h['screen_status']==M.EXCEEDS and j['first_joint_failure_step']==10
    checks.append('velocity_head_acceleration_failure_separate_from_position_trajectory')
    a0,q0,s0,g0,ag0=fixture(0.)
    p,h,j=both(a0,q0,s0,g0,ag0)
    assert j['joint_engineering_screen_passed'] and p['turn_defined_agent_steps']==0
    assert p['turn_zero_speed_NOT_APPLICABLE_agent_steps']==40 and p['turn_defined_max'] is None
    checks.append('zero_speed_turn_NOT_APPLICABLE_never_angle_zero_imputation')
    slow,qs,ss,gs,ags=fixture(.05);qs[...,:2]=slow[None,:,:2]-np.arange(1,21)[:,None,None]*.1*slow[None,:,2:];qs[...,2:]=-slow[None,:,2:]
    p,h,j=both(slow,qs,ss,gs,ags)
    assert p['turn_strict_exceed_count']>0 and p['moving_turn_screen_exceed_count']==0 and j['joint_engineering_screen_passed']
    assert p['strict_status']=='EXCEEDS_ENGINEERING_STRICT_REFERENCE'
    checks.append('low_speed_direction_change_strict_retained_but_not_moving_turn_screen')
    partial=s.copy();partial[4:]=False
    p,h,j=both(a,q,partial,g,age,4)
    assert j['joint_screen_status']=='NOT_ASSESSABLE_COMPLETE_PATH' and not j['joint_engineering_screen_passed']
    assert p['structural_status']=='PARTIAL_GENERATED_OUTPUT' and p['speed_agent_steps']==8 and j['first_joint_failure_step']==5
    checks.append('partial_path_retained_with_generated_prefix_counts_never_full_path_pass')
    changed=q.copy();changed[4:]=1e20
    pp,hh,jj=both(a,changed,partial,g,age,4)
    assert p==pp and h==hh and j==jj
    checks.append('unsupported_generated_tail_perturbation_does_not_change_prefix_metrics')
    hole=s.copy();hole[4,0]=False
    p,h,j=both(a,q,hole,g,age)
    assert not j['joint_complete_finite_support'] and j['first_joint_failure_step']==5
    checks.append('missing_fixed_member_not_dropped_to_improve_guard')
    broken=q.copy();broken[6,0,2]=np.nan
    p,h,j=both(a,broken,s,g,age)
    assert p['complete_finite_support'] and not h['complete_finite_support'] and not j['joint_engineering_screen_passed']
    assert j['first_joint_failure_step']==7
    checks.append('nonfinite_velocity_head_kept_distinct_and_joint_not_assessable')
    history=np.zeros((1,10,23,4),np.float32);valid=np.zeros((1,10,23),bool)
    history[0,:,0,0]=np.arange(10,dtype=np.float32)/10;history[0,:,0,2]=1
    valid[0,8,0]=True;valid[0,9,1]=True
    base,present,ages,last=M.current_anchor(history,valid)
    assert last[0,0]==8 and ages[0,0]==np.float32(.1) and base[0,0,0]==np.float32(.8)+np.float32(.1)
    assert ages[0,1]==0 and present[0,:2].all()
    bad=history.copy();bad[~valid]=1e6
    np.testing.assert_array_equal(base,M.current_anchor(bad,valid)[0])
    checks.append('literal_float32_causal_anchor_stale_age_and_invalid_history_invariance')
    scale=np.array([104/105,70/68,104/105,70/68]);native=a*scale;nativeq=q*scale
    pp,hh,jj=both(native,nativeq,s,g,age)
    assert abs(pp['speed_mean']-104/105)<1e-13 and abs(hh['speed_mean']-104/105)<1e-13
    assert jj['joint_engineering_screen_passed']
    checks.append('source_pitch_inverse_applied_once_to_position_and_velocity_consistently')
    outside=q.copy();outside[:,:,:2]+=200
    # Move the anchor equally so there is no new kinematic jump.
    ao=a.copy();ao[:,:2]+=200
    p,h,j=both(ao,outside,s,g,age)
    d=M.diagnostics(ao,outside,s,20,g,105,68,True)
    assert j['joint_engineering_screen_passed'] and d['field_player_outside_rectangle_agent_steps']==40 and not d['field_proximity_or_residual_used_as_reject']
    checks.append('field_rectangle_diagnostic_is_not_unvalidated_physical_reject')
    aa=a.copy();qq=q.copy();aa[1]=aa[0];qq[:,1]=qq[:,0]
    p,h,j=both(aa,qq,s,g,age);d=M.diagnostics(aa,qq,s,20,g,105,68,True)
    assert d['player_centre_min_distance']==0 and d['near_centre_lt_0p25_pair_steps']==20 and j['joint_engineering_screen_passed']
    checks.append('all_fixed_player_centre_pairs_overlap_diagnostic_not_collision_certificate')
    aq=a.copy();bq=q.copy();aq[22,2]=50;bq[:,22,0]=aq[22,0]+np.arange(1,21)*5;bq[:,22,2]=50
    p,h,j=both(aq,bq,s,g,age);d=M.diagnostics(aq,bq,s,20,g,105,68,True)
    assert d['ball_head_speed_max']==50 and d['ball_position_speed_max']>49 and not d['ball_player_limits_applied'] and j['joint_engineering_screen_passed']
    checks.append('BALL_separate_no_player_speed_acceleration_turn_limit_application')
    almost=q.copy();almost[...,:2]=a[None,:,:2]+np.arange(1,21)[:,None,None]*.1*a[None,:,2:]*1.01
    p,h,j=both(a,almost,s,g,age);d=M.diagnostics(a,almost,s,20,g,105,68,True)
    assert d['player_position_velocity_residual_max']>0 and j['joint_engineering_screen_passed']
    checks.append('residual_is_diagnostic_no_hidden_projection_or_zero_residual_gate')
    b=a0.copy();qv=q0.copy();qv[...,2]=(M.ACCELERATION+.01)*.1
    hp=M.motion_summary(b,qv,s0,20,g0,True,ag0,'head')
    assert hp['acceleration_strict_exceed_count']>0 and hp['acceleration_screen_exceed_count']==0
    qv[...,2]=(M.ACCELERATION+.03)*.1
    hp=M.motion_summary(b,qv,s0,20,g0,True,ag0,'head')
    assert hp['acceleration_screen_exceed_count']>0
    checks.append('inherited_fixed_acceleration_extra0p02_strict_and_screen_both_retained')
    goodp,goodh,goodj=both(a,q,s,g,age);badp,badh,badj=both(a,q,partial,g,age,4)
    bucket=Bucket();bucket.add(goodp,goodh,goodj);bucket.add(badp,badh,badj)
    r=bucket.row({})
    assert r['planned_windows']==2 and r['joint_screen_pass_count']==1 and r['joint_screen_pass_fraction_over_planned']==.5
    checks.append('all_planned_denominator_keeps_partial_failure_not_success_subset')
    jump=q.copy();jump[:,0,0]+=10
    failp,failh,failj=both(a,jump,s,g,age)
    bucket.add(failp,failh,failj);r=bucket.row({})
    assert r['position_any_speed_exceed_count']==1 and r['head_any_speed_exceed_count']==0
    assert r['position_any_speed_exceed_fraction_over_planned']==1/3
    assert r['position_structural_PARTIAL_GENERATED_OUTPUT_count']==1
    assert r['joint_screen_pass_fraction_over_planned']==1/3
    checks.append('marginal_violation_and_structural_reason_rates_keep_all_planned_windows')
    empty=np.zeros(23,bool);p,h,j=both(a,q,s,empty,age,supported=False)
    assert not j['joint_engineering_screen_passed'] and p['structural_status']=='UNSUPPORTED_FIXED_GROUP_BINDING'
    try:guard('NO_ROOT_LOCK_YET')
    except (FileNotFoundError,PermissionError):pass
    else:raise AssertionError('Production saved-array screen must not be authorized before root lock')
    for filename in ('feasibility_math.py','feasibility_io.py','pipeline.py','run_feasibility.py','toy_checks.py'):
        ast.parse((P/filename).read_text(encoding='utf-8'))
    checks.append('unsupported_empty_group_and_no_root_authorization_no_NPZ_access')
    if Path(output_name).name!=output_name:raise ValueError('QA output must be a new local basename')
    put(P/output_name,dict(status='PASS_HANDMADE_ONLY_FULL_OUTPUT_ENGINEERING_SCREEN_QA',utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,cases=len(checks),real_NPZ_members_decoded=0,checkpoints_read=0,world_forwards=0,fits=0,value_readouts=0,
        labels_or_actual_future_truth_read=False,engineering_reference_not_physiological_certification=True))
    print('HANDMADE_FULL_OUTPUT_QA_PASS cases='+str(len(checks))+' real_NPZ_members=0')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='toy_QA.json');args=parser.parse_args();main(args.output)
