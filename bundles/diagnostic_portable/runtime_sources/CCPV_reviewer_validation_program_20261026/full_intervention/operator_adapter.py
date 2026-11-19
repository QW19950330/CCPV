"""CPU batched input-operator integration; no model loading, value or training.

All paths keep the full23-node model. Only a known-ID subview is given to the
reference operator, and only the selected terminal velocities are scattered.
"""
from __future__ import annotations
import math
from typing import Any
import torch
from source_operator.reference_operator import (apply_reference_operator,ReferenceMode,EntityType,MetricUnits,MotionLimits,InputContractError,MotionConstraintError)

def _bits(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8))
def _list(x):return x.tolist() if hasattr(x,'tolist') else list(x)
def _id(x):
 if type(x) is int:return x
 if type(x) is str and x and x.strip()==x:return x
 return None
def _key(x):return (type(x),x)
def make_registries(entity_ids,identity_status,teams):
 ids,status=_list(entity_ids),_list(identity_status);tm=_list(teams)
 if len(ids)!=len(status):raise InputContractError('registry batch mismatch')
 result=[]
 for b,(row,st) in enumerate(zip(ids,status)):
  t=tm if len(tm)==23 and not isinstance(tm[0],list) else tm[b]
  if len(row)!=23 or len(st)!=23 or len(t)!=23:raise InputContractError('registry axes23')
  known=[];kinds=[];ball=None
  for value,s,team in zip(row,st,t):
   value=str(value) if isinstance(value,str) else value
   if s=='VERIFIED_SOURCE_PLAYER_TEAM' and team in (0,1) and _id(value) is not None:known.append(value);kinds.append('PLAYER')
   elif s=='EXPLICIT_MODEL_BALL_SOURCE_ID' and team==2 and _id(value) is not None:
    known.append(value);kinds.append('BALL');ball=value
   else:known.append(None);kinds.append(None)
  result.append({'entity_ids':known,'entity_types':kinds,'ball_entity_id':ball})
 return result

def _registry(reg,usable,team):
 if not isinstance(reg,dict):return None,'REGISTRY_NOT_EXPLICIT_DICT'
 ids=reg.get('entity_ids');types=reg.get('entity_types');ball=reg.get('ball_entity_id')
 if not isinstance(ids,(list,tuple)) or not isinstance(types,(list,tuple)) or len(ids)!=23 or len(types)!=23:return None,'REGISTRY_AXES'
 known=[];seen=set();ballslots=[]
 for i,(eid,kind) in enumerate(zip(ids,types)):
  if eid is None:
   if kind is not None:return None,'UNKNOWN_ID_HAS_ENTITY_TYPE'
   if bool(usable[i]):return None,'ACTIVE_USABLE_ID_UNKNOWN'
   continue
  if _id(eid) is None:return None,'INVALID_STRICT_SOURCE_ID'
  if _key(eid) in seen:return None,'DUPLICATE_SOURCE_ID'
  seen.add(_key(eid))
  if kind not in ('PLAYER','BALL'):return None,'UNKNOWN_ENTITY_TYPE'
  if kind=='BALL':ballslots.append(i)
  if (kind=='BALL') != (int(team[i])==2):return None,'BALL_TYPE_TEAM_CODE_CONFLICT'
  known.append(i)
 if len(ballslots)!=1 or _id(ball) is None or _key(ids[ballslots[0]])!=_key(ball):return None,'EXPLICIT_BALL_ID_UNRESOLVED'
 return {'slots':known,'ids':[ids[i] for i in known],'types':[types[i] for i in known],'ball_id':ball,'lookup':{_key(ids[i]):i for i in known}},None

def make_commands(history,registries,targets,delta0=False):
 if not isinstance(history,torch.Tensor) or history.dtype!=torch.float32 or history.device.type!='cpu':raise InputContractError('commands require CPU float32 original history')
 B=history.shape[0];out=torch.full((B,2,2),float('nan'),dtype=history.dtype)
 for b in range(B):
  ids=registries[b]['entity_ids']
  if not isinstance(targets[b],(list,tuple)) or len(targets[b])!=2:continue
  for k,eid in enumerate(targets[b]):
   ix=[i for i,x in enumerate(ids) if x is not None and _key(x)==_key(eid)]
   if len(ix)!=1:continue
   v=history[b,-1,ix[0],2:4]
   if not torch.isfinite(v).all():continue
   if delta0:out[b,k]=v
   else:
    speed=float(torch.linalg.vector_norm(v));d=.1*torch.stack((-v[1],v[0]))/speed if speed>=.1 else v.new_tensor([.1,0.])
    out[b,k]=v+d
 return out

def rollout(api,model,history,valid,teams,registries,targets,commands,coalition,semantics,limits,*,units,dt,horizon=20):
 """semantics=bypass/NONE/first/persistent (canonical lowercase strings).

 Persistent means the same absolute command is reapplied to each INPUT; it is
 not a forced output path. Rejected rows stop producing outputs; fixed-batch
 dummy forward work is counted separately and never treated as a completion.
 """
 if semantics not in ('bypass','none','first','persistent'):raise InputContractError('Unknown intervention semantics')
 if type(horizon) is not int or horizon<1:raise InputContractError('positive integer horizon')
 if type(dt) not in (int,float) or isinstance(dt,bool) or not math.isfinite(dt) or dt<=0 or float(dt)!=float(api.DT):raise InputContractError('dt must equal frozen API.DT')
 if not isinstance(units,MetricUnits) or not isinstance(limits,MotionLimits):raise InputContractError('Explicit reference MetricUnits/MotionLimits required')
 if units!=MetricUnits('m','m/s','s','m/s^2','rad/s'):raise InputContractError('Explicit physical metric units')
 lv=(limits.vmax_m_s,limits.amax_m_s2,limits.turnmax_rad_s)
 if any(type(x) not in (int,float) or not math.isfinite(x) for x in lv) or lv[0]<=0 or lv[1]<0 or lv[2]<0:raise InputContractError('Finite caller-declared physical limits')
 if not isinstance(history,torch.Tensor) or history.device.type!='cpu' or history.dtype!=torch.float32 or history.requires_grad or history.ndim!=4 or history.shape[1:]!=(10,23,4):raise InputContractError('history CPU detached float32[B,10,23,4]')
 if not isinstance(valid,torch.Tensor) or valid.device.type!='cpu' or valid.dtype!=torch.bool or valid.shape!=history.shape[:-1]:raise InputContractError('valid CPU bool[B,10,23]')
 if not torch.isfinite(history[valid]).all():raise InputContractError('usable states must be finite')
 B=history.shape[0]
 if not isinstance(teams,torch.Tensor) or teams.device.type!='cpu' or teams.dtype!=torch.long or teams.shape not in ((23,),(B,23)) or not torch.isin(teams,torch.tensor([0,1,2])).all():raise InputContractError('teams CPU int64[23] or[B,23]')
 if len(registries)!=B or len(targets)!=B:raise InputContractError('registry/targets batch')
 if type(coalition) is not tuple or any(type(i) is not int or i not in (0,1) for i in coalition) or tuple(sorted(set(coalition)))!=coalition:raise InputContractError('coalition canonical tuple of0/1')
 if semantics in ('bypass','none'):
  if coalition or commands is not None:raise InputContractError('bypass/NONE require empty coalition and commands=None')
 elif not isinstance(commands,torch.Tensor) or commands.device.type!='cpu' or commands.dtype!=history.dtype or commands.shape!=(B,2,2) or commands.requires_grad:raise InputContractError('SET commands detached CPU float32[B,2,2]')
 if model is not None:
  if getattr(model,'training',False):raise InputContractError('frozen model must already be eval')
  if hasattr(model,'parameters'):
   if any(p.device.type!='cpu' or p.dtype!=history.dtype for p in model.parameters()):raise InputContractError('model CPU float32; no implicit casting')
  cfg=getattr(model,'model_config',{})
  if 'fps' in cfg and 1./float(cfg['fps'])!=float(dt):raise InputContractError('model fps differs from operator dt')
 originals=(history.clone(),valid.clone(),teams.clone(),commands.clone() if commands is not None else None)
 h=history.clone();v=valid.clone();tm=teams.clone();usable=v.any(1);observed,age=api.initial_provenance(v)
 states=torch.full((B,horizon,23,4),float('nan'),dtype=history.dtype);support=torch.zeros((B,horizon,23),dtype=torch.bool);steps=torch.zeros(B,dtype=torch.long)
 effective=torch.zeros(B,dtype=torch.long);dummy=torch.zeros(B,dtype=torch.long);active=torch.ones(B,dtype=torch.bool);status=['PENDING']*B;reasons=[[] for _ in range(B)];maps=[];slot_targets=[];calls=[];cleared=[];rejections=[];batchcalls=0
 for b in range(B):
  rowtm=tm if tm.ndim==1 else tm[b];mp,why=_registry(registries[b],usable[b],rowtm);slots=[]
  if why is None:
   if not isinstance(targets[b],(list,tuple)) or len(targets[b])!=2:why='TWO_EXPLICIT_TARGET_IDS_REQUIRED'
   else:
    for eid in targets[b]:
     slot=mp['lookup'].get(_key(eid)) if _id(eid) is not None else None
     if slot is None:why='TARGET_ID_UNKNOWN';break
     if registries[b]['entity_types'][slot]!='PLAYER':why='TARGET_IS_NOT_PLAYER';break
     if not bool(v[b,-1,slot]):why='TARGET_T0_NOT_VALID';break
     slots.append(slot)
    if len(slots)==2 and slots[0]==slots[1]:why='TARGET_IDS_NOT_DISTINCT'
  maps.append(mp);slot_targets.append(slots)
  if why is not None:active[b]=False;status[b]='UNSUPPORTED';reasons[b].append(why)
 initial_active=active.clone()
 with torch.inference_mode():
  for step in range(horizon):
   if not active.any():break
   apply=semantics=='none' or (bool(coalition) and (semantics=='persistent' or semantics=='first' and step==0))
   if apply:
    for b in torch.where(active)[0].tolist():
     row_before=h[b].clone();trial=row_before.clone();local=[];ok=True;mp=maps[b]
     selected=(0,) if semantics=='none' else coalition
     for k in selected:
      target=targets[b][k];slot=slot_targets[b][k];known=mp['slots'];view=trial[-1:,known].clone();currentmask=v[b,-1,known]
      # This view is only for reference validation; masked fillers never enter h.
      view=torch.where(currentmask[None,:,None],view,0.)
      command=None if semantics=='none' else commands[b,k].clone()
      rec={'scene':b,'step':step,'target_index':k,'source_id':target,'mode':'NONE' if semantics=='none' else 'SET_VELOCITY','command_absolute_m_s':None if command is None else command.tolist(),'committed':False,'reference_view_slots':known,'reference_view_inactive_filler_is_not_observation':True}
      try:
       result=apply_reference_operator(view,ReferenceMode.NONE if semantics=='none' else ReferenceMode.SET_VELOCITY,entity_ids=mp['ids'],entity_types=mp['types'],ball_entity_id=mp['ball_id'],target_id=target,absolute_velocity=command,dt=float(dt),model_dt=float(api.DT),units=units,limits=limits)
       rec['reference_report']=result.report
       if semantics=='none':
        if not _bits(result.state,view):raise AssertionError('NONE reference changed bytes')
       else:
        sub=known.index(slot);expected_view=view.clone();expected_view[-1,sub,2:4]=command
        if not _bits(expected_view,result.state):raise AssertionError('Reference changed non-target subview slots')
        expected=trial.clone();expected[-1,slot,2:4]=result.state[-1,sub,2:4];trial[-1,slot,2:4]=result.state[-1,sub,2:4]
        if not _bits(expected,trial):raise AssertionError('Nonlocal input edit')
        rec['modified_slots_full']=[[9,slot,2],[9,slot,3]]
       rec['locality_ok']=True;rec['accepted']=True
       rec['numeric_velocity_changed']=semantics!='none' and not _bits(row_before[-1,slot,2:4],trial[-1,slot,2:4])
      except (InputContractError,MotionConstraintError) as exc:
       rec['accepted']=False;rec['exception']=type(exc).__name__;rec['reason']=str(exc);rec['reference_report']=getattr(exc,'report',None);ok=False
      calls.append(rec);local.append(rec)
      if not ok:break
     if ok:
      h[b]=trial
      for rec in local:rec['committed']=True
      if semantics!='none':
       for k in coalition:
        slot=slot_targets[b][k]
        if any(r.get('target_index')==k and r.get('numeric_velocity_changed',False) for r in local) and bool(observed[b,-1,slot]):
         observed[b,-1,slot]=False
         _,newage=api.initial_provenance(observed[b:b+1]);age[b,slot]=newage[0,slot]
         cleared.append({'scene':b,'step':step,'slot':slot,'reason':'edited velocity is not a measured full state'})
     else:
      # An AB step is transactional: accepted A is rolled back if B refused.
      active[b]=False;status[b]='REQUEST_REJECTED';reasons[b].append(local[-1].get('reason','request rejected'));rejections.append({'scene':b,'step':step,'completed_steps_before_rejection':int(steps[b]),'last_call_index':len(calls)-1,'transaction_rolled_back':True})
      if not _bits(h[b],row_before):raise AssertionError('Rejected scene step changed history')
   if not active.any():break
   effective+=active.long();dummy+=(~active).long();batchcalls+=1
   if model is None:
    base,present,_=api.current_anchor(h,v);velocity=base[...,2:];newstate=torch.cat((base[...,:2]+float(dt)*velocity,velocity),-1);newstate=torch.where(present[...,None],newstate,0.)
   else:newstate=api.step(model,h,v,tm,observed,age,mode='mean',integrate=False)
   if newstate.shape!=(B,23,4) or newstate.dtype!=history.dtype or newstate.device.type!='cpu':raise InputContractError('predictor output CPU float32[B,23,4]')
   if not torch.isfinite(newstate[active]).all():raise FloatingPointError('Nonfinite output on active scenes')
   states[active,step]=newstate[active];support[active,step]=v.any(1)[active];steps+=active.long()
   usable_now=v.any(1);nh=torch.cat((h[:,1:],newstate[:,None]),1);nv=torch.cat((v[:,1:],usable_now[:,None]),1)
   no,na=api.advance_provenance(observed,age)
   h=torch.where(active[:,None,None,None],nh,h);v=torch.where(active[:,None,None],nv,v);observed=torch.where(active[:,None,None],no,observed);age=torch.where(active[:,None],na,age)
 for b in range(B):
  if active[b] and int(steps[b])==horizon:status[b]='COMPLETED'
 checks={'history_input_unchanged':_bits(history,originals[0]),'valid_input_unchanged':_bits(valid,originals[1]),'teams_input_unchanged':_bits(teams,originals[2]),'commands_input_unchanged':commands is None or _bits(commands,originals[3]),'never_usable_support_not_created':not bool(support[~usable[:,None,:].expand(-1,horizon,-1)].any()),'full_model_nodes_retained':True,'only_terminal_target_velocity_scattered':all(x.get('locality_ok',True) for x in calls),'edited_states_never_marked_new_real_observations':True}
 if not all(checks.values()):raise AssertionError('adapter invariance failure')
 return {'states':states,'support':support,'completed_steps':steps,'scene_status':status,'statuses':status,'scene_reasons':reasons,'reasons':reasons,'initial_scene_supported':initial_active,'rejections':rejections,'operator_calls':calls,'call_records':calls,'operator_call_counts':torch.tensor([sum(x['scene']==b for x in calls) for b in range(B)],dtype=torch.long),'actual_operator_calls':len(calls),'batch_forward_calls':batchcalls,'learned_batch_forward_calls':batchcalls if model is not None else 0,'scene_effective_forward_counts':effective,'scene_dummy_forward_counts':dummy,'edited_observation_flags_cleared':cleared,'final_observation_valid':observed,'final_observation_age':age,'checks':checks,'semantics':semantics,'coalition':coalition,'requested_horizon_steps':horizon,'requested_duration_s':horizon*float(dt) if semantics=='persistent' else (float(dt) if semantics=='first' else 0.),'input_command_is_not_hard_output_path':True,'value_or_event_readout_computed':False}
