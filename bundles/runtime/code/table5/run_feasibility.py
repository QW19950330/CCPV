"""Root-gated saved-only full output engineering screen, all planned rows.

No world model, value head, q, optimizer, calibration, labels or true future.
No projection, revised paths, output repair, outcome filtering or success retry.
"""
from collections import defaultdict,Counter
from pathlib import Path
import argparse
import numpy as np
import feasibility_math as M
from feasibility_io import P,J,sha,put,load,guard,table,Stream,now

PREFIX_KEYS=('history_states','history_valid','history_seen','teams','entity_ids','identity_status',
             'operator_candidate_slots','operator_candidate_ids','operator_case_status','match_ids','source_indices')
CELL_KEYS=('states','support','completed_steps','paths','scene_status','source_indices','match_ids','target_source_ids')
NUMERIC=('fixed_group_entities','expected_full_agent_steps','anchor_t0_valid_entities','anchor_causally_extrapolated_entities',
         'anchor_age_max_s','anchor_age_mean_s','speed_agent_steps','acceleration_agent_steps','turn_defined_agent_steps',
         'moving_turn_agent_steps','turn_zero_speed_NOT_APPLICABLE_agent_steps','defined_below_moving_screen_agent_steps',
         'speed_exceed_count','acceleration_strict_exceed_count','acceleration_screen_exceed_count','turn_strict_exceed_count',
         'moving_turn_screen_exceed_count','speed_mean','speed_max','acceleration_mean','acceleration_max',
         'turn_defined_mean','turn_defined_max','moving_turn_max')
RATE_KEYS=('position_complete','head_complete','joint_complete','position_screen_pass','head_screen_pass','joint_screen_pass',
           'position_strict_pass','head_strict_pass')
CAUSES=('speed_exceed','acceleration_strict_exceed','acceleration_screen_exceed','turn_strict_exceed','moving_turn_screen_exceed')
STRUCTURAL=('UNSUPPORTED_FIXED_GROUP_BINDING','NONFINITE_CAUSAL_ANCHOR','NO_GENERATED_OUTPUT',
            'PARTIAL_GENERATED_OUTPUT','MISSING_OR_NONFINITE_GENERATED_SUPPORT')
RATE_KEYS=RATE_KEYS+tuple(f'{kind}_any_{cause}' for kind in ('position','head') for cause in CAUSES)+tuple(
    f'{kind}_structural_{status}' for kind in ('position','head') for status in STRUCTURAL)

class Bucket:
    def __init__(self):
        self.n=0;self.rates=Counter();self.stats={k:[0,0.,None] for k in [f'{r}_{k}' for r in ('position','head') for k in NUMERIC]}
    def add(self,position,head,joint):
        self.n+=1
        self.rates.update({k:int(v) for k,v in dict(position_complete=position['complete_finite_support'],head_complete=head['complete_finite_support'],
            joint_complete=joint['joint_complete_finite_support'],position_screen_pass=position['screen_status']==M.PASSED,
            head_screen_pass=head['screen_status']==M.PASSED,joint_screen_pass=joint['joint_engineering_screen_passed'],
            position_strict_pass=position['strict_status']=='ENGINEERING_STRICT_REFERENCE_PASSED',head_strict_pass=head['strict_status']=='ENGINEERING_STRICT_REFERENCE_PASSED').items()})
        for kind,r in [('position',position),('head',head)]:
            for cause in CAUSES:self.rates[f'{kind}_any_{cause}']+=int(r[cause+'_count']>0)
            for status in STRUCTURAL:self.rates[f'{kind}_structural_{status}']+=int(r['structural_status']==status)
            for k in NUMERIC:
                v=r[k]
                if v is None:continue
                a=self.stats[kind+'_'+k];a[0]+=1;a[1]+=float(v);a[2]=float(v) if a[2] is None else max(a[2],float(v))
    def row(self,base):
        out={**base,'planned_windows':self.n}
        for k in RATE_KEYS:
            out[k+'_count']=self.rates[k];out[k+'_fraction_over_planned']=self.rates[k]/self.n if self.n else None
        for k,(n,total,maxi) in self.stats.items():
            out[k+'_supported_windows']=n;out[k+'_window_mean']=total/n if n else None
            out[k+'_window_max']=maxi;out[k+'_sum']=total if n else None
        return out

def main(expected):
    plan,binding=guard(expected);reads=[]
    pins=[binding['original_integration_manifest'],binding['original_prepared_manifest'],*binding['source_files'],
          *binding['reviewer_reports_and_continuation'],*binding['source_pitch_metadata']]
    for c in binding['cells']:pins.append(c['metadata'])
    if not all(sha(r['path'])==r['sha256'] for r in pins):raise ValueError('Bound source metadata changed')
    source=load(binding['prepared'],PREFIX_KEYS,reads)
    if source['history_states'].shape!=(768,10,23,4) or not np.array_equal(source['teams'],[0]*11+[1]*11+[2]):
        raise ValueError('Original history or team schema changed')
    anchor,present,age,last=M.current_anchor(source['history_states'],source['history_valid'])
    if not np.array_equal(present,source['history_seen']):raise ValueError('Source historical-presence mismatch')
    mids=source['match_ids'].astype(str);matches=sorted(set(mids));ids=source['entity_ids']
    if len(matches)!=12 or not all(np.sum(mids==m)==64 for m in matches):raise ValueError('Need all12x64 windows')
    roster=(source['teams']!=2)[None]&present
    roster_supported=np.all(~roster|(source['identity_status']=='VERIFIED_SOURCE_PLAYER_TEAM'),axis=1)&roster.any(1)
    target=np.zeros((768,23),bool);target_supported=np.zeros(768,bool)
    for w in range(768):
        slots=source['operator_candidate_slots'][w]
        all_ids=ids[w][ids[w]!=''].astype(str).tolist()
        if len(all_ids)!=len(set(all_ids)):raise ValueError('Duplicate stable source ID')
        good=(source['operator_case_status'][w]=='SUPPORTED_TWO_SOURCE_HOME_T0_VALID' and len(set(slots.tolist()))==2
              and np.all((slots>=0)&(slots<11)))
        if good:
            good=bool(source['history_valid'][w,-1,slots].all() and np.array_equal(ids[w,slots],source['operator_candidate_ids'][w])
                      and np.all(source['identity_status'][w,slots]=='VERIFIED_SOURCE_PLAYER_TEAM'))
        if good:target[w,slots]=True;target_supported[w]=True
    output=P/'results';output.mkdir(exist_ok=False)
    status_stream=Stream(output/'scene_path_status.csv.gz');metric_stream=Stream(output/'scene_guard_metrics.csv.gz')
    diagnostic_stream=Stream(output/'scene_diagnostics.csv.gz');buckets=defaultdict(Bucket);failures=Counter()
    try:
        for cell in binding['cells']:
            a=load(cell['states'],CELL_KEYS,reads)
            if (a['states'].shape!=(9,768,20,23,4) or a['support'].shape!=(9,768,20,23)
                    or a['completed_steps'].shape!=(9,768) or a['paths'].astype(str).tolist()!=plan['paths']
                    or not np.array_equal(a['source_indices'],source['source_indices']) or not np.array_equal(a['match_ids'],source['match_ids'])
                    or not np.array_equal(a['target_source_ids'],source['operator_candidate_ids'])):
                raise ValueError('Saved path/source/actor alignment mismatch')
            for pi,path in enumerate(plan['paths']):
                for w in range(768):
                    n=int(a['completed_steps'][pi,w]);q=a['states'][pi,w];sup=a['support'][pi,w]
                    if not 0<=n<=20 or sup[n:].any() or np.any(sup&~present[w][None]):raise ValueError('Impossible saved support/length contract')
                    dims=plan['pitch_transforms'][mids[w]]
                    base=dict(cell=cell['cell'],arm=cell['arm'],seed=cell['seed'],path=path,scene=w,match_id=mids[w],
                              source_index=int(source['source_indices'][w]),completed_steps=n,old_scene_status=str(a['scene_status'][pi,w]))
                    status_stream.add({**base,'fixed_target_binding_supported':bool(target_supported[w]),'fixed_roster_group_supported':bool(roster_supported[w]),
                        'fixed_roster_players':int(roster[w].sum()),'source_history_ball_present':bool(present[w,22]),
                        'target_anchor_t0_source_fullstate_valid':bool(target_supported[w]),'all_roster_anchor_stale_entities':int(np.sum(roster[w]&(age[w]>0))),
                        'retained_all_planned_denominator':True})
                    for frame in plan['frames']:
                        factor=np.asarray(dims['standard_to_native'] if frame=='source_pitch_native' else [1.,1.,1.,1.])
                        aa=anchor[w].astype(float)*factor;qq=q.astype(float)*factor
                        L,W=(dims['L_m'],dims['W_m']) if frame=='source_pitch_native' else (105.,68.)
                        motion={k:M.series(aa,qq,sup,n,k) for k in ('position','head')}
                        diagnostic_stream.add({**base,'coordinate_frame':frame,**M.diagnostics(aa,qq,sup,n,roster[w],L,W,present[w,22],motion)})
                        for name,group,gs in [('two_t0valid_home_targets',target[w],target_supported[w]),
                                              ('all_history_present_roster_players',roster[w],roster_supported[w])]:
                            pos=M.motion_summary(aa,qq,sup,n,group,gs,age[w],'position',motion['position'])
                            head=M.motion_summary(aa,qq,sup,n,group,gs,age[w],'head',motion['head'])
                            joint=M.joint_summary(pos,head)
                            row={**base,'coordinate_frame':frame,'fixed_group':name}
                            row.update({'position_'+k:v for k,v in pos.items() if k!='kind'})
                            row.update({'head_'+k:v for k,v in head.items() if k!='kind'});row.update(joint)
                            for label,r in [('position',pos),('head',head)]:
                                for profile in ('strict','screen'):
                                    slot=r['first_'+profile+'_failure_slot']
                                    row[label+'_first_'+profile+'_failure_source_id']=str(ids[w,slot]) if slot is not None and slot>=0 else None
                            slot=joint['first_joint_failure_slot'];row['first_joint_failure_source_id']=str(ids[w,slot]) if slot is not None and slot>=0 else None
                            metric_stream.add(row)
                            key=(cell['arm'],cell['seed'],path,frame,name,mids[w]);buckets[key].add(pos,head,joint)
                            reason=joint['first_joint_failure_reason'] or 'NO_SCREEN_FAILURE'
                            failures[key+(reason,)]+=1
            print(f"CELL_COMPLETED {cell['cell']} all9x768 native+standard targets+roster",flush=True)
    finally:
        for f in (status_stream,metric_stream,diagnostic_stream):f.close()
    if status_stream.count!=110592 or metric_stream.count!=442368 or diagnostic_stream.count!=221184:raise ValueError('Planned Cartesian row count changed')
    matchrows=[]
    for (arm,seed,path,frame,group,mid),bucket in buckets.items():
        r=bucket.row(dict(arm=arm,seed=seed,path=path,coordinate_frame=frame,fixed_group=group,match_id=mid))
        if r['planned_windows']!=64:raise ValueError('Planned match denominator changed')
        matchrows.append(r)
    table(output/'guard_by_match.csv',matchrows)
    grouped=defaultdict(list)
    for r in matchrows:grouped[tuple(r[k] for k in ('arm','seed','path','coordinate_frame','fixed_group'))].append(r)
    numeric=[k for k in matchrows[0] if k not in ('arm','seed','path','coordinate_frame','fixed_group','match_id')]
    seedrows=[]
    for (arm,seed,path,frame,group),rr in grouped.items():
        if len(rr)!=12:raise ValueError('Need all12 matches in every planned group')
        row=dict(arm=arm,seed=seed,path=path,coordinate_frame=frame,fixed_group=group,planned_windows=768,planned_matches=12)
        for k in numeric:
            if k=='planned_windows':continue
            vv=[float(r[k]) for r in rr if r[k] is not None]
            row[k+'_supported_matches']=len(vv);row[k+'_equal12match']=float(np.mean(vv)) if len(vv)==12 else None
            row[k+'_available_match_mean']=float(np.mean(vv)) if vv else None
        seedrows.append(row)
    table(output/'guard_by_seed.csv',seedrows)
    grouped=defaultdict(list)
    for r in seedrows:grouped[tuple(r[k] for k in ('arm','path','coordinate_frame','fixed_group'))].append(r)
    summary=[];metricnames=[k for k in seedrows[0] if k.endswith('_equal12match')]
    for (arm,path,frame,group),rr in grouped.items():
        expectedn=1 if arm=='CV' else 5
        if len(rr)!=expectedn:raise ValueError('CV must be once; learned arms five fixed seeds')
        row=dict(arm=arm,path=path,coordinate_frame=frame,fixed_group=group,seeds=expectedn,
                 uncertainty='NA_CV_WORLD_ONCE' if arm=='CV' else 'SAMPLE_SD_FIVE_INITIALIZATION_SEEDS_NOT_MC_OR_POPULATION_CI')
        for k in metricnames:
            vv=[float(r[k]) for r in rr if r[k] is not None]
            row[k+'_supported_seeds']=len(vv);row[k+'_mean']=float(np.mean(vv)) if len(vv)==expectedn else None
            row[k+'_seed_SD']=float(np.std(vv,ddof=1)) if len(vv)==expectedn==5 else None
        summary.append(row)
    table(output/'guard_summary.csv',summary)
    failure_rows=[dict(arm=k[0],seed=k[1],path=k[2],coordinate_frame=k[3],fixed_group=k[4],match_id=k[5],
                       first_joint_failure_reason=k[6],windows=count,planned_match_windows=64,
                       fraction_over_planned_match_windows=count/64) for k,count in sorted(failures.items())]
    table(output/'first_failure_by_match.csv',failure_rows)
    if not all(sha(r['path'])==r['sha256'] for r in pins):raise ValueError('Input metadata mutated during readout')
    for record in [binding['prepared'],*[c['states'] for c in binding['cells']]]:
        if sha(record['path'])!=record['sha256']:raise ValueError('Old saved input mutated during screen')
    files=[dict(path=str(p.relative_to(P)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()]
    put(P/'run_receipt.json',dict(status='COMPLETED_SAVED_OUTPUT_ENGINEERING_REFERENCE_SCREEN_NOT_PHYSICAL_CERTIFICATION',utc=now(),
        implementation_lock_sha256=expected,experiment_plan_sha256=sha(P/'experiment_plan.json'),data_binding_sha256=sha(P/'data_binding.json'),
        planned_scene_path_rows=status_stream.count,frame_group_guard_rows=metric_stream.count,frame_diagnostic_rows=diagnostic_stream.count,
        by_match_rows=len(matchrows),by_seed_rows=len(seedrows),summary_rows=len(summary),files=files,array_reads=reads,
        old_inputs_unchanged=True,no_projection_or_trajectory_edit=True,no_q_based_selection=True,no_old680_or706_mask=True,
        new_model_fits=0,world_model_forwards=0,value_heads_read=0,value_readouts=0,labels_or_actual_future_truth_read=False,
        caveat='Complete scoped engineering-reference screen only; actual-output rejection counts do not establish physiological safety, recommended efficacy, value calibration or independent validation'))
    print('FULL_OUTPUT_SCREEN_COMPLETED planned=110592 guard_rows=442368 diagnostic_rows=221184')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-lock',required=True);args=parser.parse_args();main(args.root_lock)
