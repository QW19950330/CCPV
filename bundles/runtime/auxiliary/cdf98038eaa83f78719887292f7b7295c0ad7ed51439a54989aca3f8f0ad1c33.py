"""Frozen raw602 NumPy readouts after all particles/features are immutable."""
import os
for _n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_n]='1'
import argparse
import numpy as np
from io_core import P,ROOT,sha,read,put,save,selected,table,utc,guard,path
from data import readout_binding,read_head,old_q_readouts
from probe_math import GROUPS,READOUTS,head,particle_probability
FEATURE_KEYS=('seed','original_rows','cohort_positions','anchor_id','match_id','fixed_team_A','present','features_cpu','features_oldGPU','features_particles')+tuple('features_'+kind+'_'+g for g in GROUPS for kind in ('meanpath','meanfeature'))

def readout(expected):
    guard(expected,'readout');g=read(P/'generation_freeze.json')
    assert g['implementation_lock_sha256']==expected and g['K']==16 and g['original_rows']==5344
    for r in g['records']:assert sha(path(r['path']))==r['sha256']
    b=readout_binding();old=old_q_readouts(b);N=5344
    assert len(old['anchor_id'])==N
    cps={(r['seed'],r['held_match']):read_head(r) for r in b['heads']}
    assert len(cps)==30
    arrays={};diagnostics=[];ood=[];recomputed_max=0.
    def store(name,key,seed,pos,value):
        full=name+'__'+key
        if full not in arrays:arrays[full]=np.full((5,N)+value.shape[1:],np.nan)
        arrays[full][seed,pos]=value
    for rec in g['records']:
        d=selected(path(rec['path']),FEATURE_KEYS);seed=int(d['seed']);pos=d['cohort_positions'].astype(int)
        assert np.array_equal(d['anchor_id'],old['anchor_id'][pos]) and np.array_equal(d['original_rows'],old['initial_indices'][pos])
        assert np.array_equal(d['fixed_team_A'],old['fixed_team_A'][pos])
        for game in sorted(set(d['match_id'].astype(str))):
            take=d['match_id'].astype(str)==game;jj=pos[take];cp=cps[seed,game]
            cpu=head(cp,d['features_cpu'][take]);gpu=head(cp,d['features_oldGPU'][take])
            B=int(take.sum());pp=head(cp,d['features_particles'][take].reshape(B*16,36))
            part={k:v.reshape(B,16) for k,v in pp.items() if k!='z'};mix=particle_probability(part)
            for k in ('q_logp','q_log1mp','early_logp','early_log1mp'):
                literal=old['Raw_FullD_Marginal__'+k][seed,jj]
                recomputed_max=max(recomputed_max,float(np.max(np.abs(gpu[k]-literal))))
                store('OldGPUmean',k,seed,jj,literal);store('CPUmean',k,seed,jj,cpu[k])
                store('particles',k,seed,jj,part[k])
            for k in ('late_logp','late_log1mp'):store('particles',k,seed,jj,part[k])
            for group in GROUPS:
                mf=head(cp,d['features_meanfeature_'+group][take]);mp=head(cp,d['features_meanpath_'+group][take])
                for k in ('q_logp','q_log1mp','early_logp','early_log1mp'):
                    store('MC_'+group,k,seed,jj,mix[group][k]);store('MeanFeature_'+group,k,seed,jj,mf[k]);store('MeanPath_'+group,k,seed,jj,mp[k])
                mcq=np.exp(mix[group]['q_logp']);mfq=np.exp(mf['q_logp']);mpq=np.exp(mp['q_logp']);cq=np.exp(cpu['q_logp'])
                first,second,third=mcq-mfq,mfq-mpq,mpq-cq
                assert np.max(np.abs(mcq-cq-first-second-third))<1e-12
                for k,v in mix[group].items():
                    if k not in ('q_logp','q_log1mp','early_logp','early_log1mp'):store('mix_'+group,k,seed,jj,v)
                for k,v in dict(MC_minus_MeanFeature=first,MeanFeature_minus_MeanPath=second,MeanPath_minus_CPU=third).items():store('probability_'+group,k,seed,jj,v)
            zpart=pp['z'].reshape(B,16,36);zcpu=cpu['z']
            store('OOD','particle_z',seed,jj,zpart);store('OOD','CPUmean_z',seed,jj,zcpu)
            for j,row in enumerate(jj):
                ood.append(dict(seed=seed,anchor_id=str(old['anchor_id'][row]),match_id=game,
                    particle_feature_count=16*36,particle_absz_gt3=int((np.abs(zpart[j])>3).sum()),particle_absz_gt5=int((np.abs(zpart[j])>5).sum()),
                    CPUmean_feature_count=36,CPUmean_absz_gt3=int((np.abs(zcpu[j])>3).sum()),CPUmean_absz_gt5=int((np.abs(zcpu[j])>5).sum()),
                    no_clipping_or_exclusion=True))
            diagnostics.append(dict(seed=seed,match_id=game,rows=B,frozen_head_used=True,no_fit=True,no_labels_or_future=True))
    assert all(np.isfinite(v).all() for v in arrays.values()) and recomputed_max<1e-12
    for name in READOUTS:
        for k in ('q_logp','q_log1mp','early_logp','early_log1mp'):assert arrays[name+'__'+k].shape==(5,N)
    save(P/'frozen_readouts.npz',**arrays,initial_indices=old['initial_indices'],anchor_id=old['anchor_id'],match_id=old['match_id'],fixed_team_A=old['fixed_team_A'],prefix_supported=old['prefix_supported'])
    table(P/'readout_OOD_diagnostics.csv',ood)
    put(P/'readout_freeze.json',dict(status='ALL_ELEVEN_FIXED_RAW_READOUTS_FROZEN_BEFORE_LABEL_OR_MOTION_EVALUATION',utc=utc(),
        implementation_lock_sha256=expected,generation_freeze_sha256=sha(P/'generation_freeze.json'),readout_sha256=sha(P/'frozen_readouts.npz'),
        original_rows=N,models=5,readouts=READOUTS,old_GPU_q_literal_reused=True,old_GPU_recomputed_max_logpair_difference=recomputed_max,
        OOD_csv_sha256=sha(P/'readout_OOD_diagnostics.csv'),checks=diagnostics,no_fit=True,no_calibration=True,
        no_true_Y2_gate=True,no_outcomes_or_actual_future_decoded=True,no_particle_clipping_or_selection=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--implementation-lock-sha256',required=True);a=p.parse_args();readout(a.implementation_lock_sha256)
