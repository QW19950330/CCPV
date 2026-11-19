"""Locked label-free CPU two-step conditional branch from proven exact cache."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_n]='1'
import argparse
import numpy as np
from io_core import P,ROOT,sha,read,put,save,selected,utc,guard,array_sha
from data import causal_inputs,old_cache,prefix_choice
from source_bridge import load_frozen_D,to_native_coordinates,features_from_native_paths,union_features
from probe_math import GROUPS,noise_for,speed_decomposition,mean_features_literal
from sampling import two_step_branches
BATCH=32

def features(batch,rows,full,valid,X0,Zold):
    x2,z=features_from_native_paths(batch,rows,full,valid,X0)
    _,x=union_features(X0[rows],Zold[rows],z,x2)
    return x
def generate(expected):
    guard(expected,'generate')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cuda_inventory_visible=bool(torch.cuda.is_available())
    torch.set_default_device('cpu')
    device_probe=torch.empty(0,device='cpu')
    assert torch.get_default_device().type=='cpu' and device_probe.device.type=='cpu'
    runtime_device_guard=dict(cuda_inventory_visible=cuda_inventory_visible,
        default_device=str(torch.get_default_device()),actual_empty_tensor_device=str(device_probe.device),
        execution_device='cpu',inventory_visibility_is_not_device_use=True)
    print('CPU_DEVICE_GUARD',runtime_device_guard,flush=True)
    b,batch,a,ix=causal_inputs();choice,proof=prefix_choice()
    assert not (P/'generated').exists() and not (P/'generation_freeze.json').exists()
    (P/'generated').mkdir();records=[]
    for seed in range(5):
        cache=old_cache(b,batch,seed)
        api,model,cp=load_frozen_D(seed,authorized=True,device='cpu')
        assert all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
        assert cp['sha256']==b['checkpoints'][seed]['checkpoint']['sha256']
        statehash=api.state_hash(model.state_dict())
        assert statehash==proof['seed_results'][seed]['old_parameter_state_sha256']
        for j,start in enumerate(range(0,len(ix),BATCH)):
            pos=np.arange(start,min(start+BATCH,len(ix)));rows=ix[pos];B=len(rows)
            gpu=cache['generated_native'][rows].copy();valid=cache['generated_valid'][rows].copy()
            assert np.isfinite(gpu).all() and np.array_equal(valid,batch.valid[rows].any(1)[:,None].repeat(20,1))
            history=gpu[:,8:18].copy();hv=valid[:,8:18].copy();obs=np.zeros_like(hv);age=cache['real_observation_age_s'][rows,17].copy()
            assert history.dtype==np.float32 and age.dtype==np.float32
            u,eps,keys=noise_for(seed,a['anchor_id'][rows])
            initial_hashes=dict(history=array_sha(history),valid=array_sha(hv),observed=array_sha(obs),age=array_sha(age))
            branch,trace=two_step_branches(api,model,*map(torch.from_numpy,(history,hv,obs,age)),u,eps)
            assert initial_hashes==dict(history=array_sha(history),valid=array_sha(hv),observed=array_sha(obs),age=array_sha(age))
            cpu=gpu.copy();cpu[:,18:]=branch['cpu_suffix_model']
            cpu_native=to_native_coordinates(cpu,batch,rows)
            particle_native=branch['particle_suffix_model'].copy() # A proof establishes exact factor1; assert transform below.
            assert np.array_equal(to_native_coordinates(gpu,batch,rows),gpu)
            assert np.array_equal(cpu_native,cpu)
            xf_cpu=features(batch,rows,cpu_native,valid,a['X0'],a['Z_cv'])
            oldfeatures=cache['DirectUnion36'][rows].copy()
            particles=[]
            for k in range(16):
                full=gpu.copy();full[:,18:]=particle_native[:,k]
                particles.append(features(batch,rows,full,valid,a['X0'],a['Z_cv']))
            pf=np.stack(particles,1)
            assert np.array_equal(pf[:,:,:18],oldfeatures[:,None,:18].repeat(16,1),equal_nan=True)
            assert np.array_equal(xf_cpu[:,:18],oldfeatures[:,:18],equal_nan=True)
            arrays=dict(seed=np.array(seed),original_rows=rows,cohort_positions=pos,anchor_id=a['anchor_id'][rows],match_id=a['match_id'][rows],
                fixed_team_A=a['fixed_team_A'][rows],fixed_team_A_index=batch.fixed_team_A_index[rows],entity_ids=batch.entity_ids[rows],teams=batch.teams,present=valid[:,0],
                history_at1p8_model=history,history_at1p8_valid=hv,history_at1p8_observed=obs,age_at1p8=age,
                cpu_mean_model=cpu,cpu_mean_native=cpu_native,old_gpu_native=gpu,generated_valid=valid,
                particle_suffix_native=particle_native,noise_uniform=u,noise_normal=eps,noise_key_sha256=keys,
                particle_valid=np.broadcast_to(valid[:,None,18:],(B,16,2,23)).copy(),particle_observed=np.zeros((B,16,2,23),bool),
                cpu_suffix_valid=valid[:,18:].copy(),cpu_suffix_observed=np.zeros((B,2,23),bool),
                features_cpu=xf_cpu,features_oldGPU=oldfeatures,features_particles=pf,**branch)
            decomposition=speed_decomposition(particle_native,gpu[:,17,:,:2],cpu_native[:,18:],gpu[:,18:])
            arrays['particle_velocity_1p8_to2']=decomposition['particle_velocity']
            for name,(lo,hi) in GROUPS.items():
                for key,value in decomposition[name].items():arrays['speed_'+name+'__'+key]=value
                avg=particle_native[:,lo:hi].astype(np.float64).mean(1)
                meanpath=gpu.astype(np.float64);meanpath[:,18:]=avg
                arrays['mean_suffix_'+name]=avg
                arrays['features_meanpath_'+name]=features(batch,rows,meanpath,valid,a['X0'],a['Z_cv'])
                arrays['features_meanfeature_'+name]=mean_features_literal(pf,oldfeatures,lo,hi)
                assert array_sha(arrays['features_meanfeature_'+name][:,:18])==array_sha(oldfeatures[:,:18])
                assert np.array_equal(arrays['features_meanpath_'+name][:,:18],oldfeatures[:,:18],equal_nan=True)
            assert api.state_hash(model.state_dict())==statehash
            path=P/'generated'/f's{seed}_b{j:03}.npz';save(path,**arrays)
            records.append(dict(path=path.relative_to(P).as_posix(),sha256=sha(path),seed=seed,batch=j,original_rows=rows,cohort_positions=pos,
                prefix_mode=choice['prefix_mode'],input_hashes=initial_hashes,checkpoint_sha256=cp['sha256'],
                parameter_state_before=statehash,parameter_state_after=api.state_hash(model.state_dict()),
                CPUmean_actual_model_calls=2,particle_first_step_actual_model_calls=0,
                particle_second_step_actual_model_calls=int(np.ceil(B*16/128)),particle_second_max_NN_batch=128,
                trace=trace,prefix_observation_history_allFalse=True,exact_saved_age18=True,never_seen_preserved=True,
                no_target_or_future_arrays_decoded=True))
            if j%20==0:print('FROZEN_SUFFIX',seed,j,'ROWS',int(pos[-1])+1,'/5344',flush=True)
        del model
    assert len(records)==5*167
    put(P/'generation_freeze.json',dict(status='ALL_FIVE_MODELS_K16_SUFFIX_STATES_AND_FEATURES_FROZEN_BEFORE_VALUE_READOUT',utc=utc(),
        implementation_lock_sha256=expected,records=records,original_rows=5344,seeds=5,K=16,suffix_steps=2,
        prefix_selection_sha256=sha(ROOT/'prefix_selection.json'),prefix_addendum_sha256=sha(ROOT/'prefix_mode_addendum.json'),
        storage_proof_sha256=sha(ROOT/'reviews/storage_equivalence_QA.json'),readout_spec_sha256=sha(ROOT/'readout_spec.json'),
        device='cpu',dtype='float32',cpu_threads=1,runtime_device_guard=runtime_device_guard,no_new_fit=True,no_outcome_or_actual_future_read=True,
        B_cpu_replay_not_executed=True,not_full_rollout_distribution_integral=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--implementation-lock-sha256',required=True);args=p.parse_args();generate(args.implementation_lock_sha256)
