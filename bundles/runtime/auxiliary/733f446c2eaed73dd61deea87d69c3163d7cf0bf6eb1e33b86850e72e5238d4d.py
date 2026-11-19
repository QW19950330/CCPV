"""Pure NumPy local conditional-speed/probability identities, no data lookup."""
import hashlib
import numpy as np
GROUPS={'K16':(0,16),'First8':(0,8),'Last8':(8,16)}
READOUTS=('CPUmean','OldGPUmean')+tuple(kind+'_'+g for g in GROUPS for kind in ('MC','MeanFeature','MeanPath'))
def mean_features_literal(particles,old_features,lo,hi):
    result=np.asarray(particles,np.float64)[:,lo:hi].mean(1)
    result[:,:18]=np.asarray(old_features)[:,:18]
    return result
def noise_for(seed,anchors,K=16):
    if seed not in range(5) or K!=16:raise ValueError('Fixed five models and16particles only')
    u=np.empty((len(anchors),K,2,23),np.float32);eps=np.empty((len(anchors),K,2,23,2),np.float32)
    keys=np.empty((len(anchors),K,2,2),dtype='U64')
    for b,anchor in enumerate(anchors):
        for k in range(K):
            for t in range(2):
                for component in range(2):
                    key=f'20261026|model={seed}|anchor={str(anchor)}|particle={k}|suffixstep={t+19}|component={component}'
                    digest=hashlib.sha256(key.encode('utf-8')).digest();keys[b,k,t,component]=digest.hex()
                    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4'))))
                    if component==0:u[b,k,t]=rng.random(23,dtype=np.float32)
                    else:eps[b,k,t]=rng.standard_normal((23,2),dtype=np.float32)
    return u,eps,keys
def speed_decomposition(particle_suffix,xy18,cpu_suffix,old_suffix):
    # Inputs[B,K,2,N,4], [B,N,2], [B,2,N,4]; physical m,m/s.
    ps,anchor,cs,gs=[np.asarray(x,np.float64) for x in (particle_suffix,xy18,cpu_suffix,old_suffix)]
    if ps.ndim!=5 or ps.shape[1:3]!=(16,2) or ps.shape[-2:]!=(23,4):raise ValueError('Fixed16two-stepjointpaths')
    if not all(np.isfinite(x).all() for x in (ps,anchor,cs,gs)):raise ValueError('Nonfinite local paths')
    V=(ps[:,:,1,:,:2]-anchor[:,None])/.2
    vc=(cs[:,1,:,:2]-anchor)/.2;vg=(gs[:,1,:,:2]-anchor)/.2
    result={}
    for name,(lo,hi) in GROUPS.items():
        v=V[:,lo:hi];norm=np.linalg.norm(v,axis=-1);center=v.mean(1);cn=np.linalg.norm(center,axis=-1)
        j=norm.mean(1)-cn;c=cn-np.linalg.norm(vc,axis=-1);backend=np.linalg.norm(vc,axis=-1)-np.linalg.norm(vg,axis=-1)
        assert np.max(np.abs((norm.mean(1)-np.linalg.norm(vc,axis=-1))-j-c))<1e-12
        assert np.max(np.abs((norm.mean(1)-np.linalg.norm(vg,axis=-1))-j-c-backend))<1e-12
        if j.min() < -1e-12:raise AssertionError('Jensen algebra error')
        result[name]=dict(V_mean=center,V_MCSE=v.std(1,ddof=1)/np.sqrt(hi-lo),norm_mean=norm.mean(1),norm_MCSE=norm.std(1,ddof=1)/np.sqrt(hi-lo),
            J=j,C=c,backend=backend,CPUmean_speed=np.linalg.norm(vc,axis=-1),oldGPU_speed=np.linalg.norm(vg,axis=-1))
    result['particle_velocity']=V
    return result
def encoded(X,cp):
    X=np.asarray(X,np.float64);finite=np.isfinite(X)
    z=np.where(finite,(X-cp['scaler_mean'])/cp['scaler_scale'],0.)
    z[:,np.asarray(cp['scaler_completely_absent_training_feature'],bool)]=0.
    result=np.concatenate((z,(~finite).astype(float)),axis=1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite normalizedheadinput')
    return result,z
def logpair(z):return -np.logaddexp(0.,-z),-np.logaddexp(0.,z)
def head(cp,X):
    Z,z=encoded(X,cp)
    eta=np.tanh(Z@cp['hidden_weight'].T+cp['hidden_bias'])@cp['output_weight'].T+cp['output_bias']
    if not np.isfinite(eta).all():raise ValueError('Nonfinite head')
    lp,ln=logpair(eta[:,0]);ep,en=logpair(eta[:,1]);qp=np.logaddexp(ep,en+lp);qn=en+ln
    return dict(q_logp=qp,q_log1mp=qn,early_logp=ep,early_log1mp=en,late_logp=lp,late_log1mp=ln,z=z)
def logmean(a,axis):
    a=np.asarray(a,np.float64);m=np.max(a,axis=axis,keepdims=True)
    return np.squeeze(m,axis=axis)+np.log(np.exp(a-m).mean(axis=axis))
def particle_probability(p):
    # Input sameparticle logs[B,K]; trueY2 never conditions these calculations.
    out={};e=np.exp(p['early_logp']);l=np.exp(p['late_logp']);q=np.exp(p['q_logp'])
    for name,(lo,hi) in GROUPS.items():
        sl=slice(lo,hi);ee,ll,qq=e[:,sl],l[:,sl],q[:,sl]
        qp=logmean(p['q_logp'][:,sl],1);qn=logmean(p['q_log1mp'][:,sl],1)
        ep=logmean(p['early_logp'][:,sl],1);en=logmean(p['early_log1mp'][:,sl],1)
        # Risk-weighted conditional late of the probability mixture, notmean(l).
        late_num=logmean(p['early_log1mp'][:,sl]+p['late_logp'][:,sl],1)
        leff=late_num-en
        cov=((ee-ee.mean(1)[:,None])*(ll-ll.mean(1)[:,None])).mean(1)
        naive=ee.mean(1)+(1-ee.mean(1))*ll.mean(1)
        assert np.max(np.abs(qq.mean(1)-naive+cov))<1e-12
        assert np.max(np.abs(np.exp(ep)+np.exp(en)*np.exp(leff)-np.exp(qp)))<1e-12
        out[name]=dict(q_logp=qp,q_log1mp=qn,early_logp=ep,early_log1mp=en,l_eff=np.exp(leff),mean_l=ll.mean(1),
            population_covariance_e_l=cov,naive_separate_branch_q=naive,q_MCSE=qq.std(1,ddof=1)/np.sqrt(hi-lo))
    return out
