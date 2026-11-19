"""Frozen inference-only arrays. No checkpoint pickle, labels, indices or fitting."""
from pathlib import Path
import hashlib,importlib,sys,io,os
import numpy as np

def load_export(entry,root=None):
    root=Path(root or Path(__file__).resolve().parent)
    path=(root/entry['clean_payload']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Payload path escapes export root')
    prefix=chr(92)*2+'?'+chr(92)
    read_path=Path(prefix+str(path)) if os.name=='nt' and not str(path).startswith(prefix) else path
    raw=read_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['export_sha256']:raise ValueError('Export SHA256 mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        if set(z.files)!=set(entry['export_keys']):raise ValueError('Export key mismatch')
        cp={k:z[k] for k in z.files}
    for k,a in cp.items():
        if a.dtype.hasobject or a.dtype.kind not in 'biufc':raise ValueError('Numeric arrays only')
        spec=entry['arrays'][k]
        if list(a.shape)!=spec['shape'] or a.dtype.str!=spec['dtype']:raise ValueError('Export array structure mismatch')
    return cp

def transform_mlp(X,cp):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or X.shape[1]!=len(cp['scaler_mean']):raise ValueError('Raw feature dimension mismatch')
    available=np.isfinite(X)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        z=np.where(available,(X-cp['scaler_mean'])/cp['scaler_scale'],0.)
    z[:,np.asarray(cp['scaler_completely_absent_training_feature'],dtype=bool)]=0.
    result=np.concatenate((z,(~available).astype(np.float64)),axis=1)
    if not np.isfinite(result).all():raise ValueError('Normalization overflow')
    return result

def mlp_logits(cp,X):
    z=transform_mlp(X,cp)
    logits=np.tanh(z@cp['hidden_weight'].T+cp['hidden_bias'])@cp['output_weight'].T+cp['output_bias']
    if not np.isfinite(logits).all():raise ValueError('Nonfinite output')
    return logits

def binary_logpairs(logits):
    z=np.asarray(logits,np.float64)
    if not np.isfinite(z).all():raise ValueError('Finite logits required')
    return -np.logaddexp(0.,-z),-np.logaddexp(0.,z)

def two_row_logpairs(logits,semantics):
    z=np.asarray(logits,np.float64)
    if z.shape[-1]!=2 or not np.isfinite(z).all():raise ValueError('Two finite frozen logits required')
    first,second=z[...,0],z[...,1]
    if semantics in ('Direct','Marginal'):
        ep,en=binary_logpairs(second)
        if semantics=='Direct':qp,qn=binary_logpairs(first)
        else:
            lp,ln=binary_logpairs(first);qp,qn=np.logaddexp(ep,en+lp),en+ln
    elif semantics=='QFirstJoint':
        qp,qn=binary_logpairs(first);rp,rn=binary_logpairs(second)
        ep=qp+rp;en=np.logaddexp(qn,qp+rn)
    else:raise ValueError('Explicit Direct, Marginal or QFirstJoint semantics required')
    return dict(q_logp=qp,q_log1mp=qn,early_logp=ep,early_log1mp=en)

def logistic_probability(cp,X):
    X=np.asarray(X,np.float64);d=len(cp['w'])
    if X.ndim!=2 or X.shape[1]!=d:raise ValueError('Logistic feature dimension mismatch')
    z=(np.where(np.isfinite(X),X,cp['mean'][:d])-cp['mean'][:d])/cp['scale'][:d]
    logits=z@cp['w']+cp['b']
    return np.exp(-np.logaddexp(0.,-logits))

def fixed_prior(cp,rows):return np.full(int(rows),float(cp['prior']),dtype=np.float64)

def shifted_logpairs(logits,cp,key='alpha'):
    if np.asarray(cp[key]).shape!=():raise ValueError('Frozen alpha must be scalar')
    return binary_logpairs(np.asarray(logits,np.float64)+cp[key])

def load_world(entry,root=None):
    import torch
    root=Path(root or Path(__file__).resolve().parent)
    vendor=(root/'vendor_world').resolve()
    for name in ('runtime','motion_models','consistent_core.models','consistent_core.gap_model'):
        module=sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(vendor):
            raise ValueError('Conflicting world module already loaded; use a fresh process')
    sys.path.insert(0,str(vendor))
    runtime=importlib.import_module('runtime')
    cp=load_export(entry,root)
    model=runtime.make_model(entry['architecture'],int(entry['seed']))
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in cp.items()},strict=True)
    return model.cpu().eval(),runtime


def qfirst_probabilities(logits):
    """No outcome/future inputs. Categories: none, Early, late-without-Early."""
    logits = np.asarray(logits, np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError('Expected row0 total-risk and row1 conditional-Early logits')
    qp, qn = binary_logpairs(logits[:, 0])
    rp, rn = binary_logpairs(logits[:, 1])
    ep = qp + rp
    late_only = qp + rn
    # log(1-q*r) = log[(1-q)+q*(1-r)], avoiding probability subtraction.
    en = np.logaddexp(qn, late_only)
    late_conditional, no_late_conditional = late_only - en, qn - en
    category = np.stack((qn, ep, late_only), axis=-1)
    if np.max(np.abs(np.logaddexp.reduce(category, axis=1))) > 1e-12:
        raise AssertionError('Three-category normalization failed')
    if np.max(np.abs(np.logaddexp(late_conditional, no_late_conditional))) > 1e-12:
        raise AssertionError('Late-given-no-Early normalization failed')
    return dict(q_logp=qp, q_log1mp=qn, early_logp=ep, early_log1mp=en,
                r_logp=rp, r_log1mp=rn, late_only_logp=late_only,
                late_logp=late_conditional, late_log1mp=no_late_conditional,
                category_logs=category, q_logits=logits[:, 0], r_logits=logits[:, 1])


def ensemble_six_logpairs(logpairs):
 out={}
 for k,v in logpairs.items():
  a=np.asarray(v,np.float64)
  if a.shape[0]!=6:raise ValueError('All six old fixed foldheads required')
  out[k]=np.logaddexp.reduce(a,axis=0)-np.log(6.)
 return out


def logmean(a, axis=1):
    a = np.asarray(a, np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a-m).mean(axis=axis))



def mixture(particle):
    if any(np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != 16 for x in particle.values()):
        raise ValueError('Exactly all 16 particles, no split/selection')
    result = {k: logmean(v) for k, v in particle.items()}
    for stem in ('q', 'early'):
        if np.max(np.abs(np.logaddexp(result[stem+'_logp'], result[stem+'_log1mp']))) > 1e-12:
            raise ArithmeticError('Probability mixture/complement identity')
    return result



def scaler_transform(cp,X):
    X=np.asarray(X,np.float64)
    return (np.where(np.isfinite(X),X,cp['mean'])-cp['mean'])/cp['scale']

def predict_world(entry,history,valid,teams,root=None,horizon=20):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    model,runtime=load_world(entry,root);runtime.configure('cpu');model.float()
    h=torch.as_tensor(history,dtype=torch.float32);v=torch.as_tensor(valid,dtype=torch.bool);tm=torch.as_tensor(teams,dtype=torch.long)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):return runtime.predict(model,h,v,tm,horizon=int(horizon)).numpy()
