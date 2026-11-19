"""Original fit and diagnostic computation with explicit run-owned I/O."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib,json,sys
import numpy as np
from io_core import J,sha,put,save,torch_save,record,check,closed,pause_requested,now
P=DATA=None
CONTRACT=REQUESTED=None
INPUT=('states','valid','position_valid','teams','entity_ids','match_ids','source_indices','train_indices','validation_indices','source_frame_ids','frame_times_s','source_period','positions','split','pitch_L','pitch_W','probe24','validation32')
class UserPause(Exception):pass
class SoftCap(Exception):pass
def api_import(device):
    from model_source import api
    from io_core import array_hash
    api.array_hash=array_hash
    api.configure(device)
    return api
def data():
    if sha(DATA)!=CONTRACT['data']['sha256']:raise ValueError('Fixed training data differs')
    with np.load(DATA,allow_pickle=False) as z:
        if set(z.files)!=set(INPUT):raise ValueError('Exact input schema required')
        a={k:z[k] for k in INPUT}
    a['position_targets']=a['positions']
    return a
def make_initial(api,cell,seed,device):
 model=api.make_model('C_coupled_tf' if cell=='C' else 'D_coupled_ar',seed)
 expected=CONTRACT['initialization']
 if api.common_hash(model)!=expected['common_hashes'][str(seed)] or api.velocity_head_hash(model)!=expected['velocity_hashes'][str(seed)]:raise ValueError('Frozeninitialization changed; noseedrepair')
 model=model.to(device);api.seed_all(100003003+seed);return model

def optimizer_for(model):
 import torch
 return torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=0.)

def batch_to(raw,device):return {k:v.to(device) for k,v in raw.items() if k!='item_index'}

def first_batch(api,a,seed):
 import torch,windows
 from torch.utils.data import DataLoader
 windows.bind_api(api);dataset=windows.Windows(a);dataset.epoch=1;gen=torch.Generator().manual_seed(100003003+seed+7)
 return next(iter(DataLoader(dataset,batch_size=32,shuffle=True,generator=gen,num_workers=0)))

def probe_batch(api,a):
 import windows
 from torch.utils.data._utils.collate import default_collate
 windows.bind_api(api);d=windows.Windows(a);d.epoch=1;d.indices=a['probe24'];return default_collate([d[i] for i in range(24)])

def count_forwards(model):
 count={'value':0}
 def hook(module,args,out):count['value']+=1
 return count,model.register_forward_hook(hook)

def diag_save(model,optimizer,batch,folder,epoch):
 from gradient_diagnostics import gradient_diagnostic
 meta,vectors=gradient_diagnostic(model,batch,optimizer=optimizer)
 save(folder/f'gradient_epoch{epoch}.npz',**vectors);put(folder/f'gradient_epoch{epoch}.json',meta)
 return [record(folder/f'gradient_epoch{epoch}.npz'),record(folder/f'gradient_epoch{epoch}.json')]

def checkpoint_payload(api,model,optimizer,seed,cell,epoch,history,loader_state,primary_count=0,diagnostic_count=0):
 import random,torch
 return dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state_dict=optimizer.state_dict(),rng_state=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate(),loader=loader_state),epoch=epoch,history=history,seed=seed,cell=cell,canonical_model='coupled2D',initial_seed=100003003+seed,primary_forward_count=primary_count,diagnostic_forward_count=diagnostic_count)

def fit(expected):
 a=data()
 import random,torch,windows
 from torch.utils.data import DataLoader
 from torch.nn.attention import sdpa_kernel,SDPBackend
 from factorized_objective import factorized_loss
 api=api_import('cuda');windows.bind_api(api);clock=P/'training_clock.json'
 if not clock.exists():put(clock,dict(utc=now(),soft_between_closed_cells_seconds=7200))
 records=[]
 with sdpa_kernel([SDPBackend.MATH]):
  for seed,cell in REQUESTED:
   folder=P/'runs'/f'{cell}_s{seed}';manifest=folder/'manifest.json';done=closed(manifest,expected)
   if done is not None:records.append(done);continue
   if pause_requested():raise UserPause('Explicituserpause beforecell')
   if (datetime.now(timezone.utc)-datetime.fromisoformat(J(clock)['utc'])).total_seconds()>=7200:raise SoftCap('120min BETWEENclosedcells; noactiveepochkill')
   resume=[]
   if folder.exists():
    if not J(P/'runtime_authorization.json').get('allow_closed_epoch_resume',False):raise RuntimeError('Closedepochresume requires explicitrootreview authorization')
    for e in range(5):
     c=closed(folder/f'epoch{e}_closed.json',expected)
     if c is None:break
     resume.append(c)
    known={Path(r['path']).name for c in resume for r in c['files']}|{f'epoch{e}_closed.json' for e in range(len(resume))}
    if not resume or any(p.name not in known for p in folder.iterdir()):raise RuntimeError('Unclosedpartialfiles preserved; rootreview/archive before resume, nooverwrite')
   else:folder.mkdir(parents=True)
   model=make_initial(api,cell,seed,'cuda');model.train();opt=optimizer_for(model);dataset=windows.Windows(a);probe=batch_to(probe_batch(api,a),'cuda');history=[];files=[];count,hook=count_forwards(model)
   if resume:
    checkpoint=resume[-1]['checkpoint'];check(checkpoint);payload=torch.load(checkpoint['path'],map_location='cuda',weights_only=False)
    if (payload['cell'],payload['seed'],payload['epoch'])!=(cell,seed,len(resume)-1):raise ValueError('Resumeidentity mismatch')
    model.load_state_dict(payload['state_dict'],strict=True);opt.load_state_dict(payload['optimizer_state_dict']);state=payload['rng_state'];torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']]);np.random.set_state(state['numpy']);random.setstate(state['python']);history=payload['history'];primary_forwards=payload['primary_forward_count'];diagnostic_forwards=payload['diagnostic_forward_count'];files=[r for c in resume for r in c['files']];start_epoch=payload['epoch']+1
   else:
    files+=diag_save(model,opt,probe,folder,0);primary_forwards=0;diagnostic_forwards=count['value'];torch_save(folder/'epoch0.pt',checkpoint_payload(api,model,opt,seed,cell,0,history,None,primary_forwards,diagnostic_forwards));files.append(record(folder/'epoch0.pt'));put(folder/'epoch0_closed.json',dict(implementation_lock_sha256=expected,cell=cell,seed=seed,epoch=0,checkpoint=record(folder/'epoch0.pt'),files=files[-3:]));start_epoch=1
   for epoch in range(start_epoch,5):
    if pause_requested():raise UserPause('Explicituserpause beforestartingnext epoch; priorcheckpointclosed')
    dataset.epoch=epoch;gen=torch.Generator().manual_seed(100003003+seed+7+1000003*(epoch-1));loader=DataLoader(dataset,batch_size=32,shuffle=True,generator=gen,num_workers=0);order=[];mask_digest=hashlib.sha256();norms=[];ns=ps=0.;nc=pc=0;loss_sum=0.
    for raw in loader:
     order.extend(raw['item_index'].tolist());mask_digest.update(raw['valid'].numpy().tobytes());batch=batch_to(raw,'cuda');opt.zero_grad(set_to_none=True);before=count['value'];loss,d=factorized_loss(model,batch,cell);actual=count['value']-before
     if actual!=(20 if cell in ('C','D') else 40):raise AssertionError('Exactbranchforwardbudget')
     primary_forwards+=actual
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinitefixedtrainingobjective')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),200.,error_if_nonfinite=True);opt.step();norms.append(float(norm));n=int(d['nll_count']);p=int(d['position_count']);nc+=n;pc+=p;ns+=float(d['nll_loss'].detach())*n;ps+=float(d['position_loss'].detach())*p;loss_sum+=float(loss.detach())
    if len(norms)!=24 or sorted(order)!=list(range(768)) or nc!=224604 or pc!=224811:raise AssertionError('Frozenepochorder/targetcountbudget')
    row=dict(epoch=epoch,train_nll=ns/nc,train_position_huber=ps/pc,mean_batch_objective=loss_sum/24,nll_targets=nc,position_targets=pc,updates=24,updates_total=epoch*24,loader_order_hash=api.array_hash(np.asarray(order,np.int64)),augmentation_mask_hash=mask_digest.hexdigest(),gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(max(norms)),clipped_steps=int(np.sum(np.asarray(norms)>200.)))
    history.append(row);before=count['value'];files+=diag_save(model,opt,probe,folder,epoch);diagnostic_forwards+=count['value']-before;torch_save(folder/f'epoch{epoch}.pt',checkpoint_payload(api,model,opt,seed,cell,epoch,history,gen.get_state(),primary_forwards,diagnostic_forwards));files.append(record(folder/f'epoch{epoch}.pt'));put(folder/f'epoch{epoch}_closed.json',dict(implementation_lock_sha256=expected,cell=cell,seed=seed,epoch=epoch,checkpoint=record(folder/f'epoch{epoch}.pt'),files=files[-3:]));print('EPOCH_CLOSED',cell,seed,epoch,flush=True)
    if pause_requested():raise UserPause('Explicituserpause; epochmodel/Adam/RNG/gradient closed')
   if diagnostic_forwards!=200:raise AssertionError('Fixed5checkpoint diagnostic200forwardspercell')
   hook.remove();put(folder/'history.json',history);files.append(record(folder/'history.json'));files += [record(folder/f'epoch{e}_closed.json') for e in range(5)];put(manifest,dict(status='FIXED4EPOCH_CELL_COMPLETE',implementation_lock_sha256=expected,cell=cell,seed=seed,updates=96,primary_transition_forwards=primary_forwards,diagnostic_transition_forwards=diagnostic_forwards,checkpoint=record(folder/'epoch4.pt'),files=files));records.append(J(manifest));print('CELL_CLOSED',cell,seed,flush=True)
 if len(records)!=len(REQUESTED) or sum(r['updates'] for r in records)!=96*len(REQUESTED) or sum(r['primary_transition_forwards'] for r in records)!=sum(1920 if c in ('C','D') else 3840 for _,c in REQUESTED) or sum(r['diagnostic_transition_forwards'] for r in records)!=200*len(REQUESTED):raise AssertionError('All4x5 primaryfitanddiagnosticbudget')
 put(P/'weights_freeze.json',dict(status='REQUESTED_FRESH_FACTORIAL_WEIGHTS_FROZEN_BEFORE_EVALUATION',implementation_lock_sha256=expected,utc=now(),records=records,files=[r for c in records for r in c['files']],primary_updates=96*len(REQUESTED),primary_transition_forwards=sum(r['primary_transition_forwards'] for r in records),diagnostic_transition_forwards=200*len(REQUESTED),allcells_fixed4epochs=True,validation_predictions_read=False))
