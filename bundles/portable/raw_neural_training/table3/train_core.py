"""Original epoch/Adam/loader/loss computation; I/O now targets explicit outputs."""
from pathlib import Path
import hashlib,json,random
import numpy as np
import torch
from torch.utils.data import DataLoader
from runtime import *
from losses import Windows,objective
from training_io import sha,dump,save_torch,preserve_epoch,array_hash
ROOT=DATA=TRUTH=None
CONTRACT=None
class UserPaused(Exception):pass
def paused():
    p=ROOT/'user_pause.json'
    return p.exists() and json.loads(p.read_text(encoding='utf-8-sig')).get('status') in ('PAUSE_REQUESTED','PAUSING','PAUSED_USER')
def rng_state(generator):
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(),python=random.getstate(),loader=generator.get_state())

def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

def fit_cell(arm,seed,arrays,device):
    out=ROOT/'runs'/f'{arm}_seed{seed}'
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text(encoding='utf-8'))
        if result['status']=='COMPLETED' and result['epochs_completed']==4 and result['updates']==96 and sha(out/'world_last.pt')==result['checkpoint_sha256']:
            print('SKIP '+out.name,flush=True);return result
        raise ValueError('Changed/incomplete completed cell')
    model=make_model(arm,seed).to(device);ch=common_hash(model);vh=velocity_head_hash(model)
    smoke=CONTRACT['initialization']
    if ch!=smoke['common_hashes'][str(seed)] or vh!=smoke['velocity_hashes'][str(seed)]:raise ValueError('Initialization changed')
    seed_all(100_003_003+seed);optimizer=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=0.)
    config=dict(arm=arm,seed=seed,epochs=4,batch_size=32,updates=96,unroll_steps=20,anchor=0,
        lr=.0003,gradient_clip=200.,initial_common_hash=ch,initial_velocity_hash=vh,
        model_config=model.model_config,parameters=sum(p.numel() for p in model.parameters()),
        data_sha256=sha(DATA),truth_sha256=sha(TRUTH),
        augmentation_seed=AUGMENTATION_SEED,checkpoint='fixed epoch4')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Resume config mismatch')
    dump(out/'config.json',config)
    history=[];start=1
    if (out/'world_last.pt').exists():
        checkpoint=torch.load(out/'world_last.pt',map_location=device,weights_only=False)
        if checkpoint['protocol_sha256']!=CONTRACT['original_protocol_sha256']:raise ValueError('Resume protocol mismatch')
        model.load_state_dict(checkpoint['state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restore_rng(checkpoint['rng_state']);history=checkpoint['history'];start=checkpoint['epoch']+1
    dataset=Windows(arrays)
    for epoch in range(start,5):
        if paused():raise UserPaused('No new epoch while pause marker active')
        dataset.epoch=epoch;generator=torch.Generator().manual_seed(100_003_003+seed+7+1000003*(epoch-1))
        loader=DataLoader(dataset,batch_size=32,shuffle=True,generator=generator,num_workers=0)
        model.train();order=[];gradients=[];nll_sum=pos_sum=total_sum=0.;nc=pc=0;mask_digest=hashlib.sha256()
        for raw in loader:
            order.extend(raw.pop('item_index').tolist());mask_digest.update(raw['valid'].numpy().tobytes())
            batch={k:v.to(device) for k,v in raw.items()}
            optimizer.zero_grad(set_to_none=True);loss,d=objective(model,batch,arm)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),200.,error_if_nonfinite=True)
            optimizer.step();gradients.append(float(norm));nll_sum+=d['nll']*d['nll_count'];nc+=d['nll_count']
            pos_sum+=d['position_huber']*d['position_count'];pc+=d['position_count'];total_sum+=d['loss']
        if len(gradients)!=24 or sorted(order)!=list(range(768)):raise AssertionError('Wrong matched training budget')
        row=dict(epoch=epoch,train_nll=nll_sum/nc,train_position_huber=pos_sum/max(pc,1),
            mean_batch_objective=total_sum/24,nll_targets=nc,position_targets=pc,updates=24,updates_total=24*epoch,
            loader_order_hash=array_hash(np.asarray(order,dtype=np.int64)),augmentation_mask_hash=mask_digest.hexdigest(),gradient_norm_mean=float(np.mean(gradients)),
            gradient_norm_max=float(np.max(gradients)),clipped_steps=int(np.sum(np.asarray(gradients)>200.)))
        history.append(row)
        save_torch(out/'world_last.pt',dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),rng_state=rng_state(generator),epoch=epoch,history=history,
            protocol_sha256=CONTRACT['original_protocol_sha256'],arm=arm,seed=seed))
        dump(out/'history.json',history)
        preserve_epoch(out,epoch)
        print(f'{arm} seed={seed} epoch={epoch}/4 updates={24*epoch}/96 nll={row["train_nll"]:.6g} pos={row["train_position_huber"]:.6g}',flush=True)
        if paused():raise UserPaused('Saved model/Adam/RNG/history; paused after epoch')
    result=dict(status='COMPLETED',arm=arm,seed=seed,epochs_completed=4,updates=96,
        checkpoint_sha256=sha(out/'world_last.pt'),history_sha256=sha(out/'history.json'),
        common_initial_hash=ch,velocity_initial_hash=vh,checkpoint_used='fixed epoch4')
    dump(out/'result.json',result);return result
