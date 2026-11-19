"""Strict full original A checkpoint/Adam/RNG comparison; never trains."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def compare(x,y,path,stats,differences):
    if type(x)!=type(y):differences.append(dict(path=path,kind='type'));return
    if isinstance(x,torch.Tensor):
        stats['tensor_count']+=1;stats['tensor_elements']+=x.numel()
        if x.dtype!=y.dtype or x.shape!=y.shape or x.detach().cpu().contiguous().numpy().tobytes()!=y.detach().cpu().contiguous().numpy().tobytes():
            differences.append(dict(path=path,kind='tensor',max_abs=float((x.cpu().double()-y.cpu().double()).abs().max()) if x.shape==y.shape and x.numel() else None))
    elif isinstance(x,np.ndarray):
        stats['numpy_count']+=1
        if x.shape!=y.shape or x.dtype!=y.dtype or x.tobytes()!=y.tobytes():differences.append(dict(path=path,kind='array'))
    elif isinstance(x,dict):
        if x.keys()!=y.keys():differences.append(dict(path=path,kind='keys'));return
        for k in x:compare(x[k],y[k],path+'.'+str(k),stats,differences)
    elif isinstance(x,(tuple,list)):
        if len(x)!=len(y):differences.append(dict(path=path,kind='length'));return
        for i,(a,b) in enumerate(zip(x,y)):compare(a,b,path+f'[{i}]',stats,differences)
    elif x!=y:differences.append(dict(path=path,kind='scalar',new=x,old=y))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    contract=read(a.contract);expected={f'A_raw4d_seed{s}' for s in range(5)}
    if set(p.name for p in (a.run/'runs').iterdir())!=expected:raise ValueError('Expected exact five A cells')
    rows=[]
    for name in sorted(expected):
        folder=a.run/'runs'/name;ref=a.reference/name;pin=contract['reference_results'][name]
        assert sha(ref/'world_last.pt')==pin['checkpoint_sha256'] and sha(ref/'history.json')==pin['history_sha256']
        new=torch.load(folder/'world_last.pt',map_location='cpu',weights_only=False)
        old=torch.load(ref/'world_last.pt',map_location='cpu',weights_only=False)
        stats=dict(tensor_count=0,tensor_elements=0,numpy_count=0);diff=[];compare(new,old,'payload',stats,diff)
        config=read(folder/'config.json')==read(ref/'config.json');history=sha(folder/'history.json')==pin['history_sha256']
        epochs=[]
        for e in range(1,5):
            cp=torch.load(folder/f'epoch{e}/world_last.pt',map_location='cpu',weights_only=False)
            epochs.append(dict(epoch=e,history_prefix_exact=cp['history']==old['history'][:e],
                optimizer_state_retained='optimizer_state_dict' in cp,rng_retained='rng_state' in cp))
        exact=not diff and config and history and all(e['history_prefix_exact'] for e in epochs)
        rows.append(dict(cell=name,exact=exact,stats=stats,differences=diff,config_equal=config,history_bytes_equal=history,
            epoch_snapshots=epochs,new_checkpoint_sha256=sha(folder/'world_last.pt'),old_checkpoint_sha256=pin['checkpoint_sha256']))
    report=dict(status='EXACT' if all(r['exact'] for r in rows) else 'DIFFERENCE',cells=rows,optimizer_updates=480,
        model_forward_calls=9600,tolerance_used=False,scope='Complete final payload, Adam, all RNG, four-epoch history, loader/mask hashes and config; no original intermediate-weight equality claim.')
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(report['status'],len(rows),'cells',sum(len(r['differences']) for r in rows),'differences')
    return 0 if report['status']=='EXACT' else 2
if __name__=='__main__':raise SystemExit(main())
