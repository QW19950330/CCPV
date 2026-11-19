"""Compare all five checkpoints, component gradients and histories exactly."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import io_core as io

def compare(x,y,path,stats,diff):
    if type(x)!=type(y):diff.append(dict(path=path,kind='type'));return
    if isinstance(x,torch.Tensor):
        stats['tensors']+=1;stats['tensor_elements']+=x.numel()
        if x.dtype!=y.dtype or x.shape!=y.shape or not torch.equal(x,y):diff.append(dict(path=path,kind='tensor'))
    elif isinstance(x,np.ndarray):
        stats['arrays']+=1;stats['array_elements']+=x.size
        if x.shape!=y.shape or x.dtype!=y.dtype or x.tobytes()!=y.tobytes():diff.append(dict(path=path,kind='array'))
    elif isinstance(x,dict):
        if x.keys()!=y.keys():diff.append(dict(path=path,kind='keys'));return
        for k in x:compare(x[k],y[k],path+'.'+str(k),stats,diff)
    elif isinstance(x,(list,tuple)):
        if len(x)!=len(y):diff.append(dict(path=path,kind='length'));return
        for i,(a,b) in enumerate(zip(x,y)):compare(a,b,path+f'[{i}]',stats,diff)
    elif x!=y:diff.append(dict(path=path,kind='scalar',new=x,old=y))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,type=Path);ap.add_argument('--reference-root',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    here=Path(__file__).resolve().parent;contract=io.J(here/'contract.json');run=args.run.resolve(strict=True)
    reference=args.reference_root.resolve(strict=True);out=args.output.resolve()
    for root in (here,run,reference):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Report overlaps code/run/reference tree')
    done=io.J(run/'completion.json');rows=[]
    for cell in done['cells']:
        folder=run/'runs'/cell;ref=reference/cell
        if not (folder/'manifest.json').is_file():raise ValueError('Cell incomplete: '+cell)
        stats=dict(tensors=0,tensor_elements=0,arrays=0,array_elements=0);diff=[];files=[]
        pins=contract['reference_files'][cell]
        expected={f'epoch{e}.pt' for e in range(5)}|{f'gradient_epoch{e}.{suffix}' for e in range(5) for suffix in ('json','npz')}|{'history.json'}
        if set(pins)!=expected:raise ValueError('Incomplete five-epoch reference contract')
        for name,pin in sorted(pins.items()):
            a,b=folder/name,ref/name
            if io.sha(b)!=pin['sha256'] or b.stat().st_size!=pin['bytes']:raise ValueError('Trusted reference changed: '+name)
            start=len(diff)
            if a.suffix=='.pt':
                # Only pinned, locally trusted reference payloads are deserialized.
                new=torch.load(a,map_location='cpu',weights_only=False);old=torch.load(b,map_location='cpu',weights_only=False)
                compare(new,old,name,stats,diff)
            elif a.suffix=='.npz':
                with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
                    if set(x.files)!=set(y.files):diff.append(dict(path=name,kind='keys'))
                    else:
                        for k in x.files:compare(x[k],y[k],name+'.'+k,stats,diff)
            else:compare(io.J(a),io.J(b),name,stats,diff)
            files.append(dict(name=name,exact=len(diff)==start,bytes_equal=io.sha(a)==pin['sha256'],sha256=io.sha(a)))
        rows.append(dict(cell=cell,exact=not diff,stats=stats,differences=diff,files=files))
    report=dict(status='EXACT_ALL_EPOCHS' if all(r['exact'] for r in rows) else 'DIFFERENCE',cells=rows,
        checkpoints=5*len(rows),gradient_vectors=20*len(rows),tolerance_used=False,
        full_payload='Parameters, Adam, Python/NumPy/torch/CUDA/loader RNG, losses, loader order/mask hashes and gradient summaries at all epochs0–4',
        all_file_bytes_equal=all(f['bytes_equal'] for r in rows for f in r['files']),run_completion_sha256=io.sha(run/'completion.json'))
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],cells=len(rows),all_file_bytes_equal=report['all_file_bytes_equal'])))
    if report['status']!='EXACT_ALL_EPOCHS':raise SystemExit(2)
if __name__=='__main__':main()
