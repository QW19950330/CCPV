"""Fixed four-epoch training; no hyperparameter or best-checkpoint selection."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--device' not in sys.argv or sys.argv[sys.argv.index('--device')+1]=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,json,time,traceback,uuid
from pathlib import Path
import numpy as np
import psutil
import torch
import train_core as core
import training_io as io

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--development-data',required=True,type=Path)
    ap.add_argument('--development-truth',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--cells',nargs='+',help='Fixed arm_seedN names; default is all 15 retained cells')
    ap.add_argument('--resume',action='store_true',help='Resume only this launcher-owned output and identical request')
    args=ap.parse_args();here=Path(__file__).resolve().parent
    contract=read(here/'training_contract.json')
    all_cells=[f'{a}_seed{s}' for a in contract['arms'] for s in contract['seeds']]
    cells=args.cells or all_cells
    if len(cells)!=len(set(cells)) or set(cells)-set(all_cells):ap.error('Duplicate or unsupported fixed cell')
    data=args.development_data.resolve(strict=True);truth=args.development_truth.resolve(strict=True)
    for p,key in ((data,'development.npz'),(truth,'position_truth_v2.npz')):
        pin=contract['data'][key]
        if p.stat().st_size!=pin['bytes'] or io.sha(p)!=pin['sha256']:raise ValueError('Input bytes differ: '+key)
    out=args.output.resolve()
    for root in (here,data.parent,truth.parent):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Output overlaps code/input tree')
    if not out.parent.is_dir():raise FileNotFoundError('Output parent must exist')
    spec=dict(data_sha256=io.sha(data),truth_sha256=io.sha(truth),cells=cells,device=args.device,
        code_sha256={str(p.relative_to(here)).replace('\\','/'):io.sha(p) for p in here.rglob('*.py')},
        contract_sha256=io.sha(here/'training_contract.json'))
    if args.resume:
        if not out.is_dir() or read(out/'run_spec.json')!=spec:raise ValueError('Resume request/code/input differs')
        if (out/'completion.json').exists():raise FileExistsError('Run is already complete')
    else:out.mkdir(exist_ok=False)
    io.ROOT=out;core.ROOT=out;core.DATA=data;core.TRUTH=truth;core.CONTRACT=contract
    if core.paused():raise core.UserPaused('Explicit pause marker active')
    lock=out/'active_process.json'
    if lock.exists():
        previous=read(lock)
        try:
            p=psutil.Process(previous['pid'])
            if p.is_running() and abs(p.create_time()-previous['create_time'])<.05:raise RuntimeError('Matching trainer already running')
        except psutil.NoSuchProcess:pass
        os.replace(lock,out/('previous_process_'+uuid.uuid4().hex+'.json'))
    p=psutil.Process();identity=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())
    with lock.open('x',encoding='utf-8') as f:json.dump(identity,f)
    if not args.resume:io.dump(out/'run_spec.json',spec)
    attempt=uuid.uuid4().hex;started=time.monotonic()
    io.dump(out/f'attempt_{attempt}.json',dict(status='RUNNING',identity=identity,resume=args.resume))
    try:
        with np.load(data,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        with np.load(truth,allow_pickle=False) as z:
            for k in ('source_indices','position_valid','match_ids','source_frame_ids','entity_ids'):
                if not np.array_equal(a[k],z[k]):raise ValueError('Truth alignment differs: '+k)
            a['position_targets']=z['positions']
        if len(a['train_indices'])!=768 or len(a['validation_indices'])!=256:raise ValueError('Fixed cohort differs')
        device=core.configure(args.device)
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Explicit CUDA unavailable')
        results=[]
        for cell in cells:
            arm,seed=cell.rsplit('_seed',1)
            results.append(core.fit_cell(arm,int(seed),a,device))
        io.dump(out/'completion.json',dict(status='COMPLETED_FIXED_BUDGET_TRAINING',cells=len(results),
            results=results,elapsed_seconds=time.monotonic()-started,
            optimizer_updates=96*len(results),model_forward_calls=1920*len(results),
            checkpoint_rule='Fixed epoch4; no score-based selection',
            reference_parity='Must be checked separately; completion is not a parity claim'))
        print('COMPLETED_FIXED_BUDGET_TRAINING',len(results),round(time.monotonic()-started,3),flush=True)
    except BaseException as e:
        io.dump(out/f'failure_{attempt}.json',dict(status='PAUSED_USER' if isinstance(e,core.UserPaused) else 'FAILED',
            exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        # Preserve terminal process identity rather than leaving a live-looking lock.
        if lock.exists() and read(lock)['pid']==p.pid:
            os.replace(lock,out/f'finished_process_{attempt}.json')

if __name__=='__main__':main()
