"""From-scratch fixed Table4 cells, four epochs; no score-based selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback,uuid
from pathlib import Path
import psutil
import torch
import io_core as io
import train_core as core

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--cells',nargs='+',help='Fixed C_s0 ... F_s4 cells; default all20')
    ap.add_argument('--resume',action='store_true',help='Only this identical request and fully closed epochs')
    args=ap.parse_args();here=Path(__file__).resolve().parent
    contract=io.J(here/'contract.json');data=args.data.resolve(strict=True);out=args.output.resolve()
    if io.sha(data)!=contract['data']['sha256'] or data.stat().st_size!=contract['data']['bytes']:raise ValueError('Input pin differs')
    allcells=[f'{c}_s{s}' for s in range(5) for c in ('C','D','E','F')];cells=args.cells or allcells
    if len(set(cells))!=len(cells) or set(cells)-set(allcells):ap.error('Unsupported or duplicate fixed cell')
    for root in (here,data.parent):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Output overlaps input/code tree')
    if not out.parent.is_dir():raise FileNotFoundError('Output parent must already exist')
    if not torch.cuda.is_available():raise RuntimeError('Original training requires explicit CUDA environment')
    spec=dict(data_sha256=io.sha(data),cells=cells,device='cuda',contract_sha256=io.sha(here/'contract.json'),
        code_sha256={p.relative_to(here).as_posix():io.sha(p) for p in sorted(here.rglob('*.py'))})
    if args.resume:
        if not out.is_dir() or io.J(out/'run_spec.json')!=spec:raise ValueError('Resume request/code/input differs')
        if (out/'completion.json').exists() or (out/'weights_freeze.json').exists():raise FileExistsError('Training already completed')
    else:out.mkdir(exist_ok=False)
    io.P=core.P=out;core.DATA=data;core.CONTRACT=contract
    core.REQUESTED=[(int(c.rsplit('_s',1)[1]),c.split('_s')[0]) for c in cells]
    if io.pause_requested():raise core.UserPause('Explicit pause marker active')
    lock=out/'active_process.json'
    if lock.exists():
        old=io.J(lock)
        try:
            previous=psutil.Process(old['pid'])
            if previous.is_running() and abs(previous.create_time()-old['create_time'])<.05:raise RuntimeError('Matching trainer is still alive')
        except psutil.NoSuchProcess:pass
        lock.rename(out/('previous_process_'+uuid.uuid4().hex+'.json'))
    p=psutil.Process();identity=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())
    io.put(lock,identity)
    if not args.resume:
        io.put(out/'run_spec.json',spec)
        io.put(out/'runtime_authorization.json',dict(allow_closed_epoch_resume=True,scope='Same request, code and input; no unclosed-file replacement'))
    attempt=uuid.uuid4().hex;started=time.monotonic()
    io.put(out/f'attempt_{attempt}.json',dict(utc=io.now(),identity=identity,resume=args.resume))
    try:
        core.fit(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())
        fr=io.J(out/'weights_freeze.json')
        io.put(out/'completion.json',dict(status='COMPLETED_FRESH_FIXED4_TRAINING',cells=cells,
            primary_updates=fr['primary_updates'],primary_forwards=fr['primary_transition_forwards'],
            diagnostic_forwards=fr['diagnostic_transition_forwards'],elapsed_seconds=time.monotonic()-started,
            original_weights_used_for_initialization=False,validation_predictions_read=False,
            parity='Requires separate comparison of all epochs, gradients and histories'))
        print('COMPLETED',len(cells),round(time.monotonic()-started,3),flush=True)
    except BaseException as e:
        io.put(out/f'failure_{attempt}.json',dict(status='PAUSED_USER' if isinstance(e,core.UserPause) else 'FAILED',
            exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if lock.exists() and io.J(lock)['pid']==p.pid:lock.rename(out/f'finished_process_{attempt}.json')

if __name__=='__main__':main()
