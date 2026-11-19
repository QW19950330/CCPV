"""Logged fixed A x five CUDA training; fresh output and no automatic retry."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import psutil
from datetime import datetime,timezone
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def utc():return datetime.now(timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--logs',type=Path,required=True)
    ap.add_argument('--timeout-seconds',type=int,default=1800);a=ap.parse_args()
    inputs=a.inputs.resolve();out=a.output.resolve();logs=a.logs.resolve();code=Path(__file__).resolve().parent
    if out.exists() or logs.exists():raise FileExistsError('Create-only output and logs required')
    if not out.parent.is_dir():raise ValueError('Output parent must exist')
    for root in (inputs,code):
        if out.is_relative_to(root) or logs.is_relative_to(root):raise ValueError('Separate output/logs from code and inputs')
    contract=json.loads((code/'training_contract.json').read_text(encoding='utf-8'))
    if contract['arms']!=['A_raw4d'] or contract['seeds']!=list(range(5)):raise ValueError('A five-seed fixed contract required')
    for n,pin in contract['data'].items():
        if sha(inputs/'data'/n)!=pin['sha256']:raise ValueError('Input changed: '+n)
    command=[sys.executable,'-B','-X','utf8',str(code/'train.py'),'--development-data',str(inputs/'data/development.npz'),
        '--development-truth',str(inputs/'data/position_truth_v2.npz'),'--output',str(out),'--device','cuda']
    logs.mkdir(parents=True,exist_ok=False);started=utc();tick=time.monotonic()
    with (logs/'stdout.txt').open('xb') as stdout,(logs/'stderr.txt').open('xb') as stderr:
        proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        put(logs/'start.json',dict(command=command,started_utc=started,pid=proc.pid,device='cuda',models=5,epochs=4))
        print('A5_CUDA_STARTED',proc.pid,flush=True);timed_out=False;timeout_survivors=[]
        while True:
            try:exit_code=proc.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                print('A5_CUDA_ALIVE',round(time.monotonic()-tick),'seconds',flush=True)
                if time.monotonic()-tick>a.timeout_seconds:
                    print('A5_HARD_TIMEOUT_STOPPING_OWN_PROCESS_TREE',flush=True);timed_out=True
                    try:
                        owner=psutil.Process(proc.pid);owned=owner.children(recursive=True)+[owner]
                    except psutil.NoSuchProcess:owned=[]
                    for child in reversed(owned):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    _,alive=psutil.wait_procs(owned,timeout=10)
                    timeout_survivors=[dict(pid=p.pid,created=p.create_time()) for p in alive]
                    exit_code=proc.wait(timeout=10);break
    receipt=dict(started_utc=started,completed_utc=utc(),exit_code=exit_code,pid=proc.pid,timed_out=timed_out,
        elapsed_seconds=time.monotonic()-tick,timeout_survivors=timeout_survivors,stdout_sha256=sha(logs/'stdout.txt'),stderr_sha256=sha(logs/'stderr.txt'),
        stderr_bytes=(logs/'stderr.txt').stat().st_size,status='COMPLETE_PENDING_STRICT_PARITY' if exit_code==0 else 'FAILED_PRESERVED')
    put(logs/'execution.json',receipt);print(json.dumps(receipt),flush=True)
    if exit_code:print((logs/'stderr.txt').read_text(encoding='utf-8',errors='replace'),flush=True)
    return exit_code
if __name__=='__main__':raise SystemExit(main())
