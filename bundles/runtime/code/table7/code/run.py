"""Run original fixed 240-head training, readout and scoring, create-only."""
import argparse, hashlib, json, os, subprocess, sys, time, platform
from pathlib import Path
from datetime import datetime, timezone

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def utc():return datetime.now(timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout-seconds',type=int,default=1800)
    a=ap.parse_args(); inputs=a.inputs.resolve(); out=a.output.resolve(); code=Path(__file__).resolve().parent
    if out.exists():raise FileExistsError('Create-only: output must not exist')
    if out.is_relative_to(inputs) or out.is_relative_to(code):raise ValueError('Output must be separate from code and inputs')
    manifest=load(inputs/'manifest.json')
    for r in manifest['files']:
        p=(inputs/r['path']).resolve()
        if not p.is_relative_to(inputs) or sha(p)!=r['sha256']:raise ValueError('Frozen input mismatch: '+r['path'])
    study=inputs/'snapshots'/manifest['source_study']
    for n in manifest['numerical_source_files']:
        if sha(code/n)!=sha(study/n):raise ValueError('Original numerical source changed: '+n)
    out.mkdir(parents=True,exist_ok=False)
    lock=dict(status='PORTABLE_FIXED240_ORIGINAL_NUMERICAL_SOURCE',utc=utc(),
        experiment_plan_sha256=sha(study/'experiment_plan.json'),data_binding_sha256=sha(inputs/'portable_binding.json'),
        input_manifest_sha256=sha(inputs/'manifest.json'),files=[dict(path=p.name,sha256=sha(p)) for p in sorted(code.glob('*.py'))],
        original_implementation_lock_sha256=sha(study/'implementation_lock.json'),
        models=240,updates_per_model=200,total_updates=48000,threads=1,device='cpu',dtype='float64')
    put(out/'implementation_lock.json',lock); expected=sha(out/'implementation_lock.json')
    put(out/'runtime_authorization.json',dict(status='AUTHORIZED_PORTABLE_REPRODUCTION',implementation_lock_sha256=expected,
        allowed_stages=['init_audit','fit','readout','score'],scope='Reproduce original fixed factorial only; no50/noWC2018/no external tuning'))
    logs=out/'execution_logs';logs.mkdir()
    env=dict(os.environ,CCPV_TABLE7_OUTPUT=str(out),CCPV_TABLE7_INPUT=str(inputs),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    started=time.monotonic()
    for stage in ('init_audit','fit','readout','score'):
        command=[sys.executable,'-B','-X','utf8',str(code/('score_factorial.py' if stage=='score' else 'runtime.py'))]
        if stage!='score':command.append(stage)
        command+=['--implementation-lock-sha256',expected]
        tick=time.monotonic();begin=utc()
        with (logs/(stage+'.stdout.txt')).open('xb') as stdout,(logs/(stage+'.stderr.txt')).open('xb') as stderr:
            proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
            put(logs/(stage+'.start.json'),dict(stage=stage,started_utc=begin,pid=proc.pid,args=command))
            print('START',stage,'PID',proc.pid,flush=True)
            timed_out=False
            while True:
                try:exit_code=proc.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    print('ALIVE',stage,'elapsed_seconds',round(time.monotonic()-tick),flush=True)
                    if time.monotonic()-started>a.timeout_seconds:
                        print('HARD_TIMEOUT',stage,flush=True);proc.kill();exit_code=proc.wait();timed_out=True;break
        receipt=dict(stage=stage,started_utc=begin,completed_utc=utc(),pid=proc.pid,exit_code=exit_code,timed_out=timed_out,
            elapsed_seconds=time.monotonic()-tick,stdout_sha256=sha(logs/(stage+'.stdout.txt')),stderr_sha256=sha(logs/(stage+'.stderr.txt')))
        put(logs/(stage+'.execution.json'),receipt)
        print('END',stage,'exit',exit_code,'seconds',round(receipt['elapsed_seconds'],2),flush=True)
        if exit_code:
            put(out/'failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',stage=stage,receipt=receipt))
            print((logs/(stage+'.stderr.txt')).read_text(encoding='utf-8',errors='replace'),flush=True);return exit_code
    put(out/'execution_complete.json',dict(status='ACTUAL_TRAINING_READOUT_SCORING_COMPLETE_PENDING_COMPARISON',models=240,
        updates=48000,elapsed_seconds=time.monotonic()-started,python=sys.version,platform=platform.platform()))
    return 0
if __name__=='__main__':sys.exit(main())
