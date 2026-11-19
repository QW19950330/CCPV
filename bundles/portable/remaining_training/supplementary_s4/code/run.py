"""Portable create-only wrapper around the byte-identical original S4 stages."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def utc():return datetime.now(timezone.utc).isoformat()
def worker(a):
    import run_study as original
    original.ROOT=a.output.resolve()
    original_read=original.read
    def read_portable(path):
        data=original_read(path)
        if Path(path).resolve()==original.ROOT/'plan.json':
            def resolve(x):
                if isinstance(x,dict):
                    d={k:resolve(v) for k,v in x.items()}
                    if 'path' in d and isinstance(d['path'],str):
                        f=(a.input_root/d['path']).resolve()
                        if not f.is_relative_to(a.input_root):raise ValueError('Bound input path escape')
                        d['path']=str(f)
                    return d
                if isinstance(x,list):return [resolve(v) for v in x]
                return x
            return resolve(data)
        return data
    original.read=read_portable
    name={'fit':'fit01','readout':'readout01','score':'score01'}[a.stage]
    {'fit':original.fit_stage,'readout':original.readout_stage,'score':original.score_stage}[a.stage](original.ROOT/'authorization.json',original.ROOT/name)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage',choices=['fit','readout','score'],help=argparse.SUPPRESS)
    a=ap.parse_args();a.input_root=a.input_root.resolve();a.metadata=a.metadata.resolve();a.output=a.output.resolve()
    if a.stage:worker(a);return 0
    code=Path(__file__).resolve().parent
    if a.output.exists():raise FileExistsError('Create-only output must not exist')
    if a.output.is_relative_to(code) or a.output==a.input_root:raise ValueError('Output overlaps protected source')
    manifest=read(a.metadata/'input_manifest.json');plan=a.metadata/'portable_plan.json'
    if sha(plan)!=manifest['portable_plan_sha256']:raise ValueError('Portable plan changed')
    for r in manifest['numerical_source']:
        if sha(code/r['path'])!=r['sha256']:raise ValueError('Original numerical source changed')
    for r in manifest['files']:
        path=(a.input_root/r['path']).resolve()
        if not path.is_relative_to(a.input_root) or sha(path)!=r['sha256']:raise ValueError('Frozen input changed: '+r['path'])
    a.output.mkdir(parents=True,exist_ok=False)
    with plan.open('rb') as f,(a.output/'plan.json').open('xb') as g:g.write(f.read())
    put(a.output/'authorization.json',dict(status='AUTHORIZED_MATCHED_VALUE_OBJECTIVE',allowed_stages=['fit','readout','score'],
        plan_sha256=sha(a.output/'plan.json'),scope='Fixed original 30 heads only; no world forwards/no50/no external tuning'))
    put(a.output/'execution_spec.json',dict(input_root=str(a.input_root),metadata=str(a.metadata),python=sys.executable,
        code={p.name:sha(p) for p in code.glob('*.py')},input_manifest_sha256=sha(a.metadata/'input_manifest.json'),started_utc=utc()))
    logs=a.output/'logs';logs.mkdir();start=time.monotonic()
    for stage in ('fit','readout','score'):
        cmd=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--input-root',str(a.input_root),'--metadata',str(a.metadata),'--output',str(a.output),'--stage',stage]
        tick=time.monotonic();begin=utc()
        with (logs/(stage+'.stdout.txt')).open('xb') as out,(logs/(stage+'.stderr.txt')).open('xb') as err:
            p=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            put(logs/(stage+'.start.json'),dict(pid=p.pid,started_utc=begin,command=cmd));print('START',stage,p.pid,flush=True)
            timeout=False
            while True:
                try:rc=p.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    print('ALIVE',stage,round(time.monotonic()-tick),'seconds',flush=True)
                    if time.monotonic()-start>1200:
                        print('HARD_TIMEOUT',stage,flush=True);p.kill();rc=p.wait();timeout=True;break
        receipt=dict(stage=stage,pid=p.pid,exit_code=rc,timeout=timeout,started_utc=begin,completed_utc=utc(),
            elapsed_seconds=time.monotonic()-tick,stdout_sha256=sha(logs/(stage+'.stdout.txt')),stderr_sha256=sha(logs/(stage+'.stderr.txt')))
        put(logs/(stage+'.execution.json'),receipt);print('END',stage,rc,round(receipt['elapsed_seconds'],2),flush=True)
        if rc:
            put(a.output/'failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',receipt=receipt))
            print((logs/(stage+'.stderr.txt')).read_text(encoding='utf-8'),flush=True);return rc
    import numpy,torch
    put(a.output/'execution_complete.json',dict(status='ACTUAL_FIXED30_TRAINING_READOUT_SCORING_COMPLETE_PENDING_COMPARISON',
        models=30,updates=6000,elapsed_seconds=time.monotonic()-start,numpy=numpy.__version__,torch=torch.__version__,python=sys.version,
        new_trajectory_forwards=0,source_shards_rebuilt=0))
    return 0
if __name__=='__main__':raise SystemExit(main())
