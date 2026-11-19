"""Replay retained raw preprocessing into a new, disjoint directory."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from common import dump,sha,require_separate_new_output

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['metrica','skillcorner','truth'])
    ap.add_argument('--raw',required=True,type=Path)
    ap.add_argument('--development',type=Path,help='Only for separate raw truth recovery')
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    raw=args.raw.resolve(strict=True)
    if not raw.is_dir():raise NotADirectoryError(raw)
    if args.stage=='truth' and args.development is None:ap.error('truth requires --development')
    inputs=[raw]+([args.development.resolve(strict=True)] if args.development else [])
    output=require_separate_new_output(args.output,inputs)
    contract=json.loads(Path(__file__).with_name('input_contract.json').read_text(encoding='utf-8'))
    # Pin check occurs inside readers before successful output is committed.
    # Keep any partial outputs and failure receipt; never reuse them as success.
    output.mkdir()
    start=time.monotonic()
    dump(output/'execution_start.json',dict(stage=args.stage,pid=os.getpid(),raw=str(raw),
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        contract_sha256=sha(Path(__file__).with_name('input_contract.json'))))
    try:
        if args.stage=='metrica':
            from metrica_exact import prepare
            prepare(raw,output)
        elif args.stage=='skillcorner':
            from skillcorner import prepare
            prepare(raw,contract,output)
            from skillcorner_truth import recover
            recover(raw,output/'development.npz',contract,output)
        else:
            from skillcorner_truth import recover
            recover(raw,args.development.resolve(),contract,output)
        dump(output/'execution_complete.json',dict(status='COMPLETED',stage=args.stage,
            elapsed_seconds=time.monotonic()-start,optimizer_updates=0,model_forward_calls=0,
            output_npz_sha256={p.name:sha(p) for p in output.glob('*.npz')}))
        print(json.dumps(dict(status='COMPLETED',stage=args.stage,elapsed_seconds=time.monotonic()-start)),flush=True)
    except BaseException as e:
        dump(output/'execution_failure.json',dict(status='FAILED',exception=type(e).__name__,
            message=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start))
        raise

if __name__=='__main__':main()
