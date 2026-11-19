"""Explicit input/output replay for the fixed Table 5 engineering screen."""
from pathlib import Path
import argparse, copy, importlib, platform, sys, time
import feasibility_io as io

CODE=Path(__file__).resolve().parent

def records(x):
    if isinstance(x,dict):
        if 'path' in x and 'sha256' in x: yield x
        for v in x.values(): yield from records(v)
    elif isinstance(x,list):
        for v in x: yield from records(v)

def resolve_binding(data_root):
    root=Path(data_root).resolve(strict=True)
    binding=copy.deepcopy(io.J(CODE/'input_binding.json'))
    for r in records(binding):
        rel=Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Relative contained inputs required')
        p=(root/rel).resolve(strict=True)
        if not p.is_relative_to(root): raise ValueError('Input escapes explicit data root')
        if not p.is_file(): raise ValueError('Input must be a file')
        r['path']=str(p)
    return binding

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','run'])
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError('Output must not exist; old outputs are never overwritten')
    if out==CODE or CODE.is_relative_to(out) or out.is_relative_to(CODE): raise ValueError('Output must be separate from the code tree')
    code_manifest=io.J(CODE/'code_manifest.json')
    for r in code_manifest['files']:
        if io.sha(CODE/r['path'])!=r['sha256']: raise ValueError('Package file changed: '+r['path'])
    binding=resolve_binding(args.data_root)
    root=Path(args.data_root).resolve(strict=True)
    protected={root/Path(r['path']).relative_to(root).parts[0] for r in records(binding)}
    if out==root or root.is_relative_to(out) or any(out==p or out.is_relative_to(p) for p in protected):
        raise ValueError('Output must be separate from all bound input source trees')
    start=time.perf_counter()
    checked=[]
    for r in records(binding):
        digest=io.sha(r['path'])
        if digest!=r['sha256']: raise ValueError('Bound input changed: '+r['path'])
        checked.append(dict(path=r['path'],sha256=digest,bytes=Path(r['path']).stat().st_size))
    if len(binding['cells'])!=16: raise ValueError('All 16 cells are mandatory')
    out.mkdir(parents=True,exist_ok=False)
    io.put(out/'preflight.json',dict(status='PASS_FIXED_INPUT_HASHES',utc=io.now(),files=checked,
        package_manifest_sha256=io.sha(CODE/'code_manifest.json'),seconds=time.perf_counter()-start,
        stage=args.stage,python=sys.version,numpy=io.np.__version__,platform=platform.platform(),
        new_fits=0,new_world_forwards=0))
    if args.stage=='preflight':
        print('PREFLIGHT_PASS files='+str(len(checked)),flush=True); return
    # Copy runtime records only into a fresh output; no original lock is reused.
    with (out/'experiment_plan.json').open('xb') as f: f.write((CODE/'experiment_plan.json').read_bytes())
    io.put(out/'data_binding.json',binding)
    io.put(out/'execution_contract.json',dict(scope='FULL_FIXED_SAVED_OUTPUT_SCREEN_NO_TRAINING',
        code_files=code_manifest['files'],plan_sha256=io.sha(out/'experiment_plan.json'),
        binding_sha256=io.sha(out/'data_binding.json'),source_provenance=io.J(CODE/'provenance.json'),
        legacy_receipt_field_note='implementation_lock_sha256 records this NEW portable execution contract, not the original root lock'))
    contract=io.sha(out/'execution_contract.json')
    io.configure(out,contract,CODE)
    runner=importlib.import_module('run_feasibility')
    runner.main(contract)
    io.put(out/'portable_completion.json',dict(status='COMPLETE_FULL_FIXED_SAVED_OUTPUT_REPLAY',utc=io.now(),
        seconds=time.perf_counter()-start,execution_contract_sha256=contract,
        replay_receipt_sha256=io.sha(out/'run_receipt.json'),
        original_math_and_runner_byte_exact=True,changed_execution_and_path_binding_only=True,
        reference_parity='Not checked by producer; use independent verify.py',
        scope='Saved output metrics only; not model training or trajectory generation; not physiological or tactical validation'))

if __name__=='__main__': main()
