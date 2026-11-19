"""Run unchanged frozen-prediction dependence arithmetic with explicit inputs."""
import argparse,json,time,traceback
from pathlib import Path
import original_analyze as core

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();here=Path(__file__).resolve().parent;inputs=args.inputs.resolve(strict=True);out=args.output.resolve()
    contract=core.read_json(here/'contract.json');plan=core.read_json(here/'plan.json')
    if core.sha(here/'original_analyze.py')!=contract['original_source_sha256'] or core.sha(here/'plan.json')!=contract['portable_plan_sha256']:raise ValueError('Source or fixed plan changed')
    for root in (here,inputs):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Output overlaps code/input tree')
    if not out.parent.is_dir():raise FileNotFoundError('Output parent must exist')
    paths={name:(inputs/name).resolve(strict=True) for name in plan['inputs']}
    for name,p in paths.items():
        r=plan['inputs'][name]
        if not p.is_relative_to(inputs) or p.stat().st_size!=r['bytes'] or core.sha(p)!=r['sha256']:raise ValueError('Pinned input differs: '+name)
    original_read=core.read_json
    allowed_json={here/'plan.json',paths['result_freeze.json'],paths['prediction_freeze.json']}
    def relocate(x):
        if isinstance(x,dict):
            return {k:(str(paths[Path(v).name]) if k=='path' and isinstance(v,str) and Path(v).name in paths else relocate(v)) for k,v in x.items()}
        if isinstance(x,list):return [relocate(v) for v in x]
        return x
    def read_json(p):
        p=Path(p).resolve()
        if p not in allowed_json:raise ValueError('Unlisted JSON read')
        return relocate(original_read(p))
    core.read_json=read_json;core.HERE=here;core.PLAN_SHA=contract['portable_plan_sha256']
    out.mkdir(exist_ok=False);started=time.monotonic()
    try:
        core.run(out)
        files=[]
        for name,pin in contract['references'].items():
            p=out/name;exact=p.stat().st_size==pin['bytes'] and core.sha(p)==pin['sha256']
            files.append(dict(name=name,byte_exact=exact,sha256=core.sha(p)))
        passed=all(r['byte_exact'] for r in files)
        core.put(out/'parity.json',dict(status='EXACT_ALL_SAVED_STATISTICS' if passed else 'DIFFERENCE',files=files,
            input_files=7,original_source_byte_exact=True,new_fits=0,new_predictions=0,
            scope='13 full CSVs, numerical summary and Chinese report; timing and relocated provenance intentionally differ',
            elapsed_seconds=time.monotonic()-started))
        if not passed:raise AssertionError('Saved dependence statistics differ')
        print('EXACT_ALL_SAVED_STATISTICS',len(files),flush=True)
    except BaseException:
        core.put(out/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
