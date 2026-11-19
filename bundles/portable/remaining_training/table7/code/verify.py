"""Strict full-array/CSV verification after genuine training; no tolerance."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def compare_npz(new,old):
    failures=[]; max_abs=0.; fields=0
    with np.load(new,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):failures.append('member_names')
        for k in sorted(set(a.files)&set(b.files)):
            x,y=a[k],b[k];fields+=1
            exact=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
            if not exact:
                failures.append(k)
                if x.shape==y.shape and x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu':
                    max_abs=max(max_abs,float(np.nanmax(np.abs(x.astype(float)-y.astype(float)))))
    return dict(path=new.name,fields=fields,all_fields_bitwise_exact=not failures,
        differing_fields=failures,maximum_absolute_numeric_difference=max_abs,
        file_bytes_exact=sha(new)==sha(old))
def write_json(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--report-dir',type=Path,required=True);ap.add_argument('--manuscript',type=Path,required=True)
    ap.add_argument('--qfirst-output',type=Path)
    args=ap.parse_args();inp=args.inputs.resolve();out=args.output.resolve();report=args.report_dir.resolve()
    if report.exists():raise FileExistsError('Comparison report is create-only')
    report.mkdir(parents=True,exist_ok=False)
    original=inp/'snapshots/CCPV_risk_objective_factorial_20261030'
    lock=load(out/'final_weights_lock.json'); assert len(lock['records'])==240
    checkpoints=[];histories=[]
    for r in lock['records']:
        checkpoints.append(compare_npz(out/r['path'],original/r['path']))
        histories.append(dict(path=r['history'],byte_exact=sha(out/r['history'])==sha(original/r['history'])))
    init=[compare_npz(p,original/'initialization'/p.name) for p in sorted((out/'initialization').glob('*.npz'))]
    pred=compare_npz(out/'frozen_readouts.npz',original/'frozen_readouts.npz')
    tables=[]
    for name in ('by_match.csv','by_seed.csv','summary.csv','paired_comparisons.csv','coverage.csv'):
        tables.append(dict(path='evaluation/'+name,rows=len(rows(out/'evaluation'/name)),
            bytes_exact=sha(out/'evaluation'/name)==sha(original/'evaluation'/name)))
    summary=rows(out/'evaluation/summary.csv')
    display=[]
    mapping=[('D_Direct_WithEarlyBCE','D DirectWith'),('CV_Direct_WithEarlyBCE','CV DirectWith'),
        ('D_Marginal_WithEarlyBCE','D MarginalWith'),('CV_Marginal_WithEarlyBCE','CV MarginalWith')]
    if args.qfirst_output:
        summary+=rows(args.qfirst_output/'evaluation/summary.csv')
        mapping += [('D_QFirstJoint','D q-first joint'),('CV_QFirstJoint','CV q-first joint')]
    manuscript=args.manuscript.read_text(encoding='utf-8-sig')
    for arm,label in mapping:
        r=next(r for r in summary if r['arm']==arm and r['target']=='q10')
        rendered={m:f"{float(r[m+'_mean']):.9f} ± {float(r[m+'_seed_SD']):.9f}" for m in ('logloss','Brier')}
        line=f"| {label} | {rendered['logloss']} | {rendered['Brier']} |"
        display.append(dict(arm=arm,label=label,Logloss=rendered['logloss'],Brier=rendered['Brier'],
            exact_current_manuscript_row=line in manuscript))
    with (report/('Table7.csv' if args.qfirst_output else 'Table7_factorial_four_rows.csv')).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['Frontend and head','Logloss','Brier']);w.writerows((r['label'],r['Logloss'],r['Brier']) for r in display)
    primary=load(out/'evaluation/results.json')['primary']
    passed=(len(init)==60 and all(r['all_fields_bitwise_exact'] for r in checkpoints+init+[pred])
        and all(r['byte_exact'] for r in histories) and all(r['bytes_exact'] for r in tables)
        and all(r['exact_current_manuscript_row'] for r in display))
    result=dict(status='PASS_STRICT_FULL_RETRAINING_COMPARISON' if passed else 'FAIL_STRICT_COMPARISON',
        actual_models_trained=240,optimizer_updates=48000,initialization_archives=init,
        checkpoints=checkpoints,histories=histories,readouts=pred,scoring_csv=tables,
        table7_display_rows=display,manuscript_sha256=sha(args.manuscript),
        primary_negative_result_preserved=primary,scope='Cached-prefix fixed risk-head training. Frontend prediction/training not rerun.',
        complete_table7_six_rows=bool(args.qfirst_output),untrained_table7_rows=[] if args.qfirst_output else ['D q-first joint','CV q-first joint'])
    write_json(report/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('status','actual_models_trained','optimizer_updates','complete_table7_six_rows')},ensure_ascii=False))
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
