"""Compare genuinely retrained S4 outputs to the fixed original and manuscript."""
from pathlib import Path
import argparse,csv,hashlib,json,re,zipfile
from xml.etree import ElementTree as ET
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def compare_npz(a,b):
    bad=[];fields=0;err=0.
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        if set(x.files)!=set(y.files):bad.append('keys')
        for k in set(x.files)&set(y.files):
            u,v=x[k],y[k];fields+=1
            if u.dtype!=v.dtype or u.shape!=v.shape or u.tobytes()!=v.tobytes():
                bad.append(k)
                if u.shape==v.shape and u.dtype.kind in 'fiu' and v.dtype.kind in 'fiu':err=max(err,float(np.max(np.abs(u.astype(float)-v.astype(float)))))
    return dict(path=a.name,fields=fields,exact=not bad,differing_fields=bad,max_abs=err,archive_byte_exact=sha(a)==sha(b))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--manuscript',type=Path,required=True);p.add_argument('--docx',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Create-only report')
    out.mkdir(parents=True,exist_ok=False)
    cps=[compare_npz(f,a.reference/'fit01'/f.name) for f in sorted((a.run/'fit01').glob('*.npz'))]
    histories=[dict(path=f.name,byte_exact=sha(f)==sha(a.reference/'fit01'/f.name)) for f in sorted((a.run/'fit01').glob('*_history.json'))]
    pred=compare_npz(a.run/'readout01/readouts.npz',a.reference/'readout01/readouts.npz')
    csvs=[dict(path=f.parent.name+'/'+f.name,rows=len(rows(f)),byte_exact=sha(f)==sha(a.reference/f.parent.name/f.name))
        for folder in ('readout01','score01') for f in sorted((a.run/folder).glob('*.csv'))]
    summary=rows(a.run/'score01/summary.csv');paired=rows(a.run/'score01/paired_seed.csv');display=[]
    for cell in ('OldTrained_Mean','OldTrained_MC','MCTrained_Mean','MCTrained_MC'):
        r=next(r for r in summary if (r['cell'],r['target'],r['metric'])==(cell,'Y10','log_loss'))
        display.append([cell,f"{float(r['mean']):.12f} ± {float(r['seed_SD']):.12f}",'NA'])
    contrasts=[]
    for name in ('A_match','B_replace'):
        rr=[r for r in paired if r['contrast']==name and r['target']=='Y10']
        x=np.array([float(r['log_loss']) for r in rr]);br=np.array([float(r['brier']) for r in rr])
        display.append([name,f'{x.mean():.12f} ± {x.std(ddof=1):.12f}',f'{int((x<0).sum())}/5'])
        contrasts.append(dict(name=name,logloss_mean=float(x.mean()),logloss_seed_SD=float(x.std(ddof=1)),improving_seeds=int((x<0).sum()),brier_delta_mean=float(br.mean())))
    cv=rows(a.run/'score01/unchanged_historical_CV.csv')[0]
    display.append(['CV_Direct_WithEarlyBCE',f"{float(cv['logloss_mean']):.12f} ± {float(cv['logloss_seed_SD']):.12f}",'NA'])
    headers=['Training / readout or contrast','Y10 logloss mean / difference ± seed SD','Improving seeds for contrast']
    text=a.manuscript.read_text(encoding='utf-8-sig');match=re.search(r'^## Table S4\..*?(?=^## Table |\Z)',text,re.M|re.S)
    if not match:raise ValueError('Current S4 section missing')
    parsed=[[c.strip() for c in l.strip('|').split('|')] for l in match.group().splitlines() if l.startswith('|')]
    current=[parsed[0]]+parsed[2:];generated=[headers]+display;md_exact=current==generated
    with (out/'TableS4.csv').open('x',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(generated)
    docx=None
    if a.docx:
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        with zipfile.ZipFile(a.docx) as z:root=ET.fromstring(z.read('word/document.xml'))
        matches=[]
        for i,t in enumerate(root.findall('.//w:tbl',ns)):
            rr=[[''.join(x.text or '' for x in c.findall('.//w:t',ns)) for c in row.findall('w:tc',ns)] for row in t.findall('w:tr',ns)]
            if rr==generated:matches.append(i)
        docx=dict(exact=len(matches)==1,table_indices=matches,sha256=sha(a.docx))
    passed=(len(cps)==len(histories)==30 and all(r['exact'] for r in cps+[pred]) and all(r['byte_exact'] for r in histories+csvs)
        and md_exact and (docx is None or docx['exact']))
    result=dict(status='EXACT_FULL_TRAINING_READOUT_SCORING_AND_CURRENT_S4' if passed else 'DIFFERENCE',
        new_heads=30,new_optimizer_updates=6000,checkpoints=cps,histories=histories,readout=pred,scoring_csvs=csvs,
        current_markdown_exact=md_exact,manuscript_sha256=sha(a.manuscript),docx=docx,display_rows=display,
        retained_contrasts=contrasts,historical_CV='Unchanged frozen context, not an S4 newly fitted head',
        scope='Local private cached terminal features; no terminal-shard rebuild, no world forward, no raw-data public pipeline claim.')
    with (out/'comparison.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(result['status'],'heads',len(cps),'checkpoint_fields',sum(r['fields'] for r in cps),'max_error',max(r['max_abs'] for r in cps+[pred]))
    return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
