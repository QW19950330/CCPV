"""Post-hoc frozen-CSV dependence diagnostics. No model, array or raw-data loading."""
from pathlib import Path
import os
for _v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_v]='1'
import argparse, collections, csv, datetime, hashlib, json, math, statistics, time
import psutil
HERE=Path(__file__).resolve().parent
PLAN_SHA='0322755db87e45f059459808ef77fe43da09b73b336df67ad27d547c6ba3688e'
METRICS=['logloss','brier','mean_probability','observed_rate','signed_bias','absolute_bias']
ERRORS={'logloss','brier','absolute_bias','conditional_logloss','conditional_brier'}
CELLS=[('G',s) for s in range(5)]+[('GM',s) for s in range(5)]+[('PREVALENCE',-1)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rows,fields=None):
    fields=fields or list(rows[0])
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def truth(x):
    assert x in ['True','False'],x
    return x=='True'
def avg(x):return math.fsum(x)/len(x) if x and all(v is not None for v in x) else None
def diff(a,b):return a-b if a is not None and b is not None else None
def measures(ids,arm,seed,pred,ledger):
    if not ids:return {m:None for m in METRICS}
    p=[pred[g,arm,seed] for g in ids];y=[ledger[g]['y'] for g in ids]
    q=[min(1-1e-15,max(1e-15,v)) for v in p]
    pm=avg(p);rate=avg(y);bias=pm-rate
    return dict(logloss=avg([-yy*math.log(v)-(1-yy)*math.log1p(-v) for yy,v in zip(y,q)]),
                brier=avg([(v-yy)**2 for v,yy in zip(p,y)]),mean_probability=pm,
                observed_rate=rate,signed_bias=bias,absolute_bias=abs(bias))
def summarize(values):
    full=bool(values) and all(v is not None for v in values)
    return dict(mean=avg(values),seed_SD=statistics.stdev(values) if full and len(values)>1 else None,
                defined_values=sum(v is not None for v in values),negative_values=sum(v is not None and v<0 for v in values))
def run(out):
    start=time.monotonic();planpath=HERE/'plan.json';assert sha(planpath)==PLAN_SHA
    plan=read_json(planpath);proc=psutil.Process()
    put(out/'identity.json',dict(pid=os.getpid(),creation_time=proc.create_time(),executable=proc.exe(),cmdline=proc.cmdline(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=pin(planpath),source=pin(__file__),threads=1,GPU=False))
    def guard():
        assert time.monotonic()-start<plan['runtime']['hard_wall_seconds'],'time contract'
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['runtime']['max_output_bytes'],'output contract'
        assert not (HERE/'PAUSE_REQUESTED').exists(),'pause sentinel'
    # Access is confined to the seven explicit plan inputs. Referenced heads/NPZ are never opened.
    inputs={}
    for name,record in plan['inputs'].items():
        current=pin(record['path']);assert current==record,(name,current);inputs[name]=current;guard()
    allowed=set(inputs)
    assert allowed=={'predictions.csv','event_links.csv','all_anchor_ledger.csv','fixture_metrics.csv','seed_summary.csv','result_freeze.json','prediction_freeze.json'}
    original_freeze=read_json(inputs['result_freeze.json']['path'])
    frozen={Path(r['path']).name:r for r in original_freeze['files']}
    for name in allowed-{'result_freeze.json'}:
        assert frozen[name]==inputs[name],name
    pf=read_json(inputs['prediction_freeze.json']['path']);assert pf['predictions']==inputs['predictions.csv']
    ledger_rows=csvrows(inputs['all_anchor_ledger.csv']['path']);assert len(ledger_rows)==11542
    ledger={};groups=collections.defaultdict(list)
    for ordinal,row in enumerate(ledger_rows):
        r=dict(row);r.update(grid_index=int(row['grid_index']),N=int(row['N']),t0_UTC_ns=int(row['t0_UTC_ns']),y=int(row['y']),eligible=truth(row['eligible']),ordinal=ordinal)
        assert r['grid_index'] not in ledger;assert r['y'] in [-1,0,1]
        assert r['eligible']==(truth(r['player_support']) and r['y']>=0)
        ledger[r['grid_index']]=r;groups[r['match_id'],r['period']].append(r)
    matches=sorted({r['match_id'] for r in ledger.values()});assert len(matches)==6 and len(groups)==12
    assert len({(r['match_id'],r['period'],r['N']) for r in ledger.values()})==len(ledger)
    groupchecks=[];phase_assign=[]
    for (mid,period),rr in sorted(groups.items()):
        rr.sort(key=lambda r:r['t0_UTC_ns']);origin=rr[0]['t0_UTC_ns'];n0=min(r['N'] for r in rr);assert origin>=0
        assert all(b['t0_UTC_ns']-a['t0_UTC_ns']==3000000000 and b['N']-a['N']==75 for a,b in zip(rr,rr[1:]))
        for r in rr:
            dt=r['t0_UTC_ns']-origin;dn=r['N']-n0;assert dt>=0 and dt%3000000000==0 and dn%75==0
            k=dt//3000000000;assert k==dn//75;r['phase']=k%4;r['k']=k
            phase_assign.append(dict(grid_index=r['grid_index'],match_id=mid,period=period,N=r['N'],t0_UTC_ns=r['t0_UTC_ns'],origin_ns=origin,k=k,phase=k%4,eligible=r['eligible'],y=r['y'],reason=r['reason']))
        minimum_intervals=[]
        for phase in range(4):
            tt=[r['t0_UTC_ns'] for r in rr if r['eligible'] and r['phase']==phase]
            gaps=[b-a for a,b in zip(tt,tt[1:])];assert all(v>=12000000000 for v in gaps)
            minimum_intervals.append(min(gaps) if gaps else None)
        groupchecks.append(dict(match_id=mid,period=period,candidates=len(rr),origin_ns=origin,N_origin=n0,exact_3second_grid=True,minimum_phase_eligible_intervals_ns=minimum_intervals))
    eligible=[g for g,r in ledger.items() if r['eligible']];positives=[g for g in eligible if ledger[g]['y']==1]
    assert len(eligible)==9483 and len(positives)==363
    predicted=csvrows(inputs['predictions.csv']['path']);assert len(predicted)==104313
    pred={};clip_count=0;minp=1.;maxp=0.
    for r in predicted:
        g=int(r['grid_index']);s=int(r['seed']);a=r['arm'];l=ledger[g];p=float(r['p'])
        assert l['eligible'] and r['held']==l['match_id'] and int(r['row'])==l['ordinal'] and int(r['y'])==l['y']
        assert (a,s) in CELLS and (g,a,s) not in pred and math.isfinite(p) and 0<p<1
        pred[g,a,s]=p;clip_count+=int(p<1e-15 or p>1-1e-15);minp=min(minp,p);maxp=max(maxp,p)
    assert all((g,a,s) in pred for g in eligible for a,s in CELLS)
    assert len(pred)==len(eligible)*11
    # If clipping actually matters, stop rather than alter event-loss interpretation.
    assert clip_count==0,'Frozen probabilities trigger log stabilization: report before continuing under unchanged event plan'
    links=csvrows(inputs['event_links.csv']['path']);seen=set();events=collections.defaultdict(list);bywindow=collections.defaultdict(set)
    for r in links:
        g=int(r['grid_index']);mid=r['match_id'];eid=r['native_event_id'];l=ledger[g]
        assert eid and mid==l['match_id'] and truth(r['eligible'])==l['eligible']
        key=(g,mid,eid);assert key not in seen;seen.add(key)
        if l['eligible']:assert l['y']==1
        events[mid,eid].append(g);bywindow[g].add((mid,eid))
    assert all(bool(bywindow[g])==(ledger[g]['y']==1) for g in eligible)
    event_good={e:[g for g in gs if ledger[g]['eligible']] for e,gs in events.items()}
    covered=[e for e,gs in event_good.items() if gs];assert len(covered)==113
    event_support=[]
    for (mid,eid),gs in sorted(events.items()):
        good=event_good[mid,eid];halves={ledger[g]['period'] for g in gs};assert len(halves)==1
        phase_counts=collections.Counter(ledger[g]['phase'] for g in good);assert all(n<=1 for n in phase_counts.values())
        tt=[ledger[g]['t0_UTC_ns'] for g in good];reasons=collections.Counter(ledger[g]['reason'] for g in gs if not ledger[g]['eligible'])
        event_support.append(dict(match_id=mid,native_event_id=eid,period=next(iter(halves)),candidate_windows=len(gs),eligible_windows=len(good),excluded_windows=len(gs)-len(good),
            eligible_phase_count=len(phase_counts),eligible_phases='|'.join(map(str,sorted(phase_counts))),eligible_anchor_start_ns=min(tt) if tt else None,
            eligible_anchor_end_ns=max(tt) if tt else None,eligible_anchor_span_seconds=(max(tt)-min(tt))/1e9 if tt else None,
            candidate_grid_indices='|'.join(map(str,sorted(gs))),eligible_grid_indices='|'.join(map(str,sorted(good))),excluded_reasons=json.dumps(dict(sorted(reasons.items())),ensure_ascii=False)))
    multi=[dict(grid_index=g,match_id=ledger[g]['match_id'],period=ledger[g]['period'],phase=ledger[g]['phase'],eligible=ledger[g]['eligible'],y=ledger[g]['y'],
                linked_event_count=len(es),native_event_ids='|'.join(eid for _,eid in sorted(es))) for g,es in sorted(bywindow.items()) if len(es)>1]
    full={};original_metrics=csvrows(inputs['fixture_metrics.csv']['path']);metric_checks=0;max_difference=0.
    for mid in matches:
        ids=[g for g in eligible if ledger[g]['match_id']==mid]
        for a,s in CELLS:full[mid,a,s]=measures(ids,a,s,pred,ledger)
    assert len(original_metrics)==66
    for r in original_metrics:
        key=r['held'],r['arm'],int(r['seed']);assert int(r['n'])==sum(ledger[g]['match_id']==r['held'] for g in eligible)
        for m in METRICS:
            error=abs(full[key][m]-float(r[m]));max_difference=max(max_difference,error);assert error<=1e-12,(key,m,error);metric_checks+=1
    for r in csvrows(inputs['seed_summary.csv']['path']):
        a=r['arm'];s=-1 if a=='PREVALENCE' else int(r['seed']);m=r['metric'];expected=avg([full[mid,a,s][m] for mid in matches])
        err=abs(expected-float(r['equal_fixture_mean']));max_difference=max(max_difference,err);assert err<=1e-12;metric_checks+=1
    put(out/'input_validation.json',dict(status='PASS',inputs=inputs,plan_sha256=PLAN_SHA,source_sha256=sha(__file__),
        ledger_rows=len(ledger),eligible_anchors=len(eligible),positive_anchors=len(positives),prediction_rows=len(pred),
        all_candidate_event_keys=len(events),covered_event_keys=len(covered),links=len(links),log_clip_triggered_prediction_rows=clip_count,
        min_frozen_probability=minp,max_frozen_probability=maxp,original_score_comparisons=metric_checks,max_original_score_difference=max_difference,
        groups=groupchecks,model_or_NPZ_or_raw_data_loaded=False,originals_modified=False))
    table(out/'phase_assignment.csv',phase_assign)
    coverage=[]
    for (mid,period),rr in sorted(groups.items()):
        for phase in range(4):
            cc=[r for r in rr if r['phase']==phase];ee=[r for r in cc if r['eligible']]
            reasons=collections.Counter(r['reason'] for r in cc if not r['eligible'])
            known=set().union(*(bywindow[r['grid_index']] for r in ee)) if ee else set()
            coverage.append(dict(match_id=mid,period=period,phase=phase,candidate_anchors=len(cc),eligible_anchors=len(ee),positive_anchors=sum(r['y']==1 for r in ee),covered_event_keys=len(known),
                                 excluded_anchors=len(cc)-len(ee),excluded_reasons=json.dumps(dict(sorted(reasons.items())),ensure_ascii=False)))
    table(out/'phase_coverage.csv',coverage)
    phase_fixture=[];phase_macro=[];phase_pairs=[];phase_summary=[];phase_values={}
    for phase in range(4):
        for mid in matches:
            ids=[g for g in eligible if ledger[g]['match_id']==mid and ledger[g]['phase']==phase]
            known=set().union(*(bywindow[g] for g in ids)) if ids else set()
            for a,s in CELLS:
                mm=measures(ids,a,s,pred,ledger);phase_values[phase,mid,a,s]=mm
                phase_fixture.append(dict(phase=phase,held=mid,arm=a,seed=s,n=len(ids),positives=sum(ledger[g]['y'] for g in ids),covered_events=len(known),**mm))
        for m in METRICS:
            values={}
            for a,s in CELLS:
                values[a,s]=avg([phase_values[phase,mid,a,s][m] for mid in matches])
                phase_macro.append(dict(phase=phase,arm=a,seed=s,metric=m,defined_fixtures=sum(phase_values[phase,mid,a,s][m] is not None for mid in matches),value=values[a,s]))
            for s in range(5):phase_pairs.append(dict(phase=phase,seed=s,contrast='GM-G',metric=m,delta=diff(values['GM',s],values['G',s])))
            for a in ['G','GM','PREVALENCE','GM-G']:
                vv=[values[a,s] for s in range(5)] if a in ['G','GM'] else ([values[a,-1]] if a=='PREVALENCE' else [diff(values['GM',s],values['G',s]) for s in range(5)])
                phase_summary.append(dict(phase=phase,arm_or_contrast=a,metric=m,lower_is_better=m in ERRORS,**summarize(vv)))
        guard()
    table(out/'phase_fixture_metrics.csv',phase_fixture);table(out/'phase_macro_metrics.csv',phase_macro)
    table(out/'phase_paired_seed.csv',phase_pairs);table(out/'phase_summary.csv',phase_summary)
    table(out/'event_support.csv',event_support)
    table(out/'multi_event_windows.csv',multi,['grid_index','match_id','period','phase','eligible','y','linked_event_count','native_event_ids'])
    event_metrics=[];ev={};event_fixture=[];ef={};event_summary=[]
    for (mid,eid),ids in sorted(event_good.items()):
        for a,s in CELLS:
            pp=[pred[g,a,s] for g in ids]
            mm=dict(conditional_logloss=avg([-math.log(p) for p in pp]),conditional_brier=avg([(1-p)**2 for p in pp]))
            ev[mid,eid,a,s]=mm;event_metrics.append(dict(match_id=mid,native_event_id=eid,arm_or_contrast=a,seed=s,eligible_windows=len(ids),**mm))
        for s in range(5):
            mm={m:diff(ev[mid,eid,'GM',s][m],ev[mid,eid,'G',s][m]) for m in ['conditional_logloss','conditional_brier']}
            event_metrics.append(dict(match_id=mid,native_event_id=eid,arm_or_contrast='GM-G',seed=s,eligible_windows=len(ids),**mm))
    for mid in matches:
        keys=[e for e in covered if e[0]==mid]
        for a,s in CELLS:
            mm={m:avg([ev[mid,eid,a,s][m] for _,eid in keys]) for m in ['conditional_logloss','conditional_brier']}
            ef[mid,a,s]=mm;event_fixture.append(dict(held=mid,arm_or_contrast=a,seed=s,covered_events=len(keys),**mm))
        for s in range(5):
            mm={m:diff(ef[mid,'GM',s][m],ef[mid,'G',s][m]) for m in ['conditional_logloss','conditional_brier']}
            event_fixture.append(dict(held=mid,arm_or_contrast='GM-G',seed=s,covered_events=len(keys),**mm))
    for m in ['conditional_logloss','conditional_brier']:
        macro={(a,s):avg([ef[mid,a,s][m] for mid in matches]) for a,s in CELLS}
        for a in ['G','GM','PREVALENCE','GM-G']:
            vv=[macro[a,s] for s in range(5)] if a in ['G','GM'] else ([macro[a,-1]] if a=='PREVALENCE' else [diff(macro['GM',s],macro['G',s]) for s in range(5)])
            event_summary.append(dict(arm_or_contrast=a,metric=m,lower_is_better=True,**summarize(vv)))
    table(out/'event_metrics.csv',event_metrics);table(out/'event_fixture_metrics.csv',event_fixture);table(out/'event_summary.csv',event_summary)
    deletions=[];deletion_summary=[];dv={}
    for omitted in matches:
        retained=[m for m in matches if m!=omitted]
        for metric in METRICS:
            for a,s in CELLS:
                value=avg([full[m,a,s][metric] for m in retained]);original=avg([full[m,a,s][metric] for m in matches]);dv[omitted,metric,a,s]=value
                deletions.append(dict(omitted_fixture=omitted,arm_or_contrast=a,seed=s,metric=metric,retained_fixture_count=5,value=value,original_six_fixture_value=original,shift=value-original))
            for s in range(5):
                value=dv[omitted,metric,'GM',s]-dv[omitted,metric,'G',s];dv[omitted,metric,'GM-G',s]=value
                original=avg([full[m,'GM',s][metric]-full[m,'G',s][metric] for m in matches])
                deletions.append(dict(omitted_fixture=omitted,arm_or_contrast='GM-G',seed=s,metric=metric,retained_fixture_count=5,value=value,original_six_fixture_value=original,shift=value-original))
    for metric in METRICS:
        for a in ['G','GM','PREVALENCE','GM-G']:
            ss=[-1] if a=='PREVALENCE' else list(range(5))
            means=[avg([dv[omitted,metric,a,s] for s in ss]) for omitted in matches]
            deletion_summary.append(dict(arm_or_contrast=a,metric=metric,lower_is_better=metric in ERRORS,
                min_deletion_mean=min(means),max_deletion_mean=max(means),negative_deletion_mean_count=sum(v<0 for v in means),
                min_all_deletion_seed_values=min(dv[o,metric,a,s] for o in matches for s in ss),max_all_deletion_seed_values=max(dv[o,metric,a,s] for o in matches for s in ss)))
    table(out/'deletion_metrics.csv',deletions);table(out/'deletion_summary.csv',deletion_summary)
    support_hist=collections.Counter(len(event_good[e]) for e in covered)
    old_neg=all(full[mid,'GM',s]['logloss']-full[mid,'G',s]['logloss']<0 for mid in matches for s in range(5))
    summary=dict(status='COMPLETE_POSTHOC_DIAGNOSTIC',fixture_count=6,phases=[0,1,2,3],seeds=list(range(5)),candidate_anchors=len(ledger),eligible_anchors=len(eligible),positive_anchors=len(positives),
        all_candidate_event_keys=len(events),covered_event_keys=len(covered),candidate_events_without_eligible_support=len(events)-len(covered),eligible_windows_per_event_histogram=dict(sorted(support_hist.items())),
        multi_event_candidate_windows=len(multi),multi_event_eligible_windows=sum(r['eligible'] for r in multi),log_clip_trigger_count=clip_count,
        phase_GM_minus_G=[r for r in phase_summary if r['arm_or_contrast']=='GM-G'],event_GM_minus_G=[r for r in event_summary if r['arm_or_contrast']=='GM-G'],
        deletion_GM_minus_G=[r for r in deletion_summary if r['arm_or_contrast']=='GM-G'],original_all_fixture_seed_logloss_differences_negative=old_neg,
        scientific_limitations=['posthoc, outcomes previously exposed','same original heads; training overlap/dependence persists','all-negative fixture differences algebraically imply negative deletion averages',
        'conditional positive event losses are neither full binary scoring nor calibration','events may share windows; event IDs are not independent replicates','phase subsets not independent datasets; no p-values/CI/bootstrap',
        'anchor support span is not time-to-shot','no independent data qualification or formal50 authorization'])
    put(out/'summary.json',summary)
    f=lambda x:'NA' if x is None else f'{x:.10f}'
    report=['# 冻结预测的窗口与事件支持敏感性（事后诊断）','',
        '本分析在原结果已知后冻结方案，只重算已保存预测，没有训练、重新预测、校准、改标签、改资格或选择有利相位。六场都是已曝光开发比赛，原训练依赖没有消失。',
        f'保留11542候选、9483合格窗、363阳性窗；{len(covered)}个有合格支持的记录事件键，另有{len(events)-len(covered)}个候选链接事件无合格预测支持。事件键不是独立重复。',
        'PREVALENCE每场只保留一份确定预测（seed=-1）；配对展示不代表五个基线模型。数值稳定clip仅用于二分类log计算，本次触发0次，原概率/Brier/偏差未改变。','',
        '## 四相位：全部保留','',
        '每场每半场先在完整3秒候选格赋k mod4，再沿用资格。相位内合格锚至少12秒，(t0,t0+10s]不重叠；不同相位仍共享事件、比赛及训练头，不能当四个独立数据集。各相位先比赛内评分，再六比赛等权。','',
        '|相位|GM−G指标|五种子均值|配对种子SD|负差种子|','|---|---|---:|---:|---:|']
    for r in phase_summary:
        if r['arm_or_contrast']=='GM-G' and r['metric'] in ['logloss','brier','absolute_bias']:
            report.append(f"|{r['phase']}|{r['metric']}|{f(r['mean'])}|{f(r['seed_SD'])}|{r['negative_values']}/5|")
    report+=['','所有相位×比赛×半场候选、合格、阳性及事件支持见phase_coverage.csv；全部G/GM/prior及概率偏差指标保存在CSV，不以平均相位替代原总体分母。','',
        '## 事件支持等权：仅阳性条件损失','',
        '先在每个记录事件的合格关联窗上平均−log(p)和(1−p)²，再事件在比赛内等权、六比赛等权。p仍是窗口内任一记录射门的概率，并不是该特定事件的概率。这不是完整二分类proper评分，也不是校准证据。一个窗口关联多个事件时分别进入各事件均值，明确保留此复用。',
        f'有支持事件的关联窗数直方图：{json.dumps(dict(sorted(support_hist.items())),ensure_ascii=False)}。多事件候选窗{len(multi)}个，其中合格{sum(r["eligible"] for r in multi)}个。只报告支持锚点跨度；允许输入没有事件时间，未计算time-to-shot。','',
        '|条件指标|GM−G五种子均值|配对SD|负差种子|','|---|---:|---:|---:|']
    for r in event_summary:
        if r['arm_or_contrast']=='GM-G':report.append(f"|{r['metric']}|{f(r['mean'])}|{f(r['seed_SD'])}|{r['negative_values']}/5|")
    report+=['','逐事件、逐比赛及全部种子值见event_metrics.csv、event_fixture_metrics.csv；无合格支持的事件仍保留NA，不补造预测。','',
        '## 删比赛：只删除评价贡献','',
        '下面仅从原六场评价均值移除一场，剩余五场等权。没有重训LOFO；保留头的训练可能使用了被删除比赛。原各场logloss差若都为负，删任一场后均值仍负是代数必然，不是新的稳健性或泛化验证。','',
        '|省略比赛|GM−G LL（种子均值）|GM−G Brier|GM−G绝对偏差|','|---|---:|---:|---:|']
    for mid in matches:report.append('|'+mid+'|'+'|'.join(f(avg([dv[mid,m,'GM-G',s] for s in range(5)])) for m in ['logloss','brier','absolute_bias'])+'|')
    report+=['','未计算p值、置信区间或bootstrap。五种子描述优化随机性，不增加比赛样本量。完整范围与全部六种删除见deletion_metrics.csv、deletion_summary.csv。',
        '本结果只描述冻结候选在既有开发数据上的敏感性；不证明运动特征具有因果跑位价值，不增加独立比赛，也不支持自动开启50轮正式重跑。原WOY等不利结果保留。']
    (out/'报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    guard();put(out/'completion.json',dict(status='COMPLETE',seconds=time.monotonic()-start,plan_sha256=PLAN_SHA,source_sha256=sha(__file__),
        outputs=[pin(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['stdout.log','stderr.log']],new_fits=0,new_predictions=0,
        raw_or_NPZ_reads=0,reserved_matches_read=0,new_independent_matches=0))
    print(json.dumps({'status':'COMPLETE','seconds':time.monotonic()-start,'completion_sha256':sha(out/'completion.json')},ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();run(Path(args.out))
