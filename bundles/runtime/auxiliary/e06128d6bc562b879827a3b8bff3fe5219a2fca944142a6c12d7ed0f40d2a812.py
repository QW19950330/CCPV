"""Three create-only stages: fold-only training, frozen readout, then scoring."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from transport_math import head, mixture, metrics

ROOT = Path(__file__).resolve().parent
LOGS = ('q_logp', 'q_log1mp', 'early_logp', 'early_log1mp')
CELLS = ('OldTrained_Mean', 'OldTrained_MC', 'MCTrained_Mean', 'MCTrained_MC')
STATES = ('hidden_weight', 'hidden_bias', 'output_weight', 'output_bias')
SCALERS = ('mean', 'scale', 'observed_weight_mass', 'completely_absent_training_feature')
LINEAGE = ('training_seed','updates','raw_dimension','parameter_count','output_targets',
           'training_initial_rows','training_loss_rows','held_rows','training_match_ids','outer_held_match')
HEAD_KEYS = STATES+tuple('scaler_'+k for k in SCALERS)+LINEAGE
FIT_KEYS = HEAD_KEYS+tuple('initial_'+k for k in STATES)+('training_weights','actual_init_seed')
FEATURE_KEYS = ('seed','original_rows','cohort_positions','anchor_id','match_id','fixed_team_A',
                'features_oldGPU','features_particles')
IDS = ('initial_indices','anchor_id','match_id','fixed_team_A')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def put(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save(path, arrays):
    with Path(path).open('xb') as f: np.savez_compressed(f, **arrays)


def table(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def selected(rec, keys):
    if sha(rec['path']) != rec['sha256']: raise ValueError('Frozen payload changed: '+rec['path'])
    with np.load(rec['path'], allow_pickle=False) as z:
        if not set(keys) <= set(z.files): raise ValueError('Missing whitelisted members')
        return {k:z[k] for k in keys}


def guard(stage, auth, out):
    a = read(auth); p = read(ROOT/'plan.json')
    if a['status'] != 'AUTHORIZED_MATCHED_VALUE_OBJECTIVE' or stage not in a['allowed_stages']:
        raise PermissionError('Exact root stage authorization required')
    if a['plan_sha256'] != sha(ROOT/'plan.json'): raise ValueError('Plan changed')
    for rec in p['sources']+p['source_evidence']+p['metadata']+[p['historical_cv_context']]:
        if sha(rec['path']) != rec['sha256']: raise ValueError('Source/metadata changed: '+rec['path'])
    if out.parent != ROOT or out.exists(): raise ValueError('Create-only immediate study child')
    pause(); return p


def pause():
    if (ROOT/'PAUSE_REQUESTED').exists(): raise InterruptedError('User pause')
    q = ROOT/'user_pause.json'
    if q.exists() and read(q).get('status') in ('PAUSE_REQUESTED','PAUSING','PAUSED_USER'):
        raise InterruptedError('User pause')


def bounded(start, p):
    pause()
    if time.monotonic()-start > p['limits']['wall_seconds']: raise TimeoutError('1200 s cumulative producer budget')
    size = sum(x.stat().st_size for d in ('fit01','readout01','score01') for x in (ROOT/d).rglob('*') if x.is_file())
    if size > p['limits']['output_bytes']: raise RuntimeError('512 MiB output budget')


def freeze(out, status, start, p, **extra):
    bounded(start,p)
    records = [pin(x) for x in sorted(out.iterdir()) if x.is_file()]
    put(out/'freeze.json',dict(status=status,utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(ROOT/'plan.json'),
        elapsed_cumulative_s=time.monotonic()-start,records=records,**extra))


def frozen(folder, status):
    f = read(folder/'freeze.json')
    if f['status'] != status or f['plan_sha256'] != sha(ROOT/'plan.json'): raise ValueError('Previous stage not frozen')
    for r in f['records']:
        if sha(r['path']) != r['sha256']: raise ValueError('Previous stage changed')
    return f


def features(p, seed):
    """Only frozen source IDs and two raw feature arrays; never actual future/labels."""
    n = p['population']['N']; count = np.zeros(n, np.int8)
    a = dict(mean=np.empty((n,36), np.float64), particles=np.empty((n,16,36), np.float64))
    for key in ('anchor_id','match_id','fixed_team_A'): a[key] = np.empty(n, dtype='U160')
    for rec in p['shards']:
        if rec['seed'] != seed: continue
        d = selected(rec,FEATURE_KEYS); pos = np.asarray(d['cohort_positions'],int); b = len(pos)
        assert int(d['seed']) == seed and pos.tolist() == rec['cohort_positions']
        assert d['original_rows'].tolist() == rec['original_rows'] == [p['population']['original_indices'][j] for j in pos]
        assert d['features_oldGPU'].shape == (b,36) and d['features_particles'].shape == (b,16,36)
        assert np.array_equal(d['features_particles'][:,:,:18],np.repeat(d['features_oldGPU'][:,None,:18],16,axis=1),equal_nan=True)
        assert not count[pos].any(); count[pos] += 1
        a['mean'][pos],a['particles'][pos] = d['features_oldGPU'],d['features_particles']
        for key in ('anchor_id','match_id','fixed_team_A'): a[key][pos] = d[key].astype(str)
    assert (count == 1).all() and a['match_id'].tolist() == p['population']['match_for_cohort']
    assert len(set(a['anchor_id'].tolist())) == n
    a['initial_indices'] = np.asarray(p['population']['original_indices'],int)
    return a


def check_head(cp, rec, fold):
    assert int(cp['training_seed']) == rec['seed'] and int(cp['updates']) == 200
    assert int(cp['raw_dimension']) == 36 and int(cp['parameter_count']) == 602
    assert cp['output_targets'].astype(str).tolist() == ['Y10','Y2']
    assert str(cp['outer_held_match']) == fold['held_match']
    assert cp['training_match_ids'].astype(str).tolist() == fold['training_matches']
    for key,expected in (('training_initial_rows',fold['training_rows']),('training_loss_rows',fold['training_rows']),('held_rows',fold['held_rows'])):
        assert np.array_equal(cp[key],expected)
    for key,shape in zip(STATES,((8,72),(8,),(2,8),(2,))):
        assert cp[key].shape == shape and cp[key].dtype == np.float64 and np.isfinite(cp[key]).all()
    assert all(cp['scaler_'+k].shape == (36,) for k in SCALERS)
    assert (cp['scaler_scale'] > 0).all()


def training_targets(p, a, positions, fold):
    rows = a['initial_indices'][positions]
    assert rows.tolist() == fold['training_rows']
    assert set(a['match_id'][positions]) == set(fold['training_matches']) and fold['held_match'] not in set(a['match_id'][positions])
    # Whole NPZ members decode physically; only these asserted fold slices leave this function.
    d = selected(p['labels'],('anchor_id','match_id','fixed_team_A','eligible_recorded10','y_10','y_early2','y_late8'))
    for key in ('anchor_id','match_id','fixed_team_A'): assert np.array_equal(d[key][rows].astype(str),a[key][positions])
    assert d['eligible_recorded10'][rows].all()
    y,y2,l = [np.asarray(d[k][rows],np.float64).copy() for k in ('y_10','y_early2','y_late8')]
    assert all(np.isin(v,(0,1)).all() for v in (y,y2,l)) and np.array_equal(y,y2+(1-y2)*l)
    return np.c_[y,y2]


def fit_stage(auth, out):
    start=time.monotonic(); p=guard('fit',auth,out); out.mkdir()
    import torch
    import matched_model as M
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    records=[]
    for seed in range(5):
        a=features(p,seed)
        for fold in p['population']['folds']:
            bounded(start,p)
            held=fold['held_match']; take=np.flatnonzero(a['match_id'] != held)
            rec=next(r for r in p['heads'] if r['seed']==seed and r['held_match']==held)
            old=selected(rec,FIT_KEYS); check_head(old,rec,fold)
            y=training_targets(p,a,take,fold)
            ids=a['match_id'][take]
            w=np.array([1/(5*np.count_nonzero(ids==m)) for m in ids],np.float64)
            assert np.array_equal(w,old['training_weights'])
            initial={k:old['initial_'+k].copy() for k in STATES}
            norm={k:old['scaler_'+k].copy() for k in SCALERS}
            cp,hist=M.fit(initial,norm,a['particles'][take].copy(),y,w,seed,tick=lambda:bounded(start,p))
            cp.update({k:old[k].copy() for k in LINEAGE if k not in cp})
            cp['actual_init_seed']=old['actual_init_seed'].copy()
            assert all(np.array_equal(cp['initial_'+k],old['initial_'+k]) for k in STATES)
            assert all(np.array_equal(cp['scaler_'+k],old['scaler_'+k]) for k in SCALERS)
            cp_path=out/f'o{fold["fold"]}_s{seed}.npz'; hp=out/f'o{fold["fold"]}_s{seed}_history.json'
            save(cp_path,cp); put(hp,hist)
            records.append(dict(seed=seed,held_match=held,fold=fold['fold'],checkpoint=pin(cp_path),history=pin(hp),
                                original_head=rec,train_rows=len(take),updates=200))
            print('FROZEN_HEAD',len(records),'/30',held,seed,flush=True)
    assert len(records)==30
    freeze(out,'ALL30_MC_MATCHED_HEADS_FROZEN_BEFORE_HELD_PREDICTION',start,p,heads=records,new_updates=6000,
           held_predictions=0,held_label_values_passed_to_fitter=0,new_trajectory_forwards=0)


def readout_stage(auth, out):
    start=time.monotonic(); p=guard('readout',auth,out)
    f=frozen(ROOT/'fit01','ALL30_MC_MATCHED_HEADS_FROZEN_BEFORE_HELD_PREDICTION')
    start-=f['elapsed_cumulative_s']; assert len(f['heads'])==30; out.mkdir()
    reference=selected(p['old_readout'],IDS+tuple(arm+'__'+k for arm in ('OldGPUmean','MC_K16') for k in LOGS))
    n=p['population']['N']; data={cell+'__'+k:np.full((5,n),np.nan) for cell in CELLS for k in LOGS}
    for cell in ('OldTrained_MC','MCTrained_MC'): data[cell+'__q_MCSE']=np.full((5,n),np.nan)
    diagnostics=[]; ledger=[]; max_error=0.
    for seed in range(5):
        a=features(p,seed); bounded(start,p)
        for key in IDS: assert np.array_equal(a[key].astype(str),reference[key].astype(str))
        for fold in p['population']['folds']:
            held=fold['held_match']; take=np.flatnonzero(a['match_id']==held); b=len(take)
            oldrec=next(r for r in p['heads'] if r['seed']==seed and r['held_match']==held)
            newrec=next(r for r in f['heads'] if r['seed']==seed and r['held_match']==held)
            old=selected(oldrec,HEAD_KEYS); new=selected(newrec['checkpoint'],HEAD_KEYS)
            check_head(old,oldrec,fold); check_head(new,oldrec,fold)
            assert all(np.array_equal(new['scaler_'+k],old['scaler_'+k]) for k in SCALERS)
            diag=dict(seed=seed,match_id=held,windows=b)
            for family,cp in (('OldTrained',old),('MCTrained',new)):
                mean,zm,mm=head(cp,a['mean'][take]); part,zp,mp=head(cp,a['particles'][take].reshape(b*16,36))
                part={k:v.reshape(b,16) for k,v in part.items()}; mc=mixture(part)
                for mode,result in (('Mean',mean),('MC',mc)):
                    cell=family+'_'+mode
                    for k in LOGS:
                        if family=='OldTrained':
                            expected=reference[('OldGPUmean' if mode=='Mean' else 'MC_K16')+'__'+k][seed,take]
                            err=float(np.max(np.abs(result[k]-expected))); max_error=max(max_error,err)
                            if err > 1e-12: raise ArithmeticError('Old outer head readout did not replay')
                        data[cell+'__'+k][seed,take]=result[k]
                    diag[cell+'__Early_gt_q']=int((result['early_logp']>result['q_logp']).sum())
                data[family+'_MC__q_MCSE'][seed,take]=np.exp(part['q_logp']).std(1,ddof=1)/4
                if family=='OldTrained':
                    for mode,z,missing in (('Mean',zm,mm),('MC',zp,mp)):
                        diag[mode+'_missing']=int(missing.sum())
                        for lim in (3,5): diag[mode+'_z_gt'+str(lim)]=int((np.abs(z)>lim).sum())
            diagnostics.append(diag); ledger.append(dict(seed=seed,match_id=held,rows=b,held_head=held,new_checkpoint=newrec['checkpoint']))
    assert all(np.isfinite(v).all() for v in data.values())
    for cell in CELLS:
        for stem in ('q','early'): assert np.max(np.abs(np.logaddexp(data[cell+'__'+stem+'_logp'],data[cell+'__'+stem+'_log1mp'])))<1e-12
    data.update({k:reference[k] for k in IDS}); save(out/'readouts.npz',data)
    table(out/'diagnostics.csv',diagnostics); put(out/'lineage.json',ledger)
    freeze(out,'ALL4_MATCHED_READOUT_CELLS_FROZEN_BEFORE_HELD_SCORING',start,p,predictions=pin(out/'readouts.npz'),
           old_replay_max_abs=max_error,source_head_training_label_members_read=False,held_scoring=0)


def score_stage(auth, out):
    start=time.monotonic(); p=guard('score',auth,out)
    f=frozen(ROOT/'readout01','ALL4_MATCHED_READOUT_CELLS_FROZEN_BEFORE_HELD_SCORING')
    start-=f['elapsed_cumulative_s']; out.mkdir()
    d=selected(f['predictions'],IDS+tuple(cell+'__'+k for cell in CELLS for k in LOGS))
    label=selected(p['labels'],('anchor_id','match_id','fixed_team_A','period','source_frame_N','eligible_recorded10','y_10','y_early2','y_late8'))
    ix=d['initial_indices'].astype(int)
    assert np.array_equal(ix,p['population']['original_indices']) and np.array_equal(ix,np.flatnonzero(label['eligible_recorded10']))
    for k in IDS[1:]: assert np.array_equal(d[k].astype(str),label[k][ix].astype(str))
    expected=np.array([f'{m}|{period}|N:{frame}' for m,period,frame in zip(label['match_id'].astype(str),label['period'],label['source_frame_N'])])
    assert np.array_equal(expected,label['anchor_id'].astype(str))
    y,y2,late=[label[k][ix] for k in ('y_10','y_early2','y_late8')]
    assert all(np.isin(v,(0,1)).all() for v in (y,y2,late)) and np.array_equal(y,y2+(1-y2)*late)
    fixtures=[]; seeds=[]; names=p['population']['matches']
    for cell in CELLS:
        for s in range(5):
            for target,stem,truth in (('Y10','q',y),('Y2','early',y2)):
                rr=[]
                for match in names:
                    take=d['match_id'].astype(str)==match
                    v=metrics(d[cell+'__'+stem+'_logp'][s,take],d[cell+'__'+stem+'_log1mp'][s,take],truth[take]); rr.append(v)
                    fixtures.append(dict(cell=cell,seed=s,target=target,match_id=match,windows=int(take.sum()),positives=int(truth[take].sum()),**v))
                seeds.append(dict(cell=cell,seed=s,target=target,**{k:float(np.mean([r[k] for r in rr])) for k in v}))
    metrics_names=list(v); summaries=[]
    for cell in CELLS:
        for target in ('Y10','Y2'):
            rr=[r for r in seeds if r['cell']==cell and r['target']==target]
            for k in metrics_names:
                values=np.array([r[k] for r in rr]); summaries.append(dict(cell=cell,target=target,metric=k,mean=float(values.mean()),seed_SD=float(values.std(ddof=1))))
    paired=[]; paired_fixture=[]
    pairs={'A_match':('MCTrained_MC','OldTrained_MC'),'B_replace':('MCTrained_MC','OldTrained_Mean'),
           'new_readout':('MCTrained_MC','MCTrained_Mean'),'old_readout':('OldTrained_MC','OldTrained_Mean'),
           'mean_training_shift':('MCTrained_Mean','OldTrained_Mean')}
    for rows,dest,is_fixture in ((seeds,paired,False),(fixtures,paired_fixture,True)):
        keys=('seed','target','match_id') if is_fixture else ('seed','target')
        lookup={(r['cell'],)+tuple(r[k] for k in keys):r for r in rows}
        for ident in sorted({tuple(r[k] for k in keys) for r in rows}):
            for contrast,(left,right) in pairs.items():
                a,b=lookup[(left,)+ident],lookup[(right,)+ident]
                dest.append(dict(contrast=contrast,**dict(zip(keys,ident)),**{k:a[k]-b[k] for k in metrics_names}))
            q=lambda c,k:lookup[(c,)+ident][k]
            dest.append(dict(contrast='interaction_Psi',**dict(zip(keys,ident)),
                **{k:(q('MCTrained_MC',k)-q('MCTrained_Mean',k))-(q('OldTrained_MC',k)-q('OldTrained_Mean',k)) for k in metrics_names}))
    table(out/'per_fixture.csv',fixtures); table(out/'per_seed.csv',seeds); table(out/'summary.csv',summaries)
    table(out/'paired_seed.csv',paired); table(out/'paired_fixture.csv',paired_fixture)
    with Path(p['historical_cv_context']['path']).open(encoding='utf-8-sig',newline='') as z: old=list(csv.DictReader(z))
    cv=next(r for r in old if r['arm']=='CV_Direct_WithEarlyBCE' and r['target']=='q10')
    table(out/'unchanged_historical_CV.csv',[cv])
    put(out/'label_lineage.json',dict(source=p['labels'],N=len(y),positive_Y10=int(y.sum()),positive_Y2=int(y2.sum()),
        target='fixedA recorded native ShotAtGoal in (t0,t0+10]',clock='AUTHOR3_PLUS_DIRECT_OTHER4',
        omitted_zero_eligible_match='DFL-MAT-J03WOH',independent_validation=False,readout_freeze=pin(ROOT/'readout01/freeze.json')))
    report(out,p,summaries,seeds,paired,paired_fixture,cv)
    freeze(out,'MATCHED_VALUE_OBJECTIVE_SAVED_SCORING_COMPLETE',start,p,cells=4,fixtures=len(fixtures),seeds=len(seeds),
           new_heads=30,new_updates=6000,new_trajectory_forwards=0,independent_validation=False)


def report(out,p,summary,seeds,paired,byfixture,cv):
    lookup={(r['cell'],r['target'],r['metric']):r for r in summary}
    lines=['# MC匹配价值目标：五种子小规模对照','',
        '全部5344窗、6场已曝光比赛，固定1.8秒均值前缀+0.2秒K16随机后缀。30个新价值头各200次更新，旧头、轨迹和结果保留；没有新轨迹或50轮。结局是记录ShotAtGoal，不是进球或干预收益。','',
        '|训练/读出|Y10 log loss均值±SD|Brier均值±SD|每场绝对概率偏差均值±SD|','|---|---:|---:|---:|']
    for cell in CELLS:
        values=[lookup[cell,'Y10',k] for k in ('log_loss','brier','absolute_fixture_bias')]
        lines.append('|'+cell+'|'+'|'.join(f'{r["mean"]:.12f} ± {r["seed_SD"]:.12f}' for r in values)+'|')
    lines+=['','共同主要对比A=新MC/MC−旧均值/MC；B=新MC/MC−旧均值/Mean。负log loss差值有利。Psi为训练×读出描述性交互，不是第三种方法或因果证据。','',
            '|对比|log loss差值±种子SD|改善种子|Brier差值|绝对偏差差值|','|---|---:|---:|---:|---:|']
    for contrast in ('A_match','B_replace','new_readout','old_readout','mean_training_shift','interaction_Psi'):
        rr=[r for r in paired if r['contrast']==contrast and r['target']=='Y10']; delta=np.array([r['log_loss'] for r in rr])
        lines.append(f'|{contrast}|{delta.mean():+.12f} ± {delta.std(ddof=1):.12f}|{int((delta<0).sum())}/5|{np.mean([r["brier"] for r in rr]):+.12f}|{np.mean([r["absolute_fixture_bias"] for r in rr]):+.12f}|')
    lines+=['','|对比|种子|log loss差值|Brier差值|绝对概率偏差差值|','|---|---:|---:|---:|---:|']
    for r in paired:
        if r['target']=='Y10' and r['contrast'] in ('A_match','B_replace'):
            lines.append(f'|{r["contrast"]}|{r["seed"]}|{r["log_loss"]:+.12f}|{r["brier"]:+.12f}|{r["absolute_fixture_bias"]:+.12f}|')
    lines+=['','|对比|比赛|log loss差值均值±种子SD|改善种子|','|---|---|---:|---:|']
    for contrast in ('A_match','B_replace'):
        for match in p['population']['matches']:
            v=np.array([r['log_loss'] for r in byfixture if r['contrast']==contrast and r['target']=='Y10' and r['match_id']==match])
            lines.append(f'|{contrast}|{match}|{v.mean():+.12f} ± {v.std(ddof=1):.12f}|{int((v<0).sum())}/5|')
    lines+=['',f'历史CV_Direct_WithEarlyBCE：Y10 log loss {cv["logloss_mean"]} ± {cv["logloss_seed_SD"]}，Brier {cv["Brier_mean"]}，原值保留。改善神经模型不能自动等同于超过CV。',
        '全四格、Y2、六场、五种子、负面结果和预测漂移见CSV；Early>Y10、特征异常/缺失见readout01/diagnostics.csv。未裁剪概率或按结果调整校准。',
        '每场内平均再六场等权；种子不是独立数据集，粒子不是16倍标签。旧均值头训练口径为OldGPUmean；CPUmean后端桥接已单独封存，本轮不混用。新旧容量/200更新相同，但MC每步16倍价值头输入计算。',
        '本轮仅检验已曝光开发比赛上目标与读出的匹配，不证明未曝光泛化、完整20步不确定性、真实进球率或战术干预收益。有限预算梯度并不证明全局收敛；所有30头冻结前未计算持出loss。']
    with (out/'匹配价值目标对照报告.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['fit','readout','score'],required=True)
    ap.add_argument('--authorization',type=Path,required=True); args=ap.parse_args()
    folders={'fit':'fit01','readout':'readout01','score':'score01'}
    {'fit':fit_stage,'readout':readout_stage,'score':score_stage}[args.stage](args.authorization,ROOT/folders[args.stage])
