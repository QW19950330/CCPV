"""Table4 natural ADE256 and M32 projected-path EnergyU, with original NA gates.

The original pure FP64 functions are imported from score_math. This wrapper
restricts output to the specified natural-condition primary players/BALL rows;
it does not reproduce the full historical diagnostic report.
"""
import csv
import math
from pathlib import Path
import numpy as np
from score_math import (actor_mean, window_average, strict_macro,
                        source_conditional_macro, energy_components,
                        energy_from_components, jackknife_se)
from evaluation_core import required_samples


def subsets(ids):
    ids = np.asarray(ids).astype(str)
    return [(mid, ids == mid) for mid in sorted(set(ids))] + [('ALL_MATCHES', np.ones(len(ids), bool))]


def reduce(values, ids, eligible, aggregation):
    return strict_macro(values, ids) if aggregation == 'fixed_planned' else source_conditional_macro(values, ids, eligible)


def mean_rows(a, mean, cv, cell, seed):
    ix = np.where(a['split'].astype(str) == 'validation')[0]
    ids = a['match_ids'][ix].astype(str)
    seen = a['valid'][ix, :10].any(1)
    truth = a['positions'][ix, 10:30].astype(np.float64)
    observed = a['position_valid'][ix, 10:30] & seen[:, None]
    scale = np.ones((len(ix), 2))
    d = np.linalg.norm((mean[..., :2].astype(np.float64) - truth) * scale[:, None, None], axis=-1)
    cd = np.linalg.norm((cv[..., :2].astype(np.float64) - truth) * scale[:, None, None], axis=-1)
    rows = []
    for group, membership in [('players', (a['teams'] != 2)[None] & seen), ('BALL', (a['teams'] == 2)[None] & seen)]:
        mask = observed & membership[:, None]
        actor_eligible = mask.sum(1) > 0
        v = window_average(np.swapaxes(d, 1, 2), np.swapaxes(mask, 1, 2))
        vc = window_average(np.swapaxes(cd, 1, 2), np.swapaxes(mask, 1, 2))
        wv, wc = actor_mean(v, actor_eligible), actor_mean(vc, actor_eligible)
        nactor = actor_eligible.sum(-1)
        eligible = nactor > 0
        for mid, which in subsets(ids):
            for aggregation in ('fixed_planned', 'source_observed_conditional'):
                score = reduce(wv[which], ids[which], eligible[which], aggregation)
                cv_score = reduce(wc[which], ids[which], eligible[which], aggregation)
                rows.append(dict(arm=cell, seed=seed, condition='natural', coordinate_frame='standard105x68',
                                 group=group, stratum='all', metric='mean_ADE', step=0, aggregation=aggregation,
                                 match_id=mid, planned_windows=int(which.sum()), source_eligible_windows=int(eligible[which].sum()),
                                 source_eligible_actors=int(nactor[which].sum()), score=score,
                                 CV_point_score=cv_score, arm_minus_CV=score - cv_score))
    return rows


def energy_rows(a, sample, mean256, cv256, cell, seed):
    ix = np.asarray(a['validation32'], int)
    val = np.where(a['split'].astype(str) == 'validation')[0]
    lookup = {value: i for i, value in enumerate(val)}
    vi = np.array([lookup[value] for value in ix])
    ids = a['match_ids'][ix].astype(str)
    seen = a['valid'][ix, :10].any(1)
    if not np.array_equal(sample['scene_indices'], ix) or sample['hidden_players'].any():
        raise ValueError('Original natural M32 selection/mask changed.')
    sample_xy = required_samples(sample, seen)
    target = a['positions'][ix, 10:30].astype(np.float64)
    observed = a['position_valid'][ix, 10:30].astype(bool) & seen[:, None]
    scale = np.ones((len(ix), 2))
    x = sample_xy * scale[:, None, None, None, :]
    y = target * scale[:, None, None, :]
    mean = mean256[vi, ..., :2].astype(np.float64) * scale[:, None, None, :]
    cv = cv256[vi, ..., :2].astype(np.float64) * scale[:, None, None, :]
    components = {}
    for w, actor in zip(*np.where(seen)):
        mask = observed[w, :, actor].copy()
        if not mask.any():
            continue
        av, pv = energy_components(x[w, :, :, actor], y[w, :, actor], mask)
        if not (np.isfinite(mean[w, mask, actor]).all() and np.isfinite(cv[w, mask, actor]).all()):
            raise FloatingPointError('Required deterministic state nonfinite.')
        error = mean[w, mask, actor] - y[w, mask, actor]
        cv_error = cv[w, mask, actor] - y[w, mask, actor]
        components[w, actor] = (av, pv, float(np.sqrt(np.square(cv_error).sum() / mask.sum())),
                                float(np.linalg.norm(error, axis=-1).mean()), float(np.linalg.norm(cv_error, axis=-1).mean()))
    rows = []
    for group, membership in [('players', (a['teams'] != 2)[None] & seen), ('BALL', (a['teams'] == 2)[None] & seen)]:
        score = np.full(len(ix), np.nan)
        loo = np.full((len(ix), 32), np.nan)
        cval, mval, cvpoint = (np.full(len(ix), np.nan) for _ in range(3))
        nactor, neg = np.zeros(len(ix), int), np.zeros(len(ix), int)
        for w in range(len(ix)):
            ss, ll, cc, mm, cp = [], [], [], [], []
            for actor in np.where(membership[w])[0]:
                if (w, actor) not in components:
                    continue
                av, pv, c, m, point = components[w, actor]
                r = energy_from_components(av, pv)
                ss.append(r['score']); ll.append(r['loo']); cc.append(c); mm.append(m); cp.append(point)
                neg[w] += r['negative_roundoff']
            nactor[w] = len(ss)
            if ss:
                score[w] = np.mean(ss); loo[w] = np.mean(ll, axis=0)
                cval[w] = np.mean(cc); mval[w] = np.mean(mm); cvpoint[w] = np.mean(cp)
        eligible = nactor > 0
        for mid, which in subsets(ids):
            for aggregation in ('fixed_planned', 'source_observed_conditional'):
                agg = lambda value: reduce(value[which], ids[which], eligible[which], aggregation)
                value, lv, cvv, mv, cp = agg(score), agg(loo), agg(cval), agg(mval), agg(cvpoint)
                rows.append(dict(arm=cell, seed=seed, condition='natural', segment='M32', coordinate_frame='standard105x68',
                                 group=group, stratum='all', metric='energy_projected_path', aggregation=aggregation,
                                 match_id=mid, M=32, planned_windows=int(which.sum()), source_eligible_windows=int(eligible[which].sum()),
                                 planned_actors=int(membership[which].sum()), source_eligible_actors=int(nactor[which].sum()),
                                 numeric_failed_windows=int((eligible[which] & ~np.isfinite(score[which])).sum()),
                                 score=value, CV_delta_score=cvv, CV_point_metric=cp, mean_feedback_point_metric=mv,
                                 arm_minus_CV=value - cvv, MCSE=jackknife_se(lv), five_pipeline_seed_SD=math.nan,
                                 negative_roundoff_actors=int(neg[which].sum())))
    return rows


def csv_write(path, rows):
    def clean(value):
        if isinstance(value, np.generic):
            value = value.item()
        return '' if isinstance(value, float) and not math.isfinite(value) else value
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: clean(v) for k, v in row.items()} for row in rows)


def summarize(mean_scores, path_scores):
    rows = []
    for arm in ('C', 'D', 'E', 'F'):
        selected = []
        for metric, records in [('ADE256', mean_scores), ('EnergyU32_M32', path_scores)]:
            rr = [r for r in records if r['arm'] == arm and r['group'] == 'players' and r['aggregation'] == 'fixed_planned'
                  and r['match_id'] == 'ALL_MATCHES']
            if len(rr) != 5 or {r['seed'] for r in rr} != set(range(5)):
                raise ValueError('A Table4 summary requires all five seeds of all four arms.')
            rr.sort(key=lambda r: r['seed'])
            values = [r['score'] for r in rr]
            selected.append(dict(metric=metric, mean=float(np.mean(values)), sample_seed_SD=float(np.std(values, ddof=1)),
                                 seed_values=values, planned_windows=rr[0]['planned_windows'],
                                 source_eligible_windows=rr[0]['source_eligible_windows']))
        rows.append(dict(arm=arm, metrics=selected))
    return rows
