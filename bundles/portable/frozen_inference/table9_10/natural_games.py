"""Original full natural-game computation; explicit saved arrays and new output path."""
import itertools
import numpy as np
import game_math as M
from portable_io import save,table,write,output_record,now

def descriptive(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    empty = dict(count=0, mean=None, sd=None, min=None, p01=None, p05=None, p25=None,
                 median=None, p75=None, p95=None, p99=None, max=None, positive_fraction=None, negative_fraction=None, zero_fraction=None)
    if not len(x):
        return empty
    qs = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return dict(count=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x)>1 else None,
                **{k: float(v) for k, v in zip(('min','p01','p05','p25','median','p75','p95','p99','max'), qs)},
                positive_fraction=float(np.mean(x>0)), negative_fraction=float(np.mean(x<0)), zero_fraction=float(np.mean(x==0)))

def run(z,out):
    q, supported = np.exp(z['q_logp']), z['supported']
    if q.shape != (2,5,4,9,768) or supported.shape != (5,4,9,768):
        raise ValueError('Readout contract mismatch')
    mids = z['match_ids'].astype(str)
    matchlist = sorted(set(mids))
    game_names = tuple(M.GAMES)
    game_ok = np.stack([supported[:, :, [M.PATHS.index(p) for p in M.GAMES[g]]].all(2) for g in game_names], axis=2)
    common_game = game_ok.all((0,1))
    common_both = common_game.all(0)
    common_all9 = supported.all((0,1,2))
    out.mkdir(exist_ok=False)
    save(out / 'masks.npz', game_supported=game_ok, common_game=common_game,
         common_both_games=common_both, common_all9_paths=common_all9,
         match_ids=z['match_ids'], source_indices=z['source_indices'])
    scene, distribution, matchrows, seedrows = [], [], [], []
    cache = {}
    metrics = tuple(M.exact_game(np.zeros((1,4))))
    for fi, family in enumerate(M.VALUE_FAMILIES):
        for seed in range(5):
            for ai, arm in enumerate(M.WORLD_ARMS):
                for gi, game in enumerate(game_names):
                    ix = [M.PATHS.index(p) for p in M.GAMES[game]]
                    vals = q[fi,seed,ai,ix].T
                    result = M.exact_game(vals)
                    cache[fi,seed,ai,gi] = result
                    actor_gap,actor_order = M.actor_order(result['phi_A'],result['phi_B'])
                    valid = game_ok[seed,ai,gi]
                    v0gap = q[fi,seed,ai,M.PATHS.index('persistent_AB')] - q[fi,seed,ai,M.PATHS.index('persistent_delta0_AB')]
                    for w in range(768):
                        rec = dict(value_family=family, seed=seed, world_arm=arm, duration_policy=game,
                                   scene_index=w, match_id=mids[w], source_index=int(z['source_indices'][w]),
                                   actor_A_source_id=str(z['target_source_ids'][w,0]), actor_B_source_id=str(z['target_source_ids'][w,1]),
                                   game_supported=bool(valid[w]), common_game=bool(common_game[gi,w]),
                                   common_both_games=bool(common_both[w]))
                        rec.update({f'q_{name}': float(vals[w,j]) if valid[w] else None for j,name in enumerate(('empty','A','B','AB'))})
                        rec.update({k: float(result[k][w]) if valid[w] else None for k in metrics})
                        rec.update(actor_phi_A_minus_phi_B=float(actor_gap[w]) if valid[w] else None,
                                   actor_order_sign=int(actor_order[w]) if valid[w] else None,
                                   actor_order_exact_tie=bool(actor_order[w]==0) if valid[w] else None)
                        rec['persistent_AB_minus_repeated_v0_AB_joint_only'] = float(v0gap[w]) if game=='persistent' and np.isfinite(v0gap[w]) else None
                        scene.append(rec)
                    for scope, mask in [('available_cell',valid), ('common_game',common_game[gi]), ('common_both_games',common_both)]:
                        for metric in metrics:
                            v = result[metric]
                            distribution.append(dict(value_family=family, seed=seed, world_arm=arm, duration_policy=game,
                                                     scope=scope, metric=metric, planned=768, **descriptive(v[mask])))
                            mm = []
                            for mid in matchlist:
                                take = mask & (mids==mid)
                                x = v[take]
                                mean = float(x.mean()) if len(x) else None
                                matchrows.append(dict(value_family=family, seed=seed, world_arm=arm, duration_policy=game,
                                                      scope=scope, metric=metric, match_id=mid, planned=64,
                                                      supported=int(take.sum()), mean=mean))
                                if mean is not None:
                                    mm.append(mean)
                            seedrows.append(dict(value_family=family, seed=seed, world_arm=arm, duration_policy=game,
                                                 scope=scope, metric=metric, planned=768, supported=int(mask.sum()),
                                                 matches_supported=len(mm), strict12_match_macro=float(np.mean(mm)) if len(mm)==12 else None,
                                                 available_match_macro=float(np.mean(mm)) if mm else None))
    summaries = []
    for family, arm, game, scope, metric in itertools.product(M.VALUE_FAMILIES,M.WORLD_ARMS,game_names,
            ('available_cell','common_game','common_both_games'),metrics):
        rr = [r for r in seedrows if (r['value_family'],r['world_arm'],r['duration_policy'],r['scope'],r['metric']) == (family,arm,game,scope,metric)]
        x = [r['strict12_match_macro'] for r in rr if r['strict12_match_macro'] is not None]
        summaries.append(dict(value_family=family,world_arm=arm,duration_policy=game,scope=scope,metric=metric,
                              seeds_supported=len(x), five_seed_match_macro_mean=float(np.mean(x)) if len(x)==5 else None,
                              five_seed_match_macro_sd=float(np.std(x,ddof=1)) if len(x)==5 else None,
                              uncertainty='initialization-seed sample SD, NOT Monte Carlo or independent-match CI'))
    sensitivity = []
    def corr(kind, label, x, y, mask):
        rho,n = M.rank_correlation(np.where(mask,x,np.nan),np.where(mask,y,np.nan))
        sensitivity.append(dict(rank_unit='SCENE_DELTA_AB',kind=kind,comparison=label,common_rows=int(mask.sum()),paired_rows=n,spearman=rho))
    for seed,ai,gi in itertools.product(range(5),range(4),range(2)):
        corr('value_family',f's{seed}/{M.WORLD_ARMS[ai]}/{game_names[gi]}',cache[0,seed,ai,gi]['delta_AB'],cache[1,seed,ai,gi]['delta_AB'],common_both)
    for fi,seed,gi in itertools.product(range(2),range(5),range(2)):
        for ai,bi in itertools.combinations(range(4),2):
            corr('world_arm',f'{M.VALUE_FAMILIES[fi]}/s{seed}/{game_names[gi]}/{M.WORLD_ARMS[ai]}:{M.WORLD_ARMS[bi]}',cache[fi,seed,ai,gi]['delta_AB'],cache[fi,seed,bi,gi]['delta_AB'],common_both)
    for fi,ai,gi in itertools.product(range(2),range(4),range(2)):
        for s,t in itertools.combinations(range(5),2):
            corr('seed',f'{M.VALUE_FAMILIES[fi]}/{M.WORLD_ARMS[ai]}/{game_names[gi]}/s{s}:s{t}',cache[fi,s,ai,gi]['delta_AB'],cache[fi,t,ai,gi]['delta_AB'],common_both)
    for fi,seed,ai in itertools.product(range(2),range(5),range(4)):
        corr('duration',f'{M.VALUE_FAMILIES[fi]}/s{seed}/{M.WORLD_ARMS[ai]}',cache[fi,seed,ai,0]['delta_AB'],cache[fi,seed,ai,1]['delta_AB'],common_both)
        gap = q[fi,seed,ai,7]-q[fi,seed,ai,8]
        mask = common_both & supported[seed,ai,8]
        corr('reference_joint_only',f'{M.VALUE_FAMILIES[fi]}/s{seed}/{M.WORLD_ARMS[ai]}',cache[fi,seed,ai,1]['delta_AB'],gap,mask)
    # Predeclared paired numeric sensitivities; no head or trajectory winner selection.
    paired,reference = [],[]
    def paired_row(kind,label,seed,game,metric,x,y,mask):
        d=x-y
        mm=[float(d[mask&(mids==mid)].mean()) for mid in matchlist if np.any(mask&(mids==mid))]
        return dict(kind=kind,comparison=label,seed=seed,duration_policy=game,metric=metric,
                    scope='common_both_games',planned=768,supported=int(mask.sum()),matches_supported=len(mm),
                    strict12_match_macro=float(np.mean(mm)) if len(mm)==12 else None,**descriptive(d[mask]))
    for fi,seed,gi,metric in itertools.product(range(2),range(5),range(2),metrics):
        for bi in (0,1,3):
            paired.append(paired_row('world_arm',f'{M.VALUE_FAMILIES[fi]}/D_coupled_ar-minus-{M.WORLD_ARMS[bi]}',seed,game_names[gi],metric,
                          cache[fi,seed,2,gi][metric],cache[fi,seed,bi,gi][metric],common_both))
    for seed,ai,gi,metric in itertools.product(range(5),range(4),range(2),metrics):
        paired.append(paired_row('value_family',f'D_Direct_WithEarlyBCE-minus-CV_Direct_WithEarlyBCE/{M.WORLD_ARMS[ai]}',seed,game_names[gi],metric,
                      cache[1,seed,ai,gi][metric],cache[0,seed,ai,gi][metric],common_both))
    refpairs=(('first_AB_minus_natural_empty',4,0),('persistent_AB_minus_natural_empty',7,0),
              ('persistent_AB_minus_repeated_v0_AB_joint_only',7,8),('repeated_v0_AB_minus_natural_empty',8,0),
              ('no_input_edit_NONE_minus_natural_empty',1,0))
    for fi,seed,ai,(name,left,right) in itertools.product(range(2),range(5),range(4),refpairs):
        d=q[fi,seed,ai,left]-q[fi,seed,ai,right]
        for scope,mask in [('common_all9_paths',common_all9),('available_pair',supported[seed,ai,left]&supported[seed,ai,right])]:
            mm=[float(d[mask&(mids==mid)].mean()) for mid in matchlist if np.any(mask&(mids==mid))]
            reference.append(dict(value_family=M.VALUE_FAMILIES[fi],seed=seed,world_arm=M.WORLD_ARMS[ai],joint_contrast=name,
                                  scope=scope,planned=768,supported=int(mask.sum()),matches_supported=len(mm),
                                  strict12_match_macro=float(np.mean(mm)) if len(mm)==12 else None,**descriptive(d[mask])))
    # Sample SD across five paired initialization seeds of the same fixed contrast.
    paired_summary=[]
    keys=sorted({(r['kind'],r['comparison'],r['duration_policy'],r['metric']) for r in paired})
    for kind,label,game,metric in keys:
        rr=[r for r in paired if (r['kind'],r['comparison'],r['duration_policy'],r['metric'])==(kind,label,game,metric)]
        x=[r['strict12_match_macro'] for r in rr if r['strict12_match_macro'] is not None]
        paired_summary.append(dict(kind=kind,comparison=label,duration_policy=game,metric=metric,scope='common_both_games',
                                   seeds_supported=len(x),five_seed_mean=float(np.mean(x)) if len(x)==5 else None,
                                   paired_seed_sd=float(np.std(x,ddof=1)) if len(x)==5 else None,
                                   uncertainty='paired initialization seed sample SD, not MC or population CI'))
    actor_counts,actor_pairs=[],[]
    orders={k:M.actor_order(v['phi_A'],v['phi_B'])[1] for k,v in cache.items()}
    for (fi,seed,ai,gi),order in orders.items():
        for scope,mask in [('available_cell',game_ok[seed,ai,gi]),('common_game',common_game[gi]),('common_both_games',common_both)]:
            x=order[mask]
            actor_counts.append(dict(rank_unit='FIXED_ACTOR_A_VS_B_WITHIN_SCENE',value_family=M.VALUE_FAMILIES[fi],seed=seed,
                                     world_arm=M.WORLD_ARMS[ai],duration_policy=game_names[gi],scope=scope,planned=768,
                                     supported=int(mask.sum()),A_over_B=int(np.sum(x>0)),B_over_A=int(np.sum(x<0)),
                                     exact_ties=int(np.sum(x==0)),tie_rule='phi_A-minus-phi_B == 0.0; no fitted tolerance'))
    def actor_pair(kind,label,left,right):
        actor_pairs.append(dict(rank_unit='FIXED_ACTOR_A_VS_B_WITHIN_SCENE',kind=kind,comparison=label,
                                scope='common_both_games',planned=768,tie_rule='exact stored float phi_A-minus-phi_B zero',
                                **M.actor_agreement(left,right,common_both)))
    for fi,ai,gi in itertools.product(range(2),range(4),range(2)):
        for seed,t in itertools.combinations(range(5),2):
            actor_pair('seed',f'{M.VALUE_FAMILIES[fi]}/{M.WORLD_ARMS[ai]}/{game_names[gi]}/s{seed}:s{t}',
                       orders[fi,seed,ai,gi],orders[fi,t,ai,gi])
    for seed,ai,gi in itertools.product(range(5),range(4),range(2)):
        actor_pair('value_family',f's{seed}/{M.WORLD_ARMS[ai]}/{game_names[gi]}/CVvalue:Dvalue',
                   orders[0,seed,ai,gi],orders[1,seed,ai,gi])
    for fi,seed,ai in itertools.product(range(2),range(5),range(4)):
        actor_pair('duration',f'{M.VALUE_FAMILIES[fi]}/s{seed}/{M.WORLD_ARMS[ai]}/first:persistent',
                   orders[fi,seed,ai,0],orders[fi,seed,ai,1])
    for fi,seed,gi in itertools.product(range(2),range(5),range(2)):
        for ai,bi in itertools.combinations(range(4),2):
            actor_pair('world_arm',f'{M.VALUE_FAMILIES[fi]}/s{seed}/{game_names[gi]}/{M.WORLD_ARMS[ai]}:{M.WORLD_ARMS[bi]}',
                       orders[fi,seed,ai,gi],orders[fi,seed,bi,gi])
    files = []
    for name,rows in [('game_scene.csv',scene),('distributions_by_seed.csv',distribution),
                      ('game_by_match.csv',matchrows),('game_by_seed.csv',seedrows),
                      ('game_summary.csv',summaries),('rank_sensitivity.csv',sensitivity),
                      ('paired_sensitivity_by_seed.csv',paired),('paired_sensitivity_summary.csv',paired_summary),
                      ('joint_reference_by_seed.csv',reference),('actor_order_counts.csv',actor_counts),
                      ('actor_order_agreement.csv',actor_pairs)]:
        path = out/name
        table(path,rows)
        files.append(output_record(path))
    files.append(output_record(out/'masks.npz'))
    errors = [abs(r['efficiency_error']) for r in scene if r['game_supported']]
    write(out/'summary.json', dict(status='COMPLETED_SAVED_SHARED_VALUE_DIAGNOSTICS',utc=now(),
          planned_windows=768,planned_matches=12,game_scene_rows=len(scene),common_game_counts={g:int(common_game[i].sum()) for i,g in enumerate(game_names)},
          common_both_games=int(common_both.sum()),common_all9_paths=int(common_all9.sum()),max_exact_game_efficiency_error=max(errors,default=0.),
          primary_value_family=M.VALUE_FAMILIES[0],secondary_value_family=M.VALUE_FAMILIES[1],
          primary_scope='common_both_games', all_negative_zero_positive_retained=True,
          actual_future_xy_read=False,event_labels_read=False,world_forward_calls=0,fits=0,
          no_stochastic_rollout_MC=True, no_independent_test=True, no_effective_action_or_goal_claim=True,
          repeated_v0='joint AB reference sensitivity only; mixed coalitions absent, no Shapley game',
          actor_order='within-scene phi_A-phi_B sign/exact ties; paired agreement common_both_games; no v0 actor ranks',
          actor_role='two fixed roster-home actors, no verified possession/attack/offball role',
          scope='model-internal score response; factual DFL heads uncalibrated on generated SkillCorner states'))
    files.append(output_record(out/'summary.json'))
    return dict(scene_rows=len(scene), common_both=int(common_both.sum()), common_first=int(common_game[0].sum()), common_persistent=int(common_game[1].sum()), common_all9=int(common_all9.sum()))
