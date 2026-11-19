"""Original complete deterministic natural-readout report table aggregation."""
from collections import defaultdict
import numpy as np
from portable_io import rows,read,table

def run(G,O,readouts):
    O.mkdir(exist_ok=False)
    summary=read(G/'summary.json');game=rows(G/'game_summary.csv')
    primary=[r for r in game if r['value_family']=='CV_Direct_WithEarlyBCE' and r['scope']=='common_both_games' and r['metric'] in ('delta_AB','phi_A','phi_B','interaction_difference')]
    table(O/'primary_game_summary.csv',primary)
    with np.load(G/'masks.npz',allow_pickle=False) as z:mask=z['common_both_games'];mids=z['match_ids'].astype(str)
    z=readouts
    q=np.exp(z['q_logp']);arms=z['world_arms'].astype(str);families=z['value_families'].astype(str)
    scores=[]
    for fi,family in enumerate(families):
     for ai,arm in enumerate(arms):
      for pi,path in enumerate(('baseline','none_every_step','first_A','first_B','first_AB','persistent_A','persistent_B','persistent_AB','persistent_delta0_AB')):
       perseed=[];counts=[]
       for s in range(5):
        values=q[fi,s,ai,pi]
        selected=mask&np.isfinite(values)
        counts.append(int(selected.sum()))
        mm=[float(values[selected&(mids==m)].mean()) for m in sorted(set(mids)) if np.any(selected&(mids==m))]
        perseed.append(float(np.mean(mm)) if len(mm)==12 else None)
       d=dict(value_family=family,world_arm=arm,path=path,scope='common_both_games_with_path_supported',
              five_seed_match_macro_mean=float(np.mean(perseed)) if all(v is not None for v in perseed) else None,
              five_seed_match_macro_sd=float(np.std(perseed,ddof=1)) if all(v is not None for v in perseed) else None)
       d['supported_counts']=';'.join(map(str,counts))
       d.update({f'seed{s}':v for s,v in enumerate(perseed)});scores.append(d)
    table(O/'score_levels_by_seed.csv',scores)
    reference=rows(G/'joint_reference_by_seed.csv');reference_summary=[];groups=defaultdict(list)
    for r in reference:groups[(r['value_family'],r['world_arm'],r['joint_contrast'],r['scope'])].append(r)
    for (family,arm,contrast,scope),rr in groups.items():
     assert len(rr)==5
     vals=[float(r['strict12_match_macro']) for r in rr if r['strict12_match_macro']!='']
     reference_summary.append(dict(value_family=family,world_arm=arm,joint_contrast=contrast,scope=scope,
                                  supported_counts=';'.join(r['supported'] for r in sorted(rr,key=lambda r:int(r['seed']))),
                                  five_seed_match_macro_mean=float(np.mean(vals)) if len(vals)==5 else None,
                                  five_seed_match_macro_sd=float(np.std(vals,ddof=1)) if len(vals)==5 else None))
    table(O/'joint_reference_summary.csv',reference_summary)
    agreement=rows(G/'actor_order_agreement.csv');agroups=defaultdict(list)
    for r in agreement:
     label=r['comparison'];kind=r['kind']
     if '/D_coupled_ar/' not in label:continue
     if kind=='seed' and not label.startswith('CV_Direct_WithEarlyBCE/'):continue
     if kind=='duration' and not label.startswith('CV_Direct_WithEarlyBCE/'):continue
     if kind not in ('seed','duration','value_family'):continue
     agroups[kind].append(r)
    ag=[]
    for kind,rr in agroups.items():
     vals=[float(r['strict_order_agreement_fraction']) for r in rr if r['strict_order_agreement_fraction']!='']
     ag.append(dict(kind=kind,world_arm='D_coupled_ar',paired_comparisons=len(rr),
                    median_strict_order_agreement=float(np.median(vals)) if vals else None,
                    min_strict_order_agreement=min(vals) if vals else None,max_strict_order_agreement=max(vals) if vals else None,
                    min_both_nontie_count=min(int(r['both_nontie_scenes']) for r in rr),
                    max_both_nontie_count=max(int(r['both_nontie_scenes']) for r in rr),
                    interpretation='descriptive range across fixed pairs, not CI; exact machine ties reported separately in source'))
    table(O/'D_actor_order_stability.csv',ag)
