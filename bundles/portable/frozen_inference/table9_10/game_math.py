"""Original exact deterministic game/rank arithmetic only; no fit or readout API."""
import numpy as np

WORLD_ARMS = ('A_raw4d', 'C_coupled_tf', 'D_coupled_ar', 'CV')

VALUE_FAMILIES = ('CV_Direct_WithEarlyBCE', 'D_Direct_WithEarlyBCE')

PATHS = ('baseline', 'none_every_step', 'first_A', 'first_B', 'first_AB',
         'persistent_A', 'persistent_B', 'persistent_AB', 'persistent_delta0_AB')

GAMES = {'first': ('baseline', 'first_A', 'first_B', 'first_AB'),
         'persistent': ('baseline', 'persistent_A', 'persistent_B', 'persistent_AB')}

def exact_game(v):
    """Last axis empty,A,B,AB; K2 standard Shapley and half interaction."""
    v = np.asarray(v, np.float64)
    if v.shape[-1] != 4:
        raise ValueError('Need actual four coalition readouts')
    z, a, b, ab = np.moveaxis(v, -1, 0)
    phi_a = .5 * ((a - z) + (ab - b))
    phi_b = .5 * ((b - z) + (ab - a))
    interaction = ab - a - b + z
    return dict(delta_A=a-z, delta_B=b-z, delta_AB=ab-z, phi_A=phi_a, phi_B=phi_b,
                interaction_difference=interaction, shapley_pair_half=.5*interaction,
                efficiency_error=phi_a+phi_b-(ab-z))

def average_rank(values):
    """Average ties; deterministic no effect-based winner selection."""
    x = np.asarray(values, np.float64)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return result

def rank_correlation(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return None, int(good.sum())
    a, b = average_rank(np.asarray(x)[good]), average_rank(np.asarray(y)[good])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, int(good.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(good.sum())

def actor_order(phi_a, phi_b):
    """Within-scene ordering of the TWO fixed actors; exact float-zero ties.

    No tolerance/threshold is selected from effects. This is distinct from
    ranking scenes by delta_AB. Undefined values keep a NaN order.
    """
    gap = np.asarray(phi_a, np.float64) - np.asarray(phi_b, np.float64)
    return gap, np.sign(gap)

def actor_agreement(left, right, mask):
    left, right, mask = np.asarray(left), np.asarray(right), np.asarray(mask, bool)
    if left.shape != right.shape or left.shape != mask.shape:
        raise ValueError('Actor ordering alignment mismatch')
    if not np.isin(left[mask], [-1., 0., 1.]).all() or not np.isin(right[mask], [-1., 0., 1.]).all():
        raise ValueError('Undefined actor ordering on requested paired support')
    a,b=left[mask],right[mask]
    lt,rt=a==0,b==0
    non=~lt&~rt
    n=int(len(a)); nn=int(non.sum())
    same=int(np.sum((a==b)&non))
    allsame=int(np.sum(a==b))
    return dict(paired_scenes=n, both_nontie_scenes=nn, both_tie_scenes=int(np.sum(lt&rt)),
                left_only_tie_scenes=int(np.sum(lt&~rt)),right_only_tie_scenes=int(np.sum(~lt&rt)),
                at_least_one_tie_scenes=int(np.sum(lt|rt)), strict_order_agree_scenes=same,
                strict_order_reverse_scenes=int(np.sum((a!=b)&non)),
                order_agreement_including_both_ties_scenes=allsame,
                strict_order_agreement_fraction=same/nn if nn else None,
                agreement_including_both_ties_fraction=allsame/n if n else None)

V0_PATHS=('v0_empty','v0_A','v0_B','v0_AB')

def exact_v0_game(q):
 q=np.asarray(q,float)
 if q.shape[-1]!=4:raise ValueError('Need genuine empty/A/B/AB paths')
 e,a,b,ab=np.moveaxis(q,-1,0)
 pa=.5*((a-e)+(ab-b));pb=.5*((b-e)+(ab-a));interaction=ab-a-b+e
 return dict(delta_A=a-e,delta_B=b-e,delta_AB=ab-e,phi_A=pa,phi_B=pb,interaction_difference=interaction,
             shapley_pair_half=.5*interaction,efficiency_error=pa+pb-(ab-e))
