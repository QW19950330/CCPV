"""Exact old native-prefix/feature methods; import is NumPy-only, no I/O."""
from dataclasses import dataclass
import numpy as np
N=23
H=10
F=20
DT=.1
TEAMS=np.array([0]*11+[1]*11+[2],dtype=np.int64)

@dataclass(frozen=True)
class HistoryBatch:
    """Only fields causally available at t0. Labels/future arrays are absent."""
    states: np.ndarray                    # native m,m/s; [B,10,23,4]
    valid: np.ndarray                     # model-usable fullstate [B,10,23]
    position_valid: np.ndarray            # native/model target xy usable
    real_observation_valid: np.ndarray    # measured, not asof extrapolation
    observation_age_s: np.ndarray         # last measured fullstate age [B,23]
    teams: np.ndarray                     # [23], literal TEAMS
    entity_ids: np.ndarray                # [B,23], source strings; empty pads
    anchor_id: np.ndarray                 # [B]
    fixed_team_A_index: np.ndarray        # [B] original t0 home0 or away1
    pitch_length_m: np.ndarray            # [B], reviewed source metadata
    pitch_width_m: np.ndarray             # [B]
    native_to_model_sign: np.ndarray      # [B,2], reviewed +/-1, no fitted flip
    supported: np.ndarray                 # [B] prefix-only support decision
    unsupported_reason: np.ndarray        # [B] explicit, no positive rescue

    def validate(self):
        s = np.asarray(self.states)
        b = len(self.anchor_id)
        if s.shape != (b, H, N, 4) or s.dtype != np.float32:
            raise ValueError('Need float32 native history [B,10,23,4], never a summary')
        for name in ('valid', 'position_valid', 'real_observation_valid'):
            a = np.asarray(getattr(self, name))
            if a.shape != (b, H, N) or a.dtype != bool:
                raise ValueError(name + ' must be bool[B,10,23]')
        if (self.real_observation_valid & ~self.valid).any():
            raise ValueError('Real observed fullstate cannot be model-invalid')
        if (self.valid & ~self.position_valid).any():
            raise ValueError('A usable fullstate needs usable xy')
        if not np.isfinite(s[self.valid]).all():
            raise ValueError('Nonfinite usable native state')
        if not np.array_equal(self.teams, TEAMS):
            raise ValueError('Need home0×11,away1×11,ball2; A is separate')
        if self.entity_ids.shape != (b, N):
            raise ValueError('Fixed source identity slots required')
        if len(set(self.anchor_id.tolist())) != b:
            raise ValueError('Duplicate anchor IDs')
        if self.observation_age_s.shape != (b, N):
            raise ValueError('Need last real-observation ages [B,23]')
        present = self.valid.any(1)
        if not np.isfinite(self.observation_age_s).all() or (self.observation_age_s < 0).any():
            raise ValueError('Real ages must remain finite and nonnegative')
        if (self.observation_age_s[~present] != 0).any():
            raise ValueError('Never usable entities have zero age and excluded presence')
        for name in ('pitch_length_m', 'pitch_width_m', 'fixed_team_A_index',
                     'supported', 'unsupported_reason'):
            if np.asarray(getattr(self, name)).shape != (b,):
                raise ValueError('Wrong row axis for ' + name)
        if self.supported.dtype != bool or self.native_to_model_sign.shape != (b, 2):
            raise ValueError('Wrong support/sign shape')
        ix = self.supported
        if not np.isin(self.fixed_team_A_index[ix], [0, 1]).all():
            raise ValueError('Supported rows require fixed source t0 team A')
        if not np.isfinite(self.pitch_length_m[ix]).all() or not np.isfinite(self.pitch_width_m[ix]).all():
            raise ValueError('Missing reviewed pitch dimensions')
        if (self.pitch_length_m[ix] <= 0).any() or (self.pitch_width_m[ix] <= 0).any():
            raise ValueError('Invalid reviewed pitch dimensions')
        if not np.isin(self.native_to_model_sign[ix], [-1., 1.]).all():
            raise ValueError('Only reviewed orthogonal axis signs; no fitted transform')
        for row in np.flatnonzero(ix):
            ids = self.entity_ids[row]
            nonempty = ids[ids != '']
            if len(set(nonempty.tolist())) != len(nonempty):
                raise ValueError('Source identity appears in multiple slots')
            if ids[22].casefold() != 'ball' or (present[row] & (ids == '')).any():
                raise ValueError('Ball/padding identity is not explicit')
            if not present[row, 22] or not present[row, :11].any() or not present[row, 11:22].any():
                raise ValueError('Supported bridge requires ball and both teams')
        return self


def model_scale(batch: HistoryBatch):
    return np.stack([105. / batch.pitch_length_m, 68. / batch.pitch_width_m], 1) * batch.native_to_model_sign


def to_model_coordinates(batch: HistoryBatch, indices):
    indices = np.asarray(indices, np.int64)
    scale = model_scale(batch)[indices]
    factor = np.concatenate((scale, scale), 1).astype(np.float32)
    clean = np.where(batch.valid[indices, ..., None], batch.states[indices], 0.)
    return (clean * factor[:, None, None]).astype(np.float32)


def to_native_coordinates(model_states, batch: HistoryBatch, indices):
    indices = np.asarray(indices, np.int64)
    scale = model_scale(batch)[indices]
    factor = np.concatenate((scale, scale), 1).astype(np.float32)
    return (np.asarray(model_states, np.float32) / factor[:, None, None]).astype(np.float32)


def current_anchor_numpy(states, valid):
    """Exact causal contract of old current_anchor, without importing Torch."""
    time = np.arange(states.shape[1])[None, :, None]
    last_idx = np.where(valid, time, -1).max(1)
    present = last_idx >= 0
    rows = np.arange(len(states))[:, None]
    nodes = np.arange(states.shape[2])[None]
    last = states[rows, last_idx.clip(0), nodes].copy()
    # Match frozen current_anchor's float32 division byFPS, not repeated/mul dt.
    age = (states.shape[1] - 1 - last_idx.clip(0)).astype(np.float32) / np.float32(10.)
    last[..., :2] += age[..., None] * last[..., 2:]
    return np.where(present[..., None], last, 0.), present


def analytic_cv_native(batch: HistoryBatch, indices):
    """CV with the same 10-frame fullstate history/current-anchor as D."""
    indices = np.asarray(indices, np.int64)
    states = np.where(batch.valid[indices, ..., None], batch.states[indices], 0.)
    base, present = current_anchor_numpy(states, batch.valid[indices])
    elapsed = (np.arange(1, F + 1, dtype=np.float32) / np.float32(10.))[None, :, None, None]
    xy = base[:, None, :, :2] + elapsed * base[:, None, :, 2:]
    v = np.broadcast_to(base[:, None, :, 2:], xy.shape)
    output = np.concatenate((xy, v), -1)
    output = np.where(present[:, None, :, None], output, 0.)
    return output.astype(np.float32), np.broadcast_to(present[:, None], (len(indices), F, N)).copy()


def snapshot(xy, present, ids, fixed_A):
    """Use stable IDs and native home/away roster; fixed A never tracks possession."""
    A, opponent = int(fixed_A), 1 - int(fixed_A)
    players = {}
    for team, slots in ((0, range(11)), (1, range(11, 22))):
        players[team] = {str(ids[j]): np.asarray(xy[j], np.float64)
                         for j in slots if present[j] and ids[j] and np.isfinite(xy[j]).all()}
    ball = np.asarray(xy[22], np.float64) if present[22] and np.isfinite(xy[22]).all() else None
    return dict(ball=ball, players=players), A, opponent


def pair_features(previous, current, A, opponent):
    """Literal old native dictionary: current/backward .2s common-ID cohort."""
    x = np.full(10, np.nan)
    common = {t: sorted(set(previous['players'].get(t, {})) & set(current['players'].get(t, {})))
              for t in (A, opponent)}
    b0, b1 = previous['ball'], current['ball']
    if b1 is not None:
        x[:2] = np.abs(b1)
    for team, start in ((A, 2), (opponent, 4)):
        if b1 is not None and common[team]:
            xy = np.array([current['players'][team][i] for i in common[team]])
            d = np.linalg.norm(xy - b1, axis=-1)
            x[start:start + 2] = [d.min(), d.mean()]
        if common[team]:
            v = np.array([(current['players'][team][i] - previous['players'][team][i]) / .2
                          for i in common[team]])
            x[8 if team == A else 9] = np.linalg.norm(v, axis=-1).mean()
    if b0 is not None and b1 is not None:
        x[6:8] = np.sign(b1) * (b1 - b0) / .2
    return x


def decoder_summary(snapshots, A, opponent):
    """12 ball positions[-.2,0,.2,...2]; no substitution of velocity heads."""
    z = np.full(8, np.nan)
    b = np.array([s['ball'] if s['ball'] is not None else [np.nan, np.nan] for s in snapshots])
    v = np.diff(b, axis=0) / .2
    if np.isfinite(v).all():
        speed = np.linalg.norm(v, axis=-1)
        z[:2] = [speed.mean(), speed.max()]
        z[2] = np.linalg.norm(np.diff(v, axis=0) / .2, axis=-1).max()
    if np.isfinite(b[[1, -1]]).all():
        z[3] = np.linalg.norm(b[-1] - b[1])
    for team, start in ((A, 4), (opponent, 6)):
        d = [min(np.linalg.norm(p - s['ball']) for p in s['players'].get(team, {}).values())
             if s['players'].get(team) and s['ball'] is not None else np.nan for s in snapshots[1:]]
        if np.isfinite(d).all():
            z[start:start + 2] = [np.min(d), np.mean(d)]
    return z


def features_from_native_paths(batch: HistoryBatch, indices, generated, generated_valid, X0):
    indices = np.asarray(indices, np.int64)
    if generated.shape != (len(indices), F, N, 4) or generated_valid.shape != (len(indices), F, N):
        raise ValueError('Need full 20-step joint path before decoder compression')
    if X0.shape != (len(batch.anchor_id), 10):
        raise ValueError('Literal old X0 must retain every native row')
    x2, z = [], []
    for j, row in enumerate(indices):
        ids, A = batch.entity_ids[row], batch.fixed_team_A_index[row]
        # History t=-.2 at index7 and t0 at index9 are exact 25Hz-grid targets.
        snaps = [snapshot(batch.states[row, t, :, :2], batch.position_valid[row, t], ids, A)[0]
                 for t in (7, 9)]
        snaps.extend(snapshot(generated[j, t, :, :2], generated_valid[j, t], ids, A)[0]
                     for t in range(1, F, 2))
        x2.append(pair_features(snaps[-2], snaps[-1], int(A), 1 - int(A)))
        z.append(np.r_[X0[row], decoder_summary(snaps, int(A), 1 - int(A))])
    return np.asarray(x2), np.asarray(z)


def union_features(X0, old_Z_cv, generated_Z, X2_generated):
    """Fixed same-information dictionaries: E26, Direct36, with exact X0 reuse.

    The ten terminal X2 variables are not recoverable from the eight temporal
    speed/distance summaries. Their mere similar names are not duplication.
    """
    rows = len(X0)
    if (X0.shape != (rows, 10) or old_Z_cv.shape != (rows, 18)
            or generated_Z.shape != (rows, 18) or X2_generated.shape != (rows, 10)):
        raise ValueError('Wrong fixed feature dictionary shape')
    if not np.array_equal(X0, old_Z_cv[:, :10], equal_nan=True):
        raise ValueError('Old CV must use literal frozen native X0')
    if not np.array_equal(X0, generated_Z[:, :10], equal_nan=True):
        raise ValueError('Generated decoder must use same literal frozen X0')
    early26 = np.concatenate((X0, old_Z_cv[:, 10:], generated_Z[:, 10:]), 1)
    direct36 = np.concatenate((early26, X2_generated), 1)
    return early26, direct36
