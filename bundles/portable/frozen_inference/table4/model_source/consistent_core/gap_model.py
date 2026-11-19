"""Minimal causal input-gap awareness for the independent pilot only.

No state imputation, position extrapolation, spatial-layer change, MDN change, or
new observation is introduced. Two zero-initialized, bias-free projections add:

* At each valid history frame: excess inter-valid-frame elapsed time beyond one
  nominal frame, and first-valid-frame delay from the window start (seconds).
* At the temporal output: trailing gap age (seconds), and missing-frame fraction.

Fully observed histories have identically zero time features. Never-observed
entities have zero features and remain excluded by the existing presence mask.

``valid`` retains the existing *model-usable input state* contract. The persistent
variant separately receives ``observation_valid`` and ``observation_age`` so an
autoregressive prediction can be model-usable without being misrepresented as a
new sensor measurement. This is an experimental extension and does not change
the original frozen production model.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .models import AgentTemporalEncoder, WorldModel, _last_observed, _teams, _validate


GAP_FEATURE_VERSION = 1
STEP_FEATURE_NAMES = ("excess_inter_valid_elapsed_seconds", "first_valid_prefix_delay_seconds")
TRAILING_FEATURE_NAMES = ("trailing_real_observation_age_seconds", "missing_frame_fraction")


def causal_gap_features(valid: Tensor, fps: float = 25., dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    """Per-entity features from the current history mask only, without filling.

    The step-t feature uses only mask indices <=t. For the first valid entry,
    previous-observation spacing is unknown, so excess spacing is zero and its
    known delay within this window is encoded separately. No missing entity is
    assigned a fabricated observation time, coordinate, or another agent's value.
    """
    if valid.ndim != 3 or valid.dtype != torch.bool or valid.shape[1] < 1:
        raise ValueError("valid must be bool [B,H,N] with nonempty history")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not dtype.is_floating_point:
        raise ValueError("gap feature dtype must be floating point")
    time = torch.arange(valid.shape[1], device=valid.device).view(1, -1, 1)
    last_so_far = torch.where(valid, time, -1).cummax(dim=1).values
    previous = torch.cat([torch.full_like(last_so_far[:, :1], -1), last_so_far[:, :-1]], dim=1)
    has_previous = previous >= 0
    excess_frames = torch.where(has_previous, (time-previous-1).clamp_min(0), 0)
    prefix_frames = torch.where(has_previous, 0, time)
    step = torch.stack([excess_frames.to(dtype)/fps, prefix_frames.to(dtype)/fps], dim=-1)
    step = torch.where(valid[..., None], step, 0.)
    last = last_so_far[:, -1]
    present = last >= 0
    age = (valid.shape[1]-1-last).clamp_min(0).to(dtype)/fps
    fraction = (~valid).to(dtype).mean(dim=1)
    trailing = torch.where(present[..., None], torch.stack([age, fraction], dim=-1), 0.)
    return {"step": step, "trailing": trailing, "present": present, "last_valid_index": last}


class GapAwareTemporalEncoder(AgentTemporalEncoder):
    """Original recurrent/attention layers plus two small time projections.

    ``observation_valid`` tracks measured states independently from ``valid``,
    which also admits generated autoregressive states. ``observation_age`` is
    carried across rollout steps instead of being reset by a generated state.
    """

    def __init__(self, hidden_dim: int, use_temporal_attention: bool = True, fps: float = 25.):
        super().__init__(hidden_dim, use_temporal_attention)
        self.fps = float(fps)
        self.step_gap_projection = nn.Linear(2, hidden_dim, bias=False)
        self.trailing_gap_projection = nn.Linear(2, hidden_dim, bias=False)
        nn.init.zeros_(self.step_gap_projection.weight)
        nn.init.zeros_(self.trailing_gap_projection.weight)

    def forward(self, normalized: Tensor, valid: Tensor, teams: Tensor, *,
                observation_valid: Tensor | None = None,
                observation_age: Tensor | None = None,
                use_gap_features: bool = True) -> Tensor:
        b, t, n, _ = normalized.shape
        observed = valid if observation_valid is None else observation_valid
        if observed.shape != valid.shape or observed.dtype != torch.bool:
            raise ValueError("observation_valid must be bool with the same shape as valid")
        features = causal_gap_features(observed, self.fps, normalized.dtype)
        present = valid.any(dim=1)
        if observation_age is not None:
            if observation_age.shape != (b, n) or not torch.isfinite(observation_age).all() or (observation_age < 0).any():
                raise ValueError("observation_age must be finite nonnegative [B,N] seconds")
            features["trailing"] = torch.stack([observation_age.to(normalized.dtype),
                                                 (~observed).to(normalized.dtype).mean(dim=1)], dim=-1)
        if not use_gap_features:
            features["step"] = torch.zeros_like(features["step"])
            features["trailing"] = torch.zeros_like(features["trailing"])
        features["present"] = present
        clean = torch.where(valid[..., None], normalized, 0.)
        embeddings = self.mlp(clean) + self.type_embedding(teams)[:, None]
        embeddings = embeddings + self.step_gap_projection(features["step"])
        hidden = normalized.new_zeros((b*n, self.hidden_dim))
        sequence = []
        for step in range(t):
            proposed = self.gru(embeddings[:, step].reshape(b*n, -1), hidden)
            hidden = torch.where(valid[:, step].reshape(b*n, 1), proposed, hidden)
            sequence.append(hidden.reshape(b, n, -1))
        if self.temporal_attention is None:
            encoded = hidden.reshape(b, n, -1)
        else:
            states = torch.stack(sequence, dim=1)
            logits = self.temporal_attention(states).squeeze(-1)
            weights = logits.masked_fill(~valid, -1e9).softmax(dim=1)*valid
            weights = weights/weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            encoded = (states*weights[..., None]).sum(dim=1)
        encoded = encoded + self.trailing_gap_projection(features["trailing"])
        return torch.where(present[..., None], encoded, 0.)


class GapAwareWorldModel(WorldModel):
    """WorldModel with explicit gap times; common parameter names are unchanged.

    Given the same seed, all pre-existing parameters equal WorldModel's initial
    values. Only temporal.step_gap_projection.weight and
    temporal.trailing_gap_projection.weight are new, and both start at zero.
    ``model_config`` fully reconstructs this class for a version-aware loader.
    """

    def __init__(self, hidden_dim: int = 128, num_mixtures: int = 5,
                 use_temporal_attention: bool = True, use_spatial: bool = True,
                 deterministic: bool = False, neighborhood_radius: float = 20.,
                 fps: float = 25., gap_feature_version: int = GAP_FEATURE_VERSION):
        if gap_feature_version != GAP_FEATURE_VERSION:
            raise ValueError(f"Unsupported gap feature version: {gap_feature_version}")
        super().__init__(hidden_dim=hidden_dim, num_mixtures=num_mixtures,
                         use_temporal_attention=use_temporal_attention, use_spatial=use_spatial,
                         deterministic=deterministic, neighborhood_radius=neighborhood_radius, fps=fps)
        temporal = GapAwareTemporalEncoder(hidden_dim, use_temporal_attention, fps)
        loaded = temporal.load_state_dict(self.temporal.state_dict(), strict=False)
        expected = {"step_gap_projection.weight", "trailing_gap_projection.weight"}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise RuntimeError("Unexpected common temporal parameter mapping")
        self.temporal = temporal
        self.model_config["gap_feature_version"] = GAP_FEATURE_VERSION


class PersistentGapAwareWorldModel(GapAwareWorldModel):
    """Gap model carrying time since the last real observation through AR feedback."""

    def __init__(self, *args, observation_provenance_version: int = 1, **kwargs):
        if observation_provenance_version != 1:
            raise ValueError("Unsupported observation provenance version")
        super().__init__(*args, **kwargs)
        self.model_config["observation_provenance_version"] = observation_provenance_version

    def forward(self, history: Tensor, valid: Tensor, teams: Tensor, *,
                observation_valid: Tensor, observation_age: Tensor,
                use_gap_features: bool = True) -> dict[str, Tensor]:
        _validate(history, valid)
        b, _, n, _ = history.shape
        types = _teams(teams, b, n, history.device)
        clean = torch.where(valid[..., None], history, 0.0)
        last, present, _ = _last_observed(clean, valid)
        encoded = self.temporal(self.normalize(clean), valid, types,
                                observation_valid=observation_valid, observation_age=observation_age,
                                use_gap_features=use_gap_features)
        if self.spatial is not None:
            encoded = self.spatial(encoded, last[..., :2], present)
        base = self.normalize(last)
        if self.deterministic:
            return {"mean": base + self.point_head(encoded), "present": present}
        return {
            "logits": self.logits_head(encoded),
            "loc": base[..., None, :] + self.loc_head(encoded).reshape(b, n, self.num_mixtures, 4),
            "log_scale": self.log_scale_head(encoded).reshape(b, n, self.num_mixtures, 4).clamp(-7., 1.5),
            "present": present,
        }

def initial_observation_age(observation_valid: Tensor, fps: float) -> Tensor:
    if observation_valid.ndim != 3 or observation_valid.dtype != torch.bool or fps <= 0:
        raise ValueError("observation_valid must be bool [B,H,N] and fps positive")
    time = torch.arange(observation_valid.shape[1], device=observation_valid.device).view(1, -1, 1)
    last = torch.where(observation_valid, time, -1).amax(dim=1)
    present = last >= 0
    age = (observation_valid.shape[1] - 1 - last.clamp_min(0)).to(torch.float32) / float(fps)
    return torch.where(present, age, torch.zeros_like(age))
