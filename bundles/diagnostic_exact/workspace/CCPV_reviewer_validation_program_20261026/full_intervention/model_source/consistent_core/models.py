"""Models reconstructed from the manuscript, with explicit missingness and units.

States passed to public APIs are x/y in centered pitch metres and vx/vy in m/s.
MDN components are conditionally independent four-dimensional diagonals per agent;
the recurrent and graph encoders condition their parameters on all valid agents.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _teams(teams: Tensor, batch: int, agents: int, device: torch.device) -> Tensor:
    result = torch.as_tensor(teams, dtype=torch.long, device=device)
    if result.ndim == 1:
        result = result.unsqueeze(0).expand(batch, -1)
    if result.shape != (batch, agents) or ((result < 0) | (result > 2)).any():
        raise ValueError("teams must be [N] or [B,N], with types 0, 1, 2")
    return result


def _validate(history: Tensor, valid: Tensor) -> None:
    if history.ndim != 4 or history.shape[-1] != 4:
        raise ValueError("states must have shape [B,T,N,4]")
    if valid.shape != history.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B,T,N], True=observed")
    if history.shape[1] == 0:
        raise ValueError("history cannot be empty")
    if not torch.isfinite(history[valid]).all():
        raise ValueError("observed states must be finite")


def _last_observed(states: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return each agent's last observed state, presence flag and time index."""
    b, t, n, d = states.shape
    indices = torch.arange(t, device=states.device).view(1, t, 1)
    last = torch.where(valid, indices, -1).amax(dim=1)
    gather = last.clamp_min(0).view(b, 1, n, 1).expand(-1, -1, -1, d)
    value = states.gather(1, gather).squeeze(1)
    present = last >= 0
    return torch.where(present[..., None], value, 0.0), present, last


def forecast_validity(history_valid: Tensor) -> Tensor:
    """Usable predicted entities, distinct from observed future ground truth.

    Preserve any entity observed within the input history through autoregression.
    Its predicted state becomes a model input; this never makes an unseen target
    into a real observation. Never-observed entities remain masked throughout.
    """
    if history_valid.ndim != 3 or history_valid.dtype != torch.bool:
        raise ValueError("history_valid must be bool [B,H,N]")
    return history_valid.any(dim=1)


class PhysicalModel(nn.Module):
    """Shared normalization, loss and explicit sampling interface."""

    def __init__(self, fps: float = 25.0) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.register_buffer("state_scale", torch.tensor([105., 68., 10.5, 10.5]))
        self.register_buffer("state_offset", torch.tensor([.5, .5, 0., 0.]))

    def normalize(self, state: Tensor) -> Tensor:
        return state / self.state_scale + self.state_offset

    def denormalize(self, state: Tensor) -> Tensor:
        return (state - self.state_offset) * self.state_scale

    def make_noise(self, batch_size: int, num_agents: int = 23,
                   device=None, dtype=None, generator=None) -> dict[str, Tensor]:
        device = self.state_scale.device if device is None else device
        dtype = self.state_scale.dtype if dtype is None else dtype
        return {
            "mixture_uniform": torch.rand((batch_size, num_agents), device=device,
                                           dtype=dtype, generator=generator),
            "normal": torch.randn((batch_size, num_agents, 4), device=device,
                                   dtype=dtype, generator=generator),
        }

    sample_noise = make_noise

    def nll(self, output: Mapping[str, Tensor], target: Tensor,
            target_valid: Tensor) -> Tensor:
        """Mean per-observed-agent NLL (or feature-mean MSE for point models).

        Gaussian normalizers and every output feature are included. Missing targets
        are replaced before arithmetic and never contribute to loss or gradients.
        """
        if target.ndim != 3 or target.shape[-1] != 4:
            raise ValueError("target must be [B,N,4]")
        if target_valid.shape != target.shape[:-1]:
            raise ValueError("target_valid must be [B,N]")
        target_valid = target_valid.bool()
        if not torch.isfinite(target[target_valid]).all():
            raise ValueError("observed target states must be finite")
        clean = torch.where(target_valid[..., None], target, 0.0)
        y = self.normalize(clean)
        if "logits" in output:
            loc, log_scale = output["loc"], output["log_scale"]
            z = (y.unsqueeze(-2) - loc) * torch.exp(-log_scale)
            log_density = (-.5 * z.square() - log_scale
                           - .5 * math.log(2 * math.pi)).sum(-1)
            log_prob = torch.logsumexp(F.log_softmax(output["logits"], -1)
                                       + log_density, dim=-1)
            per_agent = -log_prob
        else:
            per_agent = (output["mean"] - y).square().mean(-1)
        selected = torch.where(target_valid, per_agent, 0.0)
        return selected.sum() / target_valid.sum().clamp_min(1)

    loss = nll

    def predict_next(self, history: Tensor, valid: Tensor, teams: Tensor,
                     mode: str = "mean", noise: Mapping[str, Tensor] | None = None,
                     generator=None) -> Tensor:
        """Predict physical next states. 'mean' is mixture mean, never best-of-K.

        To pair stochastic branches, pass the same two tensors in ``noise``:
        mixture_uniform [B,N] in [0,1), normal [B,N,4]. A changed mixture may
        select a different component under the same uniform quantile.
        """
        if mode not in {"mean", "sample"}:
            raise ValueError("mode must be 'mean' or 'sample'")
        output = self(history, valid, teams)
        if "logits" not in output:
            return self.denormalize(output["mean"])
        weights = output["logits"].softmax(-1)
        if mode == "mean":
            normalized = (weights[..., None] * output["loc"]).sum(-2)
        else:
            b, n, k = weights.shape
            if noise is None:
                noise = self.make_noise(b, n, history.device, history.dtype, generator)
            u = torch.as_tensor(noise["mixture_uniform"], device=history.device,
                                dtype=history.dtype)
            eps = torch.as_tensor(noise["normal"], device=history.device,
                                  dtype=history.dtype)
            if u.shape != (b, n) or eps.shape != (b, n, 4):
                raise ValueError("noise shapes must be mixture_uniform[B,N], normal[B,N,4]")
            if not torch.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
                raise ValueError("mixture uniforms must be finite in [0,1)")
            if not torch.isfinite(eps).all():
                raise ValueError("normal noise must be finite")
            index = (u[..., None] > weights.cumsum(-1)).sum(-1).clamp_max(k-1)
            gather = index[..., None, None].expand(-1, -1, 1, 4)
            loc = output["loc"].gather(-2, gather).squeeze(-2)
            sigma = output["log_scale"].gather(-2, gather).squeeze(-2).exp()
            normalized = loc + sigma * eps
        return self.denormalize(normalized)


class AgentTemporalEncoder(nn.Module):
    """Shared two-layer MLP + type embedding + per-agent masked GRU."""

    def __init__(self, hidden_dim: int, use_temporal_attention: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_temporal_attention = use_temporal_attention
        self.mlp = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.temporal_attention = nn.Linear(hidden_dim, 1) if use_temporal_attention else None

    def forward(self, normalized: Tensor, valid: Tensor, teams: Tensor) -> Tensor:
        b, t, n, _ = normalized.shape
        clean = torch.where(valid[..., None], normalized, 0.0)
        embeddings = self.mlp(clean) + self.type_embedding(teams)[:, None, :, :]
        h = normalized.new_zeros((b*n, self.hidden_dim))
        sequence = []
        for step in range(t):
            proposed = self.gru(embeddings[:, step].reshape(b*n, -1), h)
            h = torch.where(valid[:, step].reshape(b*n, 1), proposed, h)
            sequence.append(h.reshape(b, n, -1))
        encoded = torch.stack(sequence, dim=1)
        if self.temporal_attention is None:
            return h.reshape(b, n, -1)
        logits = self.temporal_attention(encoded).squeeze(-1)
        weights = logits.masked_fill(~valid, -1e9).softmax(dim=1) * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return (encoded * weights[..., None]).sum(dim=1)


class RadiusMessagePassing(nn.Module):
    """One directed radius-graph message-passing layer with sum aggregation."""

    def __init__(self, hidden_dim: int, radius: float) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError("neighborhood_radius must be positive")
        self.radius = float(radius)
        self.message = nn.Sequential(nn.Linear(2*hidden_dim+3, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.update = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoded: Tensor, positions: Tensor, present: Tensor) -> Tensor:
        b, n, d = encoded.shape
        delta = positions[:, None, :, :] - positions[:, :, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        neighbors = (distance <= self.radius) & present[:, :, None] & present[:, None, :]
        neighbors &= ~torch.eye(n, device=encoded.device, dtype=torch.bool)[None]
        relative = torch.cat((delta/self.radius, (distance/self.radius)[..., None]), -1)
        source = encoded[:, :, None, :].expand(-1, -1, n, -1)
        dest = encoded[:, None, :, :].expand(-1, n, -1, -1)
        message = self.message(torch.cat((source, dest, relative), -1))
        summed = (message * neighbors[..., None]).sum(dim=2)
        updated = self.norm(encoded + self.update(torch.cat((encoded, summed), -1)))
        return torch.where(present[..., None], updated, 0.0)


class WorldModel(PhysicalModel):
    """CCPV dynamics. Residual means use the last observed normalized state.

    Residual parameterization and diagonal per-agent factorization are explicit
    implementation choices completing details absent from the submitted Methods.
    They impose no hard kinematic constraints; simulation owns such projections.
    """

    def __init__(self, hidden_dim: int = 128, num_mixtures: int = 5,
                 use_temporal_attention: bool = True, use_spatial: bool = True,
                 deterministic: bool = False, neighborhood_radius: float = 20.0,
                 fps: float = 25.0) -> None:
        super().__init__(fps)
        if hidden_dim < 1 or num_mixtures < 1:
            raise ValueError("hidden_dim and num_mixtures must be positive")
        self.num_mixtures = int(num_mixtures)
        self.deterministic = bool(deterministic)
        self.model_config = dict(hidden_dim=hidden_dim, num_mixtures=num_mixtures,
                                 use_temporal_attention=use_temporal_attention,
                                 use_spatial=use_spatial, deterministic=deterministic,
                                 neighborhood_radius=neighborhood_radius, fps=fps)
        self.temporal = AgentTemporalEncoder(hidden_dim, use_temporal_attention)
        self.spatial = RadiusMessagePassing(hidden_dim, neighborhood_radius) if use_spatial else None
        if deterministic:
            self.point_head = nn.Linear(hidden_dim, 4)
            nn.init.zeros_(self.point_head.weight)
            nn.init.zeros_(self.point_head.bias)
        else:
            self.logits_head = nn.Linear(hidden_dim, num_mixtures)
            self.loc_head = nn.Linear(hidden_dim, num_mixtures*4)
            self.log_scale_head = nn.Linear(hidden_dim, num_mixtures*4)
            nn.init.normal_(self.loc_head.weight, std=.001)
            nn.init.zeros_(self.loc_head.bias)
            nn.init.zeros_(self.log_scale_head.weight)
            nn.init.constant_(self.log_scale_head.bias, math.log(.05))

    def forward(self, history: Tensor, valid: Tensor, teams: Tensor) -> dict[str, Tensor]:
        _validate(history, valid)
        b, _, n, _ = history.shape
        types = _teams(teams, b, n, history.device)
        clean = torch.where(valid[..., None], history, 0.0)
        last, present, _ = _last_observed(clean, valid)
        encoded = self.temporal(self.normalize(clean), valid, types)
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


class ConstantVelocity(PhysicalModel):
    def __init__(self, fps: float = 25.0, **_ignored) -> None:
        super().__init__(fps)
        self.model_config = {"fps": fps}

    def _physical_prediction(self, history: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        _validate(history, valid)
        last, present, index = _last_observed(torch.where(valid[..., None], history, 0.0), valid)
        future = last.clone()
        elapsed = (history.shape[1] - index).clamp_min(1).to(history.dtype) / self.fps
        future[..., :2] += last[..., 2:] * elapsed[..., None]
        future = torch.where(present[..., None], future, 0.0)
        return future, present

    def forward(self, history: Tensor, valid: Tensor, teams: Tensor) -> dict[str, Tensor]:
        future, present = self._physical_prediction(history, valid)
        return {"mean": self.normalize(future), "present": present}

    def predict_next(self, history: Tensor, valid: Tensor, teams: Tensor,
                     mode: str = "mean", noise=None, generator=None) -> Tensor:
        if mode not in {"mean", "sample"}:
            raise ValueError("mode must be 'mean' or 'sample'")
        # Preserve physical precision for an analytic baseline instead of an
        # unnecessary float32 normalize/denormalize round trip near x/y=0.
        return self._physical_prediction(history, valid)[0]


class SocialLSTM(PhysicalModel):
    """Declared recurrent-neighborhood baseline, not the authors' official code.

    A shared per-agent LSTMCell consumes encoded state and the mean of neighbor
    hidden states within a metric radius at each history frame. This is a local
    pooled implementation, not the original paper's discretized social grid.
    """

    def __init__(self, hidden_dim: int = 128, neighborhood_radius: float = 20.,
                 fps: float = 25., **_ignored) -> None:
        super().__init__(fps)
        self.hidden_dim, self.radius = hidden_dim, float(neighborhood_radius)
        self.model_config = dict(hidden_dim=hidden_dim, neighborhood_radius=neighborhood_radius, fps=fps)
        self.encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.lstm = nn.LSTMCell(2*hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history: Tensor, valid: Tensor, teams: Tensor) -> dict[str, Tensor]:
        _validate(history, valid)
        b, t, n, _ = history.shape
        types = _teams(teams, b, n, history.device)
        clean = torch.where(valid[..., None], history, 0.0)
        encoded = self.encoder(self.normalize(clean)) + self.type_embedding(types)[:, None]
        h = history.new_zeros((b, n, self.hidden_dim))
        c = h.clone()
        eye = torch.eye(n, device=history.device, dtype=torch.bool)[None]
        for step in range(t):
            distance = torch.cdist(clean[:, step, :, :2], clean[:, step, :, :2])
            neighbors = ((distance <= self.radius) & ~eye
                         & valid[:, step, :, None] & valid[:, step, None, :])
            pool = torch.bmm(neighbors.to(h.dtype), h) / neighbors.sum(-1, keepdim=True).clamp_min(1)
            proposed_h, proposed_c = self.lstm(torch.cat((encoded[:, step], pool), -1).reshape(b*n, -1),
                                             (h.reshape(b*n, -1), c.reshape(b*n, -1)))
            mask = valid[:, step, :, None]
            h = torch.where(mask, proposed_h.reshape(b, n, -1), h)
            c = torch.where(mask, proposed_c.reshape(b, n, -1), c)
        last, present, _ = _last_observed(clean, valid)
        return {"mean": self.normalize(last) + self.head(h), "present": present}


class VanillaMHSA(PhysicalModel):
    """Point-forecast baseline: last agent queries attend all valid history tokens."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4,
                 fps: float = 25., **_ignored) -> None:
        super().__init__(fps)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.model_config = dict(hidden_dim=hidden_dim, num_heads=num_heads, fps=fps)
        self.encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history: Tensor, valid: Tensor, teams: Tensor) -> dict[str, Tensor]:
        _validate(history, valid)
        b, t, n, _ = history.shape
        types = _teams(teams, b, n, history.device)
        clean = torch.where(valid[..., None], history, 0.0)
        tokens = self.encoder(self.normalize(clean)) + self.type_embedding(types)[:, None]
        position = torch.arange(t, device=history.device, dtype=history.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.hidden_dim, 2, device=history.device,
                                      dtype=history.dtype) * (-math.log(10000.)/self.hidden_dim))
        time = history.new_zeros((t, self.hidden_dim))
        time[:, 0::2] = torch.sin(position*freq)
        time[:, 1::2] = torch.cos(position*freq[:time[:, 1::2].shape[-1]])
        tokens = tokens + time[None, :, None]
        last, present, _ = _last_observed(clean, valid)
        queries = self.encoder(self.normalize(last)) + self.type_embedding(types)
        keys = tokens.reshape(b, t*n, -1)
        padding = ~valid.reshape(b, t*n)
        # PyTorch attention cannot softmax an all-masked row. A zero sentinel is
        # enabled for that row and the final agent predictions are zeroed below.
        all_missing = padding.all(-1)
        padding = padding.clone()
        padding[all_missing, 0] = False
        keys = keys.clone()
        keys[all_missing] = 0.
        attended, _ = self.attention(queries, keys, keys, key_padding_mask=padding, need_weights=False)
        prediction = self.normalize(last) + self.head(self.norm(queries + attended))
        zero = self.normalize(torch.zeros_like(last))
        prediction = torch.where(present[..., None], prediction, zero)
        return {"mean": prediction, "present": present}


class ValueNetwork(PhysicalModel):
    """xT-aligned value with auxiliary xG supervision from supplied labels only.

    The returned scalar is the last valid frame's xT head, not an arbitrary sum
    of xT and xG. The xG head regularizes a shared representation and is supervised
    only on aligned shot targets. No geometry-generated proxy is constructed.
    """

    def __init__(self, hidden_dim: int = 128, neighborhood_radius: float = 20.,
                 fps: float = 25.) -> None:
        super().__init__(fps)
        self.model_config = dict(hidden_dim=hidden_dim, neighborhood_radius=neighborhood_radius, fps=fps)
        self.encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.spatial = RadiusMessagePassing(hidden_dim, neighborhood_radius)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.xt_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.xg_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def heads(self, rollout: Tensor, valid: Tensor, teams: Tensor,
              attacking_team: Tensor | int | None = None) -> dict[str, Tensor]:
        _validate(rollout, valid)
        b, t, n, _ = rollout.shape
        types = _teams(teams, b, n, rollout.device)
        if attacking_team is not None:
            attack = torch.as_tensor(attacking_team, dtype=torch.long, device=rollout.device).reshape(-1)
            if attack.numel() == 1:
                attack = attack.expand(b)
            if attack.shape != (b,) or ((attack < 0) | (attack > 1)).any():
                raise ValueError("attacking_team must be 0/1 scalar or [B]")
            types = torch.where(types == 2, 2, (types != attack[:, None]).long())
        clean = torch.where(valid[..., None], rollout, 0.0)
        encoded = self.encoder(self.normalize(clean)) + self.type_embedding(types)[:, None]
        spatial = self.spatial(encoded.reshape(b*t, n, -1), clean[..., :2].reshape(b*t, n, 2),
                               valid.reshape(b*t, n)).reshape(b, t, n, -1)
        frame = spatial.sum(dim=2) / valid.sum(-1, keepdim=True).clamp_min(1)
        frame_valid = valid.any(-1)
        hidden = frame.new_zeros((1, b, frame.shape[-1]))
        temporal = []
        for step in range(t):
            _, proposed = self.temporal(frame[:, step:step+1], hidden)
            hidden = torch.where(frame_valid[:, step][None, :, None], proposed, hidden)
            temporal.append(hidden.squeeze(0))
        features = torch.stack(temporal, dim=1)
        return {"xt": self.xt_head(features).squeeze(-1),
                "xg": self.xg_head(features).squeeze(-1), "valid": frame_valid}

    def forward(self, rollout: Tensor, valid: Tensor, teams: Tensor,
                attacking_team: Tensor | int | None = None) -> Tensor:
        heads = self.heads(rollout, valid, teams, attacking_team)
        indices = torch.arange(rollout.shape[1], device=rollout.device)[None]
        last = torch.where(heads["valid"], indices, -1).amax(-1)
        value = heads["xt"].gather(1, last.clamp_min(0)[:, None]).squeeze(1)
        return torch.where(last >= 0, value, 0.0)

    @staticmethod
    def supervised_loss(heads: Mapping[str, Tensor], xt: Tensor, xg: Tensor,
                        shot: Tensor, target_valid: Tensor, lambda_xt: float = 1.,
                        lambda_xg: float = 1.) -> dict[str, Tensor]:
        if lambda_xt < 0 or lambda_xg < 0 or lambda_xt + lambda_xg <= 0:
            raise ValueError("value loss weights must be nonnegative and not both zero")
        if any(x.shape != heads["xt"].shape for x in (xt, xg, shot, target_valid)):
            raise ValueError("value labels and heads must have matching [B,F] shape")
        mask = target_valid.bool() & heads["valid"]
        shot_mask = mask & shot.bool()
        if not torch.isfinite(xt[mask]).all() or not torch.isfinite(xg[shot_mask]).all():
            raise ValueError("valid value targets must be finite")
        if ((xt[mask] < 0) | (xt[mask] > 1)).any() or ((xg[shot_mask] < 0) | (xg[shot_mask] > 1)).any():
            raise ValueError("xT state targets and xG probabilities must lie in [0,1]")
        xt_clean = torch.where(mask, xt, 0.)
        xg_clean = torch.where(shot_mask, xg, 0.)
        xt_loss = torch.where(mask, (heads["xt"]-xt_clean).square(), 0.).sum() / mask.sum().clamp_min(1)
        xg_loss = torch.where(shot_mask, (heads["xg"]-xg_clean).square(), 0.).sum() / shot_mask.sum().clamp_min(1)
        return {"loss": lambda_xt*xt_loss + lambda_xg*xg_loss,
                "xt_loss": xt_loss, "xg_loss": xg_loss,
                "xt_count": mask.sum(), "xg_count": shot_mask.sum()}


def build_model(name: str, **kwargs) -> PhysicalModel:
    aliases = {"ccpv": {}, "world_model": {}, "no_spatial": {"use_spatial": False},
               "no_temporal_attention": {"use_temporal_attention": False},
               "mse": {"deterministic": True}, "no_mdn": {"deterministic": True}}
    name = name.lower().replace("-", "_")
    if name in aliases:
        return WorldModel(**(kwargs | aliases[name]))
    if name in {"cv", "constant_velocity"}:
        return ConstantVelocity(**kwargs)
    if name in {"social_lstm", "sociallstm"}:
        return SocialLSTM(**kwargs)
    if name in {"vanilla_mhsa", "mhsa"}:
        return VanillaMHSA(**kwargs)
    raise ValueError(f"Unknown model {name!r}")
