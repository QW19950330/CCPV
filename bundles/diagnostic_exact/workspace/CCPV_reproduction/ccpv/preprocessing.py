"""Optional causal reconstruction assumptions for Method 4.1 quality control.

The manuscript did not report its exact filter, window or acceleration bound.
These explicit options are auditable alternatives, not recovered original
settings. They never use a future observation or fill an unobserved entity.
"""
from __future__ import annotations

from typing import Any
import numpy as np

SPEED_HISTOGRAM_EDGES = np.array([0, .5, 1, 2, 3, 4, 5, 6, 8, 10.5, 15, 20, 30, 50, np.inf])
QUANTILE_LEVELS = [0.0, .25, .5, .75, .9, .95, .99, 1.0]


def summarize_speeds(speeds: np.ndarray) -> dict[str, Any]:
    """JSON-safe exact quantiles and a common histogram, including high tails."""
    values = np.asarray(speeds, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if (values < 0).any():
        raise ValueError("Speeds cannot be negative")
    counts, _ = np.histogram(values, bins=SPEED_HISTOGRAM_EDGES)
    return {"count": int(len(values)), "quantile_levels": QUANTILE_LEVELS,
            "quantiles_m_s": np.quantile(values, QUANTILE_LEVELS).tolist() if len(values) else [None] * len(QUANTILE_LEVELS),
            "histogram_edges_m_s": SPEED_HISTOGRAM_EDGES[:-1].tolist() + ["inf"],
            "histogram_counts": counts.tolist()}


def _validated_options(smooth_window: int, player_speed_cap: float | None,
                       player_accel_cap: float | None) -> dict[str, Any]:
    if isinstance(smooth_window, bool) or not isinstance(smooth_window, (int, np.integer)) or smooth_window < 1:
        raise ValueError("smooth_window must be a positive integer")
    for name, cap in [("player_speed_cap", player_speed_cap), ("player_accel_cap", player_accel_cap)]:
        if cap is not None and (not np.isfinite(cap) or cap <= 0):
            raise ValueError(f"{name} must be a positive finite physical bound or None")
    return {"smooth_window": int(smooth_window),
            "player_speed_cap": None if player_speed_cap is None else float(player_speed_cap),
            "player_accel_cap": None if player_accel_cap is None else float(player_accel_cap)}


def preprocess_positions(positions: np.ndarray, observed: np.ndarray, times: np.ndarray, *,
                          teams: np.ndarray | None = None, smooth_window: int = 1,
                          player_speed_cap: float | None = None,
                          player_accel_cap: float | None = None,
                          include_speed_samples: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Causal moving mean, finite differences, then optional kinematic projection.

    Input is [frames,entities,2] in meters with an observation mask and seconds.
    Missing runs reset the mean, previous position and acceleration state.
    First observations have no causal velocity and remain full-state masked.
    When caps are enabled, velocity is projected before integrating the next
    position. Only players (team 0/1) receive speed/acceleration caps; ball
    positions may be smoothed but ball speed is never bounded as a player.

    Acceleration cannot be bounded at the first finite-difference velocity of
    a run because no previous observed velocity exists. It is bounded from
    the next pair onward. The returned metadata makes this boundary explicit.
    """
    options = _validated_options(smooth_window, player_speed_cap, player_accel_cap)
    pos = np.asarray(positions, dtype=np.float64)
    obs = np.asarray(observed, dtype=bool)
    t = np.asarray(times, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[-1] != 2 or obs.shape != pos.shape[:2] or t.shape != (len(pos),):
        raise ValueError("Expected positions[F,N,2], observed[F,N], times[F]")
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Times must be finite and strictly increasing")
    if not np.isfinite(pos[obs]).all():
        raise ValueError("Observed positions must be finite")
    if teams is None:
        if pos.shape[1] != 23:
            raise ValueError("Explicit teams needed when entity count differs from 23")
        teams = np.array([0] * 11 + [1] * 11 + [2])
    team = np.asarray(teams)
    if team.shape != (pos.shape[1],) or not np.isin(team, [0, 1, 2]).all():
        raise ValueError("teams must have one 0/1/2 role for each entity")
    smoothed = np.zeros_like(pos)
    filtered = np.zeros_like(pos)
    velocity = np.zeros_like(pos)
    raw_velocity = np.zeros_like(pos)
    valid = np.zeros_like(obs)
    counts = {"smoothing_adjusted_observations": 0, "projected_position_observations": 0,
              "total_position_adjusted_observations": 0, "velocity_adjusted_observations": 0,
              "speed_clipped_steps": 0, "acceleration_clipped_steps": 0,
              "initial_velocity_unavailable_observations": 0,
              "initial_acceleration_unconstrained_steps": 0}
    cap_enabled = player_speed_cap is not None or player_accel_cap is not None
    # Consecutive array positions may still hide an unobserved time interval.
    # A larger-than-nominal time gap is a reset rather than an imputation bridge.
    # Running minimum uses only timestamps already reached. A full-sequence
    # median would allow a change to future timestamps to alter a past reset.
    causal_nominal_dt = np.r_[np.inf, np.minimum.accumulate(np.diff(t))] if len(t) > 1 else np.array([np.inf])
    for entity in range(pos.shape[1]):
        history: list[np.ndarray] = []
        previous_index = -2
        previous_velocity_known = False
        for step in np.flatnonzero(obs[:, entity]):
            contiguous = step == previous_index + 1 and (step == 0 or t[step] - t[step-1] <= 1.5 * causal_nominal_dt[step])
            if not contiguous:
                history = []
                previous_velocity_known = False
            history.append(pos[step, entity])
            if len(history) > smooth_window:
                history.pop(0)
            smoothed[step, entity] = np.mean(history, axis=0)
            if not np.allclose(smoothed[step, entity], pos[step, entity], rtol=0, atol=1e-9):
                counts["smoothing_adjusted_observations"] += 1
            if not contiguous:
                filtered[step, entity] = smoothed[step, entity]
                counts["initial_velocity_unavailable_observations"] += 1
                previous_index = int(step)
                continue
            dt = t[step] - t[previous_index]
            raw_velocity[step, entity] = (pos[step, entity] - pos[previous_index, entity]) / dt
            candidate = (smoothed[step, entity] - smoothed[previous_index, entity]) / dt
            player = team[entity] in [0, 1]
            if player and player_speed_cap is not None:
                speed = np.linalg.norm(candidate)
                if speed > player_speed_cap:
                    candidate *= player_speed_cap / speed
                    counts["speed_clipped_steps"] += 1
            if player and player_accel_cap is not None:
                if previous_velocity_known:
                    delta = candidate - velocity[previous_index, entity]
                    change = np.linalg.norm(delta)
                    maximum = player_accel_cap * dt
                    if change > maximum:
                        candidate = velocity[previous_index, entity] + delta * (maximum / change)
                        counts["acceleration_clipped_steps"] += 1
                else:
                    counts["initial_acceleration_unconstrained_steps"] += 1
            velocity[step, entity] = candidate
            if player and cap_enabled:
                filtered[step, entity] = filtered[previous_index, entity] + candidate * dt
            else:
                filtered[step, entity] = smoothed[step, entity]
            if not np.allclose(filtered[step, entity], smoothed[step, entity], rtol=0, atol=1e-9):
                counts["projected_position_observations"] += 1
            if not np.allclose(filtered[step, entity], pos[step, entity], rtol=0, atol=1e-9):
                counts["total_position_adjusted_observations"] += 1
            if not np.allclose(candidate, raw_velocity[step, entity], rtol=0, atol=1e-9):
                counts["velocity_adjusted_observations"] += 1
            valid[step, entity] = True
            previous_velocity_known = True
            previous_index = int(step)
    state = np.concatenate([filtered, velocity], axis=-1).astype(np.float32)
    state[~valid] = 0
    role_samples: dict[str, dict[str, np.ndarray]] = {}
    role_stats: dict[str, dict[str, Any]] = {}
    for role, selection in [("players", team != 2), ("ball", team == 2)]:
        role_mask = valid[:, selection]
        raw_speed = np.linalg.norm(raw_velocity[:, selection], axis=-1)[role_mask]
        filtered_speed = np.linalg.norm(velocity[:, selection], axis=-1)[role_mask]
        role_samples[role] = {"raw": raw_speed, "filtered": filtered_speed}
        role_stats[role] = {"raw": summarize_speeds(raw_speed), "filtered": summarize_speeds(filtered_speed)}
    diagnostics = {"enabled": True, "parameters": options,
                   "implementation": "causal_coordinate_mean_then_backward_velocity_then_player_projection_and_integration_v1",
                   "reconstruction_assumption": True,
                   "assumption_note": "Exact smoothing and acceleration settings were not reported in the manuscript; these are explicit reconstruction options.",
                   "missing_observations_imputed": 0,
                   "first_velocity_policy": "unavailable at each observed-run start; full state masked",
                   "first_acceleration_policy": "not constrained until two finite-difference velocities exist",
                   "provided_velocity_policy": "when this filter is enabled velocities are recomputed from positions for consistency",
                   "raw_speed_source": "causal backward differences of original observed positions, using the same valid transitions",
                   "counts": counts, "speed_m_s": role_stats}
    if include_speed_samples:
        diagnostics["_speed_samples"] = role_samples
    return state, valid, diagnostics
