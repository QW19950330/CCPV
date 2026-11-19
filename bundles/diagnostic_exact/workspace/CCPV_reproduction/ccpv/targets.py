"""Train-only event targets with explicit provenance and clock alignment.

No event model is constructed from tracking geometry alone.  The xT grid is
estimated from labelled passes/carries/shots in training matches; the xG model
uses labelled training shots or supplied event xG values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

MOVE_TYPES = {"pass", "carry"}


def _bool_column(values: pd.Series, name: str) -> np.ndarray:
    lookup = {"true": True, "1": True, "1.0": True, "yes": True,
              "false": False, "0": False, "0.0": False, "no": False}
    out = values.astype(str).str.lower().map(lookup)
    if out.isna().any():
        raise ValueError(f"{name} must contain explicit booleans/0/1, without missing labels")
    return out.to_numpy(dtype=bool)


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "time_s", "type", "x", "y", "team", "attack_direction"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event CSV missing columns: {sorted(missing)}")
    e = events.copy()
    e["match_id"] = e["match_id"].astype(str)
    if "clip_id" in e:
        e["clip_id"] = e["clip_id"].fillna("").astype(str)
    e["type"] = e["type"].astype(str).str.lower()
    for key in ["time_s", "x", "y", "team", "attack_direction"]:
        e[key] = pd.to_numeric(e[key], errors="raise")
        if not np.isfinite(e[key]).all():
            raise ValueError(f"Event column {key} contains nonfinite values")
    if not e["team"].isin([0, 1]).all() or not e["attack_direction"].isin([-1, 1]).all():
        raise ValueError("Event teams must be 0/1 and attack_direction must be -1/+1")
    if e.duplicated(subset=[c for c in ["match_id", "clip_id", "time_s", "type", "team", "event_id"] if c in e]).any():
        raise ValueError("Duplicate event keys; resolve duplicate/simultaneous records with event_id")
    return e.sort_values(["match_id", "time_s"]).reset_index(drop=True)


def _oriented_xy(events: pd.DataFrame, end: bool = False) -> tuple[np.ndarray, np.ndarray]:
    prefix = "end_" if end else ""
    direction = events["attack_direction"].to_numpy(float)
    # Rotate by 180 degrees when attacking left, not a reflection in x alone.
    return events[prefix + "x"].to_numpy(float) * direction, events[prefix + "y"].to_numpy(float) * direction


def _cells(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> np.ndarray:
    xbin = np.clip(((x + 52.5) / 105.0 * nx).astype(int), 0, nx - 1)
    ybin = np.clip(((y + 34.0) / 68.0 * ny).astype(int), 0, ny - 1)
    return ybin * nx + xbin


@dataclass
class XTGrid:
    grid: np.ndarray
    nx: int
    ny: int
    metadata: dict[str, Any]

    def predict(self, x: np.ndarray, y: np.ndarray, attack_direction: Any = 1) -> np.ndarray:
        x, y, d = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(attack_direction))
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(d, [-1, 1]).all():
            raise ValueError("xT input requires finite pitch coordinates and explicit direction")
        return self.grid.ravel()[_cells(x * d, y * d, self.nx, self.ny)].astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"nx": self.nx, "ny": self.ny, "grid": self.grid.tolist(), **self.metadata}


def fit_xt(events: pd.DataFrame, train_match_ids: Any, *, nx: int = 12, ny: int = 8,
           tolerance: float = 1e-9, max_iterations: int = 10000) -> XTGrid:
    """Empirical xT Bellman fixed point; failed moves transition to zero value.

    An unvisited cell remains zero and is identified in metadata.  No goals or
    action probabilities are fabricated for sparse cells.
    """
    if nx < 1 or ny < 1:
        raise ValueError("xT grid dimensions must be positive")
    e = validate_events(events)
    train_ids = set(map(str, train_match_ids))
    e = e[e.match_id.isin(train_ids) & e.type.isin(MOVE_TYPES | {"shot"})].copy()
    if len(e) == 0:
        raise ValueError("No training-match pass/carry/shot events: xT cannot be fitted")
    if (abs(e.x) > 52.5).any() or (abs(e.y) > 34).any():
        raise ValueError("Event origins must use centered 105x68 m pitch coordinates")
    x, y = _oriented_xy(e)
    origins = _cells(x, y, nx, ny)
    nc = nx * ny
    shots = e.type.eq("shot").to_numpy()
    if shots.any():
        if "goal" not in e:
            raise ValueError("Training shot events need observed goal=0/1 labels for xT")
        goals = pd.to_numeric(e.loc[e.type.eq("shot"), "goal"], errors="raise").to_numpy()
        if not np.isin(goals, [0, 1]).all():
            raise ValueError("Every training shot must have a binary observed goal label")
    else:
        goals = np.empty(0)
    total = np.bincount(origins, minlength=nc).astype(float)
    shot_count = np.bincount(origins[shots], minlength=nc).astype(float)
    goal_count = np.bincount(origins[shots], weights=goals, minlength=nc).astype(float)
    reward = np.divide(goal_count, total, out=np.zeros(nc), where=total > 0)
    transitions = np.zeros((nc, nc), dtype=float)
    moves = e.loc[~shots].copy()
    if len(moves):
        if not {"successful", "end_x", "end_y"} <= set(moves.columns):
            raise ValueError("Pass/carry events require successful, end_x and end_y")
        success = _bool_column(moves.successful, "successful")
        destination = moves.loc[success].copy()
        if len(destination):
            for c in ["end_x", "end_y"]:
                destination[c] = pd.to_numeric(destination[c], errors="raise")
            if not np.isfinite(destination[["end_x", "end_y"]]).all().all():
                raise ValueError("Successful moves require finite end coordinates")
            if (abs(destination.end_x) > 52.5).any() or (abs(destination.end_y) > 34).any():
                raise ValueError("Move destinations are outside the 105x68 m pitch")
            dx, dy = _oriented_xy(destination, end=True)
            to = _cells(dx, dy, nx, ny)
            from_cells = origins[~shots][success]
            np.add.at(transitions, (from_cells, to), 1)
    transitions = np.divide(transitions, total[:, None], out=np.zeros_like(transitions), where=total[:, None] > 0)
    value = np.zeros(nc)
    for iteration in range(1, max_iterations + 1):
        new = reward + transitions @ value
        delta = float(np.max(abs(new - value)))
        value = new
        if delta < tolerance:
            break
    else:
        raise RuntimeError("xT Bellman iteration did not converge; inspect nonterminating action chains")
    meta = {"method": "empirical_train_only_action_grid", "train_match_ids": sorted(train_ids),
            "events_used": int(len(e)), "shots": int(shots.sum()), "goals": int(goals.sum()),
            "empty_cells": int((total == 0).sum()), "iterations": iteration, "max_delta": delta,
            "action_counts": total.reshape(ny, nx).tolist(),
            "shot_counts": shot_count.reshape(ny, nx).tolist(),
            "warning": "Sparse/empty cells have unsmoothed empirical estimates; not validated calibration."}
    return XTGrid(value.reshape(ny, nx).astype(np.float32), nx, ny, meta)


def shot_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Intercept, distance to goal, and goal-mouth opening angle (radians)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    dx = 52.5 - x
    distance = np.hypot(dx, y)
    a = np.stack([dx, 3.66 - y], axis=-1)
    b = np.stack([dx, -3.66 - y], axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.divide((a * b).sum(axis=-1), denom, out=np.ones_like(dx), where=denom > 1e-10)
    angle = np.arccos(np.clip(cosine, -1, 1))
    return np.stack([np.ones_like(distance), distance / 30.0, angle], axis=-1)


@dataclass
class XGModel:
    coefficients: np.ndarray
    metadata: dict[str, Any]

    def predict(self, x: Any, y: Any, attack_direction: Any = 1) -> np.ndarray:
        x, y, d = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(attack_direction))
        return expit(shot_features(x * d, y * d) @ self.coefficients).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"coefficients": self.coefficients.tolist(), **self.metadata}


def fit_xg(events: pd.DataFrame, train_match_ids: Any, *, ridge: float = 1e-3) -> XGModel:
    e = validate_events(events)
    ids = set(map(str, train_match_ids))
    e = e[e.match_id.isin(ids) & e.type.eq("shot")]
    if "goal" not in e or len(e) < 2:
        raise ValueError("xG logistic fit needs observed training shot labels")
    goal = pd.to_numeric(e.goal, errors="raise").to_numpy(float)
    if not np.isin(goal, [0, 1]).all() or len(np.unique(goal)) != 2:
        raise ValueError("xG fit requires both scored and missed observed training shots")
    x, y = _oriented_xy(e)
    design = shot_features(x, y)
    penalty = np.array([0.0, ridge, ridge])
    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ w
        loss = np.logaddexp(0, logits).sum() - goal @ logits + .5 * (penalty * w * w).sum()
        grad = design.T @ (expit(logits) - goal) + penalty * w
        return float(loss), grad
    fit = minimize(objective, np.zeros(3), jac=True, method="BFGS", options={"maxiter": 1000})
    if not np.isfinite(fit.fun) or (not fit.success and np.linalg.norm(fit.jac) > 1e-4):
        raise RuntimeError("Observed-shot xG fit failed: " + str(fit.message))
    return XGModel(fit.x, {"method": "observed_shot_logistic_distance_angle", "ridge": ridge,
                          "train_match_ids": sorted(ids), "shots": len(e), "goals": int(goal.sum()),
                          "training_log_loss": float(np.mean(np.logaddexp(0, design @ fit.x) - goal * (design @ fit.x))),
                          "warning": "Training loss is not held-out validation or calibration evidence."})


def align_targets(arrays: dict[str, np.ndarray], events: pd.DataFrame, xt_model: XTGrid,
                  xg_model: XGModel | None, *, history_frames: int, tolerance_s: float | None = None,
                  fps: float = 25.0, clock_basis: str = "match") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """State xT labels plus shot-end xG aligned on the explicitly shared clock.

    xT is available only where the ball, attacking team, and direction are
    observed. Missing ball/clock never receives a fabricated value target.
    """
    e = validate_events(events)
    if "frame_times_s" not in arrays or "attack_direction" not in arrays:
        raise ValueError("Value alignment requires frame_times_s and explicit attack_direction")
    if clock_basis not in {"match", "clip"}:
        raise ValueError("clock_basis must be match or clip")
    if clock_basis == "clip" and "clip_id" not in e:
        raise ValueError("Clip-local tracking clock requires clip_id on every event")
    times = arrays["frame_times_s"][:, history_frames:]
    ball = arrays["states"][:, history_frames:, 22, :2]
    valid = arrays["valid"][:, history_frames:, 22].copy()
    team = arrays["attacking_team"]
    direction = arrays["attack_direction"]
    valid &= np.isin(team, [0, 1])[:, None] & np.isin(direction, [-1, 1])[:, None]
    valid &= np.isfinite(times)
    xt = np.zeros(times.shape, dtype=np.float32)
    xg = np.zeros_like(xt)
    shot = np.zeros(times.shape, dtype=bool)
    tolerance = .5 / fps + 1e-8 if tolerance_s is None else tolerance_s
    if tolerance < 0:
        raise ValueError("Event alignment tolerance cannot be negative")
    for i in range(len(times)):
        if not valid[i].any():
            continue
        xt[i, valid[i]] = xt_model.predict(ball[i, valid[i], 0], ball[i, valid[i], 1], direction[i])
        match = str(arrays["match_ids"][i])
        candidates = e[e.match_id.eq(match) & e.team.eq(int(team[i])) & e.type.eq("shot")]
        if clock_basis == "clip":
            candidates = candidates[candidates.clip_id.eq(str(arrays["clip_ids"][i]))]
        for _, ev in candidates.iterrows():
            j = int(np.argmin(abs(times[i] - ev.time_s)))
            if abs(times[i, j] - ev.time_s) > tolerance or not valid[i, j]:
                continue
            if shot[i, j]:
                raise ValueError("Two shots align to one frame; choose an explicit finer alignment rule")
            supplied = ev.get("xg", np.nan)
            if pd.notna(supplied):
                value = float(supplied)
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("External xg must be a probability in [0,1]")
            elif xg_model is not None:
                value = float(xg_model.predict(ev.x, ev.y, ev.attack_direction))
            else:
                # xT supervision remains valid; no shot-end target is invented.
                continue
            shot[i, j] = True
            xg[i, j] = value
    result = {"xt": xt, "xg": xg, "shot": shot, "target_valid": valid}
    meta = {"clock_basis": clock_basis, "tolerance_s": tolerance,
            "valid_future_frames": int(valid.sum()), "xg_shot_targets": int(shot.sum()),
            "xt_definition": "train-event Bellman grid evaluated at observed ball state in attacking orientation",
            "xg_definition": "external event probability when supplied, otherwise train-only observed-shot logistic model",
            "xg_inactive_for_nonshot_or_unlabelled_shot": True}
    return result, meta
