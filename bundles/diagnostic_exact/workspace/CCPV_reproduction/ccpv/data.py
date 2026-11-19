"""Auditable tracking preparation; missing observations remain missing.

All saved arrays are numeric/bool/Unicode and can be loaded with
``np.load(..., allow_pickle=False)``.  Coordinates are centered physical meters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from collections import Counter
import hashlib
import json
import math
import re
from zipfile import ZipFile
import numpy as np
import pandas as pd

from .targets import validate_events, fit_xt, fit_xg, align_targets
from .preprocessing import preprocess_positions, summarize_speeds, _validated_options

TEAMS = np.array([0] * 11 + [1] * 11 + [2], dtype=np.int64)


def source_hash(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def validate_prepared(arrays: Mapping[str, np.ndarray]) -> None:
    required = {"states", "valid", "teams", "match_ids", "clip_ids", "scenario", "attacking_team"}
    if required - set(arrays):
        raise ValueError(f"Prepared data missing arrays: {sorted(required - set(arrays))}")
    s = np.asarray(arrays["states"])
    v = np.asarray(arrays["valid"])
    if s.ndim != 4 or s.shape[2:] != (23, 4) or v.shape != s.shape[:3]:
        raise ValueError("states must be [samples,frames,23,4] and valid [samples,frames,23]")
    if s.dtype != np.float32 or v.dtype != bool:
        raise ValueError("states must be float32 and valid must be bool")
    if not np.isfinite(s).all() or np.any(s[~v] != 0):
        raise ValueError("States must be finite, with all missing states exactly zero")
    if np.asarray(arrays["teams"]).shape != (23,) or not np.array_equal(arrays["teams"], TEAMS):
        raise ValueError("teams must use slots 0:11 team0, 11:22 team1, 22 ball")
    for key in ["match_ids", "clip_ids", "scenario", "attacking_team"]:
        if np.asarray(arrays[key]).shape != (len(s),):
            raise ValueError(f"{key} must have one entry per window")
    for key in ["match_ids", "clip_ids", "scenario"]:
        if np.asarray(arrays[key]).dtype.kind != "U":
            raise ValueError(f"{key} must be Unicode, not object arrays")
    labels = {"xt", "xg", "shot", "target_valid"}
    if labels & set(arrays):
        if not labels <= set(arrays):
            raise ValueError("Value arrays must include xt/xg/shot/target_valid together")
        shape = np.asarray(arrays["xt"]).shape
        if len(shape) != 2 or shape[0] != len(s) or any(np.asarray(arrays[k]).shape != shape for k in labels):
            raise ValueError("All value targets must have shape [samples,future_frames]")
        if not np.isfinite(arrays["xt"]).all() or not np.isfinite(arrays["xg"]).all():
            raise ValueError("Value targets must be finite; missing targets use target_valid=False")
    for key, a in arrays.items():
        if np.asarray(a).dtype.kind == "O":
            raise ValueError(f"Object array {key} would require unsafe pickle")
    splits = [np.asarray(arrays.get(k + "_indices", []), dtype=int) for k in ["train", "validation", "test"]]
    index_sets = [set(x.tolist()) for x in splits]
    if any((x < 0).any() or (x >= len(s)).any() for x in splits):
        raise ValueError("Split indices are outside prepared data")
    for i in range(3):
        for j in range(i + 1, 3):
            if index_sets[i] & index_sets[j]:
                raise ValueError("Overlapping split window indices")
            if set(arrays["match_ids"][splits[i]]) & set(arrays["match_ids"][splits[j]]):
                raise ValueError("Match identity leakage across splits")


def save_prepared(path: str | Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".npz":
        p = p.with_suffix(".npz")
    if metadata.get("data_kind") not in {"real", "synthetic_test"}:
        raise ValueError("Metadata must explicitly declare real or synthetic_test")
    validate_prepared(arrays)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    p.with_suffix(".json").write_text(json.dumps(_jsonable(dict(metadata)), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return p


def load_prepared(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    validate_prepared(arrays)
    if meta.get("data_kind") not in {"real", "synthetic_test"}:
        raise ValueError("Missing explicit data_kind in adjacent metadata")
    return arrays, meta


def split_by_match(match_ids: Any, *, train_fraction: float = .7, val_fraction: float = .15,
                   seed: int = 0) -> dict[str, np.ndarray]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must leave nonempty training, validation and test fractions")
    ids = np.asarray(match_ids, dtype=str)
    unique = np.unique(ids)
    if len(unique) < 3:
        raise ValueError(f"At least 3 independently identified matches are required; found {len(unique)}")
    rng = np.random.default_rng(seed)
    order = unique[rng.permutation(len(unique))]
    ntrain = max(1, min(len(unique) - 2, int(math.floor(len(unique) * train_fraction))))
    nval = max(1, min(len(unique) - ntrain - 1, int(math.floor(len(unique) * val_fraction))))
    groups = {"train": order[:ntrain], "validation": order[ntrain:ntrain+nval], "test": order[ntrain+nval:]}
    return {k: np.flatnonzero(np.isin(ids, matches)).astype(np.int64) for k, matches in groups.items()}


def _attach_split(arrays: dict[str, np.ndarray], meta: dict[str, Any], *, seed: int,
                  train_fraction: float, val_fraction: float, identities_resolved: bool = True) -> None:
    if identities_resolved:
        try:
            splits = split_by_match(arrays["match_ids"], train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
            reason = None
        except ValueError as exc:
            # Do not hide malformed fractions, but permit a one-match audit export.
            if len(np.unique(arrays["match_ids"])) >= 3:
                raise
            reason = str(exc)
            splits = {k: np.empty(0, np.int64) for k in ["train", "validation", "test"]}
    else:
        reason = "Source clip-to-match identities are unresolved; clips are not independent match groups"
        splits = {k: np.empty(0, np.int64) for k in ["train", "validation", "test"]}
    for k, indices in splits.items():
        arrays[k + "_indices"] = indices
    meta["split"] = {"method": "group_by_real_match_id", "seed": seed, "requested_train_fraction": train_fraction,
                     "requested_validation_fraction": val_fraction, "evaluation_blocked": reason is not None,
                     "blocked_reason": reason, "identities_resolved": identities_resolved,
                     "groups": {k: {"samples": len(ix), "match_ids": sorted(set(arrays["match_ids"][ix].tolist())),
                                    "window_indices": ix.tolist()} for k, ix in splits.items()}}


def _team(value: Any) -> int:
    text = str(value).strip().lower()
    mapping = {"0": 0, "0.0": 0, "left": 0, "home": 0,
               "1": 1, "1.0": 1, "right": 1, "away": 1,
               "2": 2, "2.0": 2, "ball": 2}
    if text not in mapping:
        raise ValueError(f"Unknown team value {value!r}")
    return mapping[text]


def _identity_order(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _velocities(positions: np.ndarray, observed: np.ndarray, times: np.ndarray,
                supplied: np.ndarray | None = None, supplied_valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(positions)
    known = np.zeros(observed.shape, dtype=bool)
    for entity in range(23):
        for t in np.flatnonzero(observed[:, entity]):
            # A central/forward difference at the history boundary would leak
            # the first future observation into the model input. Use only past.
            if t == 0 or not observed[t-1, entity]:
                continue
            a, b = t-1, t
            velocity[t, entity] = (positions[b, entity] - positions[a, entity]) / (times[b] - times[a])
            known[t, entity] = True
    if supplied is not None and supplied_valid is not None:
        velocity[supplied_valid] = supplied[supplied_valid]
        known[supplied_valid] = True
    return velocity, observed & known


def _prepare_frame_table(table: pd.DataFrame, *, history_frames: int, future_frames: int,
                         fps: float, stride: int, coordinate_system: str,
                         identity_conflicts: str = "error",
                         preprocessing: Mapping[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if min(history_frames, future_frames, stride) < 1 or fps <= 0:
        raise ValueError("History/future/stride/fps must be positive")
    if identity_conflicts not in {"error", "skip_clip"}:
        raise ValueError("identity_conflicts must be error or skip_clip")
    preprocessing_options = None
    if preprocessing is not None:
        unknown = set(preprocessing) - {"smooth_window", "player_speed_cap", "player_accel_cap"}
        if unknown:
            raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
        preprocessing_options = _validated_options(**{ "smooth_window": 1, "player_speed_cap": None,
                                                       "player_accel_cap": None, **dict(preprocessing)})
    required = {"match_id", "clip_id", "frame", "entity_id", "team", "x", "y"}
    if required - set(table.columns):
        raise ValueError(f"Tracking CSV missing columns: {sorted(required - set(table.columns))}")
    d = table.copy()
    for c in ["match_id", "clip_id", "entity_id"]:
        if d[c].isna().any():
            raise ValueError(f"{c} cannot be missing")
        d[c] = d[c].astype(str)
    d["team"] = d.team.map(_team)
    frames = pd.to_numeric(d.frame, errors="raise").to_numpy(float)
    if not np.isfinite(frames).all() or not np.equal(frames, np.round(frames)).all():
        raise ValueError("Frame must contain finite integer frame indices")
    d["frame"] = frames.astype(np.int64)
    for c in ["x", "y"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    if coordinate_system == "normalized":
        d["x"] = d.x * 105 - 52.5
        d["y"] = d.y * 68 - 34
    elif coordinate_system == "corner":
        d["x"] -= 52.5
        d["y"] -= 34
    elif coordinate_system != "centered":
        raise ValueError("coordinate_system must be centered, corner or normalized")
    if d.duplicated(["match_id", "clip_id", "frame", "entity_id"]).any():
        raise ValueError("Duplicate tracking observation for the same frame/entity")
    if ("vx" in d) != ("vy" in d):
        raise ValueError("Supply both physical vx/vy or neither")
    if "vx" in d:
        d["vx"] = pd.to_numeric(d.vx, errors="raise")
        d["vy"] = pd.to_numeric(d.vy, errors="raise")
    if "observed" in d:
        text = d.observed.astype(str).str.lower()
        if not text.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
            raise ValueError("observed must be boolean/0/1")
        declared = text.isin(["true", "1", "1.0"])
    else:
        declared = pd.Series(True, index=d.index)
    d["_observed"] = declared & np.isfinite(d.x) & np.isfinite(d.y)
    if "time_s" in d:
        d["time_s"] = pd.to_numeric(d.time_s, errors="raise")
        if not np.isfinite(d.time_s).all():
            raise ValueError("time_s must be finite")
    total = history_frames + future_frames
    sequences, valids, position_valids, matches, clips, scenarios, teams, directions, frame_times, identities = ([] for _ in range(10))
    mappings = []
    excluded_clips = []
    qc_clips = []
    qc_samples = {role: {kind: [] for kind in ["raw", "filtered"]} for role in ["players", "ball"]}
    skipped_short = 0
    for (match, clip), group in d.groupby(["match_id", "clip_id"], sort=True):
        start, end = int(group.frame.min()), int(group.frame.max())
        if end - start + 1 < total:
            skipped_short += 1
            continue
        if end - start > 5_000_000:
            raise ValueError("Unreasonably large frame range; check clip grouping")
        ids = group[["entity_id", "team"]].drop_duplicates()
        if ids.duplicated("entity_id").any():
            raise ValueError(f"Identity changes team within {clip}; resolve annotations explicitly")
        slots: dict[str, int] = {}
        if "slot" in group:
            specified = group[["entity_id", "slot"]].drop_duplicates()
            if specified.duplicated("entity_id").any():
                raise ValueError("Entity slot changes within clip")
            for _, row in specified.iterrows():
                if not np.isfinite(row.slot) or float(row.slot) != int(row.slot):
                    raise ValueError("Explicit slots must be finite integers")
                slot = int(row.slot)
                entity_team = int(ids.loc[ids.entity_id.eq(row.entity_id), "team"].iloc[0])
                if slot < 0 or slot >= 23 or TEAMS[slot] != entity_team:
                    raise ValueError("Explicit slot is inconsistent with entity team")
                slots[row.entity_id] = slot
        else:
            counts = {tm: int(ids.loc[ids.team.eq(tm), "entity_id"].nunique()) for tm in [0, 1]}
            if any(count > 11 for count in counts.values()):
                reason = f"{clip}: distinct player track counts {counts} exceed 11-team capacity; unresolved substitutions/track identities"
                if identity_conflicts == "error":
                    raise ValueError(reason)
                excluded_clips.append({"match_id": match, "clip_id": clip, "reason": reason,
                                       "action": "whole clip excluded; no observed player selected/dropped within admitted clips"})
                continue
            for tm, offset, capacity in [(0, 0, 11), (1, 11, 11), (2, 22, 1)]:
                entity_ids = sorted(ids.loc[ids.team.eq(tm), "entity_id"].tolist(), key=_identity_order)
                if tm == 2:
                    # Multiple nonoverlapping ball tracklets still refer to the
                    # one physical ball. Simultaneous duplicate records are
                    # rejected below; no location is filled or selected.
                    slots.update({eid: 22 for eid in entity_ids})
                    continue
                if len(entity_ids) > capacity:
                    raise ValueError(f"{clip}: {len(entity_ids)} tracks for team {tm}, capacity {capacity}; provide resolved stable slot identities")
                slots.update({eid: offset + i for i, eid in enumerate(entity_ids)})
        slotted = group.assign(_slot=group.entity_id.map(slots))
        duplicate = slotted.duplicated(["frame", "_slot"], keep=False)
        if (duplicate & slotted._slot.ne(22)).any():
            raise ValueError(f"{clip}: two simultaneous player observations map to the same entity slot")
        ambiguous_ball_frames = sorted(slotted.loc[duplicate & slotted._slot.eq(22), "frame"].unique().tolist())
        if ambiguous_ball_frames:
            # Two possible ball detections are uncertainty, not permission to
            # choose the most convenient trajectory as ground truth.
            group = group.copy()
            group.loc[group.team.eq(2) & group.frame.isin(ambiguous_ball_frames), "_observed"] = False
        length = end - start + 1
        base = float(group.time_s.iloc[0]) - (int(group.frame.iloc[0]) - start) / fps if "time_s" in group else 0.0
        times = base + np.arange(length, dtype=np.float64) / fps
        if "time_s" in group:
            expected = base + (group.frame.to_numpy() - start) / fps
            if np.max(abs(expected - group.time_s.to_numpy())) > .05 / fps:
                raise ValueError("Tracking timestamps disagree with uniform frame/fps clock")
        pos = np.zeros((length, 23, 2), dtype=np.float32)
        observed = np.zeros((length, 23), bool)
        supplied = np.zeros_like(pos)
        supplied_known = np.zeros_like(observed)
        for _, row in group.iterrows():
            t, sl = int(row.frame) - start, slots[row.entity_id]
            if row._observed:
                pos[t, sl] = [row.x, row.y]
                observed[t, sl] = True
                if "vx" in d and np.isfinite(row.vx) and np.isfinite(row.vy):
                    supplied[t, sl] = [row.vx, row.vy]
                    supplied_known[t, sl] = True
        velocity, valid = _velocities(pos, observed, times, supplied, supplied_known)
        state = np.concatenate([pos, velocity], axis=-1)
        state[~valid] = 0
        if preprocessing_options is not None:
            state, valid, quality = preprocess_positions(pos, observed, times, teams=TEAMS,
                                                         include_speed_samples=True, **preprocessing_options)
            samples_for_summary = quality.pop("_speed_samples")
            for role in ["players", "ball"]:
                for kind in ["raw", "filtered"]:
                    qc_samples[role][kind].append(samples_for_summary[role][kind])
            qc_clips.append({"match_id": match, "clip_id": clip, **quality})
        idarray = np.full(23, "", dtype=f"<U{max(1,sum(map(len,slots))+len(slots))}")
        for eid, sl in slots.items():
            idarray[sl] = eid if not idarray[sl] else idarray[sl] + "/" + eid
        mappings.append({"match_id": match, "clip_id": clip, "entity_to_slot": slots,
                         "ambiguous_ball_frames_masked": ambiguous_ball_frames})
        for first in range(0, length - total + 1, stride):
            last = first + total
            if not valid[first:first+history_frames].any() or not valid[first+history_frames:last].any():
                continue
            rows_in_window = group[group.frame.between(start + first, start + last - 1)]
            attack = rows_in_window.attacking_team.dropna().unique() if "attacking_team" in group else []
            attack = [-1 if str(a) in ["-1", "-1.0", "unknown"] else _team(a) for a in attack]
            attack = attack[0] if len(attack) == 1 and attack[0] in [0, 1] else -1
            direct = rows_in_window.attack_direction.dropna().unique() if "attack_direction" in group else []
            direct = int(direct[0]) if len(direct) == 1 and direct[0] in [-1, 1] else 0
            sc = rows_in_window.scenario.dropna().astype(str).unique() if "scenario" in group else []
            scenario = sc[0] if len(sc) == 1 else "unlabelled"
            sequences.append(state[first:last]);valids.append(valid[first:last]);position_valids.append(observed[first:last])
            matches.append(match);clips.append(clip);scenarios.append(scenario);teams.append(attack);directions.append(direct)
            frame_times.append(times[first:last]);identities.append(idarray.copy())
    if not sequences:
        raise ValueError("No admissible windows: check history+future length and observed agents")
    arrays = {"states": np.asarray(sequences, np.float32), "valid": np.asarray(valids, bool),
              "position_valid": np.asarray(position_valids, bool), "teams": TEAMS.copy(),
              "match_ids": np.asarray(matches, dtype=str), "clip_ids": np.asarray(clips, dtype=str),
              "scenario": np.asarray(scenarios, dtype=str), "attacking_team": np.asarray(teams, np.int64),
              "attack_direction": np.asarray(directions, np.int64), "frame_times_s": np.asarray(frame_times, np.float64),
              "entity_ids": np.asarray(identities, dtype=str)}
    meta = {"fps": fps, "history_frames": history_frames, "future_frames": future_frames, "stride_frames": stride,
            "samples": len(sequences), "matches": len(set(matches)), "clips": len(set(clips)),
            "missing_ball_rate": float(1 - arrays["valid"][:, :, 22].mean()),
            "observed_state_fraction": float(arrays["valid"].mean()), "entity_slot_maps": mappings,
            "skipped_short_clips": skipped_short, "excluded_identity_conflict_clips": excluded_clips,
            "identity_conflict_policy": identity_conflicts,
            "preprocessing": {"coordinates": "centered_105x68_m", "input_coordinates": coordinate_system,
                              "velocity": "supplied physical m/s, otherwise causal backward finite differences over contiguous observed frames",
                              "singleton_position_without_velocity": "position_valid=True but valid=False and zero state",
                              "smoothing": "none (paper does not specify a reproducible filter)",
                              "clipping": "none; no offscreen coordinates or missing ball are imputed",
                              "ambiguous_ball_frames_masked": sum(len(m["ambiguous_ball_frames_masked"]) for m in mappings)}}
    if preprocessing_options is not None:
        meta["quality_control"] = {
            "enabled": True, "parameters": preprocessing_options, "reconstruction_assumption": True,
            "assumption_note": "The manuscript did not specify these exact filtering settings; they are explicit reconstruction assumptions.",
            "per_clip": qc_clips,
            "counts": {key: sum(q["counts"][key] for q in qc_clips) for key in qc_clips[0]["counts"]},
            "speed_m_s": {role: {kind: summarize_speeds(np.concatenate(qc_samples[role][kind]))
                                      for kind in ["raw", "filtered"]} for role in ["players", "ball"]},
            "statistics_unit": "source observed contiguous transitions, counted once per clip before overlapping window extraction",
            "raw_states_saved": False}
        meta["preprocessing"]["smoothing"] = f"causal trailing coordinate mean, window={preprocessing_options['smooth_window']}, reset at missing gaps"
        meta["preprocessing"]["velocity"] = "causal backward finite differences after smoothing, optional player velocity projection and consistent position integration"
        meta["preprocessing"]["clipping"] = "optional explicit speed/acceleration projection for team0/1 only; never ball; missing observations remain masked"
        meta["preprocessing"]["reconstruction_assumption"] = True
    else:
        meta["quality_control"] = {"enabled": False, "parameters": None}
    return arrays, meta


def _value_labels(arrays: dict[str, np.ndarray], meta: dict[str, Any], events_csv: str | Path | None,
                  *, clock_basis: str, external_xg_provenance: str | None = None) -> None:
    meta["value_labels"] = {"enabled": False, "reason": "No synchronized event file was supplied"}
    if events_csv is None:
        return
    meta["source_hashes"].append(source_hash(events_csv))
    if meta["split"]["evaluation_blocked"]:
        meta["value_labels"]["reason"] = "Event fitting disabled: independent match split unavailable"
        return
    if not arrays["valid"][:, :, 22].any():
        meta["value_labels"]["reason"] = "No observed ball states; value supervision disabled"
        return
    events = validate_events(pd.read_csv(events_csv, dtype={"match_id": str, "clip_id": str}))
    if "xg" in events and events.xg.notna().any() and not external_xg_provenance:
        raise ValueError("External xg labels require external_xg_provenance (provider/model/version)")
    train_ids = np.unique(arrays["match_ids"][arrays["train_indices"]])
    xt = fit_xt(events, train_ids)
    try:
        xg = fit_xg(events, train_ids)
        xg_warning = None
    except ValueError as exc:
        xg = None
        xg_warning = str(exc)
    labels, alignment = align_targets(arrays, events, xt, xg, history_frames=meta["history_frames"], fps=meta["fps"], clock_basis=clock_basis)
    arrays.update(labels)
    meta["value_labels"] = {"enabled": bool(labels["target_valid"].any()),
                            "reason": None if labels["target_valid"].any() else "No aligned ball/team/direction targets",
                            "xt_model": xt.to_dict(), "xg_model": xg.to_dict() if xg else None,
                            "xg_warning": xg_warning, "external_xg_provenance": external_xg_provenance, **alignment}


def prepare_csv(tracking_csv: str | Path, output_npz: str | Path, *, events_csv: str | Path | None = None,
                history_frames: int = 25, future_frames: int = 50, fps: float = 25.0, stride: int = 25,
                coordinate_system: str = "centered", seed: int = 0, train_fraction: float = .7,
                val_fraction: float = .15, clock_basis: str | None = None,
                external_xg_provenance: str | None = None, identity_conflicts: str = "error",
                preprocessing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    table = pd.read_csv(tracking_csv, dtype={"match_id": str, "clip_id": str, "entity_id": str})
    basis = clock_basis or ("match" if "time_s" in table else "clip")
    arrays, meta = _prepare_frame_table(table, history_frames=history_frames, future_frames=future_frames,
                                       fps=fps, stride=stride, coordinate_system=coordinate_system, identity_conflicts=identity_conflicts,
                                       preprocessing=preprocessing)
    meta.update({"data_kind": "real", "provenance": {"format": "canonical_tracking_csv", "clock_basis": basis},
                 "source_hashes": [source_hash(tracking_csv)]})
    _attach_split(arrays, meta, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction)
    _value_labels(arrays, meta, events_csv, clock_basis=basis, external_xg_provenance=external_xg_provenance)
    save_prepared(output_npz, arrays, meta)
    return meta


def prepare_gsr(gsr_json: str | Path, output_npz: str | Path, *, match_mapping: Mapping[str, Any] | str | Path | None = None,
                events_csv: str | Path | None = None, history_frames: int = 25, future_frames: int = 50,
                fps: float | None = None, stride: int = 25, seed: int = 0, train_fraction: float = .7,
                val_fraction: float = .15, external_xg_provenance: str | None = None,
                identity_conflicts: str = "error", preprocessing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Adapt observed SoccerNet-GSR pitch locations; no video/ball imputation.

    Accept a JSON file, directory of Labels-GameState.json files, or a list path
    supplied as a directory. Explicit match mappings override info.game_id.
    """
    source = Path(gsr_json)
    is_zip = source.suffix.lower() == ".zip"
    files = sorted(source.rglob("Labels-GameState.json")) if source.is_dir() else [source]
    if not files:
        raise ValueError("No readable Labels-GameState.json files were found")
    def sources():
        if is_zip:
            with ZipFile(source) as archive:
                names = sorted(n for n in archive.namelist() if n.endswith("Labels-GameState.json") and not n.startswith("__MACOSX/"))
                if not names:
                    raise ValueError("ZIP contains no Labels-GameState.json entries")
                for member in names:
                    payload = archive.read(member)
                    ref = str(source.resolve()) + "::" + member
                    yield ref, Path(member).parent.name, json.loads(payload), {"path": ref, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        else:
            for file in files:
                with file.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                yield str(file.resolve()), file.parent.name, payload, source_hash(file)
    mapping_sources = []
    if isinstance(match_mapping, (str, Path)):
        mapping_sources.append(source_hash(match_mapping))
        mapping = json.loads(Path(match_mapping).read_text(encoding="utf-8"))
    else:
        mapping = dict(match_mapping or {})
    records, provenance, hashes, warnings = [], [], [source_hash(source)] if is_zip else [], []
    resolved = True
    source_fps = []
    clock_bases = []
    for source_reference, parent_name, data, item_hash in sources():
        info = data.get("info", {})
        clip = str(info.get("name") or parent_name)
        entry = mapping.get(clip, {})
        if isinstance(entry, (str, int)):
            entry = {"match_id": str(entry)}
        match = entry.get("match_id", info.get("game_id"))
        identity_source = "explicit_mapping" if "match_id" in entry else "info.game_id"
        if match is None or str(match).strip() == "":
            match = "UNRESOLVED_CLIP:" + clip
            resolved = False
            identity_source = "unresolved"
            warnings.append(f"{clip}: clip identity cannot substitute for a match identity")
        rate = float(fps if fps is not None else info.get("frame_rate", 25))
        if rate <= 0:
            raise ValueError("Invalid GSR frame_rate")
        source_fps.append(rate)
        images = {}
        for image in data.get("images", []):
            image_id = str(image.get("image_id", image.get("id")))
            basename = Path(str(image.get("file_name", ""))).stem
            frame = image.get("frame_id", image.get("frame"))
            if frame is None and basename.isdigit():
                frame = int(basename)
            if frame is None:
                raise ValueError(f"{clip}: image {image_id} has no explicit frame or numeric filename")
            images[image_id] = int(frame)
        if not images:
            raise ValueError(f"{clip}: no images/frame mapping")
        first_frame = min(images.values())
        offset = entry.get("start_time_s")
        clock_bases.append("match" if offset is not None else "clip")
        # clip_start is preserved for audit. Its match/period clock semantics
        # are not silently assumed for event synchronization.
        offset = float(offset) if offset is not None else 0.0
        categories = {int(c["id"]): str(c.get("name", "")) for c in data.get("categories", []) if "id" in c}
        role_counts = Counter()
        ball_pitch = 0
        skipped = Counter()
        for ann in data.get("annotations", []):
            attrs = ann.get("attributes") or {}
            role = str(attrs.get("role", ann.get("role", categories.get(ann.get("category_id"), "")))).lower()
            role_counts[role] += 1
            if role not in {"player", "goalkeeper", "ball"}:
                continue
            image = str(ann.get("image_id"))
            if image not in images:
                skipped["unknown_image"] += 1
                continue
            xy = ann.get("bbox_pitch") or {}
            if not isinstance(xy, dict) or "x_bottom_middle" not in xy or "y_bottom_middle" not in xy:
                skipped["missing_pitch_coordinate"] += 1
                continue
            x, y = xy["x_bottom_middle"], xy["y_bottom_middle"]
            if x is None or y is None or not np.isfinite([x, y]).all():
                skipped["nonfinite_pitch_coordinate"] += 1
                continue
            if "track_id" not in ann or ann["track_id"] is None:
                raise ValueError(f"{clip}: no stable track_id on {role}; detection IDs are not identities")
            entity = str(ann["track_id"])
            tm = 2 if role == "ball" else _team(attrs.get("team", ann.get("team")))
            frame = images[image]
            record = {"match_id": str(match), "clip_id": clip, "frame": frame,
                      "time_s": offset + (frame-first_frame)/rate, "entity_id": entity,
                      "team": tm, "x": float(x), "y": float(y)}
            for key in ["attacking_team", "attack_direction", "scenario"]:
                if key in entry:
                    record[key] = entry[key]
            slot_map = entry.get("track_slots", {})
            if slot_map:
                if entity not in slot_map:
                    raise ValueError(f"Explicit track_slots missing {clip} track {entity}")
                record["slot"] = slot_map[entity]
            records.append(record)
            ball_pitch += int(role == "ball")
        hashes.append(item_hash)
        provenance.append({"file": source_reference, "clip_id": clip, "match_id": str(match),
                           "match_identity_source": identity_source, "info": info, "role_counts": dict(role_counts),
                           "observed_ball_pitch_records": ball_pitch, "skipped_records": dict(skipped),
                           "clock_basis": clock_bases[-1], "explicit_start_time_s": entry.get("start_time_s"),
                           "ball_origin_verified": False,
                           "ball_origin_note": "Only the supplied annotations were inspected; original release/enhancement provenance is not inferred."})
    if len(set(source_fps)) != 1:
        raise ValueError("Mixed fps needs an explicit resampling implementation; not silently resampled")
    if len(set(clock_bases)) != 1:
        raise ValueError("Mixed match/clip clocks; provide explicit start_time_s for all clips or none")
    arrays, meta = _prepare_frame_table(pd.DataFrame(records), history_frames=history_frames, future_frames=future_frames,
                                       fps=source_fps[0], stride=stride, coordinate_system="centered", identity_conflicts=identity_conflicts,
                                       preprocessing=preprocessing)
    meta.update({"data_kind": "real", "provenance": {"format": "SoccerNet_GSR_JSON", "sources": provenance,
                                                        "clock_basis": clock_bases[0], "warnings": warnings},
                 "source_hashes": hashes + mapping_sources})
    _attach_split(arrays, meta, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction, identities_resolved=resolved)
    _value_labels(arrays, meta, events_csv, clock_basis=clock_bases[0], external_xg_provenance=external_xg_provenance)
    save_prepared(output_npz, arrays, meta)
    return meta


def make_synthetic_fixture(output_npz: str | Path, *, seed: int = 0, num_matches: int = 8,
                           windows_per_match: int = 4, history_frames: int = 25, future_frames: int = 50,
                           fps: float = 25.0, train_fraction: float = .7, val_fraction: float = .15) -> dict[str, Any]:
    """Small known stochastic dynamics for software integration, NEVER football evidence."""
    if num_matches < 3 or windows_per_match < 1 or min(history_frames, future_frames) < 1 or fps <= 0:
        raise ValueError("Fixture needs >=3 groups, positive windows/history/future/fps")
    rng = np.random.default_rng(seed)
    total = history_frames + future_frames
    states, matches, clips = [], [], []
    for match in range(num_matches):
        for window in range(windows_per_match):
            position = rng.uniform([-30, -20], [30, 20], size=(23, 2))
            velocity = rng.normal(0, 1.4, size=(23, 2))
            seq = np.zeros((total, 23, 4), np.float32)
            for t in range(total):
                seq[t, :, :2] = position
                seq[t, :, 2:] = velocity
                # Damped, weakly coupled motion with a known conditional law.
                center = position[:11].mean(axis=0)
                coupling = .025 * (center - position)
                coupling[11:22] *= .6
                acceleration = -.35 * velocity + coupling + rng.normal(0, .25, size=(23, 2))
                velocity = velocity + acceleration / fps
                speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
                velocity *= np.minimum(1, 5 / np.maximum(speed, 1e-6))
                position = position + velocity / fps
            states.append(seq);matches.append(f"synthetic_match_{match:03d}");clips.append(f"synthetic_clip_{match:03d}_{window:03d}")
    states = np.asarray(states, np.float32)
    count = len(states)
    valid = np.ones(states.shape[:3], bool)
    # A small observed missing segment exercises masking without inventing states.
    for i in range(count):
        if i % 3 == 0:
            entity = int(rng.integers(0, 22));valid[i, 5:9, entity] = False
    states[~valid] = 0
    ball_x = states[:, history_frames:, 22, 0]
    # These are simulator targets called xt/xg only to exercise the value API.
    # They are neither empirical xT/xG nor fits to match/event observations.
    synthetic_xt = (.12 / (1 + np.exp(-ball_x / 16))).astype(np.float32)
    synthetic_xg = (.25 / (1 + np.exp(-(ball_x - 10) / 10))).astype(np.float32)
    shot = rng.random(ball_x.shape) < .04
    synthetic_xg[~shot] = 0
    arrays = {"states": states, "valid": valid, "position_valid": valid.copy(), "teams": TEAMS.copy(),
              "match_ids": np.asarray(matches, str), "clip_ids": np.asarray(clips, str),
              "scenario": np.full(count, "synthetic_dynamics", dtype="<U18"), "attacking_team": np.zeros(count, np.int64),
              "attack_direction": np.ones(count, np.int64),
              "frame_times_s": np.broadcast_to(np.arange(total)/fps, (count,total)).copy(),
              "xt": synthetic_xt, "xg": synthetic_xg, "shot": shot,
              "target_valid": valid[:, history_frames:, 22].copy()}
    meta = {"data_kind": "synthetic_test", "fps": fps, "history_frames": history_frames,
            "future_frames": future_frames, "samples": count, "matches": num_matches, "clips": count,
            "missing_ball_rate": 0.0, "source_hashes": [], "seed": seed,
            "provenance": {"generator": "damped_weakly_coupled_stochastic_dynamics_v1",
                           "purpose": "software integration and masking test only", "real_football_evidence": False,
                           "value_targets": "explicit simulator-generated potentials and shot flags; not empirical xT/xG"},
            "preprocessing": {"coordinates": "centered_105x68_m", "velocity": "simulated physical m/s", "imputation": "none"},
            "value_labels": {"enabled": True, "source": "synthetic_test_simulator", "empirical_xt_xg": False}}
    _attach_split(arrays, meta, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction)
    save_prepared(output_npz, arrays, meta)
    return meta
