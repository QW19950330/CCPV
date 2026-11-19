"""Retained SkillCorner adapter, including historical behavior; see limitations."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
TEAMS = np.array([0]*11+[1]*11+[2], dtype=np.int64)
FPS = 10.0

HISTORY = 10

FUTURE = 20

WINDOW = HISTORY + FUTURE

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def finite_xy(value: dict[str, Any]) -> tuple[float, float] | None:
    x, y = value.get("x"), value.get("y")
    if x is None or y is None:
        return None
    xy = np.asarray([x, y], dtype=np.float64)
    if not np.isfinite(xy).all():
        return None
    return float(xy[0]), float(xy[1])

def make_window(
    match_id: int,
    period: int,
    frames: list[dict[str, Any]],
    previous: dict[int, tuple[float, float]],
    player_team: dict[int, int],
    home_id: int,
    pitch_length: float,
    pitch_width: float,
    period_start_frame: int,
) -> dict[str, np.ndarray] | None:
    ids_by_team = [set(), set()]
    for frame in frames:
        for player_id, _, _ in frame["players"]:
            team_id = player_team.get(player_id)
            if team_id is None:
                raise ValueError(f"Unknown player identity {player_id} in match {match_id}")
            ids_by_team[0 if team_id == home_id else 1].add(player_id)
    if any(len(ids) > 11 for ids in ids_by_team):
        return None

    slots = [{player_id: offset + slot for slot, player_id in enumerate(sorted(ids))}
             for offset, ids in zip((0, 11), ids_by_team)]
    slot_map = {player_id: slot for team_slots in slots for player_id, slot in team_slots.items()}
    for player_id, team_id in player_team.items():
        if player_id in slot_map and ((team_id == home_id) != (slot_map[player_id] < 11)):
            raise AssertionError("Slot/team assignment mismatch")

    positions = np.zeros((WINDOW, 23, 2), dtype=np.float32)
    observed = np.zeros((WINDOW, 23), dtype=bool)
    source_frames = np.empty(WINDOW, dtype=np.int64)
    frame_times = np.empty(WINDOW, dtype=np.float64)
    for t, frame in enumerate(frames):
        source_frames[t] = frame["frame"]
        frame_times[t] = (frame["frame"] - period_start_frame) / FPS
        seen: set[int] = set()
        for player_id, detected, xy in frame["players"]:
            if player_id in seen:
                raise ValueError(f"Duplicate player identity {player_id} in frame {frame['frame']}")
            seen.add(player_id)
            if detected and xy is not None:
                slot = slot_map[player_id]
                positions[t, slot] = (xy[0] * 105.0 / pitch_length,
                                      xy[1] * 68.0 / pitch_width)
                observed[t, slot] = True
        if frame["ball_detected"] and frame["ball_xy"] is not None:
            xy = frame["ball_xy"]
            positions[t, 22] = (xy[0] * 105.0 / pitch_length,
                                xy[1] * 68.0 / pitch_width)
            observed[t, 22] = True

    velocity = np.zeros_like(positions)
    velocity_known = np.zeros_like(observed)
    for t in range(WINDOW):
        for slot, player_id in enumerate([*sorted(ids_by_team[0]), *sorted(ids_by_team[1])]):
            if not observed[t, slot]:
                continue
            if t > 0 and observed[t - 1, slot]:
                velocity[t, slot] = (positions[t, slot] - positions[t - 1, slot]) * FPS
                velocity_known[t, slot] = True
            elif t == 0 and player_id in previous:
                prev = np.asarray(previous[player_id], dtype=np.float32)
                prev = prev * np.asarray([105.0 / pitch_length, 68.0 / pitch_width], dtype=np.float32)
                velocity[t, slot] = (positions[t, slot] - prev) * FPS
                velocity_known[t, slot] = True
        if observed[t, 22]:
            prev_ball = frames[t].get("previous_ball_xy") if t == 0 else None
            if t > 0 and observed[t - 1, 22]:
                velocity[t, 22] = (positions[t, 22] - positions[t - 1, 22]) * FPS
                velocity_known[t, 22] = True
            elif t == 0 and prev_ball is not None:
                prev_ball = np.asarray(prev_ball, dtype=np.float32)
                prev_ball = prev_ball * np.asarray([105.0 / pitch_length, 68.0 / pitch_width], dtype=np.float32)
                velocity[t, 22] = (positions[t, 22] - prev_ball) * FPS
                velocity_known[t, 22] = True

    valid = observed & velocity_known
    if not valid[:HISTORY].any() or not valid[HISTORY:].any():
        return None
    states = np.concatenate([positions, velocity], axis=-1)
    states[~valid] = 0.0
    entity_ids = np.full(23, "", dtype="U32")
    for player_id, slot in slot_map.items():
        entity_ids[slot] = str(player_id)
    entity_ids[22] = "ball"
    return {
        "states": states,
        "valid": valid,
        "position_valid": observed,
        "real_observed_mask": observed.copy(),
        "entity_ids": entity_ids,
        "match_id": np.asarray(f"skillcorner:{match_id}"),
        "clip_id": np.asarray(f"skillcorner:{match_id}:period:{period}:window:{frames[0]['frame']}"),
        "scenario": np.asarray("unlabelled"),
        "attacking_team": np.asarray(-1, dtype=np.int64),
        "attack_direction": np.asarray(0, dtype=np.int64),
        "frame_times_s": frame_times,
        "source_frame_ids": source_frames,
        "source_period": np.asarray(period, dtype=np.int64),
    }

def prepare_one(raw: Path, match_id: int, rows_by_id: dict[str, dict[str, Any]],
                period_starts: dict[int, int], window_records: list[dict[str, np.ndarray]],
                match_qc: list[dict[str, Any]]) -> None:
    folder = raw / str(match_id)
    match_path = folder / f"{match_id}_match.json"
    tracking_path = folder / f"{match_id}_tracking_extrapolated.jsonl"
    match_row = rows_by_id[f"data/matches/{match_id}/{match_id}_match.json"]
    tracking_row = rows_by_id[f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"]
    if sha256_file(match_path) != match_row["sha256"]:
        raise ValueError(f"Pinned match metadata checksum mismatch: {match_path}")
    metadata = read_json(match_path)
    player_team = {int(p["id"]): int(p["team_id"]) for p in metadata["players"]
                   if p.get("id") is not None and p.get("team_id") is not None}
    home_id = int(metadata["home_team"]["id"])
    away_id = int(metadata["away_team"]["id"])
    if set(player_team.values()) - {home_id, away_id}:
        raise ValueError(f"Unexpected team identity in match {match_id}")
    pitch_length, pitch_width = float(metadata["pitch_length"]), float(metadata["pitch_width"])
    starts = {int(p["period"]): int(p["start_frame"]) for p in metadata["match_periods"]}
    if starts != period_starts:
        raise ValueError(f"Period start mismatch for match {match_id}: {starts} != {period_starts}")

    source_hash = hashlib.sha256()
    emitted = rejected_roster = rejected_support = chunks = 0
    current_period: int | None = None
    chunk: list[dict[str, Any]] = []
    previous_frame_number: int | None = None
    previous_positions: dict[int, tuple[float, float]] = {}
    previous_ball: tuple[float, float] | None = None
    last_frame: dict[str, Any] | None = None

    def flush_chunk(period: int, rows: list[dict[str, Any]],
                    previous: dict[int, tuple[float, float]],
                    prev_ball: tuple[float, float] | None) -> tuple[int, dict[int, tuple[float, float]], tuple[float, float] | None]:
        nonlocal emitted, rejected_roster, rejected_support, chunks
        if len(rows) != WINDOW:
            return 0, previous, prev_ball
        chunks += 1
        rows[0]["previous_ball_xy"] = prev_ball
        ids = {int(x[0]) for fr in rows for x in fr["players"]}
        team_counts = [sum(player_team.get(pid) == team for pid in ids) for team in (home_id, away_id)]
        if max(team_counts) > 11:
            rejected_roster += 1
            return 0, {pid: xy for pid, xy in previous.items()}, prev_ball
        sample = make_window(match_id, period, rows, previous, player_team, home_id,
                             pitch_length, pitch_width, period_starts[period])
        detected_now = {pid: xy for pid, detected, xy in rows[-1]["players"] if detected and xy is not None}
        ball_now = rows[-1]["ball_xy"] if rows[-1]["ball_detected"] else None
        if sample is None:
            rejected_support += 1
        else:
            window_records.append(sample)
            emitted += 1
        return 1, detected_now, ball_now

    with tracking_path.open("rb") as source:
        for raw_line in source:
            source_hash.update(raw_line)
            frame_data = json.loads(raw_line)
            period = frame_data.get("period")
            if period not in (1, 2):
                if chunk:
                    chunk = []
                current_period = None
                previous_frame_number = None
                previous_positions = {}
                previous_ball = None
                last_frame = None
                continue
            period = int(period)
            frame_number = int(frame_data["frame"])
            if current_period != period or (previous_frame_number is not None and frame_number != previous_frame_number + 1):
                chunk = []
                previous_positions = {}
                previous_ball = None
                last_frame = None
                current_period = period
            players = []
            for item in frame_data.get("player_data") or []:
                player_id = int(item["player_id"])
                if player_id not in player_team:
                    raise ValueError(f"Tracking id missing from pinned match metadata: {match_id}:{player_id}")
                xy = finite_xy(item)
                players.append((player_id, item.get("is_detected") is True, xy))
            ball = frame_data.get("ball_data") or {}
            ball_xy = finite_xy(ball)
            chunk.append({"frame": frame_number, "players": players,
                          "ball_detected": ball.get("is_detected") is True,
                          "ball_xy": ball_xy})
            previous_frame_number = frame_number
            last_frame = chunk[-1]
            if len(chunk) == WINDOW:
                _, previous_positions, previous_ball = flush_chunk(period, chunk, previous_positions, previous_ball)
                chunk = []
    if source_hash.hexdigest() != tracking_row["sha256"]:
        raise ValueError(f"Pinned tracking checksum mismatch: {tracking_path}")
    match_qc.append({"match_id": match_id, "home_team_id": home_id, "away_team_id": away_id,
                     "pitch_length_m": pitch_length, "pitch_width_m": pitch_width,
                     "period_frames": {str(k): int(v["duration_frames"]) for k, v in
                                        ((int(p["period"]), p) for p in metadata["match_periods"])},
                     "complete_3s_chunks": chunks, "admitted_windows": emitted,
                     "excluded_roster_capacity": rejected_roster,
                     "excluded_no_valid_state_in_history_or_future": rejected_support,
                     "trailing_incomplete_frames_discarded": len(chunk),
                     "tracking_sha256": tracking_row["sha256"],
                     "match_metadata_sha256": match_row["sha256"]})
    print(f"match={match_id} windows={emitted} roster_excluded={rejected_roster} "
          f"support_excluded={rejected_support} chunks={chunks}", flush=True)

def assemble(records: list[dict[str, np.ndarray]], matches: list[int], split: dict[str, Any],
             source_metadata: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not records:
        raise ValueError("No windows were prepared")
    arrays: dict[str, np.ndarray] = {
        "states": np.stack([x["states"] for x in records]).astype(np.float32, copy=False),
        "valid": np.stack([x["valid"] for x in records]).astype(bool, copy=False),
        "position_valid": np.stack([x["position_valid"] for x in records]).astype(bool, copy=False),
        "real_observed_mask": np.stack([x["real_observed_mask"] for x in records]).astype(bool, copy=False),
        "teams": TEAMS.copy(),
        "match_ids": np.asarray([x["match_id"].item() for x in records], dtype=str),
        "clip_ids": np.asarray([x["clip_id"].item() for x in records], dtype=str),
        "scenario": np.asarray([x["scenario"].item() for x in records], dtype=str),
        "attacking_team": np.asarray([x["attacking_team"].item() for x in records], dtype=np.int64),
        "attack_direction": np.asarray([x["attack_direction"].item() for x in records], dtype=np.int64),
        "frame_times_s": np.stack([x["frame_times_s"] for x in records]),
        "source_frame_ids": np.stack([x["source_frame_ids"] for x in records]),
        "source_period": np.asarray([x["source_period"].item() for x in records], dtype=np.int64),
        "entity_ids": np.stack([x["entity_ids"] for x in records]),
    }
    matches_array = arrays["match_ids"]
    indices: dict[str, np.ndarray] = {}
    for key, group in split["groups"].items():
        expected = set(group)
        actual = set(matches_array[np.isin(matches_array, sorted(expected))].tolist())
        if actual != expected:
            raise ValueError(f"Locked {key} matches not all represented: expected={sorted(expected)} actual={sorted(actual)}")
        indices[key] = np.flatnonzero(np.isin(matches_array, sorted(expected))).astype(np.int64)
    arrays.update({f"{k}_indices": v for k, v in indices.items()})
    counts = {key: {str(mid): int(np.sum(matches_array[ix] == mid)) for mid in sorted(set(matches_array[ix]))}
              for key, ix in indices.items()}
    meta = {
        "data_kind": "real", "source": "SkillCorner Open Data",
        "source_url": source_metadata["source"], "source_commit": source_metadata["commit"],
        "license_record": "Repository LICENSE is MIT; repository README requests SkillCorner attribution."
                         " The data is provider-produced broadcast tracking; detected-only points are used as observations and targets.",
        "fps": FPS, "history_frames": HISTORY, "future_frames": FUTURE, "stride_frames": WINDOW,
        "coordinate_system": "centered physical meters scaled per source pitch to 105 x 68 m",
        "state_definition": "xy from finite provider coordinates with is_detected=true; vx/vy are backward differences only when consecutive frames are detected; unobserved/extrapolated positions are not targets",
        "valid_definition": "position is provider-detected now and in the immediately preceding source frame; velocity therefore has causal support",
        "position_valid_definition": "finite xy explicitly flagged is_detected=true",
        "windowing": "non-overlapping contiguous 30-frame windows within a single period; no period boundary crossing; 10 history + 20 future frames at 10 Hz",
        "split": {"method": "pre-locked match-level SHA256 assignment; no same-match overlap across splits",
                  "created_utc": split["created_utc"], "rule": split["rule"],
                  "prior_model_exposure": False,
                  "groups": {k: {"match_ids": split["groups"][k], "windows": counts[k]}
                             for k in ["train", "validation", "test"]}},
        "provenance": {"format": "SkillCorner Open Data tracking_extrapolated JSONL",
                       "matches": source_metadata["match_qc"]},
        "source_hashes": [{"match_id": mid, "tracking_sha256": source_metadata["tracking_hashes"][str(mid)],
                            "metadata_sha256": source_metadata["metadata_hashes"][str(mid)]} for mid in matches],
        "n_samples": int(len(records)), "n_matches": int(len(set(matches_array.tolist()))),
        "observed_state_fraction": float(arrays["valid"].mean()),
        "observed_position_fraction": float(arrays["position_valid"].mean()),
        "ball_position_fraction": float(arrays["position_valid"][:, :, 22].mean()),
    }
    return arrays, meta
