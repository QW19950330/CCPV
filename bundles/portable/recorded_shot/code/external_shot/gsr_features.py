"""Contract-driven released GSR player-history adapter; no event or model reads.

Only synthetic inputs are authorized at authoring time. A source contract and a
separate production execution authorization are required for real fixtures.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STREAM_SHA256 = "0b37e58916fea220d0f56ce8dcceadb3d65f6d3dfa6f9819419dfb3d774e9236"
FEATURE_NAMES = ["mean_abs_x", "mean_abs_y", "mean_x_squared", "mean_y_squared",
                 "mean_distance_nearer_goal", "min_distance_nearer_goal",
                 "n_current_players", "n_backward_velocity_players", "n_radial_velocity_players",
                 "fraction_backward_velocity", "fraction_radial_velocity", "speed_missing",
                 "radial_missing", "mean_speed", "rms_speed", "mean_toward_nearer_goal_velocity"]
VARIANTS = ["200ms_shared", "OLS1s_shared"]
OFFSETS = np.array([-25, -20, -15, -10, -5, 0], dtype=np.int64)
TIMES = OFFSETS / 25.
CENTRED = TIMES - TIMES.mean()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def _stream_class():
    path = HERE / "stream_json.py"
    if digest(path) != STREAM_SHA256:
        raise ValueError("pinned Stream source changed")
    spec = importlib.util.spec_from_file_location("pinned_gsr_stream", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Stream


def opaque(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str) and v.strip():
        return v
    return None


def exact_frame(image, prefix):
    iid, fn = image.get("image_id"), image.get("file_name")
    if not isinstance(fn, str) or not re.fullmatch(r"\d{6}\.jpg", fn):
        raise ValueError("unexpected image file_name")
    g = int(fn[:-4])
    if g < 1 or not isinstance(iid, str) or iid != prefix + fn[:-4]:
        raise ValueError("image_id/file_name mapping mismatch")
    return iid, g


def player_fields(record):
    if record.get("supercategory") != "object":
        return None
    attrs = record.get("attributes")
    if not isinstance(attrs, dict) or attrs.get("role") not in ("player", "goalkeeper"):
        return None
    box, xy = record.get("bbox_pitch"), None
    if isinstance(box, dict):
        x, y = box.get("x_bottom_middle"), box.get("y_bottom_middle")
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
            xy = (float(x), float(y))
    # No track_id fallback, team field, ball coordinate or future event access.
    return opaque(attrs.get("player_id")), xy, attrs["role"]


def geometry(x, v, L):
    """Exact preserved sixteen-feature formula; d=0 radial motion is missing."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    assert x.ndim == 2 and x.shape == v.shape and x.shape[1] == 2 and len(x) > 0 and np.isfinite(x).all()
    d = np.hypot(L/2-np.abs(x[:, 0]), x[:, 1]); vm = np.isfinite(v).all(1); rm = vm & (d > 0)
    speed = np.linalg.norm(v[vm], axis=1)
    radial = ((L/2-np.abs(x[rm, 0]))*np.sign(x[rm, 0])*v[rm, 0]-x[rm, 1]*v[rm, 1])/d[rm]
    return np.array([np.abs(x[:, 0]).mean(), np.abs(x[:, 1]).mean(), (x[:, 0]**2).mean(),
                     (x[:, 1]**2).mean(), d.mean(), d.min(), len(x), vm.sum(), rm.sum(), vm.mean(),
                     rm.mean(), int(not vm.any()), int(not rm.any()), speed.mean() if len(speed) else np.nan,
                     np.sqrt((speed**2).mean()) if len(speed) else np.nan, radial.mean() if len(radial) else np.nan], float)


def slopes(history):
    assert history.ndim == 3 and history.shape[1:] == (6, 2)
    return ((history[:, -1]-history[:, -2])/.2,
            np.sum(history*CENTRED[None, :, None], axis=1)/.7)


def validate_contract(c):
    if not isinstance(c, dict):
        raise ValueError("explicit contract required")
    if not isinstance(c.get("fixture"), str) or not c["fixture"]:
        raise ValueError("fixture must be an explicit nonempty string")
    if type(c.get("half")) is not int or c["half"] not in (1, 2):
        raise ValueError("half must be 1 or 2")
    if not isinstance(c.get("image_prefix"), str) or not c["image_prefix"]:
        raise ValueError("image prefix must be explicit")
    for k in ("first_frame", "last_frame", "input_bytes"):
        if type(c.get(k)) is not int or c[k] < 1:
            raise ValueError("invalid " + k)
    if c["last_frame"] < c["first_frame"] or not re.fullmatch(r"[0-9a-f]{64}", c.get("input_sha256", "")):
        raise ValueError("invalid domain or input hash")
    e = c.get("expected_info")
    if not isinstance(e, dict) or not {"name", "version", "frame_rate", "seq_length"} <= set(e):
        raise ValueError("explicit info name/version/frame_rate/seq_length required")
    if not set(e) <= {"version", "game_id", "id", "name", "frame_rate", "seq_length"}:
        raise ValueError("info contract field outside whitelist")
    if e["frame_rate"] != 25 or isinstance(e["frame_rate"], bool):
        raise ValueError("this fixed representation requires declared 25Hz")
    if type(e["seq_length"]) is not int or e["seq_length"] != c["last_frame"]-c["first_frame"]+1:
        raise ValueError("declared seq_length/domain mismatch")
    # Coordinates are already the provider's metre-valued pitch plane.
    # For this frozen deployment contract only the documented 105x68 frame is supported.
    if c.get("pitch_length_m") != 105.0 or c.get("pitch_width_m") != 68.0:
        raise ValueError("explicit released 105x68 metre coordinate contract required")
    if c.get("coordinate_units") != "metres_centered_pitch" or c.get("coordinate_transform") != "identity":
        raise ValueError("unplanned coordinate transform")


def extract_features(path, contract, *, max_seconds=600, max_keys=1000000):
    validate_contract(contract)
    started = time.monotonic()
    path = Path(path); before = path.stat()
    if before.st_size != contract["input_bytes"]:
        raise ValueError("input size mismatch")
    first, last = contract["first_frame"], contract["last_frame"]
    anchors = list(range(first+24, last+1, 75))
    wanted = {int(g+d) for g in anchors for d in OFFSETS if first <= g+d <= last}
    counts = Counter(); images = {}; image_frames = set(); selected = {}; info = None; images_done = False

    def guard():
        if time.monotonic()-started > max_seconds:
            raise TimeoutError("bounded GSR feature extraction timeout")
        if len(selected) > max_keys:
            raise RuntimeError("bounded selected player-key limit")

    Stream = _stream_class()
    with path.open("rb") as f:
        stream = Stream(f)
        for key, row in stream.records():
            if key == "info":
                info = {k: row.get(k) for k in contract["expected_info"]}
                if info != contract["expected_info"]:
                    raise ValueError("explicit info mismatch")
            elif key == "images":
                iid, g = exact_frame(row, contract["image_prefix"])
                if iid in images or g in image_frames:
                    raise ValueError("duplicate image_id or frame")
                if not first <= g <= last:
                    raise ValueError("image outside declared frame domain")
                images[iid] = g; image_frames.add(g); counts["images"] += 1
                if counts["images"] % 10000 == 0: guard()
            elif key == "images__end":
                images_done = True
            elif key == "annotations":
                if not images_done:
                    raise ValueError("annotations precede complete image map")
                counts["annotations"] += 1
                iid = row.get("image_id")
                if not isinstance(iid, str) or iid not in images:
                    raise ValueError("annotation missing image mapping")
                g = images[iid]; fields = player_fields(row)
                if fields is None:
                    counts["excluded_role_annotations"] += 1
                else:
                    pid, xy, role = fields; counts[role + "_annotations"] += 1
                    if pid is None: counts["missing_player_id_rows"] += 1
                    if xy is None: counts["invalid_coordinate_rows"] += 1
                    elif abs(xy[0]) > 52.5 or abs(xy[1]) > 34:
                        counts["finite_outside_pitch_rows_not_filtered"] += 1
                    if g in wanted and pid is not None:
                        k = (g, pid)
                        if k in selected:
                            old_n, _ = selected[k]
                            selected[k] = (old_n+1, None)
                            if old_n == 1: counts["duplicate_selected_identity_keys"] += 1
                        else:
                            selected[k] = (1, xy)
                if counts["annotations"] % 5000 == 0: guard()
            elif key == "categories":
                counts["category_records"] += 1
        if stream.bytes != contract["input_bytes"] or stream.hash.hexdigest() != contract["input_sha256"]:
            raise ValueError("complete input hash/size mismatch")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("input changed while reading")
    if info is None: raise ValueError("missing info")
    guard()
    by_frame = {}
    for (g, pid), (n, xy) in selected.items():
        if n == 1 and xy is not None:
            by_frame.setdefault(g, {})[pid] = xy
    X = np.full((len(anchors), 2, 16), np.nan, dtype=np.float64); ledger = []
    for i, g in enumerate(anchors):
        ids = sorted(by_frame.get(g, {}))
        history_frames = [int(g+d) for d in OFFSETS]
        hids = [pid for pid in ids if all(f in image_frames and pid in by_frame.get(f, {}) for f in history_frames)]
        if ids:
            xx = np.array([by_frame[g][pid] for pid in ids], float)
            vv = np.full((2, len(ids), 2), np.nan)
            if hids:
                hh = np.array([[by_frame[f][pid] for f in history_frames] for pid in hids], float)
                v200, vols = slopes(hh)
                if not np.isfinite(v200).all() or not np.isfinite(vols).all():
                    raise ValueError("nonfinite velocity arithmetic")
                lookup = {pid: j for j, pid in enumerate(ids)}
                j = [lookup[pid] for pid in hids]
                vv[0, j] = v200; vv[1, j] = vols
            X[i] = np.stack([geometry(xx, v, contract["pitch_length_m"]) for v in vv])
            if not np.array_equal(X[i, 0, :13], X[i, 1, :13], equal_nan=True):
                raise ValueError("shared first13 mismatch")
        missing_future = sum(f not in image_frames for f in range(g, g+251))
        eligible = bool(np.isfinite(X[i, :, :13]).all())
        ledger.append(dict(fixture=contract["fixture"], half=contract["half"], g=g,
                           row_key=f'{contract["fixture"]}|{contract["half"]}|{g}', candidate_index=i,
                           t0_ms=40*(g-1), end_ms=40*(g-1)+10000,
                           input_eligible=eligible, future_record_clock_complete=(missing_future == 0),
                           current_player_count=len(ids), six_point_history_count=len(hids),
                           history_image_frames_present=sum(f in image_frames for f in history_frames),
                           future_missing_image_count=missing_future,
                           input_unknown_reason="" if eligible else "NO_FINITE_CURRENT_PLAYER_OR_NONFINITE_FEATURE",
                           future_clock_reason="" if missing_future == 0 else "INCOMPLETE_RELEASED_IMAGE_CLOCK10"))
    missing_images = sorted(set(range(first, last+1))-image_frames)
    summary = dict(status="SOURCE_FEATURES_ONLY_NOT_OUTCOME_OR_EXTERNAL_SUCCESS", fixture=contract["fixture"],
                   half=contract["half"], input_sha256=stream.hash.hexdigest(), input_bytes=stream.bytes,
                   info=info, counts=dict(counts), selected_identity_keys=len(selected),
                   duplicate_scope="Selected current/history frames only; other roles/frames never affect inputs.",
                   actual_image_count=len(image_frames), declared_domain=[first, last],
                   missing_image_frames=missing_images, candidate_count=len(anchors),
                   input_eligible_count=sum(r["input_eligible"] for r in ledger),
                   future_complete_count=sum(r["future_record_clock_complete"] for r in ledger),
                   motion_missing_input_retained=sum(r["input_eligible"] and r["six_point_history_count"] == 0 for r in ledger),
                   no_coordinate_transform=True, no_future_xy_in_features=True,
                   no_ball_team_or_event_values_used=True, elapsed_seconds=time.monotonic()-started)
    return {"X": X, "ledger": ledger, "summary": summary}


def save_features(result, out):
    out = Path(out); out.mkdir(exist_ok=False)
    rows = result["ledger"]
    with (out/"anchor_ledger.csv").open("x", encoding="utf-8", newline="") as f:
        fields = list(rows[0]) if rows else ["fixture", "half", "g", "row_key", "t0_ms", "end_ms", "input_eligible", "future_record_clock_complete"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out/"features.npz", X=result["X"], motion_variants=np.array(VARIANTS),
                        feature_names=np.array(FEATURE_NAMES),
                        **{k: np.array([r[k] for r in rows]) for k in ("fixture", "half", "g", "row_key", "candidate_index", "t0_ms", "input_eligible", "future_record_clock_complete")})
    with (out/"summary.json").open("x", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True); parser.add_argument("--contract", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    save_features(extract_features(args.input, contract), args.out)
