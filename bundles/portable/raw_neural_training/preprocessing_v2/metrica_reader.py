"""Retained raw readers only; no downloading or implicit local paths."""
from __future__ import annotations
import csv, hashlib, json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"

FILE_SPECS = {
    "README.md": (2901, "03fb97acc3545cc7428d0a07485921c5a1d405c1"),
    "documentation/events-definitions.pdf": (802916, "5f0d279e6b5d3a8028727c129190245b4c5e13e8"),
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv": (139446, "3c426dd4f442dd433bccf89e3634f91aa310cd8f"),
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": (32782073, "ff409a5680fd75acc7cebcf786827214c687478b"),
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": (32782961, "ce2e91c0e633426cec1146f4e728b72417454e3d"),
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv": (153897, "decccd47c1d53960d290cb6679337eec83a8a97f"),
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": (29627711, "e7334d8d762954a215c49e55f211855000243fdc"),
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": (31890338, "8b4d3648888cc8b033e7142919ebaea04f6328e0"),
    "data/Sample_Game_3/Sample_Game_3_events.json": (1709486, "992b4cb502f0b0bc1244174f014f3f7ecb6d38ef"),
    "data/Sample_Game_3/Sample_Game_3_metadata.xml": (81806, "74b55a80f6aeeb8f8df7fd26e4b3a94e2cc1738d"),
    "data/Sample_Game_3/Sample_Game_3_tracking.txt": (53443115, "7062b7ceaa44d53345a4625b8f447efc74895b1d"),
}

def fingerprints(path: Path) -> dict:
    size = path.stat().st_size
    git = hashlib.sha1(f"blob {size}\0".encode())
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            git.update(block)
            sha.update(block)
    return {"bytes": size, "git_blob_sha1": git.hexdigest(), "sha256": sha.hexdigest()}

@dataclass
class TrackingGame:
    match_id: str
    positions: np.ndarray  # [frame, source identity including ball, xy], physical m / NaN
    identities: list[str]
    teams: np.ndarray
    frames: np.ndarray
    periods: np.ndarray
    times: np.ndarray
    forced_break_frames: list[int]
    provenance: dict

def normalized_to_meters(xy: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    """Preserve source top-left convention: +x right, +y down; center at zero."""
    return ((np.asarray(xy, np.float32) - np.float32(.5)) * np.asarray(pitch, np.float32)).astype(np.float32)

def read_csv_game(raw: Path, game: int) -> TrackingGame:
    tables, names = [], []
    for side in ["Home", "Away"]:
        p = raw / f"data/Sample_Game_{game}/Sample_Game_{game}_RawTrackingData_{side}_Team.csv"
        with p.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            next(reader); next(reader)
            header = next(reader)
        d = pd.read_csv(p, skiprows=2)
        if (len(header) - 3) % 2 or header[-2] != "Ball":
            raise ValueError("Unexpected Metrica CSV paired xy header")
        ids = [f"{side.lower()}:{header[i].strip()}" for i in range(3, len(header)-2, 2)]
        if len(ids) != len(set(ids)) or any(not x.split(":", 1)[1] for x in ids):
            raise ValueError("Player identities must be nonempty and unique per team")
        tables.append(d); names.append(ids)
    home, away = tables
    clocks = ["Period", "Frame", "Time [s]"]
    if home.shape[0] != away.shape[0] or not np.array_equal(home[clocks].to_numpy(), away[clocks].to_numpy()):
        raise ValueError("Home/away clocks differ; refusing to discard or invent rows")
    frames = home.Frame.to_numpy(np.int64)
    if len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError("Source frame IDs must be strictly increasing")
    times = home["Time [s]"].to_numpy(np.float64)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), np.diff(frames)/25, atol=1e-7, rtol=0):
        raise ValueError("CSV source clock must be native 25 Hz")
    arrays = [d.iloc[:, 3:].to_numpy(np.float32).reshape(len(d), -1, 2) for d in tables]
    hb, ab = arrays[0][:, -1], arrays[1][:, -1]
    hv, av = np.isfinite(hb).all(1), np.isfinite(ab).all(1)
    conflict = hv & av & ~np.isclose(hb, ab, atol=1e-7, rtol=0).all(1)
    ball = np.where(hv[:, None], hb, ab).copy()
    ball[~(hv | av) | conflict] = np.nan
    xy = np.concatenate([arrays[0][:, :-1], arrays[1][:, :-1], ball[:, None]], axis=1)
    ids = names[0] + names[1] + ["ball"]
    teams = np.asarray([0]*len(names[0]) + [1]*len(names[1]) + [2], np.int64)
    return TrackingGame(f"metrica:{game}", normalized_to_meters(xy, (105, 68)), ids, teams,
                        frames, home.Period.to_numpy(np.int64), times, [],
                        {"format": "Metrica CSV", "fps": 25, "pitch_m": [105, 68],
                         "pitch_source": "pinned official README", "ball_duplicate_conflict_frames": int(conflict.sum()),
                         "clock": "original Frame and Time [s]; no halftime wall-clock interpolation"})

def epts_metadata(path: Path) -> dict:
    root = ET.parse(path).getroot()
    fps = float(root.findtext("./Metadata/GlobalConfig/FrameRate"))
    if fps != 25:
        raise ValueError("Only native 25 Hz EPTS is supported")
    pitch = tuple(float(root.findtext(f"./Metadata/Sessions/Session/FieldSize/{key}")) for key in ["Width", "Height"])
    channels = {c.attrib["id"]: c for c in root.iter("Channel")}
    if any(channels[k].findtext("Unit") != "normalized" for k in ["x", "y"]):
        raise ValueError("EPTS channel units must explicitly be normalized")
    score = root.find("./Metadata/Sessions/Session/MatchParameters/Score")
    home, away = score.attrib["idLocalTeam"], score.attrib["idVisitingTeam"]
    player_team = {p.attrib["id"]: {home: 0, away: 1}[p.attrib["teamId"]] for p in root.findall("./Metadata/Players/Player")}
    channel_map = {c.attrib["id"]: (c.attrib["playerId"], c.attrib["channelId"]) for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.findall("./Metadata/GlobalConfig/ProviderGlobalParameters/ProviderParameter")}
    periods = []
    for number, label in enumerate(["first_half", "second_half", "first_extra_half", "second_extra_half"], 1):
        if params.get(label + "_start") and params.get(label + "_end"):
            periods.append((number, int(params[label+"_start"]), int(params[label+"_end"])))
    formats = []
    for e in root.findall("./DataFormatSpecifications/DataFormatSpecification"):
        if e.attrib["separator"] != ":":
            raise ValueError("Unsupported EPTS line separator")
        refs = list(e.iter("PlayerChannelRef"))
        ids = []
        for i in range(0, len(refs), 2):
            px, py = [channel_map[x.attrib["playerChannelId"]] for x in refs[i:i+2]]
            if px[0] != py[0] or (px[1], py[1]) != ("x", "y"):
                raise ValueError("EPTS player xy channels do not identify the same player")
            ids.append(px[0])
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate EPTS player channel identity")
        if [x.attrib["channelId"] for x in e.iter("BallChannelRef")] != ["x", "y"]:
            raise ValueError("Unsupported EPTS ball channels")
        formats.append({"start": int(e.attrib["startFrame"]), "end": int(e.attrib["endFrame"]), "players": ids})
    formats.sort(key=lambda x: x["start"])
    if not periods or not formats or any(a["end"] >= b["start"] for a,b in zip(formats,formats[1:])):
        raise ValueError("Missing or overlapping EPTS identity/period metadata")
    return {"fps": fps, "pitch_m": pitch, "player_team": player_team, "period_ranges": periods, "formats": formats}

def read_epts_game(raw: Path) -> TrackingGame:
    base = raw / "data/Sample_Game_3"
    meta = epts_metadata(base / "Sample_Game_3_metadata.xml")
    ids = sorted(meta["player_team"]) + ["ball"]
    lookup = {p:i for i,p in enumerate(ids)}
    frames = np.arange(min(x[1] for x in meta["period_ranges"]), max(x[2] for x in meta["period_ranges"])+1, dtype=np.int64)
    xy = np.full((len(frames), len(ids), 2), np.nan, np.float32)
    seen = np.zeros(len(frames), bool)
    periods = np.zeros(len(frames), np.int64)
    for period, first, last in meta["period_ranges"]:
        selected = (frames >= first) & (frames <= last)
        if (periods[selected] != 0).any():
            raise ValueError("EPTS periods overlap")
        periods[selected] = period
    formats = meta["formats"]; fi=0
    with (base / "Sample_Game_3_tracking.txt").open(encoding="utf-8-sig") as stream:
        for line in stream:
            fields = line.strip().split(":")
            if len(fields) != 3:
                raise ValueError("Malformed EPTS line")
            frame = int(fields[0]); row = frame - int(frames[0])
            while fi+1 < len(formats) and frame > formats[fi]["end"]:
                fi += 1
            f = formats[fi]
            if not (0 <= row < len(frames)) or seen[row] or not f["start"] <= frame <= f["end"] or periods[row] == 0:
                raise ValueError("EPTS frame duplicate, out of order or outside declared identity/period range")
            values = np.fromstring(fields[1].replace(";", ","), sep=",", dtype=np.float32)
            ball = np.fromstring(fields[2], sep=",", dtype=np.float32)
            if len(values) != 2*len(f["players"]) or len(ball) != 2:
                raise ValueError("EPTS data width disagrees with frame-specific XML identities")
            xy[row, [lookup[p] for p in f["players"]]] = values.reshape(-1,2)
            xy[row, -1] = ball; seen[row] = True
    return TrackingGame("metrica:3", normalized_to_meters(xy, meta["pitch_m"]), ids,
                        np.asarray([meta["player_team"][p] for p in ids[:-1]]+[2], np.int64),
                        frames, periods, frames.astype(np.float64)/25,
                        [f["start"] for f in formats[1:]],
                        {"format": "FIFA EPTS", "fps": 25, "pitch_m": list(meta["pitch_m"]),
                         "pitch_source": "original XML FieldSize", "period_ranges": meta["period_ranges"],
                         "identity_format_ranges": formats, "missing_source_frame_rows": int((~seen).sum()),
                         "clock": "original frameCount/25; not wall clock; period from XML"})
