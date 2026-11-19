"""Rebuild frozen development motion features. No fitting or event parsing."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import frozen_geometry as old
from frozen_motion import OFFSETS, CONDITIONS, from_history
HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def pinned(path, expected):
    path = Path(path)
    if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
        raise ValueError("Input byte pin mismatch: " + path.name)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": expected["sha256"]}

def array_pin(a):
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha256_c_order_bytes": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}

def ledger_inputs(path, contract):
    record = pinned(path, contract["ledger"])
    with Path(path).open(encoding="utf-8-sig", newline="") as f: ledger = list(csv.DictReader(f))
    if len(ledger) != contract["candidate_rows"]: raise ValueError("Candidate count changed")
    y = np.array([int(r["y"]) for r in ledger], dtype=np.dtype(contract["original_feature_arrays"]["y"]["dtype"]))
    eligible = np.array([old.yes(r["eligible"]) for r in ledger], dtype=np.dtype(contract["original_feature_arrays"]["eligible"]["dtype"]))
    matches = np.array([r["match_id"] for r in ledger], dtype=np.dtype(contract["original_feature_arrays"]["match"]["dtype"]))
    grid = np.array([int(r["grid_index"]) for r in ledger], dtype=np.dtype(contract["original_feature_arrays"]["grid_index"]["dtype"]))
    if eligible.sum() != contract["eligible_rows"] or set(matches) != set(contract["matches"]):
        raise ValueError("Frozen cohort changed")
    if len(set(grid.tolist())) != len(grid): raise ValueError("Duplicate candidate grid")
    return ledger, y, eligible, matches, grid, record

def cached_histories(args, contract, ledger, matches, grid, inputs):
    for item in contract["histories"]:
        path = Path(args.history_root) / item["file"]
        inputs.append(pinned(path, item))
        with np.load(path, allow_pickle=False) as z:
            data = {name: z[name] for name in z.files}
        idx = np.flatnonzero(matches == item["match"])
        if not np.array_equal(data["rows"], idx): raise ValueError("History row order changed")
        if not np.array_equal(data["grid_index"], grid[idx]): raise ValueError("History grid order changed")
        if not np.array_equal(data["offsets"], OFFSETS): raise ValueError("History offsets changed")
        expected_t0 = np.array([int(ledger[i]["t0_UTC_ns"]) for i in idx])
        if not np.array_equal(data["t0_ns"], expected_t0): raise ValueError("History t0 changed")
        yield item["match"], idx, data

def raw_histories(args, contract, ledger, matches, grid, inputs, out, guard):
    # The caller must possess authorization to read these provider files.
    if not args.authorize_raw_xml:
        raise PermissionError("Raw mode requires --authorize-raw-xml")
    history_out = out / "past_history"
    history_out.mkdir(exist_ok=False)
    for mid in contract["matches"]:
        idx = np.flatnonzero(matches == mid)
        rr = [ledger[i] for i in idx]
        wanted = {(r["period"], int(r["N"]) + int(d)) for r in rr for d in OFFSETS}
        records = [r for r in contract["raw_sources"] if r["match"] == mid]
        meta = next(r for r in records if r["role"] == "matchinformation")
        pos = next(r for r in records if r["role"] == "positions")
        meta_record = dict(meta, path=str(Path(args.raw_root) / meta["file"]))
        pos_record = dict(pos, path=str(Path(args.raw_root) / pos["file"]))
        inputs.append(pinned(meta_record["path"], meta))
        roster, length = old.metadata(meta_record, mid)
        if length != 105.0: raise ValueError("Frozen pitch changed")
        ids = sorted(roster)
        points, clock, scan_record = old.scan(pos_record, mid, wanted, roster, guard)
        inputs.append(dict(scan_record, path=str(Path(pos_record["path"]).resolve())))
        hist = np.full((len(idx), len(ids), 6, 2), np.nan)
        pt = np.full((len(idx), len(ids), 6), -1, dtype=np.int64)
        ft = np.full((len(idx), 6), -1, dtype=np.int64)
        cm = np.zeros((len(idx), len(ids)), bool)
        hm = cm.copy()
        for j, (i, r) in enumerate(zip(idx, rr)):
            period, n, t0 = r["period"], int(r["N"]), int(r["t0_UTC_ns"])
            legacy, actual_t0 = old.make_features(points, clock, period, n, roster, length)
            if actual_t0 != t0 or bool(np.isfinite(legacy[:13]).all()) != old.yes(r["player_support"]):
                raise ValueError("Raw PLAYER clock/current support differs from frozen ledger")
            complete = t0 >= 0 and all(clock.get((period, n+k), -1) == t0+k*40_000_000 for k in range(251))
            if complete != old.yes(r["complete_recorded_clock10"]):
                raise ValueError("Raw PLAYER future-clock coverage differs from frozen ledger")
            for k, offset in enumerate(OFFSETS):
                ft[j, k] = clock.get((period, n+int(offset)), -1)
                for z, pid in enumerate(ids):
                    stamp, xy = points.get((period, n+int(offset), pid), (-1, np.full(2, np.nan)))
                    pt[j, z, k], hist[j, z, k] = stamp, xy
            _, cm[j], hm[j] = from_history(old, hist[j], pt[j], ft[j], t0, length, "native")
        data = dict(rows=idx, grid_index=grid[idx], player_ids=np.array(ids), offsets=OFFSETS,
                    xy=hist, player_times_ns=pt, frame_times_ns=ft,
                    current_mask=cm, history_mask=hm,
                    t0_ns=np.array([int(r["t0_UTC_ns"]) for r in rr]))
        with (history_out / (mid + ".npz")).open("xb") as f: np.savez_compressed(f, **data)
        del points, clock
        yield mid, idx, data
        guard()

def run(args):
    if sys.flags.optimize: raise RuntimeError("Run without -O; frozen function assertions are required")
    started = time.monotonic()
    contract = load_json(HERE / "input_contract.json")
    ledger, y, eligible, matches, grid, ledger_record = ledger_inputs(args.ledger, contract)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    inputs = [ledger_record]
    def guard():
        if (out / "PAUSE_REQUESTED").exists(): raise RuntimeError("Pause requested; partial outputs retained")
        if time.monotonic() - started > 1800: raise TimeoutError("1800-second resource guard; no fallback")
    receipt = {"status": "RUNNING", "mode": args.mode, "python": platform.python_version(),
               "numpy": np.__version__, "raw_xml_read_this_run": args.mode == "raw",
               "training": False, "inference": False, "raw_events_parsed": False,
               "labels_and_eligibility_recomputed_from_raw_events": False}
    try:
        X = np.full((2, 2, len(ledger), 16), np.nan)
        seen = np.zeros(len(ledger), bool)
        summary = []
        histories = cached_histories(args, contract, ledger, matches, grid, inputs) if args.mode == "history" else raw_histories(args, contract, ledger, matches, grid, inputs, out, guard)
        for mid, idx, data in histories:
            if seen[idx].any(): raise ValueError("Overlapping history rows")
            seen[idx] = True
            for j, i in enumerate(idx):
                guard()
                t0 = int(ledger[i]["t0_UTC_ns"])
                for ci, condition in enumerate(CONDITIONS):
                    features, current, history = from_history(old, data["xy"][j],
                        data["player_times_ns"][j], data["frame_times_ns"][j], t0, 105.0, condition)
                    X[ci, :, i] = features
                    if not np.array_equal(current, data["current_mask"][j]) or not np.array_equal(history, data["history_mask"][j]):
                        raise ValueError("History/current support mask changed")
                if not np.array_equal(X[0, 0, i, 6:8], X[1, 0, i, 6:8], equal_nan=True):
                    raise ValueError("Quantization changed current/history counts")
            summary.append({"match": mid, "rows": len(idx), "history_masks_exact": True})
            print("FEATURE_MATCH_COMPLETE", mid, len(idx), flush=True)
        if not seen.all() or not np.isfinite(X[:, :, :, :13][:, :, eligible]).all():
            raise ValueError("Incomplete candidate support")
        # Field order, dtype and save function follow the original executed source.
        arrays = dict(X=X, y=y, eligible=eligible, match=matches, grid_index=grid,
                      conditions=np.array(CONDITIONS),
                      motion_variants=np.array(["200ms_shared", "OLS1s_shared"]),
                      feature_names=np.array(old.NAMES))
        comparisons = []
        for name, a in arrays.items():
            actual = array_pin(a)
            expected = contract["original_feature_arrays"][name]
            comparisons.append(dict(array=name, expected=expected, actual=actual, bit_exact=actual == expected))
        if not all(r["bit_exact"] for r in comparisons):
            receipt["array_checks"] = comparisons
            raise ValueError("Feature array bit identity failed; no tolerance or input pin relaxation")
        feature_path = out / "features.npz"
        with feature_path.open("xb") as f: np.savez_compressed(f, **arrays)
        feature_record = {"path": str(feature_path.resolve()), "bytes": feature_path.stat().st_size, "sha256": digest(feature_path)}
        original = contract["original_features"]
        container_exact = all(feature_record[k] == original[k] for k in ("sha256", "bytes"))
        reference_checks = []
        if args.reference_features:
            reference_record = pinned(args.reference_features, original)
            inputs.append(reference_record)
            with np.load(args.reference_features, allow_pickle=False) as ref:
                if list(ref.files) != list(arrays): raise ValueError("Reference array field order differs")
                for name, a in arrays.items():
                    b = ref[name]
                    exact = a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")
                    reference_checks.append({"array": name, "bit_exact": exact, "shape": list(a.shape), "dtype": a.dtype.str})
            if not all(r["bit_exact"] for r in reference_checks): raise ValueError("Actual reference comparison failed")
        receipt.update(status="PASS_ALL_8_ARRAYS_BIT_EXACT", wall_seconds=time.monotonic()-started,
            candidate_rows=len(ledger), eligible_rows=int(eligible.sum()), inputs=inputs,
            per_match=summary, array_checks=comparisons, actual_reference_checks=reference_checks,
            output=feature_record, original_feature_byte_pin=original,
            npz_container_byte_exact=container_exact,
            downstream_original_input_pin_automatically_satisfied=container_exact,
            input_pin_relaxed=False,
            code_files=[{"file": p.name, "sha256": digest(p)} for p in sorted(HERE.glob("*.py"))],
            scope_limit="Features reconstructed from saved histories or raw PLAYER coordinates plus the exact audited candidate/label ledger. Event-clock mapping and y/eligibility remain inherited; no raw-event-to-full-chain replay claim.")
        write_json(out / "receipt.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "candidate_rows", "eligible_rows", "npz_container_byte_exact")}))
    except Exception as exc:
        receipt.update(status="FAILED_NO_SOURCE_REPAIR", error=repr(exc), inputs=inputs,
                       wall_seconds=time.monotonic()-started)
        write_json(out / "failure.json", receipt)
        raise

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("history", "raw"))
    p.add_argument("--ledger", required=True)
    p.add_argument("--history-root")
    p.add_argument("--raw-root")
    p.add_argument("--authorize-raw-xml", action="store_true")
    p.add_argument("--reference-features")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.mode == "history" and not a.history_root: p.error("--history-root required")
    if a.mode == "raw" and not a.raw_root: p.error("--raw-root required")
    run(a)

if __name__ == "__main__": main()
