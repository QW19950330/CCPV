"""Verify labels against the already audited event-clock cache, not raw events."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.dont_write_bytecode = True
import frozen_geometry as old
from build_features import HERE, load_json, ledger_inputs, pinned, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", required=True)
    p.add_argument("--event-clocks", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    contract = load_json(HERE / "input_contract.json")
    ledger, y, eligible, matches, grid, ledger_pin = ledger_inputs(a.ledger, contract)
    event_pin = pinned(a.event_clocks, contract["event_clocks"])
    with Path(a.event_clocks).open(encoding="utf-8-sig", newline="") as f:
        events = [r for r in csv.DictReader(f) if r["scenario"] == "AUTHOR3_PLUS_DIRECT_OTHER4" and r["match_id"] in contract["matches"]]
    if len({(r["match_id"], r["native_event_id"]) for r in events}) != len(events):
        raise ValueError("Duplicate audited event identity")
    by_match = {mid: [e for e in events if e["match_id"] == mid] for mid in contract["matches"]}
    for r in ledger:
        label, unknown, _ = old.anyshot(by_match[r["match_id"]], r["period"],
            int(r["t0_UTC_ns"]), old.yes(r["complete_recorded_clock10"]))
        keep = old.yes(r["player_support"]) and label >= 0
        if label != int(r["y"]) or keep != old.yes(r["eligible"]) or "|".join(unknown) != r["unknown_shot_ids"]:
            raise ValueError("Cached label/eligibility mismatch at grid " + r["grid_index"])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "receipt.json", {
        "status": "PASS_ALL_11542_CACHED_LABEL_ROWS", "rows": len(ledger),
        "eligible": int(eligible.sum()), "eligible_positive_windows": int(y[eligible].sum()),
        "ledger": ledger_pin, "event_clocks": event_pin,
        "raw_event_parse": False, "raw_tracking_scan": False,
        "future_PLAYER_clock_and_player_support_rederived": False,
        "clock_scenario": "AUTHOR3_PLUS_DIRECT_OTHER4",
        "scope": "Exact original anyshot rule replay on the audited clock cache, using frozen ledger PLAYER support/coverage. Does not recreate native event validity, direct/author identity mapping, physical event timing or candidate acquisition."
    })
    print("PASS_ALL_11542_CACHED_LABEL_ROWS")

if __name__ == "__main__": main()
