"""Portable replay of the fixed nine-fixture player-only BAS Shot10 task.

No fitting, model selection, data acquisition or publication. All output
folders and files are create-only. Existing observations are exposed; a new
freeze made here is a replay boundary, not a new blinded external experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
from pathlib import Path, PurePosixPath
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = ["117093", "118575", "118576", "118577", "118578",
            "128057", "128058", "132831", "132877"]
MODULES = ("gsr_features", "frozen_predict", "event_labels",
           "cohort_contract", "stream_json")
REVISION = "325bd4c57ee8e135ab652f3921d274526ada496b"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def relative_path(root, name):
    p = PurePosixPath(name)
    need(not p.is_absolute() and p.parts and ".." not in p.parts
         and "\\" not in name and ":" not in name, "unsafe relative input path")
    root = Path(root).resolve()
    result = root.joinpath(*p.parts).resolve()
    need(result.is_relative_to(root), "input escapes explicit root")
    return result


def pin(path, name=None):
    path = Path(path)
    raw = path.read_bytes()
    return {"name": name or path.name, "bytes": len(raw), "sha256": sha(raw)}


def read_pinned(root, rec):
    path = relative_path(root, rec["name"])
    need(path.stat().st_size == rec["bytes"], "input bytes mismatch: " + rec["name"])
    raw = path.read_bytes()
    need(sha(raw) == rec["sha256"], "input SHA256 mismatch: " + rec["name"])
    return raw


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_npz(path, **arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def fresh_output(path):
    path = Path(path).resolve()
    need(not path.is_relative_to(HERE), "outputs must be outside the code package")
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_context():
    manifest = read_json(HERE / "package_manifest.json")
    for rec in manifest["files"]:
        read_pinned(HERE, rec)
    config = read_json(HERE / "fixed_inputs.json")
    need(config["planned_fixtures"] == FIXTURES, "fixed nine fixtures changed")
    need(config["source_revision"] == REVISION, "source revision changed")
    need(config["clock"] == {"frame_rate": 25, "anchor": "40*(g-1)",
                            "half_local": True, "native_shift_frames": 0},
         "native clock contract changed")
    need([(r["fixture"], r["half"]) for r in config["gsr"]] ==
         [(f, h) for f in FIXTURES for h in (1, 2)], "18 ordered halves required")
    need([r["fixture"] for r in config["bas"]] == FIXTURES, "nine BAS files required")
    for rec in config["gsr"]:
        need(rec["file"]["name"] == "gsr/" + rec["fixture"] + "/" +
             rec["fixture"] + ("_1st.json" if rec["half"] == 1 else "_2nd.json"),
             "fixed GSR relative filename changed")
        need(rec["contract"]["input_bytes"] == rec["file"]["bytes"] and
             rec["contract"]["input_sha256"] == rec["file"]["sha256"],
             "GSR contract file pin differs")
    for rec in config["bas"]:
        need(rec["file"]["name"] ==
             "bas/" + rec["fixture"] + "/" + rec["fixture"] + "_12_class_events.json",
             "fixed BAS filename changed")
    modules = {name: importlib.import_module(name) for name in MODULES}
    for rec in config["gsr"]:
        modules["gsr_features"].validate_contract(rec["contract"])
    return config, modules, sha((HERE / "fixed_inputs.json").read_bytes())


def load_batch(root, feature_rec, metadata_rec):
    raw = read_pinned(root, feature_rec)
    metadata = json.loads(read_pinned(root, metadata_rec))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(z.files == ["X"], "feature cache must contain exactly X")
        metadata["X"] = z["X"].copy()
    return metadata


def write_batch(out, fixture, half, batch):
    stem = fixture + "_" + str(half)
    write_npz(out / (stem + "_features.npz"), X=batch["X"])
    write_json(out / (stem + "_batch.json"),
               {"ledger": batch["ledger"], "summary": batch["summary"]})
    return {"fixture": fixture, "half": half,
            "features": pin(out / (stem + "_features.npz")),
            "batch": pin(out / (stem + "_batch.json"))}


def semantic_pins(bundle):
    values = []
    for h in bundle.heads:
        raw = (h.arm + "|" + str(h.seed) + "|" + str(h.variant)).encode("ascii")
        for a in (h.w, np.asarray(h.b), h.mean, h.scale):
            raw += np.asarray(a, dtype="<f8").tobytes()
        values.append({"arm": h.arm, "seed": h.seed, "sha256": sha(raw)})
    return {"heads": values, "prior_sha256": sha(np.asarray(bundle.prior, dtype="<f8").tobytes())}


def predict(args, replay):
    config, m, config_sha = load_context()
    # Explicit required path: never use the preserved module's historical default.
    bundle = m["frozen_predict"].load_frozen_heads(
        args.head_manifest, expected_manifest_sha256=(
            args.head_manifest_sha256 or config["head_manifest"]["sha256"]))
    need(semantic_pins(bundle) == config["inference_semantic_pins"],
         "frozen inference values differ from the original15; no model substitution")
    out = fresh_output(args.out)
    batches, half_records = [], []
    mode = "EXPOSED_FIXED9_CACHE_REPLAY" if replay else "EXPOSED_FIXED9_RAW_RECONSTRUCTION"
    write_json(out / "started.json",
               {"status": "PREDICTION_STARTED", "mode": mode, "utc": utc(),
                "fits": 0, "bas_payloads_opened": 0, "new_blinded_test": False})
    for rec in config["gsr"]:
        fixture, half = rec["fixture"], rec["half"]
        if replay:
            batch = load_batch(args.feature_root,
                               rec["cache"]["features"], rec["cache"]["batch"])
        else:
            # One streaming read, including source SHA/bytes; no BAS is opened.
            path = relative_path(args.data_root, rec["file"]["name"])
            batch = m["gsr_features"].extract_features(
                path, rec["contract"], max_seconds=args.max_seconds_per_half)
        batches.append(batch)
        half_records.append(write_batch(out, fixture, half, batch))
        print(json.dumps({"stage": "FEATURE_HALF", "fixture": fixture,
                          "half": half, "rows": len(batch["ledger"])}), flush=True)
    cohort = m["cohort_contract"].assemble_cohort(
        batches, [r["contract"] for r in config["gsr"]], FIXTURES)
    predictions = m["frozen_predict"].predict_frozen(bundle, cohort["X"], cohort["row_keys"])
    write_json(out / "full_candidate_ledger.json", cohort["all_ledger"])
    write_json(out / "eligible_keys.json", list(cohort["row_keys"]))
    write_npz(out / "predictions.npz", row_keys=np.array(predictions.row_keys, dtype="U"),
              probabilities=predictions.probabilities, prior=predictions.prior)
    need(load_context()[2] == config_sha, "code/config changed during prediction")
    receipt = {
        "status": "REPLAY_PREDICTIONS_FROZEN", "mode": mode, "utc": utc(),
        "new_blinded_test": False, "fits": 0, "bas_payloads_opened": 0,
        "planned_fixtures": FIXTURES, "config_sha256": config_sha,
        "head_manifest_sha256": bundle.manifest_sha256,
        "inference_semantic_pins": semantic_pins(bundle),
        "half_records": half_records,
        "n_candidates": cohort["n_candidates"], "n_input_eligible": cohort["n_input_eligible"],
        "n_input_ineligible": cohort["n_input_ineligible"],
        "files": [pin(out / n) for n in ("full_candidate_ledger.json",
                                        "eligible_keys.json", "predictions.npz")],
        "probability_sha256": sha(predictions.probabilities.astype("<f8").tobytes()),
        "limit": "Already-exposed fixed sources; no causal or full-CCPV validation."}
    write_json(out / "prediction_freeze.json", receipt)  # Commit last.
    result = pin(out / "prediction_freeze.json")
    print(json.dumps({"stage": "PREDICTION_FROZEN", **result,
                      "n_candidates": cohort["n_candidates"]}), flush=True)
    return result


def verify_freeze(root, expected_sha, config, modules, config_sha):
    root = Path(root)
    raw = (root / "prediction_freeze.json").read_bytes()
    need(sha(raw) == expected_sha, "prediction freeze SHA256 mismatch")
    freeze = json.loads(raw)
    need(freeze["status"] == "REPLAY_PREDICTIONS_FROZEN"
         and freeze["new_blinded_test"] is False
         and freeze["fits"] == freeze["bas_payloads_opened"] == 0
         and freeze["planned_fixtures"] == FIXTURES
         and freeze["config_sha256"] == config_sha
         and re.fullmatch("[0-9a-f]{64}", freeze["head_manifest_sha256"]),
         "prediction freeze scope changed")
    need(freeze["inference_semantic_pins"] == config["inference_semantic_pins"],
         "frozen numerical head identity changed")
    need([(r["fixture"], r["half"]) for r in freeze["half_records"]] ==
         [(f, h) for f in FIXTURES for h in (1, 2)], "frozen half set changed")
    need([r["name"] for r in freeze["files"]] ==
         ["full_candidate_ledger.json", "eligible_keys.json", "predictions.npz"],
         "frozen artifact set changed")
    files = {r["name"]: read_pinned(root, r) for r in freeze["files"]}
    batches = []
    for rec in freeze["half_records"]:
        stem = rec["fixture"] + "_" + str(rec["half"])
        need(rec["features"]["name"] == stem + "_features.npz" and
             rec["batch"]["name"] == stem + "_batch.json", "half cache names changed")
        batches.append(load_batch(root, rec["features"], rec["batch"]))
    cohort = modules["cohort_contract"].assemble_cohort(
        batches, [r["contract"] for r in config["gsr"]], FIXTURES)
    need(json.loads(files["full_candidate_ledger.json"]) == cohort["all_ledger"] and
         tuple(json.loads(files["eligible_keys.json"])) == cohort["row_keys"],
         "frozen complete cohort changed")
    need(all(freeze[k] == cohort[k] for k in
             ("n_candidates", "n_input_eligible", "n_input_ineligible")),
         "frozen denominators changed")
    with np.load(io.BytesIO(files["predictions.npz"]), allow_pickle=False) as z:
        need(set(z.files) == {"row_keys", "probabilities", "prior"},
             "prediction array set changed")
        keys, p, prior = tuple(z["row_keys"].tolist()), z["probabilities"].copy(), z["prior"].copy()
    need(keys == cohort["row_keys"] and p.dtype == prior.dtype == np.dtype("float64")
         and p.shape == (len(keys), 3, 5) and prior.shape == (len(keys),),
         "prediction structure changed")
    need(sha(p.astype("<f8").tobytes()) == freeze["probability_sha256"],
         "frozen probability array changed")
    predicted = modules["frozen_predict"].PredictionBatch(
        keys, p, prior, freeze["head_manifest_sha256"])
    return cohort, predicted


def score(args):
    config, m, config_sha = load_context()
    cohort, predictions = verify_freeze(
        args.prediction_root, args.freeze_sha256, config, m, config_sha)
    out = fresh_output(args.out)
    write_json(out / "bas_read_started.json",
               {"status": "EXPOSED_REPLAY_OUTCOME_READ_STARTED", "utc": utc(),
                "prediction_freeze_sha256": args.freeze_sha256,
                "new_blinded_test": False, "fits": 0})
    all_labels, bas_pins = [], []
    for rec in config["bas"]:
        raw = read_pinned(args.data_root, rec["file"])
        table = m["event_labels"].parse_event_payload(
            json.loads(raw.decode("utf-8-sig")), sha(raw), rec["fixture"])
        labels = m["event_labels"].label_windows(
            table, [r for r in cohort["all_ledger"] if r["fixture"] == rec["fixture"]])
        all_labels.extend(labels)
        bas_pins.append({"fixture": rec["fixture"], **rec["file"],
                         "label_rows": len(labels)})
    aligned = m["cohort_contract"].aligned_scoring_labels(cohort, all_labels)
    scored = m["frozen_predict"].score_frozen(predictions, aligned, FIXTURES)
    write_json(out / "full_candidate_labels.json", all_labels)
    write_json(out / "scores.json",
               {"mode": "EXPOSED_FIXED9_REPLAY", "new_blinded_test": False,
                "qualification": "PLAYER_ONLY_RELEASED_BAS_SHOT10_NOT_FULL_CCPV",
                "n_all_candidates": cohort["n_candidates"],
                "n_input_ineligible": cohort["n_input_ineligible"], "scores": scored})
    # Score rechecks persisted predictions; it never loads heads or does inference.
    verify_freeze(args.prediction_root, args.freeze_sha256, config, m, config_sha)
    write_json(out / "score_receipt.json",
               {"status": "REPLAY_SCORE_COMPLETE", "utc": utc(),
                "prediction_freeze_sha256": args.freeze_sha256,
                "config_sha256": config_sha, "bas_files": bas_pins,
                "files": [pin(out / n) for n in ("scores.json", "full_candidate_labels.json")],
                "fits": 0, "prediction_regenerations": 0, "new_blinded_test": False})
    print(json.dumps({"stage": "SCORE_COMPLETE", "known": scored["n_known"],
                      "unknown": scored["n_unknown_excluded"],
                      **pin(out / "score_receipt.json")}), flush=True)


def compare_tree(a, b, path="root", numbers=None):
    if numbers is None:
        numbers = []
    need(type(a) is type(b), "comparison type differs at " + path)
    if isinstance(a, dict):
        need(set(a) == set(b), "comparison keys differ at " + path)
        for key in a:
            compare_tree(a[key], b[key], path + "." + key, numbers)
    elif isinstance(a, list):
        need(len(a) == len(b), "comparison length differs at " + path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, path + "[" + str(i) + "]", numbers)
    elif isinstance(a, float):
        need(np.isfinite(a) and np.isfinite(b), "nonfinite scalar at " + path)
        numbers.append(abs(a-b))
        need(abs(a-b) <= 1e-14, "numerical mismatch at " + path)
    else:
        need(a == b, "comparison value differs at " + path)
    return numbers


def verify(args):
    config, _, _ = load_context()
    for rec in config["reference_outputs"]["prediction"]:
        read_pinned(args.reference_prediction_root, rec)
    for rec in config["reference_outputs"]["score"]:
        read_pinned(args.reference_score_root, rec)
    out = fresh_output(args.out)
    comparisons = {}
    with np.load(Path(args.prediction_root) / "predictions.npz", allow_pickle=False) as new, \
         np.load(Path(args.reference_prediction_root) / "predictions.npz", allow_pickle=False) as old:
        need(set(new.files) == set(old.files) == {"row_keys", "probabilities", "prior"},
             "prediction artifact arrays differ")
        need(np.array_equal(new["row_keys"], old["row_keys"]), "all prediction keys differ")
        for key in ("probabilities", "prior"):
            a, b = new[key], old[key]
            need(a.shape == b.shape and a.dtype == b.dtype, "prediction shape/dtype differs")
            delta = np.abs(a-b)
            need(np.isfinite(delta).all() and np.max(delta, initial=0) <= 1e-14,
                 "prediction mismatch")
            comparisons[key] = {"shape": list(a.shape), "entries": a.size,
                                "bitwise_numeric_equal": bool(np.array_equal(a, b)),
                                "max_abs_difference": float(np.max(delta, initial=0))}
    new_labels = read_json(Path(args.score_root) / "full_candidate_labels.json")
    old_labels = read_json(Path(args.reference_score_root) / "full_candidate_labels.json")
    need(new_labels == old_labels, "candidate labels or unknown reasons differ")
    new_score = read_json(Path(args.score_root) / "scores.json")
    old_score = read_json(Path(args.reference_score_root) / "scores.json")
    differences = compare_tree(new_score["scores"], old_score["scores"])
    for key in ("n_all_candidates", "n_input_ineligible"):
        need(new_score[key] == old_score[key], "candidate count mismatch")
    result = {"status": "PASS_FIXED9_REPLAY", "utc": utc(),
              "comparison_tolerance_abs": 1e-14, "predictions": comparisons,
              "candidate_labels_exactly_equal": len(new_labels),
              "score_numeric_scalars_compared": len(differences),
              "score_max_abs_difference": max(differences, default=0),
              "score_tree_exactly_equal": new_score["scores"] == old_score["scores"],
              "reference_files": [pin(Path(args.reference_prediction_root) / "predictions.npz"),
                                  pin(Path(args.reference_score_root) / "scores.json"),
                                  pin(Path(args.reference_score_root) / "full_candidate_labels.json")],
              "raw_gsr_reread": False, "training": False, "new_blinded_test": False,
              "scope": "Cache -> full cohort -> frozen15 predictions -> BAS labels -> all stored metrics. Not raw extraction replay, full training or full CCPV."}
    write_json(out / "comparison.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    raw = sub.add_parser("predict", help="stream all fixed18 GSR files; never read BAS")
    raw.add_argument("--data-root", required=True)
    raw.add_argument("--head-manifest", required=True)
    raw.add_argument("--head-manifest-sha256", help="explicit relocated/rebuilt manifest pin; inference values must equal original15")
    raw.add_argument("--out", required=True)
    raw.add_argument("--max-seconds-per-half", type=float, default=600)
    replay = sub.add_parser("replay", help="use all18 exact pinned original feature caches")
    replay.add_argument("--feature-root", required=True)
    replay.add_argument("--head-manifest", required=True)
    replay.add_argument("--head-manifest-sha256", help="explicit relocated/rebuilt manifest pin; inference values must equal original15")
    replay.add_argument("--out", required=True)
    scoring = sub.add_parser("score", help="read9 pinned BAS only after verifying prediction freeze")
    scoring.add_argument("--data-root", required=True)
    scoring.add_argument("--prediction-root", required=True)
    scoring.add_argument("--freeze-sha256", required=True)
    scoring.add_argument("--out", required=True)
    verification = sub.add_parser("verify", help="compare full saved replay with original outputs")
    for flag in ("prediction-root", "score-root", "reference-prediction-root",
                 "reference-score-root", "out"):
        verification.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args, False)
    elif args.command == "replay":
        predict(args, True)
    elif args.command == "score":
        score(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
