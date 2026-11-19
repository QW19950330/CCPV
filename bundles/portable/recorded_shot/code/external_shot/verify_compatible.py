"""Compare a semantically identical rebuilt-head replay, retaining manifest identity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prediction-root", "score-root", "reference-prediction-root",
                 "reference-score-root", "freeze-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config, modules, config_sha = p.load_context()
    supplement = p.read_json(p.HERE / "compatibility_manifest.json")
    for rec in supplement["files"]:
        p.read_pinned(p.HERE, rec)
    cohort, current = p.verify_freeze(
        args.prediction_root, args.freeze_sha256, config, modules, config_sha)
    for rec in config["reference_outputs"]["prediction"]:
        p.read_pinned(args.reference_prediction_root, rec)
    for rec in config["reference_outputs"]["score"]:
        p.read_pinned(args.reference_score_root, rec)
    score_root = Path(args.score_root)
    sr = p.read_json(score_root / "score_receipt.json")
    p.need(sr["prediction_freeze_sha256"] == args.freeze_sha256 and
           sr["config_sha256"] == config_sha and sr["fits"] == 0 and
           sr["prediction_regenerations"] == 0 and sr["new_blinded_test"] is False,
           "score receipt lineage differs")
    p.need([r["name"] for r in sr["files"]] ==
           ["scores.json", "full_candidate_labels.json"], "score artifact set differs")
    for rec in sr["files"]:
        p.read_pinned(score_root, rec)
    out = p.fresh_output(args.out)
    comparisons = {}
    with np.load(Path(args.reference_prediction_root) / "predictions.npz",
                 allow_pickle=False) as old:
        p.need(set(old.files) == {"row_keys", "probabilities", "prior"},
               "reference arrays differ")
        p.need(np.array_equal(np.array(current.row_keys), old["row_keys"]),
               "all row keys must match")
        for key, a in (("probabilities", current.probabilities), ("prior", current.prior)):
            b = old[key]
            p.need(a.shape == b.shape and a.dtype == b.dtype,
                   "array shape/dtype mismatch")
            delta = np.abs(a-b)
            p.need(np.isfinite(delta).all() and np.max(delta, initial=0) <= 1e-14,
                   "full-array numerical mismatch")
            comparisons[key] = {
                "shape": list(a.shape), "entries": a.size,
                "array_equal": bool(np.array_equal(a, b)),
                "array_bytes_equal": a.tobytes() == b.tobytes(),
                "max_abs_difference": float(np.max(delta, initial=0))}
    labels = p.read_json(score_root / "full_candidate_labels.json")
    reference_labels = p.read_json(
        Path(args.reference_score_root) / "full_candidate_labels.json")
    p.need(labels == reference_labels, "full labels/unknown reasons differ")
    current_score = p.read_json(score_root / "scores.json")
    reference_score = p.read_json(Path(args.reference_score_root) / "scores.json")
    for key in ("n_all_candidates", "n_input_ineligible"):
        p.need(current_score[key] == reference_score[key], "full cohort count differs")
    a, b = current_score["scores"].copy(), reference_score["scores"].copy()
    current_manifest, reference_manifest = a.pop("manifest_sha256"), b.pop("manifest_sha256")
    p.need(current_manifest == current.manifest_sha256 and
           reference_manifest == config["head_manifest"]["sha256"],
           "unexpected manifest substitution")
    # Only this identity field differs legitimately. Its inferred numerical
    # parameter fingerprint was verified above against the original15 contract.
    delta = p.compare_tree(a, b)
    result = {
        "status": "PASS_COMPATIBLE_MANIFEST_REPLAY", "utc": p.utc(),
        "prediction_freeze_sha256": args.freeze_sha256,
        "current_head_manifest_sha256": current_manifest,
        "reference_head_manifest_sha256": reference_manifest,
        "inference_parameter_fingerprints_match_original15": True,
        "excluded_comparison_field": "scores.manifest_sha256 only; both identities verified and retained",
        "predictions": comparisons, "candidate_labels_exactly_equal": len(labels),
        "score_numeric_scalars_compared": len(delta),
        "score_max_abs_difference": max(delta, default=0),
        "score_tree_equal_except_verified_manifest_identity": a == b,
        "comparison_tolerance_abs": 1e-14,
        "training": False, "new_blinded_test": False,
        "scope": "All saved rows, original15 inference values, BAS labels and metrics. Does not itself rerun raw GSR extraction or training."}
    p.write_json(out / "comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
