# Portable clean-weight diagnostic adapter v2

This new entry closes the old-checkpoint loader/path dependency for the Table15/S11/S12 response runner. The v1 archival code ZIP and all original scripts, locks, checkpoints and results remain unchanged. This is a local, unpublished candidate; author code/model licensing is still a proposal.

## What now runs independently

run_clean_diagnostic.py uses only this extracted adapter package, an explicitly supplied existing clean inference-weight package and the exact explicitly supplied prepared prefix NPZ. It does not read the original assets.json, old pickle checkpoints, old absolute paths, old workspace model code, labels, future targets or saved historical responses. The original runtime sources are included byte-exact; the existing clean inference_adapter.load_export is included byte-exact and verifies NPZ hash/keys/shapes/dtypes. The bridge additionally verifies every tensor numeric-byte hash and strict-loads all31 tensors into the original D architecture. Original runner.paths and compact implement all six paths, candidate eligibility, masks, NONE checks, sham, request limits, rejections and metrics without mathematical edits.

clean_diagnostic_binding.json is a new explicit identity chain, not a replacement historical SHA. It binds all5 original checkpoint SHAs, epoch4/96-update evidence, clean NPZ SHA/array specs, exact source/protocol hashes and the original4992-prefix SHA. Original result.json epoch evidence is included without change. The original assets and implementation-lock SHAs are retained as provenance identities; old_guard_satisfied remains false. No fake epoch pickle is synthesized and no historical guard is bypassed under its old identity.

## Actual validation and its limit

validation_protocol.json was sealed before opening the prefix arrays or running this bounded check. It fixed original row positions[0,1], batch2, all5 D seeds plus CV and all6 paths. These are source_indices5094/5095 from skillcorner:1927964, both eligible. The new clean CLI ran successfully. The original checkpoint counterpart then ran the same rows at the same batch2 using the same byte-exact model/API/operator. All72 output arrays and155 state tensors were bitwise equal; statuses, support, rejection reasons, checks and parameter hashes were equal. None of the five original checkpoints or15 source pins changed.

See same_batch_equivalence_receipt.json and adapter_seal.json. No training, future-truth read or full4992 forward occurred. The test does not compare batch2 with historical batch32 results and does not establish full-cohort or cross-platform bitwise equivalence. The two windows are fixed technical examples from one previously exposed source-test fixture, not independent scientific validation or new efficacy evidence. Final extraction preflight verifies all5 clean models load from a new directory without any forward.

## Explicit external files

1. Extract CCPV_clean_inference_weights_local_candidate_20261119_v1.zip from ../inference_weights/ into any chosen weights directory. Its ZIP SHA is abef2dcb7742e36c0e4f9a50b40001390b6eb22f674eee3b34e72e30bf948f40. Only the five clean payloads referenced in the binding are read. No original checkpoint is required to use this portable entry.
2. Supply test_prefixes.npz, SHA a14904ed5a215805ce390fa0885b99e2e49752da096d47b119ba0019b0b8b656 (8,217,989 bytes), containing all4992 original planned rows. Its known local source is CCPV_external_response_audit_20261119/skillcorner_prepared/test_prefixes.npz; the pathname is not hardcoded or resolved implicitly by this entry. No prefix/data payload is duplicated in this code ZIP.
3. For rebuilding that input, use the v1 code candidate source/route metadata: fixed SkillCorner Open Data revision4340d274572876239c154c90bc507a9b3250a656, original20-match cache parser/split, and original test-prefix extractor. Provider tracking/rosters and the cache remain separate inputs; rebuilding the old entire source-cache SHA requires all20 source matches. The final source hashes, slot/identity/support contract and historical future-qualified cohort remain unchanged. Existing manifest/receipt files are historical evidence; create-only preparation must be run in a fresh destination with code/protocol inputs, not over archived outputs.

Runtime validation used Python3.11, NumPy1.26.4, PyTorch2.7.0+cu126 on CPU with original deterministic FP32/MATH settings. CPU PyTorch is sufficient. Very long Windows extraction paths need extended-path support or a short extraction root; no path is embedded in the binding.

## Commands

Run from this extracted adapter directory; replace the two explicit input paths. The binding SHA is d4ba1fa402cbff99424c006fd44dfb7f50a558e05443b0e4eb7dc0be285fa85f.

```text
python -B run_clean_diagnostic.py preflight --binding-sha256 d4ba1fa402cbff99424c006fd44dfb7f50a558e05443b0e4eb7dc0be285fa85f --weights-root WEIGHTS_DIRECTORY
python -B run_clean_diagnostic.py run --binding-sha256 d4ba1fa402cbff99424c006fd44dfb7f50a558e05443b0e4eb7dc0be285fa85f --weights-root WEIGHTS_DIRECTORY --prefixes TEST_PREFIXES_FILE --out NEW_TWO_ROW_DIRECTORY --rows 0 1
```

The optional --all-planned flag replaces --rows and enables all4992 source-order rows in batch32, all5 D seeds and CV. That full-cohort mode has not been executed in this task. It emits the original analysis schema plus an explicit new-binding identity; only a completed full cohort can use the original full-completion status. The two-row run uses a distinct bounded status and is deliberately rejected by the original full-cohort analysis gate. Outputs must not preexist; this minimal adapter does not resume or overwrite partial outputs.

The original analyze_responses.py can read a completed full run with its explicit completion SHA. recover_truth.py and score_factual.py still require pinned provider truth/prefix support and their own explicit hashes after prediction freeze. They are in the v1 code candidate; no real counterfactual truth or full value-chain validity is inferred from this bridge.

## Verification-only legacy comparison

validate_same_batch.py is the transparent local comparison source, not a runtime dependency. It deliberately requires an explicitly supplied original workspace/checkpoint set to reproduce the recorded old-versus-clean validation. Public use of run_clean_diagnostic.py does not import or run that file. Local portable_run01 arrays are excluded from the code ZIP, as are all raw tracking, future truth, private absolute-path metadata and original pickle checkpoints.

## Licensing

No author licence is granted by this candidate. MIT for author code and a separate model licence remain proposals for owner confirmation. The SkillCorner fixed provider MIT notice accompanies the code candidate for provenance and must remain with covered data; it does not assign the authors model or code licence. No StatsBomb, Metrica or uncertified legacy raw data are included.
