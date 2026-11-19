# Table 15 / S11 / S12 diagnostic code candidate

This is a local review candidate for the completed fixed four-match SkillCorner response diagnostic. It contains actual source code, historical protocols/locks, finite dependency sources, tests and route metadata. It has not been uploaded, licensed on behalf of the authors, or validated as a complete public raw-to-result replay. No original scripts, locks, data, checkpoints or results were modified during packaging.

## Contents and integrity

The 52 finite source records comprise 50 byte-identical files and two separately identified normalized JSON copies. All 22 Python sources are unchanged. SOURCE_MANIFEST.json records each original workspace-relative identity, original SHA/size, package member, supplied SHA/size and transformations. workspace/ preserves source directory relationships; public_metadata/ contains only derived metadata for review and is deliberately not installed over historical files. Historical status fields describe their original freeze stage, not the current state of the completed experiment.

The five main entry points are run_skillcorner_response.py, skillcorner_route/prepare_test_prefixes.py and analysis/analyze_responses.py, recover_truth.py, score_factual.py beneath workspace/CCPV_external_response_audit_20261119/. The package also contains original full_intervention API/operator/reader/architecture dependencies and the original SkillCorner cache parser plus its import dependencies. Importing ccpv.data also imports targets.py; no event labels, fitting routine or event pipeline was run. Only source code is included for that dependency.

## What was verified here

verification/source_entry_validation.json records AST validation of 22 Python files, six --help/import entry checks, and four isolated synthetic checks: response aggregation/support, detected-only truth and factual denominators, prefix streaming/slot audit, and runner/CV/resume invariants. The isolated workspace contained only code and the small protocol, with no provider tracking, scientific prefix cache, response arrays or saved checkpoint. NumPy 1.26.4, PyTorch 2.7.0+cu126 (CPU calculations), pandas 2.3.1 and SciPy 1.11.1 were installed. Zero real predictions, saved-checkpoint loads, training updates or downloads occurred. The first validation-directory creation encountered Windows MAX_PATH; it ended before entry-point execution and was retained. The successful validation used extended Windows paths; no mathematical source was changed.

The bounded privacy audit scans this package only for common private home paths, access-token formats and private-key headers. Zero matches is a finite pattern result, not a universal privacy guarantee. The ZIP receipt verifies every member SHA and rehashes the 52 original files unchanged.

## Reuse and exact remaining boundaries

DEPENDENCY_ROUTES.json maps byte-identical architecture files to the existing portable source candidate and the five D seed checkpoints to existing clean inference NPZ payloads. It also maps every implementation-lock pin to supplied code, an existing local asset reference or an explicitly absent input. These maps are identities, not permission to rewrite historical SHA guards.

1. The original runner reads full_intervention/assets.json, verifies the original checkpoint SHA, and calls torch.load expecting epoch=4 and state_dict. The clean inference package contains exact state tensors in numeric NPZ, with the outer training checkpoint and epoch field removed. It therefore cannot be substituted for the old checkpoint or pass its SHA guard. The independent clean-weight adapter is a reusable starting point; a separately versioned explicit loader/binding and all-six-path synthetic equivalence check are the next minimal implementation. No such bridge or full diagnostic replay is claimed here.
2. assets.json has 39 private absolute-path strings. Its normalized review copy uses WORKSPACE_ROOT placeholders, changes its bytes and cannot satisfy the historical assets SHA. The historical implementation lock itself is preserved exactly, including its original expected hash. run_commands.json has three path replacements and is likewise a derived template. The manifest records JSON pointers, original string hashes and replacement strings without republishing private path values. Never place these normalized files at the original path and call the historical lock satisfied.
3. The original lock uses Windows backslashes. Windows with appropriate long-path handling is the tested layout. Cross-platform execution requires an explicit new path-binding manifest; it is not certified by an AST/import check.
4. The complete original 20-match SkillCorner cache (116,411,793 bytes), new 4992-row test_prefixes.npz and its 6.4 MB metadata are not duplicated in this code-only package. Their exact hashes remain in the included protocol, prepared manifest and implementation lock. The original full-cache SHA requires rebuilding all 20 pinned matches with the supplied original parser/split, not a newly assembled four-match archive. Once the exact cache and four rosters are supplied, the original prefix extractor and hash gates define the next data-preparation route. The data files have known sources; absence from this small code package is not an unexplained scientific failure.
5. Raw truth recovery requires the four pinned tracking/roster pairs listed in analysis/truth_source_spec.json. It checks an explicit completed-prediction SHA before opening prefix/future data, and retains detected-only truth with NaN plus a false mask for missing positions. The source paths/SHA are known. Obtain the files from the fixed provider revision; no videos, events, Metrica or uncertified legacy GSR data are included or needed here.
6. The response and factual CLI scripts consume the full saved six-cell blocks, completion receipts and recovered truth/prefix support. The separate diagnostic_statistics candidate provides statistical results; it does not contain all state arrays and is not a substitute for the complete prediction input to these scripts. Factual scoring reads the prefix path recorded in the truth receipt: relocation needs newly verified recovery or an explicit provenance-preserving receipt mapping with a new SHA, never an unrecorded edit to the old receipt.

## Fixed source and licensing boundary

SkillCorner Open Data, revision 4340d274572876239c154c90bc507a9b3250a656: https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656 . The original MIT notice is included at workspace/CCPV_data_expansion_20261022/raw/skillcorner/LICENSE. The original download manifest pins the 20-match inventory and per-file sources; the truth source spec pins matches 1927964, 2006363, 2015213 and 2017461. These matches were previously exposed source-test fixtures, and the historical cache used future-qualified admission/identity support. This code does not transform them into a fresh blind test or causal counterfactual truth.

Author code MIT and model licensing remain proposals for owner confirmation; no new licence grant is made by this candidate. Provider notices apply only to their covered materials and do not automatically license author weights. See LICENSING_PROPOSAL.md.

## Safe code-only commands

From an extracted workspace/ directory, the three analysis entry points accept --help without opening real outputs. The two analysis test scripts accept a fresh receipt filename, for example:

```text
python -B CCPV_external_response_audit_20261119/analysis/analyze_responses.py --help
python -B CCPV_external_response_audit_20261119/analysis/test_response_analysis.py new_response_synthetic.json
python -B CCPV_external_response_audit_20261119/analysis/test_truth_analysis.py new_truth_synthetic.json
```

Use a new extraction for tests with fixed create-only receipt names. Do not invoke prepare or run merely to test installation: their real-data/checkpoint guards and missing inputs above must first be satisfied. The original protocol and numerical results are not altered by this packaging task.
