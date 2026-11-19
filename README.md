# CCPV: frozen code and reproduction routes

Source repository for **Trajectory-based model-input interventions and reference-dependent player attribution in football**.

Repository: https://github.com/QW19950330/CCPV  
Companion archive collection: **CCPV revision: frozen code, derived inputs, and evaluation artifacts**. Zenodo draft 24370617 has reserved DOI `10.5281/zenodo.24370617`; it is not yet published or registered in this snapshot. This reservation is not a claim that the files are publicly downloadable. See [release/public_manifest.json](release/public_manifest.json) for the exact archive filenames, sizes and SHA-256 identities.

## What is in this clone

| Directory | Purpose |
|---|---|
| `bundles/portable/` | Original portable sources for trajectory models, training preparation, frozen readouts and recorded-shot analyses. |
| `bundles/runtime/` | Original public input-composition and bounded preflight/saved-statistics code. |
| `bundles/diagnostic_exact/` | Archived Table 15/S11/S12 runner, prefix preparation, response analysis, detected-only truth recovery, factual scoring and their finite source dependencies. |
| `bundles/diagnostic_portable/` | Independent clean-NPZ diagnostic entry, exact architecture/operator sources, explicit source binding and epoch provenance. |

The Git tree is a **source view**, not the full execution-archive layout. It includes every Python source from the four approved code archives, byte-for-byte. Arrays, large execution manifests and archived verification receipts remain in the fixed-SHA companion archives. The full runtime archive includes 90 explicitly identified compatibility NPZ derivatives; those are not duplicated in Git.

[PUBLIC_FILE_MANIFEST.json](PUBLIC_FILE_MANIFEST.json) identifies the current source snapshot. [SOURCE_ARCHIVES.json](SOURCE_ARCHIVES.json) identifies its four source archives. Original bundled README and `LICENSING_PROPOSAL` documents preserve their historical freeze state. Their “pending licence”, “not published”, “large archive planned” and “adapter missing” statements are historical; current scope and rights are defined by this README, the root LICENSE and [release/license_scope.json](release/license_scope.json). Original hashes and guard semantics are unchanged.

## Start from a clone

Use Python 3.11 for the supplied helpers. A short checkout/output path is helpful on Windows; preserve exact bytes (`.gitattributes` disables line-ending conversion).

```sh
git clone https://github.com/QW19950330/CCPV.git
cd CCPV
python -B verify_source_tree.py
python -B materialize_code.py --help
```

The helpers use only the Python standard library. Each scientific component retains its own requirements. The previously validated diagnostic runtime used Python 3.11, NumPy 1.26.4 and PyTorch 2.7.0 on CPU; bundled analysis checks also used pandas 2.3.1 and SciPy 1.11.1. CPU calculations preserve the original deterministic FP32/MATH settings. A source syntax check does not certify other environments.

## Restore a complete execution package

Obtain the named companion ZIPs, without renaming them, and put them in one `DOWNLOADS` directory. `materialize_code.py` verifies the selected archive's size, SHA and member count before restoring it into a **new directory outside the clone**. It does not download, train, run models, change metadata, overwrite an output or replace old identities.

```sh
python -B materialize_code.py --bundle runtime --archives-root DOWNLOADS --out ../ccpv-runtime
python -B materialize_code.py --bundle diagnostic_exact --archives-root DOWNLOADS --out ../ccpv-diagnostic-exact
```

Other bundle names are `portable` and `diagnostic_portable`. The restored runtime includes its required compatibility arrays and complete manifests; these are deliberately absent from the Git source view. This restoration uses the approved clean runtime archive. Original dirty workspace ZIPs are not public inputs.

## Bounded public preflight and saved statistics

After restoring `../ccpv-runtime`, all required clean data/reference ZIPs must also be present in `DOWNLOADS`. The current flat [release/RELEASE_ARCHIVE_MAP.json](release/RELEASE_ARCHIVE_MAP.json) preserves original logical archive IDs and actual ZIP SHA values while resolving files by their published basenames. It is an explicit new location map, not an assertion that old private metadata bytes are unchanged.

Run from the clone, using new output directories:

```sh
python -B ../ccpv-runtime/compose_inputs.py --candidate ../ccpv-runtime --archive-root DOWNLOADS --public-archive-map release/RELEASE_ARCHIVE_MAP.json --components table6 --output ../ccpv-inputs
python -B ../ccpv-runtime/check_entry.py --candidate ../ccpv-runtime --inputs ../ccpv-inputs --component table6 --output ../ccpv-check-table6
```

The other component names are `table5`, `supplementary`, `table7`, `qfirst`, `native`, `s4` and `s5`. Read each original entry's documentation before choosing an execution stage. The supported checks stop at bounded preflight or reaggregate saved statistics; they do not establish full training replay.

Existing evidence: all eight components were assembled from the final clean archives without original-workspace fallback or hardlinks, and their bounded entries exited successfully. S5's 15 saved-statistics outputs and the supplementary route's eight outputs matched their frozen references byte-for-byte. This Git preparation did not repeat those scientific calculations.

## Table 15 / S11 / S12 clean-weight entry

The directly usable source entry is:

`bundles/diagnostic_portable/run_clean_diagnostic.py`

It uses explicit clean inference NPZ files and the pinned original prefix; it does not require old pickle checkpoints. Extract `CCPV_clean_inference_weights_local_candidate_20261119_v1.zip` into `WEIGHTS_DIRECTORY` (archive SHA `abef2dcb7742e36c0e4f9a50b40001390b6eb22f674eee3b34e72e30bf948f40`).

```sh
python -B bundles/diagnostic_portable/run_clean_diagnostic.py preflight --binding-sha256 d4ba1fa402cbff99424c006fd44dfb7f50a558e05443b0e4eb7dc0be285fa85f --weights-root WEIGHTS_DIRECTORY
```

This verifies and loads all five fixed D models without a model forward. The optional bounded run requires `test_prefixes.npz`, exactly 8,217,989 bytes and SHA `a14904ed5a215805ce390fa0885b99e2e49752da096d47b119ba0019b0b8b656`:

```sh
python -B bundles/diagnostic_portable/run_clean_diagnostic.py run --binding-sha256 d4ba1fa402cbff99424c006fd44dfb7f50a558e05443b0e4eb7dc0be285fa85f --weights-root WEIGHTS_DIRECTORY --prefixes TEST_PREFIXES_FILE --out ../ccpv-new-two-row-run --rows 0 1
```

The prefix is a separate input, not embedded in the code or statistical ZIP. Its reconstruction route uses the original fixed SkillCorner cache/parser and `skillcorner_route/prepare_test_prefixes.py` in the full `diagnostic_exact` archive. [SkillCorner Open Data revision 4340d274572876239c154c90bc507a9b3250a656](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656) supplies the pinned roster/tracking sources. Exact cache identity requires the original 20-match parser/split, not a new four-match shortcut.

Previously completed equivalence used predeclared original rows 0/1, all five D seeds plus CV and all six paths, with batch 2 on both old-checkpoint and clean-NPZ sides: 72 output arrays and 155 state tensors were bitwise equal. This is a bounded technical comparison, not a full 4,992-window rerun or a claim of equivalence across batch sizes/platforms. The new binding explicitly keeps `old_guard_satisfied=false`.

The original analysis, truth-recovery and factual-scoring entries are under:
`bundles/diagnostic_exact/workspace/CCPV_external_response_audit_20261119/analysis/`.

Their full-cohort completion/hash gates still apply. The two-row output is deliberately rejected by the original full-cohort analysis gate. Future truth recovery requires its explicit prediction-freeze SHA and pinned provider sources; statistical CSVs alone do not provide complete response/truth arrays. No observed trajectory is counterfactual truth for an edited movement.

## Evidence limits and licensing

This release provides bounded input checks, saved-statistic replay, frozen sources and limited clean-weight equivalence. **Whole-raw-data training, complete raw-to-paper reproduction and cross-platform numerical equivalence remain unverified.** Previously exposed source-test fixtures remain exposed, and their historical future-qualified admission is not converted into a new blind evaluation by packaging.

Author-owned code is licensed under [MIT](LICENSE), Copyright (c) 2026 Fang Chen and Jiqian Lin. Third-party code/data retain their own terms and attributions; see [release/THIRD_PARTY_NOTICES.md](release/THIRD_PARTY_NOTICES.md) and the original notices within the bundles. Author-owned eligible inference weights, documentation and statistics follow the approved CC BY 4.0 scope in [release/license_scope.json](release/license_scope.json); that grant does not relicense provider materials.

No provider raw mirrors, videos, original private-path checkpoint ZIPs, local execution logs, credentials or large data archives are included in this Git source tree. Publication of source/assets does not assert final manuscript approval or journal submission.
