# Development event clocks and player-only labels

This package reconstructs the **existing** seven-match event-clock audit and the six-match, two-point PLAYER feature/label data used by the original 60-head development study. It does not train, select models, establish physical synchronization, or add independent matches. Existing outputs and raw inputs remain read-only. All output directories must be new.

Forty helper definitions originate from `audit_events.py` and `study.py`: 38 remain verbatim; two use basename-only file selection so unrelated parent directory names cannot change routing. The event assembly and player data blocks retain their computation statements. Execution adaptations supply the explicitly selected clock-case JSON and use basename-only input selection. `source_provenance.json` records original and adapted hashes and all routing replacements. Original entry points are never called.

Use Python 3.11.5 and the versions in `requirements.txt` for the verified environment. `numpy` operations and unknown-label rules are unchanged. CPU thread environment variables are fixed to one. Cross-platform or new-library numerical identity is not claimed.

## Inputs and execution

First use the companion `development_grid` and `development_clock_sources` packages to regenerate the candidate grid and raw clock proofs. Keep their source scan receipts. The explicit clock case is selected before running and is an existing diagnostic case, not an offset search.

```text
python -B aggregate.py --raw-root RAW_XML_DIR --author-root ELASTIC_DIR --direct-shot-mapping DIRECT_DIR/shot_mapping.csv --author-shots AUTHOR_DIR/author_shots.csv --explicit-clock-case CLOCK_DIR/explicit_clock_case.json --output NEW_AGGREGATE_DIR

python -B build_ledger.py --raw-root RAW_XML_DIR --candidate-grid GRID_DIR/prefix_grid.csv.gz --event-clocks NEW_AGGREGATE_DIR/event_clocks.csv --authorize-raw-xml --output NEW_LEDGER_DIR

python -B verify.py --aggregate NEW_AGGREGATE_DIR --reference-event-dir ORIGINAL_EVENT_DIR --ledger NEW_LEDGER_DIR --reference-player-dir ORIGINAL_PLAYER_DIR --output NEW_COMPARISON.json
```

`RAW_XML_DIR` contains the original files named in `input_contract.json`. `ELASTIC_DIR` contains `gt/` and `unsynced/` subdirectories. The first command reads 14 metadata/event XML and six author parquet files plus three explicit clock-proof inputs. The second streams six positions XML, reads six metadata XML, and consumes the newly generated complete seven-match candidate grid and event clocks. It does not consume any old label ledger or old feature arrays.

The reference contract contains only source identities, table digests and array fingerprints, not source rows, labels or arrays. The independent verifier reads every cell and every array byte from separately supplied original references; it imports no producer code or producer contract.

## Preserved semantics

- Candidate numbering is assigned on the original 13,372-row seven-match grid before selecting the six matches; 11,542 candidates retain their original global indices.
- Native event revisions, final deletions, cancellation ambiguity, roster checks, and strict author identity matching are unchanged. No nearest-frame match or fitted time offset is introduced.
- Two event-clock scenarios are retained. Development labels use `AUTHOR3_PLUS_DIRECT_OTHER4` only, with the originally specified six-match scope.
- Features use current and exact 200 ms previous PLAYER coordinates, not the newer shared-six-history criterion. Support remains exactly `isfinite(x[:13]).all()`: a missing previous point changes velocity counts/missing indicators and does not automatically exclude an otherwise supported current state. Future PLAYER timestamps are used only to determine full 10-second recorded-clock coverage; future coordinates are not prediction inputs or eligibility filters.
- Event membership is `(t0, t0 + 10 seconds]`. Unresolved event clock/validity anywhere in the relevant half yields an unknown label. A known positive does not rescue an incomplete timeline.
- Original labels, missingness, associations, feature arrays, row and field ordering are compared without numerical tolerance. Container-only differences, if any, are reported separately.

## Limits and release status

These files are a local author-code candidate. Third-party XML/parquet, event/person identifiers, feature arrays, labels, weights and credentials are not included. Obtain authorized source data through the documented providers and preserve their terms. Code ownership, licensing and a public release have not been asserted by creating this package.

Reconstructing a recorded clock does not validate true physical event time or completeness of all real-world shots. Replaying already exposed development data is not an independent experiment and does not support causal running recommendations, goal-probability gains, or the full CCPV value chain.
