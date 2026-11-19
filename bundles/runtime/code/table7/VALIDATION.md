## Material Passport

- Mode: deterministic local run and reproducibility verification.
- Source: `CCPV_risk_objective_factorial_20261030`; same six exposed development matches and frozen cached prefix features.
- Deliverable role: reproducibility of fixed existing results, with no new external validation or model selection.

## Completed execution

The relocated code at `validation/table7/relocated_code` completed 240 real model fits, 200 Adam updates each, on CPU float64 with one thread. Initialization, training, readout and scoring took 144.406 seconds in total; fitting took 126.23 seconds. Every stage exited 0 with empty stderr. Environment: Python 3.11.5, NumPy 1.26.4, PyTorch 2.7.0+cu126.

`validation/table7/completion.json` reports `COMPLETE_STRICT_RETRAINING_AND_CURRENT_TABLE7_MATCH`. No training failure occurred. A separate expected overwrite refusal was captured by `create_only_guard.json`; the completed output was unchanged.

## Strict numerical checks

- All 240 checkpoint archives: 13,440 members bitwise identical, including weights, optimizer state, training labels, fold rows and gradient diagnostics.
- All 240 complete training histories: byte-identical.
- All 60 initialization archives: 1,140 members bitwise identical.
- All 11 frozen-readout members: bitwise identical.
- Five scoring CSV files: byte-identical, retaining 480 by-match, 80 by-seed, 16 summary, 84 paired-comparison and six coverage rows.
- Maximum absolute array difference: 0.
- All 825 original source/reference files and their copied snapshots: original SHA-256 unchanged.

Full records are in `validation/table7/comparison01/comparison.json` and `source_postflight.json`. Original metadata timestamps and process paths are not expected to repeat; result arrays and numerical CSV files are checked without tolerance.

## Current Table 7

`validation/table7/comparison01/Table7.csv` contains six rows. Its first four come from the new 240-head output at `validation/table7/run01`; its two q-first rows come from the separate new 60-head output at `validation/table8_relocated/run01`. `Table7_six_row_provenance.json` records both new weights, readout and summary identities row by row.

Every displayed row equals the current `CCPV_revision_v10_external9_20261118/content/manuscript_body.md`. The complete six-row table also matches the current clean DOCX table at zero-based index 6 cell for cell; see `docx_comparison.json`.

## Preserved interpretation and limits

All original adverse outcomes and all eight factorial cells remain in the output. The primary D-Marginal NoEarly-minus-WithEarly logloss difference remains positive, with all five seeds worsening after auxiliary Early BCE removal. Early-prior initialization remains present in NoEarly models. The q-first negative primary finding is retained by its separate entry.

This run begins with frozen cached frontend features. It does not train or run trajectory frontends, repeat 50-epoch experiments, tune WC2018, or evaluate external-nine fixtures. The same six exposed matches cannot provide new independent confirmation. Current manuscript Table 8 is a separate transfer/intercept workflow.

This result certifies local execution with an available private 649 MB input/reference bundle. It does not certify a public code-only reproduction or the raw-data-to-prefix-cache preparation chain. A code ZIP excludes those data and old models. Its metadata and `prepare_inputs.py` support reassembly only for a reader who already has the authorized source files. Historical WithEarly references remain mandatory post-fit/post-readout equality checks in the unchanged original pipeline.

Executable code has no hardcoded host paths. The actual training used relocated code plus explicit independent input/output roots, with the research workspace as its working directory. Historical path strings remain only in byte-preserved provenance JSON; runtime loads the portable binding instead.
