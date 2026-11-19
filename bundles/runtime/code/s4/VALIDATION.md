## Material Passport

- Mode: actual fixed-budget CPU training and deterministic reproducibility verification.
- Source: `CCPV_matched_value_objective_pilot_20261118`, unchanged exposed-development population and frozen cached terminal features.
- Role: reproduce existing S4 results; no new external data, world forward, terminal generation, or parameter selection.

## Completed execution

The relocated code ran from `validation/s4` with explicit input-root, metadata and new-output arguments. All 30 heads were trained for 200 updates each (6,000 updates). Fitting took 182.30 seconds; the complete fit/readout/score execution took 192.94 seconds. All three stages exited 0 with empty stderr. Environment: Python 3.11.5, NumPy 1.26.4, PyTorch 2.7.0+cu126, CPU float64, one thread.

The completed output is `validation/s4/run01`. The original numerical modules remain byte-identical. The original plan's 907 absolute path records were converted to relative paths; every nonpath value was verified unchanged. No running code or plan was edited during execution.

## Strict comparison

`validation/s4/comparison01/comparison.json` reports `EXACT_FULL_TRAINING_READOUT_SCORING_AND_CURRENT_S4`:

- All 30 new checkpoint archives: all 1,140 fields bitwise identical, including model, inherited initial-state/scaler arrays, Adam and training metadata.
- All 30 full training-history files: byte-identical.
- All 22 frozen readout fields: bitwise identical.
- Every scoring CSV and readout diagnostic CSV: byte-identical (seven files total).
- Maximum absolute array difference: 0.
- Current supplementary Markdown and `Supporting_Information_working_v2.docx` S4 table: all seven rows exactly equal. The Word table has zero-based index 3.

`comparison01/TableS4.csv` was generated from this new run. `source_postflight.json` verified all 905 bound source records and all 71 original reference-freeze records. `create_only_guard.json` records the expected refusal to reuse the completed output, whose fitted-weight freeze remained unchanged.

## Retained adverse outcomes

The A_match logloss change remains −0.000304782469 ± 0.000094507133, improving in all five seeds but only three of six fixture means; its Brier difference remains adverse at +0.000013830701. B_replace remains −0.000044380265 ± 0.000084067271 with only two of five improving seeds. The new MC/MC result remains above the historical CV DirectWith logloss. All four cells, both targets, full fixture/seed outputs, and interaction contrasts remain in the run; no favorable selection was made.

## Failure and scope record

There were no training-stage failures. One separate one-line metadata helper failed at Python parsing before execution because of shell quoting. Its receipt is `validation/s4/metadata_helper_attempt1_failure.txt`; the corrected standalone helper verified the same path-only transformation without changing the active run or numerical sources.

This entry binds approximately 3.2 GB of already available authorized local inputs read-only, including 835 terminal shards. It does not copy or regenerate them. The code-only ZIP must exclude local `provenance/plan.json` and contains no raw/cache/model reference payloads. The portable metadata identifies required inputs but does not establish permission to redistribute them or a complete public raw-data-to-table pipeline.

The original saved `initial_*` head arrays and scalers are part of the fixed initialization contract; fresh Adam is used, and original final weights do not initialize the new fit. The old-head and historical-CV components of the factorial remain frozen context. These local, previously exposed six-match findings do not constitute independent new validation or real football intervention benefits.
