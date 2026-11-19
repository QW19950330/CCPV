# Table 7 fixed risk-head training

This entry reruns **240 fitted heads and 48,000 optimizer updates**, then predicts and scores. It does not merely aggregate saved predictions. It uses the original six exposed development matches, five seeds, eight cells, fixed 200 Adam updates, CPU float64 and one thread. The cached frontend features are frozen inputs; their trajectory models are not retrained here.

`code/runtime.py`, `risk_models.py`, `risk_data.py`, `risk_math.py`, and `score_factorial.py` are byte-for-byte copies of the original numerical source. Only `risk_io.py` supplies portable paths and create-only I/O. `run.py` verifies the complete input manifest and original numerical source before launching the original initialization, fit, readout and score stages. The input bundle includes original historical WithEarly checkpoints needed by the original exact-replay assertions and separately identified comparison references; none initialize fitted final weights.

Use Python 3 with the original numerical environment: PyTorch 2.7.0+cu126, NumPy version recorded in each run. All runtime paths are CLI arguments or derived from those arguments. Inputs, code and outputs may be placed independently; the working directory is irrelevant.

The verified bundle is a **local private-input reproduction**, not a completed public raw-data reproduction. A code-only release excludes `inputs/snapshots`, which contains about 649 MB of cached data and historical references. The manifest, portable binding and reference metadata specify the exact required files but do not replace permission to obtain them. `prepare_inputs.py --source-root AUTHORIZED_ORIGINAL_WORKSPACE --manifest MANIFEST_JSON --binding PORTABLE_BINDING_JSON --references REFERENCE_JSON --output NEW_INPUT_DIRECTORY` reassembles the frozen local bundle using only those listed sources. It does not regenerate cached prefixes from raw tracking data. That earlier cache-generation chain remains an upstream dependency of this entry.

Ordinary training still requires the historical WithEarly checkpoint/readout references because the unchanged original pipeline asserts their exact equality **after** independently fitting each WithEarly head and after predicting. Those references are comparison targets, never final-weight initialization. Removing these required assertions would change the original audit contract, so this delivery retains and identifies the dependency.

```powershell
python -B -X utf8 PATH_TO_CODE/run.py --inputs PATH_TO_INPUTS --output NEW_OUTPUT_DIRECTORY
python -B -X utf8 PATH_TO_CODE/verify.py --inputs PATH_TO_INPUTS --output NEW_OUTPUT_DIRECTORY --report-dir NEW_REPORT_DIRECTORY --manuscript CURRENT_MANUSCRIPT_MD
```

Output directories must not exist. A failed run is preserved and never automatically resumed or overwritten. Stages log PIDs, start/end times, exit codes, stdout/stderr hashes, and 30-second process-alive checks; the default hard timeout is 1,800 seconds. No new 50-epoch run, external-nine evaluation, WC2018 tuning, parameter search, frontend forward or frontend fit is performed.

Verification compares all 240 checkpoint members bitwise (including initial/final weights, optimizer state, labels, rows and gradients), all 240 histories bytewise, 60 initialization archives bitwise, all frozen readout members bitwise, and all five scoring CSV files bytewise. Changing timestamps and absolute runtime provenance in lock/process JSON files are deliberately not asserted identical; their arrays and referenced results are checked.

The current manuscript Table 7 contains four WithEarly factorial rows and two q-first joint rows. This entry alone reproduces the first four. The separate q-first entry supplies its 60 retrained heads; pass its completed output using `--qfirst-output` to export and check all six displayed rows. This does not relabel q-first fits as factorial fits.

The unfavorable original outcomes remain in all eight-arm tables and all 84 paired comparisons. Removing Early BCE retains the Early-prior initialization and is not pure Y10-only supervision. These exposed-match algorithmic comparisons are not independent external confirmation or verified football causal effects.
