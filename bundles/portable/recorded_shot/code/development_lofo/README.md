# Fixed-cache LOFO60 development reconstruction

This entry point reconstructs the original six-fold, five-seed, two-arm player-only development experiment underlying Table 13 and Tables S1-S2. It is separate from the later 15 deployment heads and their shared six-point-history features. It is a replay of already-exposed evidence, not a new held-out or external validation study.

## Input and execution

Use only an authorized copy of the original `features.npz`, exactly 784,832 bytes, SHA256 `b0c9cca42cb64e140afe9f4071ff16cae6ce3b01e9b36ebc9c703e6faff43797`. This code package does not include the feature cache, labels, source fixture or player identifiers, weights, credentials, or local source paths. Access and redistribution permission must be resolved separately.

The original numerical environment is Python 3.11.5, NumPy 1.26.4, CPU float64 and one BLAS thread. Other numerical environments may differ in their last bits. Install the single dependency in an appropriate environment, then run from any working directory:

```text
python -B /path/to/development_lofo/replay.py --features /authorized/path/features.npz --output /new/private/path/reconstruction
```

The output directory must not exist. All files use exclusive creation. The supplied input is read only. A `PAUSE_REQUESTED` file in the new output directory stops before the next head; completed output remains. The replay has the original 1,800-second and 512-MiB output budgets and no automatic retry.

## Preserved numerical definition

- All 11,542 cached candidates are retained, including 2,059 unknown labels; 9,483 eligible windows include 363 positives. Cached eligibility and labels are not regenerated or changed.
- Each of six fixtures is held once. Each fit uses only the other five fixtures, weighted equally; rows within each training fixture have equal weight.
- One 16-column finite-value, weighted training normalizer is shared within each fold. Missing values are imputed to the training mean; wholly missing columns use mean zero and scale one. G uses the first 13 normalized columns and GM all 16. This is the original 200-ms-motion cache, not the later shared-history cache.
- Each seed draws 16 coefficients from `default_rng(20261118 + seed).normal(0, 0.01, 16)`; G takes the first 13. The intercept is the logit of the training-only, equal-fixture prevalence.
- Each learned head has 200 fixed Adam updates: learning rate 0.003, betas 0.9/0.999, epsilon 1e-8, weight ridge `0.005 * sum(w*w) / 2`, no intercept penalty, and global gradient-norm cap 10. No loss-driven selection, early stopping or recalibration is added.
- All 60 planned head statuses are frozen before held readout, and all readouts before scoring. Unsupported folds retain their planned statuses and NA scores. PREVALENCE is one deterministic control per fold, repeated only for paired seed display.
- Logloss uses the original probability clip `[1e-15, 1-1e-15]`. The original fixture metrics, equal-six-fixture seed means and paired contrasts are retained without dropping adverse outcomes. Five-seed variation is not a match-level confidence interval.

The mathematical functions and scoring block are extracted from the original source without arithmetic edits; see `source_provenance.json`. Runtime outputs include private fold records, training indices, weights, optimizer states, all 200 history steps, held predictions, six-fixture metrics and all five-seed contrasts. Do not redistribute these generated data or weights merely because the source code is portable.

## Scope

This reconstructs frozen development features to 60 heads and their held results. It does not re-extract raw player histories, verify event clocks or raw labels, recreate every source-ledger exclusion reason, reproduce the full CCPV pipeline, or demonstrate intervention or goal benefit. There is no external dataset or external-label input. Original results remain read only. Code ownership, licensing, data permissions and a public repository/version remain author responsibilities; no release or third-party permission is asserted here.
