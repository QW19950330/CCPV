# Fixed-cache motion LOFO90 reconstruction

This entry point reconstructs the complete original development head and readout matrix underlying Table S6: six held fixtures, five seeds and three model arms, for 90 native-only fitted heads. Each frozen head is evaluated under both native and grid conditions. The grid condition does not introduce another 90 trained heads. This study is separate from the earlier Table 13 LOFO60 experiment and the later 15 deployment heads.

## Authorized inputs and execution

Use authorized copies of these exact original files:

| Input | Bytes | SHA256 |
|---|---:|---|
| Six-point `features.npz` | 2803791 | b5f94c95aa8d14d60a1b7d5600e2b6a6f090e81b4a64958ee0aa2f8dd8528d07 |
| `history_coverage.csv` | 859428 | 0781f1789082190a58dab27304a613a8d75d6f1e6746a42b18f25ba1e8651e37 |

The original numerical environment is Python 3.11.5 and NumPy 1.26.4, CPU float64 with one BLAS thread. Other environments may change last-bit arithmetic. The portable code package includes no data, source fixture/player identifiers, trained weights or machine-specific paths.

```text
python -B /path/to/development_motion_lofo/replay.py --features /authorized/path/features.npz --coverage /authorized/path/history_coverage.csv --output /new/private/path/reconstruction
```

Run from any working directory. The output directory must not exist; all outputs use exclusive creation. Inputs remain read only. Creating `PAUSE_REQUESTED` inside the new output directory stops before the next fit without removing completed outputs. Wall and output limits are 1,800 seconds and 512 MiB, with no automatic retry.

## Preserved original definition

- The cache has shape `[2 conditions, 2 motion variants, 11542 candidates, 16 features]`. All original labels and eligibility states remain fixed: 9,483 eligible windows, 363 positives and 2,059 unknown labels. Missing motion does not introduce new window exclusions.
- The three arms are `G_shared` (13 features), `GM200_shared` (16) and `GM_OLS1s` (16). They share six-point history availability. The two motion representations retain different temporal information; OLS uses six samples over one second and the 200-ms variant the last pair. This reconstruction does not reinterpret their difference as pure denoising or instantaneous-velocity accuracy.
- Every fit excludes its held fixture and weights the other five fixtures equally, with uniform weights within each fixture. Each arm fits its normalizer only on its native training features. The first 13 means/scales must match exactly across arms. Missing values use weighted finite training means, with zero/one fallback for wholly missing columns. Frozen native normalizers also serve grid readout.
- Initialization is the original `default_rng(20261118 + seed).normal(0, 0.01, 16)`, truncated to 13 for G, with training-only prior-logit intercept. Each head uses exactly 200 Adam updates at learning rate 0.003, betas 0.9/0.999 and epsilon 1e-8, global gradient-norm cap 10, and ridge `0.005 * sum(w*w) / 2` excluding the intercept. There is no adaptive stopping, checkpoint selection, calibration or grid training.
- All 90 heads freeze before any held readout. Both conditions then use the same original eligible rows. The grid features already represent the original fixed 1.05 m by 0.68 m, origin-zero, ties-to-even quantization of all six history coordinates and current geometry; no new stress conditions are added here.
- All predictions freeze before scoring. Every original fixture metric, five-seed equal-six-fixture mean, paired contrast, fixture contrast and degradation interaction is preserved. Logloss uses the original `[1e-15, 1-1e-15]` probability clip; Brier and bias use unmodified probabilities. Prevalence is one deterministic training prior per fold, repeated across conditions and only for paired-seed display.

The four numerical functions, feature names, scoring constants, metric function and aggregate block are extracted verbatim from the frozen original sources; `source_provenance.json` records fragment and actual-file hashes. Outputs include 90 heads with initialization, optimizer state, all 200 history steps and final diagnostics, 18 arm/fold normalizers, 303,456 held prediction rows, and all original score tables. Unsupported fixed folds fail without substitution, as in the original motion study.

The supplied coverage CSV is copied byte-for-byte, checked against cache identities and available feature counts, and aggregated into `coverage_summary.json`. It remains a frozen input, not regenerated history evidence. Old 200-ms support, source periods and frame numbers in that ledger are not independently re-extracted by this CLI.

## Interpretation and release boundary

This is a replay of already-exposed development evidence from a fixed feature cache. Adverse native costs, adverse grid fixtures, nonzero final gradients and unknown labels remain present. Five-seed variability is not an independent-fixture confidence interval. The replay does not certify raw labels, event clocks, source acquisition, upstream history extraction, a production error distribution, complete CCPV performance or real-world intervention/goal benefit. No external fixtures, external labels, model-selection input or formal50 training is used.

Generated private outputs contain source identities and fitted weights. Portability of the code does not authorize their redistribution. Code ownership, licensing, third-party input access and redistribution permissions, and an actual public versioned release remain author responsibilities; no public release or submission readiness is asserted.
