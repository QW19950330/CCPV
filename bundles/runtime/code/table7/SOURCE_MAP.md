# Exact source and scope

| Role | Workspace-relative original | Portable location |
|---|---|---|
| Fixed protocol and original source | `CCPV_risk_objective_factorial_20261030` | `inputs/snapshots/CCPV_risk_objective_factorial_20261030` |
| Eligible rows, match IDs and labels | `CCPV_generated_domain_pilot_20261025/prefix_only_data.npz` | Same relative suffix under `inputs/snapshots` |
| Five D frontend feature caches and one CV cache | `CCPV_native_neural_bridge_20261026/neural_bridge/generated` | Same relative suffix under `inputs/snapshots` |
| Historical WithEarly replay and provenance | Original `data_binding.json` records | Identical bytes under their workspace-relative suffixes in `inputs/snapshots` |
| Portable binding | Same original binding, only `path`/`history` values relocated | `inputs/portable_binding.json` |
| Original risk weights/readouts/metrics | `CCPV_risk_objective_factorial_20261030/{models,initialization,evaluation}` and frozen JSON/NPZ files | Frozen comparison references under `inputs/snapshots` |
| Current displayed Table 7 | `CCPV_revision_v10_external9_20261118/content/manuscript_body.md`, rows 312–317 at validation | Passed explicitly to `verify.py`; source hash is in `comparison.json` |

Each copied asset has a SHA-256 and exact original workspace-relative source in `inputs/manifest.json`. Original JSON provenance is preserved bytewise, including historical absolute-path strings. Executable code does not depend on those old paths: runtime reads the relative portable binding, while comparison-reference paths come from explicit CLI roots. The five copied numerical modules and original immutable source are required to have identical SHA-256 hashes before launch.

The original risk study is a 2×2×2 factorial: frontend D/CV, Direct/Marginal parameterization, and With/No Early BCE. It trains 6 folds × 5 seeds × 8 cells = 240 heads. The current Table 7 displays only its four WithEarly cells plus two q-first rows from the independently reproduced q-first study. The full eight-cell scoring outputs, 84 contrasts, 480 by-match rows and 80 by-seed rows are retained.

The ordinary pipeline retains the original strict order: 60 training-only initialization audits, all 240 fitted weights, all frozen held-match predictions, then scoring. Existing WithEarly model weights are read only by the historical replay equality check after a head has been fitted from its original seeded initialization. Final reference weights are not used as starting weights.
