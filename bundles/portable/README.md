# CCPV revision code — local release candidate

This directory contains the four previously executed portable code packages as 291 unchanged text files. SOURCE_MANIFEST.json binds each file to its original archive and execution receipt. It is prepared for author review; it is not an existing public repository and no author reuse licence has yet been granted.

## Entry points and verified scope

| Directory | Scope |
|---|---|
| recorded_shot/code | Fixed raw-data extraction, label/clock alignment, development cross-validation and the frozen external recorded-shot task. Start with its README and per-module input contracts. |
| frozen_inference | Saved-model trajectory inference and saved-path/intervention/result reconstruction. Saved trajectories do not count as newly generated worlds. |
| raw_neural_training | Fixed raw SkillCorner/Metrica preprocessing, small five-seed trajectory training, and saved-path same-window kinematic scoring. |
| remaining_training | Fixed Table3 legacy arm, Table4 objective ablations, Table7 factorial and q-first heads, Table8 frozen-world transfer/calibration, and supplemental matched/dependence analyses. |

The historical directory named remaining_training/table8 supplies q-first rows in current Table7; remaining_training/table8_native supplies current Table8. Preserve this distinction.

Use the entrypoint-specific README for exact command arguments and manifests. The verified numerical environment is Python3.11.5, NumPy1.26.4, PyTorch2.7.0+cu126 where required; neural FP32 uses MATH SDPA, with deterministic settings and TF32 disabled. Some readouts require CPU float64 and one thread. This is not a cross-platform bitwise guarantee.

## Inputs and interpretation

Code alone does not supply all provider data. Each module identifies its frozen inputs and legal provider acquisition route. The companion local data/model/result assets preserve all selected seeds, missingness and unfavorable comparisons. Private original checkpoints are separate from clean inference candidates; the latter must not be substituted into historical byte-hash guards.

Do not interpret verified extraction, weight replay or table reconstruction as a new performance improvement or independent confirmation. The nine-fixture frozen external result concerns player-only recorded Shot within ten seconds. It does not demonstrate real Goal improvement, causal intervention benefit, real player credit or independent validity of the entire CCPV chain. The unsupported original420-match corpus is not reconstructed by these packages.

## Release and attribution

LICENSING_PROPOSAL.md records an unapproved author licence proposal, not a licence grant. Provider and third-party software terms remain separate. An actual public URL/version and author approval must be supplied before the manuscript claims public availability. No credentials are required to inspect the code; do not put access tokens into source or manifests.

The additional historical sources, source maps and provider-rights evidence remain in the author review delivery. This source candidate does not claim that every historical absolute-path program has become portable merely because it is archived.
