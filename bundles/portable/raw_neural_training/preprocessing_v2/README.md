# Retained trajectory input reconstruction

This local code candidate rebuilds the exact inputs used by the retained Table 3 experiments. It reads explicitly supplied, already-authorized source files. It contains no downloader, token, raw data, outcome labels or checkpoint. Python 3.11.5, NumPy 1.26.4 and pandas 2.3.1 were used for parity verification.

The input contract pins SkillCorner commit `4340d274572876239c154c90bc507a9b3250a656`. The Metrica reader pins commit `e706dd506b360d69d9d123d5b8026e7294b13996` and verifies Git blob SHA1 plus byte length for every required file. Supply a separate new output directory whose parent exists. Existing outputs and code/source trees are rejected. Failed outputs remain available for diagnosis.

## SkillCorner

`SKILLCORNER_MATCHES` contains the 20 match-number directories from the pinned source. The complete historical cache is reconstructed to preserve its exact global row numbering. This includes processing the four historical test-match raw files; no model fitting, test scoring or test-based choice occurs. The original fixed split and selection then retain exactly 768 training windows from 12 matches and 256 exposed development windows from four matches.

    python -B run.py skillcorner --raw SKILLCORNER_MATCHES --output NEW_SKILLCORNER_OUTPUT

Outputs include `skillcorner_10hz.npz`, `development.npz` and `position_truth_v2.npz`. The last file independently recovers observed positions from exact raw frame and entity IDs, preserving the distinction between an observed position and a supported velocity. No interpolation or zero-position target is substituted for an unavailable observation. Three output hashes are pinned in the input contract.

This is intentionally a historical reconstruction, not a new general-purpose adapter. The original roster/window admission, floating-point arithmetic, zero-erased full-state convention and fixed split are preserved. Some old metadata describes the historical time of first preparation; `skillcorner_historical_metadata.json` is explicitly historical and does not certify that the data remain unexposed today. The source split's descriptive stride field differs from the actual adapter's nonoverlapping 30-frame stride; computation follows the retained adapter. No data were made newly independent by rebuilding them.

The original velocity loop can misalign slots if a window contains fewer than 11 home identities, and a rejected roster chunk retains an earlier prefix state. These branches are not silently changed in this fixed-input reproduction. The separate raw-source audit found neither a materially incorrect velocity nor an incorrect velocity-support mask in the 1024 selected windows; all selected rosters contain 11 identities per side. This scope does not certify those branches for arbitrary new data or every row of the full historical cache.

## Metrica

`METRICA_ROOT` contains the pinned repository README and `data/Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3` tracking and metadata files.

    python -B run.py metrica --raw METRICA_ROOT --output NEW_METRICA_OUTPUT

This reads native 25 Hz CSV/EPTS clocks, rejects discontinuities/identity boundaries, selects slots using history only, and selects 64 windows per sample with the original deterministic rule. Exact native frames populate only the corresponding nominal 10 Hz slots; unobserved intermediate slots remain unavailable. Future positions are not interpolated. `external_exact.npz` contains all 192 retained windows and source-time/entity metadata.

## Separate truth recovery

    python -B run.py truth --raw SKILLCORNER_MATCHES --development DEVELOPMENT_NPZ --output NEW_TRUTH_OUTPUT

The integrated SkillCorner command already performs this step. A separate invocation is useful for checking position recovery against an existing fixed selection. The raw truth recovery checks original masks, frame periods, teams, entity IDs, source hashes and equality at fully observed states. It does not fit any model or change original arrays.

The validation receipts outside this distributable directory state exactly which full executions were compared. Source-code reproducibility is not permission to redistribute third-party data. No public repository or author-license clearance is claimed by this local candidate.
