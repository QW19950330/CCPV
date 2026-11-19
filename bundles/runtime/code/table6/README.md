# Table 6 saved-world and same-window kinematics

This standalone entry recomputes the retained kinematic comparison from actual saved trajectory arrays and the matched recorded states. It does not copy old Table 6 values into an output table, generate new trajectories, fit models, read event outcomes or estimate causal effects.

`INPUT_ROOT` supplies the relative paths and byte hashes listed in `input_contract.json`: 137 files, including 63 trajectory arrays, their manifests, the original freeze metadata, recorded development/truth arrays, fixed prefixes and the historical summary used solely as an interoperability reference. Required inputs total 173,506,114 bytes. No data are bundled in the code candidate.

    python -B run.py --input-root INPUT_ROOT --output NEW_RESULT_DIR
    python -B table6.py --results NEW_RESULT_DIR --output NEW_TABLE_DIR

Each output directory must be absent with an existing parent. Runtime refuses output overlap with code and input files. Historical absolute references inside pinned metadata are resolved beneath the explicit input root; original metadata bytes are verified before use and are not edited. The complete 63-case product and 334-manifest original freeze membership are checked. No old execution locks or old output directories are reused.

The original metric module and complete calculation body are retained. All 13 result CSVs, including agent-step metrics, observed-reference masks, sampled-world own-anchor metrics, matched-observability comparisons, pooled/equal-match results and five-seed summaries, are compared byte-for-byte against the retained outputs. This covers 24 fixed training windows from 12 matches, both native and standardized coordinates, three distinct player groups, head/position-derived kinematics, recursive mean and 32-sample output, CV and five seeds for D/E. Unknown observations and zero-speed turn handling retain their original definitions.

`table6.py` calculates the six displayed rows from the emitted tables. The first two rows use complete AB draws, 24 × 32 per seed, for the two commanded actors. The natural-background rows use the factual observability mask; their recorded denominators differ by derivative support. Recorded physical observations are not counted 32 times as independent data. The p99 statistic pools within each seed before taking the five-seed mean and sample SD; it is not the mean of match quantiles.

This port starts at saved trajectories. It does not certify the upstream D/E training, stochastic world generator or full runner implementation; those remain separate dependencies. Passing commanded-actor constraints is imposed by the command law and is not evidence of learned action validity. Neither a 100% controlled-actor pass rate nor a whole-roster failure rate is a physiological classification. The new controlled path has no observed counterfactual truth. Current scientific limitations and adverse natural-background results remain unchanged.

Use Python 3.11.5, NumPy 1.26.4 and pandas 2.3.1. The code and inputs have not been publicly released by this local packaging step; data/weight redistribution rights and author approval remain separate.
