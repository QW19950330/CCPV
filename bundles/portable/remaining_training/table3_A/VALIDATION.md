## Material Passport

- Mode: actual fixed-budget CUDA training and deterministic verification.
- Source: retained `CCPV_consistent_dynamics_outcomes_pilot_20261024/forecasting/runs/A_raw4d_seed0` through `seed4`.
- Role: complete an existing model-training dependency for feasibility/game outputs; no new selection or independent evaluation.

## Completed execution

The relocated code and working directory at `validation/table3_A` completed all five A models from the original seeded random initialization. Each ran four epochs, 24 batches per epoch and 20-step unroll: **480 real Adam updates and 9,600 forward calls** in total. The new run is `validation/table3_A/run01`, with logs in `logs01`.

The CUDA process completed in 192.172 seconds with exit code 0 and empty stderr. There was no training failure or automatic retry. `validation/table3_A/completion.json` reports `COMPLETE_A5_ACTUAL_CUDA_TRAINING_AND_STRICT_PARITY`.

## Strict equality

`validation/table3_A/comparison.json` reports `EXACT`, five cells and zero differences. The complete final payload comparison includes 605 tensors plus NumPy arrays and scalars: final model state, Adam state, Python/NumPy/torch/CUDA/loader RNG state and full four-epoch history. All five configurations match, and every history JSON is byte-identical to its retained original.

All 20 new immutable epoch snapshots retain model/optimizer/RNG state and have exactly matching historical history prefixes. The original intermediate model weights were not retained, so this delivery does not claim equality of those unavailable weights. Checkpoint container bytes can differ because of serialization names; the full payload comparison uses exact tensor and array bytes, without a numerical tolerance.

`source_postflight.json` confirms all 34 original/source-reference files and copied inputs are unchanged, and all 13 relocated code/contract/environment files match `prefit_freeze_v2.json`.

## Source and process audit

The nine original training/model/loss modules are byte-identical to the already verified Table 3 training entry. The contract differs only in `arms` and `reference_results`, selecting the original A arm and five seeds. Runtime and loss already support the raw four-dimensional NLL branch, so no mathematical adaptation was needed.

An independent static review is preserved as `peer_prefit_review.json`. It identified that a hard timeout should stop the owned Windows process tree. The logging wrapper was repaired before any training, with the prior freeze retained and a new `prefit_freeze_v2.json`. `peer_prefit_review_v2.json` closes that operational finding. No numerical source or contract changed during that repair. The actual run completed normally and did not invoke the timeout path.

## Boundaries

The original private development NPZ and position-truth NPZ are required inputs; original A checkpoints are comparison targets and never initialize the new final weights. Private inputs/references are excluded from the code-only ZIP. This is a local exact reproduction of the original four-epoch A dependency, not a new 50-epoch run or proof of independent external performance. No WC2018 tuning, external-nine evaluation or new parameter selection occurred.
