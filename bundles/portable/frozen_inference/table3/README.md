# Table 3 frozen neural prediction entry

This standalone code candidate implements the models, their original 20-step loss, persistent observation provenance, causal masking, rollout, exact-time adapter, scoring and match/seed aggregation used by v10 Table 3. It depends only on the pinned Python packages and explicitly supplied frozen NPZ/checkpoint files. Original project entry points and local paths are not required. No data, window/entity identifiers, labels, predictions or weights are included in this code directory.

## Exact evidence scope

Table 3 uses the fixed final epoch-4 checkpoints (96 Adam updates per checkpoint), seeds 0,1,2,3,4, for B_free2d, C_coupled_tf and D_coupled_ar. It is not the historical 50-epoch experiment and does not select a best seed/epoch. Training used 768 windows from 12 SkillCorner matches. The repeated development evaluation retains 256 windows from four matches. The earlier Metrica evaluation retains 192 windows from three samples and reuses the same 15 trained checkpoints.

B has a learned position residual. C couples positions to the velocity mixture and was trained with teacher forcing; D uses the same coupling with autoregressive training. The original velocity Gaussian normalizer, scale bounds, SmoothL1 position auxiliary, separate observed-target denominators, missing-state fallback, and real-observation age semantics are retained. The coupled model has a rank-two position/velocity pushforward; it does not define a nonsingular 4D likelihood.

The displayed endpoint is natural-condition players at 2 seconds in standardized 105 x 68 metre-equivalent coordinates. Windows average within each match, matches receive equal weight within each seed, and the five seed values yield a mean and sample SD. CV is deterministic. Development displays B/C/D/CV; Metrica displays C/D/CV. The executable retains the original B external control as well, with all 16 cells in each cohort, both natural and tailgap2 conditions and all original reporting horizons/groups. The two populations remain separate.

Full source identities, exact excerpt provenance, reference data/checkpoint pins, expected Table 3 numbers and per-cell output fingerprints are recorded in the JSON files. The development summary source is forecasting/analysis/arm_summary.csv; the external summary source is forecast_confirmation/analysis/arm_summary.csv in the original studies.

## Environment and explicit inputs

Use Python 3.11.5 with requirements.txt. The original evaluation environment uses torch 2.7.0+cu126. CPU runs hide CUDA before importing torch. The code preserves deterministic algorithms, math-only SDPA, TF32 disabled and one CPU thread.

WEIGHTS_ROOT contains the 15 directories B_free2d_seed0 through D_coupled_ar_seed4, each with the original world_last.pt. The complete NPZs are development.npz, position_truth_v2.npz and external_exact.npz. Their exact filenames/hashes and sizes are in input_contract.json. Checkpoint bytes are verified before the original full checkpoint payload is deserialized.

Each output directory must be absent and its parent must exist. Existing artifacts are never overwritten. Do not place output inside WEIGHTS_ROOT or any input NPZ directory.

Input verification performs no model forward:

    python -B evaluate_table3.py check-inputs --development-data DEV/development.npz --development-truth DEV/position_truth_v2.npz --external-data EXT/external_exact.npz --weights-root WEIGHTS_ROOT --output NEW_CHECK_DIR --device cpu

Small CPU-only model, loss, metric and adapter checks use the first two fixed rows and seed 0 as a technical test, plus one batch32 timing measurement. They make no optimizer update or full-cohort scientific estimate:

    python -B cpu_smoke.py --development-data DEV/development.npz --development-truth DEV/position_truth_v2.npz --external-data EXT/external_exact.npz --weights-root WEIGHTS_ROOT --output NEW_CPU_CHECK_DIR

The public loss API is losses.Windows and losses.objective; it contains the original data batching and objective, without an optimizer/training launcher.

## Frozen full evaluation

Run only after the coordinating task authorizes the full evaluation and allocates its device. The required flag prevents accidental execution. There is no arm, seed, epoch, batch-size or tuning selector.

    python -B evaluate_table3.py evaluate --development-data DEV/development.npz --development-truth DEV/position_truth_v2.npz --external-data EXT/external_exact.npz --weights-root WEIGHTS_ROOT --output NEW_EVALUATION_DIR --device cuda --authorize-full-evaluation

Use --device cpu for explicitly authorized CPU execution. The full workload is 32 cells, 420 learned batch rollouts of 20 steps (8,400 model forward calls), zero optimizer updates, 253,952 scored rows, and about 101 MiB of uncompressed float32 predictions. Input NPZs total about 11.57 MB and 15 checkpoints about 3.48 MB.

Outputs include each cell's predictions/metrics, both cohort summaries, table3_numeric.csv, and immutable receipts. Reference array fingerprints and metric CSV bytes are checked without a numerical tolerance. Any discrepancy is preserved, reported as EVALUATED_REFERENCE_DIFFERENCE_REQUIRES_REVIEW and returns exit code 2; the code never adjusts predictions or silently accepts a numerical tolerance. CPU-vs-original-GPU floating point identity is not assumed.

This entry starts at the frozen prepared arrays and frozen weights. It does not certify raw tracking preparation, reproduce training, establish physical event timing, validate tactical causal effects, or authorize redistribution of third-party data/checkpoints. Full frozen evaluation has not been claimed merely by packaging the code or completing a small CPU test.
