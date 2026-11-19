# Table 4: frozen C/D/E/F objective comparison

This package supplies explicit-path preparation, frozen-weight natural-condition evaluation, and a bounded disposable one-step interface check for Table 4. It does not contain a full-training command. The original model and objective definitions are preserved; a preparation or one-step receipt does not establish training or full evaluation reproduction.

## Original matrix and numeric sources

| Cell | Velocity NLL history | Position Huber history |
|---|---|---|
| C | Teacher forcing with missing-state prediction fallback | Same |
| D | Autoregressive mixture mean | Same |
| E | Teacher forcing with missing-state prediction fallback | Autoregressive mixture mean |
| F | Autoregressive mixture mean | Teacher forcing with missing-state prediction fallback |

Each cell has five independent initialization pipelines (seeds 0–4), fixed four epochs, 24 batches per epoch and 96 Adam updates. The original complete training budget was 20 models and 1,920 updates, not a 50-epoch run. C/D each use 20 forward transitions per objective; E/F use two separate 20-transition branches. Both branches remain differentiable through feedback. No new full training is performed by this package's interface checks.

Table 4's player ADE is the natural, standard105x68, all-player, fixed-planned, ALL_MATCHES `mean_ADE` row at step 0 in the original `analysis_results/mean_point_all_validation.csv`. The five pipeline scores are summarized with mean and sample SD. Complete-path EnergyU is the natural, M32, standard105x68, all-player, fixed-planned, ALL_MATCHES `energy_projected_path` row in `analysis_results/trajectory_five_seed_summary.csv`; it averages the nonlinear scores of five pipelines. Their per-seed source is `trajectory_scores.csv.gz`. The separate populations are ADE256 and Energy32; they are not interchangeable.

Expected published display, used only for subsequent comparison:

| Cell | Player ADE256 | Player path EnergyU32 |
|---|---:|---:|
| C | 0.861791 ± 0.039604 | 0.706269 ± 0.030733 |
| D | 0.800769 ± 0.011955 | 1.032592 ± 0.049872 |
| E | 0.843036 ± 0.040294 | 0.689828 ± 0.032118 |
| F | 0.816172 ± 0.031308 | 1.043457 ± 0.052666 |

## Required explicit inputs

`input_manifest.json` pins the exact prepared factual cache, reused noise bank, and all 20 epoch0/epoch4 checkpoint pairs. It includes only hashes, sizes and generic cell/seed paths, not payloads or fixture/player identifiers. The checkpoint root contains `C_s0/epoch0.pt`, `C_s0/epoch4.pt`, and the corresponding folders for the other cells and seeds. Only allowlisted checkpoint bytes are deserialized.

The factual input has 1,024 windows, a fixed 768/256 train/development split, 23 entities, ten history frames and twenty forecast frames at 10 Hz. Its SHA256 is `19970d4cbc56bd4c1eb98c772b2465925214aef6f77515246154dd9013ece7c3`. The original float32 M32 noise bank has SHA256 `52b5e41d9630f5e3d2e9e4c0999ba84494b9936732f2a5140e1bcd5e07fd219b`; no new noise is generated. Raw provider records and upstream cache extraction are outside this entry point.

Use the original numerical environment when testing exact identity: Python 3.11.5, NumPy 1.26.4 and PyTorch 2.7.0+cu126. The dependency file identifies the library versions; selecting an appropriate PyTorch wheel is environment-specific. The original evaluation uses CPU float32, one Torch thread, deterministic algorithms and the MATH attention backend. Switching a computation to GPU is not a claim of CPU bitwise identity.

```text
python -B /path/to/table4/runner.py prepare --data /authorized/input.npz --checkpoint-root /authorized/runs --noise /authorized/reused_validation_noise.npz --output /new/private/prepare
python -B /path/to/table4/runner.py evaluate --data /authorized/input.npz --checkpoint-root /authorized/runs --noise /authorized/reused_validation_noise.npz --output /new/private/table4_evaluation
python -B /path/to/table4/runner.py step-check --data /authorized/input.npz --checkpoint-root /authorized/runs --noise /authorized/reused_validation_noise.npz --output /new/private/C_seed0_step --cell C --seed 0 --batch-size 2 --device cpu
```

All input paths are explicit. The output directory must not exist and cannot sit inside the code or input/checkpoint directories. Outputs use exclusive creation. A partial output is preserved, without automatic retry. The evaluation checks `PAUSE_REQUESTED` between closed model cells. Code and original checkpoints remain unchanged.

`prepare` verifies the fixed data, support, clock, split, noise layout and all 40 checkpoint files. It performs zero model forwards or optimizer steps. `step-check` uses the original epoch1 shuffled first batch, optionally restricted to its first two items for a small CPU check. It preserves the original target masks, loss, initialization checkpoint, Adam learning rate 0.0003, zero weight decay and FP32 norm clip 200. C/D are compared with the original legacy loss, gradients, clip and first Adam state. E/F exercise the original dual-branch objective. Its disposable single-step artifact is explicitly unsuitable as a Table 4 trained checkpoint. `--batch-size 32 --device cuda` is available for a separately coordinated original-size GPU interface check; it still performs only one disposable primary step plus a C/D shadow step where applicable.

## Evaluation budget and preserved mathematics

The natural primary table needs 800 network forwards: 20 models × (20 mean-feedback steps at B256 + 20 sampled-feedback steps at B1024, representing 32 windows × 32 draws). Analytic CV uses no network forwards. The original complete evaluation used 4,400 forwards because it also generated two contexts over all 1,024 windows, missing-tail samples, missing-tail means, and two CV conditions. This Table 4 entry point does not claim to repeat those additional density and missingness diagnostics or the surrounding manuscript's complete experiment.

The unchanged model is the coupled two-dimensional velocity MDN with hidden size 32, three components, temporal attention and a spatial neighborhood radius of 20 m. State normalization is `[105, 68, 10.5, 10.5]` with offsets `[0.5, 0.5, 0, 0]`; generated positions follow the coupled affine update at 0.1 s. The standardized velocity sigma floor is 0.1 m/s, not 0.1 normalized units. Mean feedback, sampled feedback, current-anchor handling and persistent real-observation age are copied from the frozen source. No clipping, draw filtering, noise replacement or output-based eligibility choice is added to sampling.

FP64 scoring retains the original arithmetic: observed time points average within an actor, actors within a window, windows within each match, then matches equally. The EnergyU norm is the observed-time projected path RMS norm, distinct from ordinary ADE. Its unordered-pair U correction and shared-draw deletion formula are preserved. A source-empty planned window yields NA in `fixed_planned`; `source_observed_conditional` excludes only such truth-empty windows. Numeric failures are never dropped. Both player and BALL support and both aggregation variants are exported, retaining unsupported BALL cases.

`source_provenance.json` records byte-exact copied files and verbatim function fragments. Explicit input/output orchestration and the primary-table-only scoring wrapper are new. Comparison with original saved outputs remains required before claiming a completed evaluation reproduction. No training reproduction, raw-source qualification, independent external validation, real intervention benefit, code ownership clearance, data redistribution permission or public release is implied by these local artifacts.
