# Table 3 fixed-budget training

This entry trains the retained B_free2d, C_coupled_tf and D_coupled_ar models from their original deterministic random initializations. It never initializes from an old checkpoint. The three arms use seeds 0–4, four epochs, 24 batches per epoch, batch size 32, 20-step unroll, Adam learning rate 0.0003, zero weight decay and gradient clipping at 200. Only fixed epoch 4 is used. There is no best-checkpoint, seed, learning-rate or architecture search.

The original model modules, dataset/missingness augmentation, teacher-forcing versus autoregressive objective, position/velocity denominators and batch order are retained. The two-dimensional velocity density has a 0.1 m/s sigma floor in standardized pitch coordinates, equivalent to 0.1/10.5 normalized velocity units. Training uses deterministic algorithms, FP32, TF32 disabled, MATH SDPA and one CPU thread.

Use Python 3.11.5 and the pinned package versions in `requirements.txt`. The CUDA build is the one used by the original retained experiment; CPU execution is available, but cross-device bit identity is not promised.

    python -B train.py --development-data DEVELOPMENT_NPZ --development-truth POSITION_TRUTH_NPZ --output NEW_RUN --device cuda

The two inputs must match the fixed byte hashes in `training_contract.json`. They can be obtained with the accompanying raw preprocessing entry. The script verifies row IDs, entities, frame IDs and masks before using the recovered position truth. A fresh output directory with an existing parent is required; code and input trees cannot be outputs.

The default is all 15 models: 1440 Adam updates and 28800 forward calls. For a staged reproduction, use `--cells B_free2d_seed0`, then a distinct run for the other 14 cells. This selector changes execution scope only; it cannot change seeds, settings, input cohort or reporting criteria.

Every epoch preserves a separate snapshot containing model weights, Adam state, Python/NumPy/torch/CUDA/loader RNG state and cumulative training history. `world_last.pt` is also kept as a bounded-retry checkpoint. A `user_pause.json` in the new run with status `PAUSE_REQUESTED` stops before a new epoch or after the current epoch is safely saved. `--resume` requires the identical request, input pins and code pins and rejects a live PID/create-time identity. The current certification covers complete fresh runs; it does not claim an independently tested interruption/resume scenario.

The original protocol hash is retained in checkpoint payloads solely for scientific lineage. The new launcher separately records its current code hashes in `run_spec.json`. The old results and execution locks are never modified. Final payload, configuration, four-epoch history, loader/mask hashes and old byte pins are checked by a separate verification script; merely finishing training is not a claim of exact reproduction.

These are the retained small development experiments, not the earlier 50-epoch formal run and not a new independent performance test. Reproducing their weights does not demonstrate overall CCPV superiority, real running benefits, causal player credit or goal-probability improvement. The paired frozen-weight evaluation entry is supplied in the separate retained-inference package.
