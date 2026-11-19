# Table 4: original four-objective training

This is a portable entry point for the retained four-epoch C/D/E/F study. It is
not a new tuning experiment or a claim of improved scientific performance.
All 20 cells (four objectives × five seeds) use the fixed epoch-4 checkpoint.
No old learned weights are used to initialize training.

## Inputs and environment

Use Python 3.11, NumPy 1.26.4, PyTorch 2.7.0+cu126 and psutil 5.9.0. The recorded
exact-training comparison uses an RTX 4070 Laptop GPU, deterministic FP32, MATH
SDPA, TF32 disabled and one CPU thread. Exact bytes on other hardware or library
versions are not promised. Dependencies are listed in `requirements.txt`; the
CUDA-specific PyTorch wheel must be installed for the user's own platform.

`contract.json` pins the development/truth inputs, fixed selections,
initialization hashes and all original epoch checkpoints and diagnostics.
Data, model checkpoints and credentials are not bundled. The existing raw-data
entry point is `preprocessing_v2` in the preceding raw-training delivery.
It recreates the two inputs used below; access to source data remains subject to
its providers' terms. The prepared Table 4 NPZ contains 768 TRAIN and 256 exposed
development windows. The 24 gradient probes are TRAIN only. The 32 evaluation
window IDs are stored for lineage and are not scored by this training command.

## Commands

Run commands with this directory as the current directory. Replace the sample
paths with actual locations. The output parent must exist, and a new output
must be outside code and input directories. Existing output is never silently
overwritten.

```text
python prepare.py --development INPUT/development.npz --truth INPUT/position_truth_v2.npz --output RUNS/prepared
python train.py --data RUNS/prepared/input.npz --output RUNS/training
python verify.py --run RUNS/training --reference-root REFERENCES/runs --output REPORTS/exact_comparison.json
```

The verifier needs locally trusted, hash-matching retained original checkpoints;
it checks their hashes before deserializing them. It is a lineage check, not a
replacement for new-outcome evaluation. It compares all epochs 0–4, parameters,
Adam, random states, history, masks, four component-gradient vectors and gradient
metadata. No numerical tolerance is used.

For a bounded diagnostic, `--cells C_s0` requests that fixed cell. Combining
disjoint, complete requests can cover the full predefined matrix; choosing a
subset based on scores is not supported by the study. Every cell has 96 Adam
updates, 1,920 primary transition forwards (C/D) or 3,840 (E/F), and 200 read-only
diagnostic forwards. The full matrix has 1,920 updates, 57,600 primary forwards
and 4,000 diagnostic forwards.

## Objective and retention rules

C uses TF/TF, D AR/AR, E TF/AR, and F AR/TF for the NLL/position losses. The
original masks, twenty-step unroll, component denominators, learning rate
0.0003, clip 200 and missing-observation fallback are retained. The complete
epoch-loop AST and diagnostic/model modules are checked against the original
source in the delivery's validation records. Read-only component gradients do
not mutate model, optimizer, RNG or existing parameter gradients.

A `user_pause.json` in the run directory with status `PAUSE_REQUESTED` stops
before a next epoch or after closing the current epoch. `--resume` accepts only
an identical code/input/request and fully closed epochs, with PID and process
creation identity checks. Unclosed partial files cause a refusal and are kept.
The delivery validation demonstrates fresh complete runs; it does not separately
claim an interruption/resume experiment. Windows rename retries are bounded to
30 seconds and affect output I/O only.

## Evaluation scope

The preceding `CCPV_neural_reproduction_delivery_20261119/table4` package supplies
the retained CPU natural-mean/sample evaluation and its fixed noise contract.
Its actual evaluation was already compared against the retained predictions and
current Table 4. This delivery verifies training back to those same checkpoints;
it does not repeat the entire larger historical condition matrix or establish
independent whole-pipeline generalization, calibrated generated-state value,
causal intervention benefit, or a recovered 420-match corpus.
