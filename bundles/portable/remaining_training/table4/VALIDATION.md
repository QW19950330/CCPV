# Actual training and exact comparison

All twenty fixed C/D/E/F × seed0–4 cells completed from original random
initialization on CUDA for four epochs. C_s0 was executed first as a bounded
implementation check (42.141 seconds); the other nineteen cells then completed
in 1,278.485 seconds. All requests exited normally. No trained checkpoint was
used to initialize a fit, and no validation score selected cells or epochs.

The data input was rebuilt from the preceding delivery's raw-reconstructed
SkillCorner development/truth outputs. The full prepared NPZ exactly matches
the original SHA-256. Gradient probes remain the same24 TRAIN windows; training
uses the same768 rows. The entire original epoch-loop AST is unchanged, and
the objective, Windows, motion model and gradient-diagnostic files match the
original source bytes.

`validation/table4_train01_parity.json` and `table4_train02_parity.json` both
report `EXACT_ALL_EPOCHS` and `all_file_bytes_equal=true`. Together they compare
all100 epoch0–4 checkpoints, 400 component-gradient vectors, 100 diagnostic JSON
files and20 full histories. Parameters, Adam, Python/NumPy/torch/CUDA/loader RNG,
loader order, augmentation masks, losses and gradient summaries are exact,
without tolerance. All320 result files in the comparison contract also match
their original file bytes. New process/closure/provenance records correctly use
new paths and times and are not claimed to be byte-identical.

The full request executed 1,920 Adam updates, 57,600 primary transition forwards
and4,000 read-only diagnostic forwards. MATH SDPA, FP32, disabled TF32, the fixed
learning rate and clip threshold remained unchanged. Every actual execution
source pinned by each run is checked after training; adding this validation
document and the external verifier does not change the training computation.

This establishes the retained Table4 training path. The preceding delivery
already evaluated the identical fixed checkpoints and inputs for current Table4.
There is no new forecasting improvement, new test set or new claim of full
historical-condition coverage from this reproduction.
