# Fixed A_raw4d training dependency

This entry fills the five retained `A_raw4d_seed0`–`seed4` training dependencies used by the saved feasibility and game outputs. It trains the original four-epoch pilot, not the 50-epoch experiment. All training, model, loss, augmentation and RNG code is copied byte-for-byte from the already verified `CCPV_neural_training_delivery_20261119/table3` entry. The only fixed-contract changes are selecting arm `A_raw4d` and its five original reference records.

The existing runtime already constructs A as `PersistentGapAwareWorldModel` with teacher forcing. The existing loss already applies its full-state four-dimensional NLL branch; the velocity-residual position-Huber branch is not used for A. No numerical implementation was added or changed. Initialization common/velocity hashes, 768 training windows, 256 development windows, masks, 20-step unroll, 24 batches per epoch, batch size 32, learning rate 0.0003, gradient clip 200, and seeds 0–4 remain fixed. Total: five models, 480 Adam updates and 9,600 forward calls.

```powershell
python -B -X utf8 PATH_TO_CODE/run_fixed_A.py --inputs PATH_TO_INPUTS --output NEW_RUN --logs NEW_LOGS
python -B -X utf8 PATH_TO_CODE/train.py --development-data PATH_TO_DATA/development.npz --development-truth PATH_TO_DATA/position_truth_v2.npz --output NEW_RUN --device cuda
python -B -X utf8 PATH_TO_CODE/verify.py --run NEW_RUN --reference PATH_TO_INPUTS/reference/runs --contract PATH_TO_CODE/training_contract.json --output NEW_COMPARISON_JSON
```

The logged wrapper is the preferred invocation; the `train.py` line is its equivalent direct command. Choose one, using a fresh output directory. The wrapper records process identity, 30-second alive checks, exit code and logs, with a 30-minute hard timeout and no automatic retry.

Use the original CUDA numerical environment. CPU is not used as a substitute for strict original CUDA weight, Adam or RNG equality. The copied generic training CLI still mentions 15 cells in its help; this entry's frozen contract contains only the five A cells, so its default executes exactly those five. Code, private inputs and new outputs are separate. A fresh run directory is required, and each epoch receives an immutable checkpoint snapshot. Local private inputs/references are excluded from the code-only release.

The comparison covers the complete final checkpoint payload (weights, optimizer, Python/NumPy/torch/CUDA/loader RNG, history), configuration, byte-exact history, and each new epoch's history prefix. Original intermediate weights were not retained, so their equality is not claimed. The original A weight hashes are comparison targets only; a fresh run initializes its model from the original fixed RNG seeds.
