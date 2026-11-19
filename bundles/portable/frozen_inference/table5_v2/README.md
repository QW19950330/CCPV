# Table 5: fixed saved-output engineering diagnostics

This executable reconstructs all 16 world/seed cells, all 768 training diagnostic windows,
all nine paths, both coordinate frames and both fixed player groups. It produces 110,592
path rows, 442,368 guard rows and 221,184 diagnostic rows, plus all original summaries.
It does not train a model, generate trajectories, read future labels, repair outputs,
select by value, certify physiological feasibility or establish real tactical efficacy.

Use Python 3.11 and NumPy 1.26.4 for the historical parity check. No Torch/GPU is needed.
The input binding lists 17 required array files and the original source/provenance pins.
Supply an authorized root containing those relative paths; no data is distributed here.
The input contract is deliberately fixed to the retained experiment, not a new dataset.

```powershell
python -B pipeline.py preflight --data-root <authorized-root> --output <new-preflight-folder>
python -B pipeline.py run --data-root <authorized-root> --output <new-replay-folder>
python -B verify.py --original <original-results> --replay <new-replay-folder>/results --output <new-verification.json>
```

`feasibility_math.py`, `run_feasibility.py` and the plan are byte-identical to the retained
producer. The I/O primitives are unchanged. Only path binding and the execution guard
are adapted to explicit inputs and a fresh output directory. The original root lock is
not reused, altered or claimed to authorize a relocated job. In the unchanged legacy
run receipt, `implementation_lock_sha256` refers to the new `execution_contract.json`;
`portable_completion.json` and provenance distinguish the two.

`toy_checks.py` retains the original handmade support, missingness, zero-speed, coordinate,
position/head disagreement and all-planned-denominator tests. Its source-parse list alone
uses `pipeline.py` instead of the original metadata preparer. Run it in a disposable copy
before production; it creates a new local toy receipt and does not read real arrays.

The verifier reads every output row and compares complete field order, row order, cells
and uncompressed bytes. Gzip timestamps are not scientific results. Exact replay is
technical reproducibility of the saved-output screen, not a new independent experiment.
Source data rights and the separate upstream trajectory generation route remain required.


## Execution-only safeguard revision

This v2 adds rejection of outputs inside the code tree or any bound source tree.
The separately preserved v1 completed the full relocated replay. The mathematical
module, runner, plan, array reader and all output serializers are unchanged in v2;
v2 additionally passed the complete input preflight and explicit refusal tests.
The v1 full replay is not misrepresented as a second full v2 execution.
