# Supplementary Table S4 actual matched-objective training

This entry independently fits the original 30 MC-matched risk heads (six outer folds × five seeds), 200 fixed Adam updates per head: **6,000 real updates**. It then freezes all heads, recomputes all four old/new-training × mean/MC-readout cells, and scores both original targets. It does not merely reaggregate an old table.

Validated environment: Python 3.11.5, NumPy 1.26.4, PyTorch 2.7.0+cu126; all head computation is CPU float64 with one thread.

The three numerical source files `run_study.py`, `matched_model.py`, and `transport_math.py` are byte-identical to the original `CCPV_matched_value_objective_pilot_20261118` files. `run.py` only supplies explicit input/output roots, relocates path records in the unchanged plan structure, and logs the three original stages. Training remains CPU float64 with one thread, original initial-state tensors and original train-fold scaler, fresh Adam, 602 parameters, lr 0.003, 0.005 weight ridge, global clip 10 and all 16 particles. Old final weights are not the new head's initial weights; only the explicitly saved original `initial_*` arrays initialize it.

```powershell
python -B -X utf8 CODE/run.py --input-root AUTHORIZED_FROZEN_SOURCE_TREE --metadata METADATA_DIRECTORY --output NEW_RUN
python -B -X utf8 CODE/verify.py --run NEW_RUN --reference ORIGINAL_MATCHED_STUDY --manuscript CURRENT_SUPPLEMENTARY_MD --docx CURRENT_SUPPLEMENTARY_DOCX --output NEW_COMPARISON_DIRECTORY
```

All paths are explicit. The input root can be moved if its manifest-relative tree and hashes are retained. `metadata/input_manifest.json` lists the exact dependencies, including 835 frozen terminal shards, old initial/scaler/head records, labels, old readout and source evidence. The local validation binds those existing private sources read-only; it does not duplicate or regenerate them. A code-only release does not contain this data, and local success does not establish a public raw-data-to-table reconstruction. The prior terminal generation and original heads remain upstream requirements.

The public `metadata/portable_plan.json` converts the original 907 path records to workspace-relative paths and preserves every other value exactly. `metadata/plan_public_lineage.json` records both hashes and that comparison. Keep `provenance/plan.json` local: it preserves the original historical paths and must be excluded from the code-only ZIP.

Outputs are create-only; all stage stdout, stderr, exit codes and failures are retained. Each 30-second progress receipt checks the running process, with the original 1,200-second cumulative stage budget. Source/input identities are checked before execution and by the original loader. No 50-epoch run, external-nine or WC2018 tuning, world inference, or terminal-shard regeneration occurs.

Strict verification compares every member of all 30 final checkpoint archives, all 30 full histories, all readout arrays, and every scoring CSV without tolerance. Paths and time fields in stage-freeze metadata legitimately differ and are not falsely claimed identical. The final historical CV row is preserved context rather than one of these 30 newly fitted heads.

All four cells, both targets, fixture failures and contrasts remain. Matching the MC objective improves A_match logloss while its Brier difference is adverse; B_replace improves only two of five seeds. These are already exposed six-fixture development results, not new independent validation or real intervention benefits.
