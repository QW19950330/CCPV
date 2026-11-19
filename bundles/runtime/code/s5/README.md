# S5: dependence of saved recorded-shot predictions

This entry recalculates the original four-phase, recorded-event-support and
fixture-deletion statistics from seven pinned CSV/JSON inputs. It does not fit a
model, create predictions, load NPZ/raw trajectories, or select a favourable
phase. `original_analyze.py` is byte-identical to the retained numerical source;
`run.py` only relocates the seven input paths and verifies results. Historical
freeze JSON files remain byte-preserved on disk and are rebased in memory for
their original identity comparisons.

```text
python run.py --inputs INPUTS --output RUNS/new_s5
```

Use Python 3.11 and psutil 5.9.0. `INPUTS` must contain the seven flat files named
and hash-pinned in `plan.json`. They are outputs of the recorded-shot development
LOFO chain in the preceding recorded-shot reproduction delivery. Original
workspace source: `CCPV_ballfree_recorded_shot_pilot_20261118/result01`. Input
data are not bundled or implicitly licensed for redistribution. Output must be
new, outside code/inputs, with its parent already present.

The seven inputs preserve 11,542 candidate anchors, 9,483 eligible anchors,
363 positive windows and 113 covered event keys. Windows are not independent
events; phases and seeds do not create new independent matches. Event-conditioned
loss is not a full binary proper score or probability calibration. Deleting an
evaluation fixture is not retrained LOFO. These are previously exposed,
post-hoc descriptive diagnostics without p-values or confidence intervals.

The new parity check covers all thirteen numerical CSVs, the entire numerical
summary and the original Chinese report byte for byte. It connects the upstream
saved predictions to the previously reproduced S5 display. Process paths,
timings and identity metadata are correctly new, not forged to match the old run.
