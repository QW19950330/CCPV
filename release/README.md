# CCPV revision: frozen code, derived inputs, and evaluation artifacts

Version **v2026.11.19**. Authors: **Fang Chen and Jiqian Lin**, Xi’an Physical Education University, Xi’an 710068, China.

This release accompanies *Trajectory-based model-input interventions and reference-dependent player attribution in football*. The code repository is [QW19950330/CCPV](https://github.com/QW19950330/CCPV). Zenodo record DOI: [10.5281/zenodo.24370617](https://doi.org/10.5281/zenodo.24370617). Release availability is established by the public record and its listed files; this identifier alone is not a publication-status assertion.

## Files and integrity

`public_manifest.json` lists the exact 17 approved archives: filename, bytes, SHA-256, purpose, source links and licence scope. Their total compressed size is **5,243,091,104 bytes** (about 5.24 GB). Original basenames are retained for cross-package compatibility. Terms such as “candidate” or “local” in a filename are historical naming, not instructions to upload other local archives. Only these exact archive hashes are covered by this manifest.

The archives contain derived inputs, saved predictions/features, control histories, unrounded statistics, clean inference weights and code. Full non-pickle replay references are a separate package from pure inference weights. Statistical rows and Monte Carlo draws are not independent matches. Consult each package’s field dictionary, units, masks, support and denominator records; seed SD describes variation across the fixed five seeds, not an independent-match confidence interval.

## Download layout and bounded entry checks

Place the downloaded ZIPs together in a `downloads` directory, keeping their filenames. Extract the public-runtime ZIP to a separate `runtime` directory. `RELEASE_ARCHIVE_MAP.json` maps original logical archive identifiers to the eleven clean input ZIP basenames and archive SHA values. It changes only the public download location mapping, not sealed code, guards or archive bytes.

With the dependencies stated in the source packages:

```text
python runtime/compose_inputs.py --candidate runtime --archive-root downloads --public-archive-map RELEASE_ARCHIVE_MAP.json --output composed_inputs
python runtime/check_entry.py --candidate runtime --inputs composed_inputs --output checks_table5 --component table5
```

Other supported `--component` values are `table6`, `supplementary`, `table7`, `qfirst`, `native`, `s4` and `s5`; use a new output directory for each. Outputs are create-only. Do not provide a local-source fallback. These commands run original preflight prefixes or saved-statistic arithmetic. Per-component contracts document further execution requirements.

The separate diagnostic bridge requires explicit prefix data and the documented provider preparation route. The exact 4,992-window prefix is not embedded in the diagnostic code ZIP. Its prepared SHA-256 is `a14904ed5a215805ce390fa0885b99e2e49752da096d47b119ba0019b0b8b656`. Full response and recovered-truth arrays are not represented by the statistical ZIP. Passing preflight does not establish that all external inputs for a full study are present.

## What was actually checked

- Eight clean-only entry/preflight or saved-statistic routes completed using **3,089 inputs / 6,257,480,344 materialized bytes**, with no original-workspace fallback or hardlinks.
- Saved S5 outputs (15 files) and supplementary outputs (8 files) matched frozen references byte-for-byte.
- Clean inference readers passed for **21 families**, **11,345 checked arrays** and **984 unique inference payloads**; these are bounded reader/synthetic checks.
- The clean diagnostic bridge was checked on fixed prefix rows **0 and 1**, five D seeds plus CV and six paths. Both sides used batch size two: **72 arrays and 155 state tensors** matched exactly. This was not a new full-cohort replay and does not establish equivalence to the historical batch-32 run.
- The frozen four-match statistics archive preserves **19 statistical files and 841,570 rows**, including NA and rejected cases.

These checks establish the stated bindings and computations, not complete raw-data-to-training-to-paper reproduction, cross-platform equivalence or validity of every historical claim. All seeds, fixtures, unfavourable results and missingness remain. The 50-epoch reconstruction did not reproduce the original forecasting advantage; Metrica, transfer failures and adverse ball results remain. The four diagnostic matches were previously exposed; the fixed cache was admitted using support in both history and future. It is not a new blind test. Factual ADE/FDE improvements do not establish real Goal benefit, intervention efficacy, causal player credit or independent whole-pipeline generalization.

## Licences and sources

Original author software is **MIT** (`LICENSE`). Author-owned weights, documentation, schemas and eligible statistics are **CC BY 4.0** (`CC-BY-4.0-NOTICE.txt`). Source-specific notices control their own material: see `license_scope.json` and `THIRD_PARTY_NOTICES.md`. SkillCorner provider material retains its pinned MIT notice; IDSSE retains CC BY attribution. StatsBomb-derived analyses retain their custom agreement, commercial-use limitation and branding requirements, and are excluded from the author CC BY/MIT grant. Metrica raw records and unverified legacy source records are not added here.

Historical proposal/pending labels inside unchanged archives are superseded only for author-owned material covered by this release-level grant. No third-party rights are reassigned. Historical manuscript snippets or reports in support archives are source evidence; this deposit is not a final approved journal-submission package. No journal submission is part of this release.

## Citation and provenance

Use `CITATION.cff` and the Zenodo record DOI 10.5281/zenodo.24370617, retaining provider attribution. Preserve filename/SHA identities when reporting the version used. Public metadata contain portable source identifiers; chat transcripts, credentials and local authorization/upload records are excluded.

## StatsBomb acknowledgement

<img src="assets/statsbomb-logo.png" alt="StatsBomb logo" width="420">

The included shot-conditional Goal analyses use StatsBomb Open Data. The official logo is shown unchanged for the source acknowledgement requested by the provider. Logo and brand rights are excluded from the author MIT/CC BY grants; no endorsement is implied. See [StatsBomb attribution](STATSBOMB_ATTRIBUTION.md) and `license_scope_statsbomb_addendum.json`.
