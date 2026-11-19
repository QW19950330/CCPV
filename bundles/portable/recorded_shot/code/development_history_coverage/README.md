# Frozen history coverage table

This small entry point regenerates only the original 11,542-row history_coverage.csv from explicitly supplied reconstructed history archives, motion features, original two-point features, and the label ledger. It does not open raw XML, fit models, recompute features, or modify other code packages.

The original coverage dictionary expression and CSV writer are copied verbatim from the frozen motion study. Match order, global indices, column order, integer/float/boolean conversions, empty cells for nonfinite counts, and CSV bytes are retained. Stored current_mask/history_mask arrays supply the two counts; coordinate arrays are not loaded.

Use Python 3.11.5 and NumPy 1.26.4:

    python -B build_coverage.py --history-dir REBUILT_HISTORY_DIR --motion-features REBUILT_MOTION/features.npz --legacy-features REBUILT_LEDGER/features.npz --ledger REBUILT_LEDGER/all_anchor_ledger.csv --output NEW_OUTPUT_DIR

The output directory must not exist and its parent must already exist. The supplied input file bytes must match input_contract.json. The contract contains relative filenames and hashes, not input payloads or local paths. The CLI verifies cross-input row, label, eligibility, fixture and global-index alignment before deriving the table. It checks the resulting complete CSV byte hash against the original, then writes receipt.json.

This is a derived-table closure step. Source reconstruction evidence belongs to the upstream development_features and development_event_labels receipts. No new raw event, candidate, label, feature, training or independent validation claim follows from regenerating this table. The optional legacy_bridge table is outside this entry point.
