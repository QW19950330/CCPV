# Frozen development candidate grid

This candidate code rebuilds the original 13,372 metadata candidates from seven previously exposed DFL fixtures. It is not a model experiment, a new holdout, or a licence grant. Supply only raw files you are authorized to use. The package contains code, relative source filenames and hashes, and the frozen protocol; it contains no window/player data, labels or weights.

The seven fixture order includes J03WOH. Period order follows the original BALL FrameSet encounter order. For each period, the first syntactically valid source N defines the origin; candidates are origin+24+75*k through max valid N. Gaps do not reset the origin; unknowns and future-clock failures never remove a candidate. The original 25 CSV fields and their order are preserved. The original metadata scanner and row expression are extracted from the locked source; the legacy run function is not invoked.

Use Python 3.11.5, NumPy 1.26.4 and lxml 4.9.3. NumPy is checked to record the original environment, but candidate construction is integer/CSV logic and does not compute model features.

Cache interface check (new output directory must not exist):

    python -B build_grid.py cache --reference-grid <original-prefix_grid.csv.gz> --output <new-cache-output>

One authorized raw replay:

    python -B build_grid.py raw --raw-root <authorized-raw-directory> --reference-grid <original-prefix_grid.csv.gz> --output <new-raw-output> --authorize-raw-ball-metadata-scan

The 14 raw filenames and byte hashes are in input_contract.json. Positions XML is streamed once and fully hashed: 2,622,603,272 bytes across seven files; seven matchinformation XML files total 83,026 bytes. The scanner queries only BALL Frame N, T, BallPossession and BallStatus, and FrameSet TeamId, GameSection and MatchId. It does not query BALL X/Y/Z or player Frame attributes. The parser still reads all source XML bytes. All metadata writer calls are counted and discarded instead of emitting the old million-row metadata cache; this does not change any candidate calculation.

Outputs are private validation artifacts: prefix_grid.csv.gz, candidates_six.csv, identity.json and receipt.json. The full grid retains the original 25 fields without adding an index column. Its zero-based row number defines global grid_index. candidates_six.csv exports exactly grid_index, match_id, period, N for the fixed six-match scope (11,542 rows), retaining the full seven-match indices. The candidate whitelist expression is extracted from the original ball-free source (four output fields); it does not inspect training outcomes or old eligibility, team, live, ball-T or label columns.

The original reference gzip has a strict file-byte pin. Rebuilt CSV must have exactly the original header, all 13,372 rows, every string cell, and decompressed bytes. Gzip header time can change: receipts report container equality, timestamps, deflate stream and footer independently; a container difference never permits changed CSV content.

Independent comparison:

    python -B verify_grid.py --reference-grid <original-prefix_grid.csv.gz> --candidate-grid <new-raw-output>/prefix_grid.csv.gz --receipt <new-comparison.json>

This closes only raw BALL metadata to fixed candidate-grid reconstruction. It does not reconstruct event validity, author clock crosswalks, event-clock mapping, player coverage, labels, training or public data permissions. The six exported candidate identifiers are an interface for downstream label reconstruction, not labels. Original source datasets and all result payloads stay outside this code directory.
