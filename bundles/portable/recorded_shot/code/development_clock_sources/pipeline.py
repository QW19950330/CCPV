"""Portable reconstruction of already-exposed source clock evidence. No model use."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import lxml
import pyarrow
import direct_logic as direct
import author_logic as author
from explicit_case import prepare_observer

PACKAGE = Path(__file__).resolve().parent


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or '\\' in relative or ':' in relative or any(x in ('..', '.') for x in rel.parts):
        raise ValueError('Only fixed safe relative input paths are accepted')
    target = (root / Path(*rel.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Input path escapes the explicit data root')
    return target


def fixed_inputs(root):
    cfg = read(PACKAGE / 'fixed_inputs.json')
    paths = {}
    for r in cfg['files']:
        p = safe_path(root, r['relative_path'])
        if p.stat().st_size != r['bytes']:
            raise ValueError('Input byte size mismatch: ' + r['relative_path'])
        # Big positions are hashed in each unmodified streaming parser.
        if r['kind'] != 'positions' and sha(p) != r['sha256']:
            raise ValueError('Input digest mismatch: ' + r['relative_path'])
        if r['kind'] in ('gt', 'unsynced'):
            b = p.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
            if blob != r['git_blob_sha1']:
                raise ValueError('Author source Git blob identity mismatch')
        paths[(r['match_id'], r['kind'])] = p
    data = safe_path(root, cfg['data_subdirectory'])
    source = {'selected_match_ids': cfg['direct_match_order'], 'downloaded_files': [
        {'path': paths[(r['match_id'], r['kind'])].relative_to(data).as_posix(),
         'bytes': r['bytes'], 'sha256': r['sha256']}
        for r in cfg['files'] if r['kind'] in ('events_raw', 'matchinformation', 'positions')]}
    return cfg, paths, data, source


def csv_write(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def rebuild(args):
    root = Path(args.data_root).resolve(strict=True)
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Refusing existing output: ' + str(out))
    if out.is_relative_to(PACKAGE):
        raise ValueError('Evidence outputs must be outside the code package')
    start = time.monotonic()
    cfg, paths, data, source = fixed_inputs(root)
    observer = prepare_observer(PACKAGE, args.explicit_selector, paths, cfg)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'opening.json', {
        'status': 'STARTED_EXPOSED_RAW_SOURCE_RECONSTRUCTION', 'utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'data_root': str(root), 'output': str(out),
        'package': str(PACKAGE), 'package_files_sha256': {p.name: sha(p) for p in PACKAGE.iterdir() if p.is_file()},
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'lxml': lxml.__version__, 'pyarrow': pyarrow.__version__},
        'fixed_inputs': cfg['files'], 'positions_reverified_during_stream': True,
        'scope': 'Direct seven-match raw evidence then author three-match exact crosswalk; replay of exposed inputs.',
        'original_windows_or_event_clock_caches_used': False,
        'new_fit': 0, 'model_forward': 0, 'label_rule_changes': False})
    (out / 'direct').mkdir(exist_ok=False)
    print(json.dumps({'stage': 'direct_started', 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    original_etree = direct.etree
    direct.etree = observer.facade(original_etree)
    try:
        shots, direct_matches = direct.rebuild(source, data)
    finally:
        direct.etree = original_etree
    csv_write(out / 'direct/shot_mapping.csv', shots)
    case, signed = observer.result()
    write(out / 'explicit_clock_case.json', case)
    write(out / 'explicit_clock_residual.json', signed)
    old = pd.read_csv(out / 'direct/shot_mapping.csv', dtype={'event_id': str}, encoding='utf-8-sig')
    files = {(m, k): p for (m, k), p in paths.items() if k in ('gt', 'unsynced')}
    (out / 'author').mkdir(exist_ok=False)
    print(json.dumps({'stage': 'author_started', 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    author_rows, author_matches = author.rebuild(files, source, data, old)
    for name, rows in author_rows.items():
        csv_write(out / 'author' / name, rows)
    outputs = {p.relative_to(out).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in out.rglob('*') if p.is_file()}
    write(out / 'completion.json', {
        'status': 'COMPLETED_RAW_EVIDENCE_RECONSTRUCTION_AWAITING_INDEPENDENT_COMPARISON',
        'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - start,
        'direct_per_match': direct_matches, 'author_per_match': author_matches,
        'direct_rows': len(shots), 'direct_unmapped_preserved': int((~old.mapped).sum()),
        'author_rows': len(author_rows['author_shots.csv']),
        'same_frame_signed_residual_s': signed['signed_residual_s'], 'outputs': outputs,
        'distinct_input_bytes': sum(r['bytes'] for r in cfg['files']),
        'positions_bytes_streamed': sum(r['bytes'] * (2 if r['match_id'] in cfg['author_match_order'] else 1)
                                        for r in cfg['files'] if r['kind'] == 'positions'),
        'window_coverage_rebuilt': False, 'final_event_clocks_or_labels_rebuilt': False,
        'physical_clock_truth_validated': False, 'new_blind_validation': False,
        'new_fit': 0, 'model_forward': 0})
    print(json.dumps({'status': 'COMPLETED', 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - start}), flush=True)


def csv_matrix(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))


def compare(args):
    root = Path(args.reference_root).resolve(strict=True)
    run = Path(args.run_dir).resolve(strict=True)
    dest = Path(args.output).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    cfg = read(PACKAGE / 'fixed_inputs.json')
    checks = []
    for r in cfg['original_evidence_references']:
        baseline = safe_path(root, r['reference_relative_path'])
        current = safe_path(run, r['output_relative_path'])
        if baseline.stat().st_size != r['bytes'] or sha(baseline) != r['sha256']:
            raise ValueError('Original evidence reference identity mismatch')
        a, b = csv_matrix(baseline), csv_matrix(current)
        same = a == b
        checks.append({'file': r['output_relative_path'], 'rows': len(a) - 1, 'columns': len(a[0]),
                       'all_cells_compared': sum(map(len, a)), 'column_and_row_order_equal': same,
                       'every_field_exact': same, 'original_sha256': r['sha256'], 'rebuilt_sha256': sha(current),
                       'byte_identical': sha(current) == r['sha256']})
    explicit = read(PACKAGE / 'explicit_inputs.json')
    baseline = safe_path(root, explicit['original_evidence']['relative_path'])
    if sha(baseline) != explicit['original_evidence']['sha256']:
        raise ValueError('Original explicit case identity mismatch')
    a, b = read(baseline), read(run / 'explicit_clock_case.json')
    old_path = a['source_pin'].pop('path')
    new_path = b['source_pin'].pop('path')
    # The sole expected location change is independently bound to the same pinned raw file.
    expected_name = PurePosixPath(new_path).name
    if old_path.replace('\\', '/').rsplit('/', 1)[-1] != expected_name:
        raise ValueError('Explicit source path basename changed')
    equal = a == b
    checks.append({'file': 'explicit_clock_case.json', 'every_nonlocation_field_exact': equal,
                   'excluded_field': 'source_pin.path', 'excluded_field_reason': 'Portable relative input path; basename, bytes and SHA remain exact.',
                   'record_count': len(a['records_at_explicit_and_two_adjacent_frames']),
                   'actor_frameset_count': len(a['actor_framesets'])})
    passed = all(r.get('every_field_exact', r.get('every_nonlocation_field_exact', False)) for r in checks)
    write(dest, {'status': 'PASS_EXACT_EVIDENCE_PARITY' if passed else 'FAIL', 'checks': checks,
                 'csv_data_cells_compared': sum(r.get('all_cells_compared', 0) - r.get('columns', 0) for r in checks),
                 'scope': 'All five original CSVs and every nonlocation field of the fixed explicit case. No score/model claims.',
                 'same_frame_signed_residual_s': read(run / 'explicit_clock_residual.json')['signed_residual_s']})
    print(json.dumps({'status': 'PASS' if passed else 'FAIL', 'checks': len(checks)}), flush=True)
    if not passed:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('rebuild')
    r.add_argument('--data-root', required=True)
    r.add_argument('--explicit-selector', required=True, help='Private fixed selector JSON; exact canonical SHA pin is enforced.')
    r.add_argument('--output', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--reference-root', required=True)
    c.add_argument('--run-dir', required=True)
    c.add_argument('--output', required=True)
    args = p.parse_args()
    rebuild(args) if args.command == 'rebuild' else compare(args)


if __name__ == '__main__':
    main()
