"""Complete original deterministic games from frozen q readouts; no model or labels."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import game_math as M
import natural_games
import natural_report
import v0_games
from portable_io import read, write, rows, sha, now

PACKAGE = Path(__file__).resolve().parent
MEMBERS = ('q_logp', 'q_log1mp', 'supported', 'value_families', 'world_arms', 'seeds',
           'paths', 'match_ids', 'source_indices', 'target_source_ids', 'prefix_supported')


def fixed_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or ':' in relative or '\\' in relative:
        raise ValueError('Only fixed safe relative filenames are accepted')
    target = (root / Path(*p.parts)).resolve(strict=True)
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Input escapes explicit root')
    return target


def load_inputs(root, contract):
    arrays, identity = {}, []
    for rec in contract['inputs']:
        path = fixed_path(root, rec['relative_path'])
        if path.stat().st_size != rec['bytes'] or sha(path) != rec['sha256']:
            raise ValueError('Frozen saved readout identity changed')
        with np.load(path, allow_pickle=False) as archive:
            z = {k: archive[k] for k in MEMBERS}
        paths = M.PATHS if rec['role'] == 'natural' else M.V0_PATHS
        if z['value_families'].astype(str).tolist() != list(M.VALUE_FAMILIES) or z['world_arms'].astype(str).tolist() != list(M.WORLD_ARMS):
            raise ValueError('Frozen family or world order changed')
        if z['paths'].astype(str).tolist() != list(paths) or z['seeds'].tolist() != list(range(5)):
            raise ValueError('Frozen path or seed order changed')
        if z['q_logp'].shape != (2, 5, 4, len(paths), 768) or z['q_logp'].dtype != np.float64:
            raise ValueError('Frozen q array contract changed')
        if z['supported'].dtype != bool or z['supported'].shape != (5, 4, len(paths), 768):
            raise ValueError('Frozen support contract changed')
        if z['prefix_supported'].shape != (768,) or len(set(z['match_ids'].astype(str))) != 12:
            raise ValueError('Frozen 768 windows / twelve fixtures changed')
        arrays[rec['role']] = z
        identity.append(rec | {'q_positions': int(z['q_logp'].size), 'finite_q_logp': int(np.isfinite(z['q_logp']).sum()),
                                'prefix_supported': int(z['prefix_supported'].sum())})
    for k in ('source_indices', 'match_ids', 'target_source_ids', 'prefix_supported'):
        if not np.array_equal(arrays['natural'][k], arrays['v0'][k]):
            raise ValueError('Natural/v0 identity alignment differs: ' + k)
    return arrays, identity


def only(rr, **selector):
    hits = [r for r in rr if all(r[k] == str(v) for k, v in selector.items())]
    if len(hits) != 1:
        raise ValueError('Table selector is not unique: ' + repr(selector))
    return hits[0]


def make_tables(out):
    reference = rows(out / 'natural/report_tables/joint_reference_summary.csv')
    summary = rows(out / 'v0/games/game_summary.csv')
    labels = {'CV_Direct_WithEarlyBCE': 'Primary CV-DirectWith', 'D_Direct_WithEarlyBCE': 'Sensitivity D-DirectWith'}
    choices = [('CV_Direct_WithEarlyBCE', 'first_AB_minus_natural_empty', 'first-AB − natural'),
               ('CV_Direct_WithEarlyBCE', 'persistent_AB_minus_natural_empty', 'persistent-AB − natural'),
               ('CV_Direct_WithEarlyBCE', 'repeated_v0_AB_minus_natural_empty', 'repeated-original-velocity AB − natural'),
               ('CV_Direct_WithEarlyBCE', 'persistent_AB_minus_repeated_v0_AB_joint_only', 'persistent-AB − repeated-original-velocity AB'),
               ('D_Direct_WithEarlyBCE', 'persistent_AB_minus_natural_empty', 'persistent-AB − natural'),
               ('D_Direct_WithEarlyBCE', 'persistent_AB_minus_repeated_v0_AB_joint_only', 'persistent-AB − repeated-original-velocity AB')]
    table9 = []
    for family, contrast, label in choices:
        r = only(reference, value_family=family, world_arm='D_coupled_ar', joint_contrast=contrast, scope='common_all9_paths')
        value = f"{float(r['five_seed_match_macro_mean']):+.9f} ± {float(r['five_seed_match_macro_sd']):.9f}".replace('-', '−')
        table9.append([labels[family], label, value])
    table10 = []
    for family in M.VALUE_FAMILIES:
        for arm in M.WORLD_ARMS:
            cells = [labels[family], arm]
            for metric in ('phi_A', 'phi_B', 'delta_AB', 'interaction_difference'):
                r = only(summary, value_family=family, world_arm=arm, scope='common_all_world_seed_families', metric=metric)
                cells.append(f"{float(r['five_seed_mean']) * 1e6:.6f} ± {float(r['paired_initialization_seed_SD']) * 1e6:.6f}")
            table10.append(cells)
    tables = {'Table9': {'headers': ['Frozen score family', 'Paired policies', 'Joint difference'], 'rows': table9},
              'Table10': {'headers': ['Frozen value family', 'World', 'phiA', 'phiB', 'Joint ΔV', 'Full interaction I_AB'], 'rows': table10}}
    write(out / 'current_tables.json', tables)
    lines = []
    for name, t in tables.items():
        with (out / (name + '.csv')).open('x', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow(t['headers']); w.writerows(t['rows'])
        lines += ['## ' + name, '', '| ' + ' | '.join(t['headers']) + ' |', '| ' + ' | '.join('---' for _ in t['headers']) + ' |']
        lines += ['| ' + ' | '.join(r) + ' |' for r in t['rows']]
        lines.append('')
    with (out / 'current_tables.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def run(args):
    root, out = Path(args.data_root).resolve(strict=True), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    if out.is_relative_to(PACKAGE):
        raise ValueError('Private numerical outputs must be outside the code package')
    start = time.monotonic()
    contract = read(PACKAGE / 'input_contract.json')
    z, identity = load_inputs(root, contract)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'opening.json', {'status': 'STARTED_SAVED_Q_DETERMINISTIC_GAMES', 'utc': now(), 'pid': os.getpid(),
        'python': sys.version, 'numpy': np.__version__, 'inputs': identity,
        'code_sha256': {p.name: sha(p) for p in PACKAGE.iterdir() if p.is_file()},
        'no_head_readout_inference': True, 'event_labels_read': False, 'world_forwards': 0, 'fits': 0})
    (out / 'natural').mkdir(exist_ok=False)
    (out / 'v0').mkdir(exist_ok=False)
    print(json.dumps({'stage': 'natural_games', 'pid': os.getpid()}), flush=True)
    natural = natural_games.run(z['natural'], out / 'natural/games')
    natural_report.run(out / 'natural/games', out / 'natural/report_tables', z['natural'])
    with np.load(out / 'natural/games/masks.npz', allow_pickle=False) as a:
        oldmask = a['common_both_games']
    print(json.dumps({'stage': 'maintained_v0_games', 'pid': os.getpid()}), flush=True)
    v0 = v0_games.run(z['v0'], oldmask, out / 'v0/games')
    make_tables(out)
    write(out / 'completion.json', {'status': 'COMPLETE_FULL_SAVED_Q_GAMES_AWAITING_REFERENCE_COMPARISON', 'utc': now(),
        'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - start, 'natural': natural, 'maintained_v0': v0,
        'inputs': identity, 'old_natural_mask_used_to_select_v0': False,
        'scope': 'Original deterministic q-to-game computation and report tables; not upstream heads, features or world trajectories. No D1 stochastic results used.',
        'fits': 0, 'world_forwards': 0, 'value_head_forwards': 0, 'labels_read': False, 'GPU': False})
    print(json.dumps({'status': 'COMPLETE', 'pid': os.getpid(), 'seconds': time.monotonic() - start,
                      'natural_rows': natural['scene_rows'], 'v0_rows': v0['scene_rows']}), flush=True)


def csv_compare(a, b):
    with a.open(encoding='utf-8-sig', newline='') as fa, b.open(encoding='utf-8-sig', newline='') as fb:
        aa, bb = csv.reader(fa), csv.reader(fb)
        count = cells = 0
        while True:
            x, y = next(aa, None), next(bb, None)
            if x is None or y is None:
                if x != y:
                    raise ValueError('CSV length mismatch: ' + a.name)
                break
            if x != y:
                where = [i for i, (u, v) in enumerate(zip(x, y)) if u != v]
                raise ValueError(f'CSV field mismatch {a.name} row={count} columns={where[:8]}')
            count += 1; cells += len(x)
    return {'rows_including_header': count, 'cells_including_header': cells, 'all_fields_exact': True,
            'byte_identical': sha(a) == sha(b)}


def verify(args):
    root, run_dir = Path(args.reference_root).resolve(strict=True), Path(args.run_dir).resolve(strict=True)
    receipt = Path(args.output).resolve()
    if receipt.exists():
        raise FileExistsError(receipt)
    contract = read(PACKAGE / 'input_contract.json')
    comparisons = []
    for rec in contract['reference_outputs']:
        original = fixed_path(root, rec['reference_relative_path'])
        actual = fixed_path(run_dir, rec['output_relative_path'])
        if original.stat().st_size != rec['bytes'] or sha(original) != rec['sha256']:
            raise ValueError('Original output reference pin changed')
        item = {'file': rec['output_relative_path']}
        if original.suffix == '.csv':
            item |= csv_compare(original, actual)
        elif original.suffix == '.npz':
            members = []
            with np.load(original, allow_pickle=False) as a, np.load(actual, allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError('Mask member order differs')
                for key in a.files:
                    x, y = a[key], b[key]
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        raise ValueError('Mask array differs: ' + key)
                    members.append({'member': key, 'shape': list(x.shape), 'dtype': str(x.dtype), 'array_bytes_exact': True})
            item['members'] = members
        else:
            a, b = read(original), read(actual)
            a.pop('utc'); b.pop('utc')
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                raise ValueError('Scientific JSON summary differs: ' + original.name)
            item |= {'all_non_timestamp_fields_type_exact': True, 'excluded_field': 'utc'}
        comparisons.append(item)
    manuscript = Path(args.manuscript).resolve(strict=True)
    if sha(manuscript) != contract['current_manuscript']['sha256']:
        raise ValueError('Current v10 reference manuscript changed')
    text = manuscript.read_text('utf-8-sig')
    tables = read(run_dir / 'current_tables.json')
    checks = []
    for name, table in tables.items():
        number = name.removeprefix('Table')
        match = re.search(r'^Table ' + number + r'\..*?(?=^Table |^### |\Z)', text, re.M | re.S)
        if not match:
            raise ValueError('Current table not found: ' + name)
        lines = [l for l in match.group().splitlines() if l.startswith('|')]
        expected = [[x.strip() for x in l.strip().strip('|').split('|')] for l in lines]
        expected = [expected[0]] + expected[2:]
        actual = [table['headers']] + table['rows']
        if expected != actual:
            raise ValueError('Current v10 table cells differ: ' + name)
        checks.append({'table': name, 'data_rows': len(table['rows']), 'cells_including_header': sum(map(len, actual)), 'every_cell_exact': True})
    write(receipt, {'status': 'PASS_FULL_DETERMINISTIC_GAMES_AND_CURRENT_TABLES_EXACT', 'utc': now(),
        'reference_files': comparisons, 'current_tables': checks,
        'csv_cells_including_headers': sum(x.get('cells_including_header', 0) for x in comparisons),
        'scope': 'All original saved-q games and natural report CSVs, masks array bytes, non-time summary fields, and current v10 Table9/10 cells. No upstream fit/inference or D1 substitution.'})
    print(json.dumps({'status': 'PASS', 'reference_files': len(comparisons),
                      'csv_cells': sum(x.get('cells_including_header', 0) for x in comparisons),
                      'current_table_cells': sum(x['cells_including_header'] for x in checks)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run'); p.add_argument('--data-root', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--reference-root', required=True); p.add_argument('--run-dir', required=True)
    p.add_argument('--manuscript', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args) if args.command == 'run' else verify(args)


if __name__ == '__main__':
    main()
