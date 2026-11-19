"""Recompute manuscript Table8 from saved natural-world states and frozen heads.

No neural/world fitting occurs. Native features and readouts are computed again;
450 inner-head calls reconstruct E OOF logits, then 120 scalar roots are refit.
Original scientific modules remain byte identical; only I/O binding is adapted.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import contextlib
import csv
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def child(root, relative):
    p = Path(relative)
    if p.is_absolute():
        raise ValueError('Portable paths must be relative')
    p = (root / p).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Path escapes supplied input root')
    return p


def resolve_records(value, root):
    if isinstance(value, dict):
        out = {k: resolve_records(v, root) for k, v in value.items()}
        if 'path' in out and 'sha256' in out:
            out['path'] = str(child(root, out['path']))
        return out
    if isinstance(value, list):
        return [resolve_records(v, root) for v in value]
    return value


def check_npz(new, old):
    import numpy as np
    bad, error, count = [], 0., 0
    with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            bad.append({'key_set_mismatch': True})
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key], b[key]
            count += 1
            numeric = x.dtype.kind in 'biufc' and y.dtype.kind in 'biufc'
            equal = (x.shape == y.shape and x.dtype == y.dtype
                     and np.array_equal(x, y, equal_nan=True) if numeric
                     else x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y))
            maximum = None
            if numeric and x.shape == y.shape:
                delta = np.abs(x.astype(float) - y.astype(float))
                maximum = float(np.nanmax(delta)) if delta.size and np.isfinite(delta).any() else 0.
                error = max(error, maximum)
            if not equal:
                bad.append({'key': key, 'max_absolute_error': maximum})
    return {'arrays': count, 'exact': not bad, 'max_absolute_error': error, 'mismatches': bad,
            'archive_bytes_equal': sha(new) == sha(old)}


def verify(out, inputs, config):
    arrays, csvs, histories = [], [], []
    for record in config['comparison_npz']:
        arrays.append({'path': record['new_path'],
                       **check_npz(out / record['new_path'], child(inputs, record['reference']))})
    for record in config['comparison_csv']:
        new, old = out / record['new_path'], child(inputs, record['reference'])
        with new.open(encoding='utf-8-sig', newline='') as f:
            a = list(csv.DictReader(f))
        with old.open(encoding='utf-8-sig', newline='') as f:
            b = list(csv.DictReader(f))
        csvs.append({'path': record['new_path'], 'rows': len(a), 'all_cells_exact': a == b,
                     'bytes_exact': sha(new) == sha(old)})
    for record in config['comparison_histories']:
        new, old = out / record['new_path'], child(inputs, record['reference'])
        histories.append({'path': record['new_path'], 'all_values_exact': read(new) == read(old),
                          'bytes_exact': sha(new) == sha(old)})
    with (out / 'adaptive/evaluation/arm_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        summary = list(csv.DictReader(f))
    rows = []
    for r in summary:
        label = r['arm'].replace('_', ' ')
        values = [f"{float(r[k + '_mean']):.9f} ± {float(r[k + '_seed_SD']):.9f}" for k in ('logloss', 'Brier')]
        values += [f"{100 * float(r[k + '_mean']):.5f} ± {100 * float(r[k + '_seed_SD']):.5f}" for k in ('bias', 'macro_abs_bias')]
        rows.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    display = read(child(inputs, config['manuscript_display']))
    display_exact = rows == display['rows']
    with (out / 'manuscript_table8_rebuilt.md').open('x', encoding='utf-8') as f:
        f.write(display['caption'] + '\n\n| Arm | Logloss | Brier | Signed bias (pp) | Macro absolute bias (pp) |\n')
        f.write('| --- | ---: | ---: | ---: | ---: |\n' + '\n'.join(rows) + '\n')
    with (out / 'native/evaluation/paired_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        negative = next(r for r in csv.DictReader(f) if r['primary'] == 'True')
    with (out / 'adaptive/evaluation/paired_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        adaptation = next(r for r in csv.DictReader(f) if r['primary'] == 'True')
    good = (all(r['exact'] for r in arrays) and all(r['all_cells_exact'] for r in csvs)
            and all(r['all_values_exact'] for r in histories) and display_exact)
    report = {'status': 'EXACT_REPRODUCTION' if good else 'MISMATCH',
              'npz': arrays, 'csv': csvs, 'histories': histories,
              'arrays_compared': sum(r['arrays'] for r in arrays),
              'max_absolute_array_error': max(r['max_absolute_error'] for r in arrays),
              'manuscript_table8_display_exact': display_exact,
              'original_negative_native_transfer': negative, 'subsequent_E_OOF_adaptation': adaptation,
              'no_retroactive_replacement_of_negative_transfer': True,
              'scope': {'saved_world_shards': 11, 'features_recomputed': 11,
                        'frozen_outer_heads_reused': 90, 'old_intercepts_reused': 120,
                        'inner_head_matrix_calls': 450, 'E_OOF_scalar_intercepts_refitted': 120,
                        'bisections': 12000, 'world_model_updates': 0, 'MLP_updates': 0,
                        'neural_world_forward_calls': 0},
              'limits': ['Saved natural-world states are inputs; world rollout is not rerun.',
                         'Old outer and inner heads and old intercepts are frozen inputs, not retrained.',
                         'Six previously exposed development matches; no independent confirmation.',
                         'No new 50-epoch run or WC2018 tuning.']}
    put(out / 'comparison.json', report)
    if not good:
        raise AssertionError('Strict native/adaptive comparison failed; inspect retained comparison.json')
    return report


def execute(args):
    import numpy as np
    code = Path(__file__).resolve().parent
    inputs, out = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError('Create-only output exists: ' + str(out))
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in (code, inputs)):
        raise ValueError('Code, inputs and output must be separate directories')
    if np.__version__ != '1.26.4':
        raise RuntimeError('Exact numerical reproduction requires NumPy 1.26.4')
    manifest, sources = read(inputs / 'input_manifest.json'), read(code / 'source_manifest.json')
    for root, records in ((inputs, manifest['files']), (code, sources['files'])):
        for r in records:
            if sha(child(root, r['path'])) != r['sha256']:
                raise ValueError('Input/source hash mismatch: ' + r['path'])
    config = read(inputs / 'portable_config.json')
    out.mkdir(parents=True, exist_ok=False)
    started, stage, accesses, durations = time.time(), 'bootstrap', [], []
    status = {'status': 'RUNNING', 'python': sys.executable, 'python_version': sys.version,
              'platform': platform.platform(), 'numpy': np.__version__, 'pid': os.getpid(),
              'input_dir': str(inputs), 'output_dir': str(out), 'cwd': str(Path.cwd()), 'argv': sys.argv,
              'source_manifest_sha256': sha(code / 'source_manifest.json'),
              'input_manifest_sha256': sha(inputs / 'input_manifest.json'), 'runner_sha256': sha(__file__),
              'device': 'cpu', 'threads': 1, 'world_forward_calls': 0, 'world_MLP_updates': 0}
    put(out / 'invocation.json', status)
    expected = sha(out / 'invocation.json')
    def prepare_io(module):
        # Adapt only the declared input/output boundary; arithmetic remains original.
        old_load = module.load_members if hasattr(module, 'load_members') else module.load
        def load(record, members):
            p = Path(record['path']).resolve()
            if not (p.is_relative_to(inputs) or p.is_relative_to(out)):
                raise ValueError('Payload access outside explicit roots')
            accesses.append({'stage': stage, 'path': str(p), 'members': list(members)})
            return old_load(record, members)
        if hasattr(module, 'load_members'):
            module.load_members = load
        else:
            module.load = load
    def run_stage(name, function, io, c, authority_key):
        nonlocal stage
        stage = name
        local_stage = name.split('_', 1)[-1]
        # E closure_validation retains its original exact authority-path checks.
        path = io.P / 'execution' / local_stage / 'portable01' / 'authorization_snapshot.json'
        put(path, {'status': 'AUTHORIZED_SAVED_WORLD_REPRODUCTION', 'stage': name,
                   'scope': 'same saved inputs, fixed mathematics, create-only output',
                   'implementation_lock_sha256': expected})
        c[authority_key] = io.rec(path)
        t0 = time.time()
        print('START_STAGE', stage, flush=True)
        with (out / (name + '.stdout.log')).open('x', encoding='utf-8') as stdout, \
             (out / (name + '.stderr.log')).open('x', encoding='utf-8') as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                function(expected, c)
        durations.append({'stage': name, 'seconds': time.time() - t0})
        print('COMPLETED_STAGE', name, durations[-1]['seconds'], flush=True)
    try:
        native_dir = code / 'vendor/native/payload'
        sys.path.insert(0, str(native_dir))
        nio = importlib.import_module('io_core')
        nio.P, nio.S, nio.BASE = out / 'native', code / 'vendor/native', inputs
        prepare_io(nio)
        n = importlib.import_module('runtime')
        n.P = nio.P
        c = resolve_records(config['native'], inputs)
        c['bound_inputs']['value_study_root'] = str(child(inputs, config['old_value_root']))
        cohort = read(c['cohort']['path'])
        # Original gate refers to historical absolute metadata paths; this is
        # replaced by the complete portable source/input hash verification above.
        n.binding = lambda _c: (_c['bound_inputs'], cohort)
        def rebuild_features(lock, native_config):
            stage_inputs = native_config['bound_inputs']
            h = nio.load_members(stage_inputs['history'], tuple(n.HISTORY_MEMBERS.values()))
            batch = n.HistoryBatch(**{k: h[v] for k, v in n.HISTORY_MEMBERS.items()}).validate()
            prefix = nio.load_members(stage_inputs['prefix'], n.PREFIX_MEMBERS)
            ix = np.asarray(cohort['selected_rows'], np.int64)
            if not np.array_equal(ix, np.flatnonzero(prefix['eligible_prefix'] & prefix['eligible_recorded10'])):
                raise ValueError('Original 5344-row eligibility changed')
            route = n.held_only_route(cohort['folds'], ix, prefix['match_id'][ix])
            records = []
            for record in native_config['saved_worlds']:
                world, seed = record['world'], record['seed']
                z = nio.load_members(record['array'], ('native_states', 'support', 'observation_age_s',
                                                      'initial_indices', 'anchor_id', 'match_id', 'fixed_team_A', 'fold_assignment'))
                for key, value in n.output_metadata(batch, prefix, ix, route).items():
                    if not np.array_equal(z[key], value):
                        raise ValueError('Saved-world metadata mismatch: ' + key)
                x2 = np.full((5344, 10), np.nan)
                zz = np.full((5344, 18), np.nan)
                for lo in range(0, 5344, 128):
                    rows = ix[lo:lo + 128]
                    xx, features = n.features_from_native_paths(batch, rows, z['native_states'][lo:lo + 128],
                                                               z['support'][lo:lo + 128], prefix['X0'])
                    x2[lo:lo + len(rows)], zz[lo:lo + len(rows)] = xx, features
                early, direct = n.union_features(prefix['X0'][ix], prefix['Z_cv'][ix], zz, x2)
                dest = nio.P / 'trajectories' / f'{world}_s{seed}' / 'states_features.npz'
                nio.save(dest, native_states=z['native_states'], support=z['support'], observation_age_s=z['observation_age_s'],
                         X2=x2, Z=zz, Early26=early, Direct36=direct, **n.output_metadata(batch, prefix, ix, route))
                records.append({'world': world, 'seed': seed, 'array': nio.rec(dest)})
                print('REBUILT_SAVED_WORLD_FEATURES', world, seed, flush=True)
            nio.put(nio.P / 'trajectory_freeze.json', {'status': 'ALL11_SAVED_WORLD_FEATURES_RECOMPUTED',
                    'implementation_lock_sha256': lock, 'records': records, 'world_forward_calls': 0,
                    'world_updates': 0, 'labels_decoded': False})
        run_stage('native_features', rebuild_features, nio, c, '_execution_authorization')
        run_stage('native_readout', n.readout, nio, c, '_execution_authorization')
        run_stage('native_score', n.score, nio, c, '_execution_authorization')
        # Separate original module namespace for the later E-OOF program.
        for key in ('runtime', 'io_core', 'score_math', 'literal_math', 'closure_validation'):
            sys.modules.pop(key, None)
        sys.path.insert(0, str(code / 'vendor/adaptive'))
        eio = importlib.import_module('io_core')
        eio.P = out / 'adaptive'
        prepare_io(eio)
        e = importlib.import_module('runtime')
        e.P = eio.P
        ec = resolve_records(config['adaptive'], inputs)
        ec['binding']['old_value_root'] = str(child(inputs, config['old_value_root']))
        for r in ec['binding']['Efeatures']:
            r['array'] = eio.rec(nio.P / 'trajectories' / f"E_s{r['seed']}" / 'states_features.npz')
        members = ec['binding']['current_frozen_readouts']['allowed_members']
        ec['binding']['current_frozen_readouts'] = {**eio.rec(nio.P / 'frozen_readouts.npz'), 'allowed_members': members}
        ec['historical_controls'] = eio.rec(nio.P / 'historical_controls.npz')
        e.getbinding = lambda _c: _c['binding']
        for name in ('oof_logits', 'calibrate', 'readout', 'score'):
            run_stage('adaptive_' + name, getattr(e, name), eio, ec, '_authority')
        stage = 'verify'
        report = verify(out, inputs, config)
        status.update(status='COMPLETED_EXACT', seconds=time.time() - started, stages=durations,
                      max_absolute_array_error=report['max_absolute_array_error'], scope=report['scope'])
    except BaseException as exc:
        status.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RESTART', failed_stage=stage,
                      seconds=time.time() - started, exception_type=type(exc).__name__, exception=str(exc),
                      traceback=traceback.format_exc(), completed_stages=durations)
        put(out / 'exception.json', status)
        raise
    finally:
        put(out / 'read_access.json', {'reads': accesses})
        put(out / 'completion.json', status)
        files = [{'path': p.relative_to(out).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(out.rglob('*')) if p.is_file()]
        put(out / 'output_manifest.json', {'files': files, 'status': status['status']})
    print(json.dumps({'status': status['status'], 'seconds': status['seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    execute(parser.parse_args())
