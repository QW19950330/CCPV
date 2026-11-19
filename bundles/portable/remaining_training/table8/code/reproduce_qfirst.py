"""Portable create-only orchestration of the byte-identical 2026-10-30 Qfirst code.

This is the Qfirst head-training task historically called delivery table8. In the
current v10 manuscript its two rows belong to Table 7, not manuscript Table 8.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import contextlib
import csv
import hashlib
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


def safe_child(root, relative):
    rel = Path(relative)
    if rel.is_absolute():
        raise ValueError('Input manifest must contain relative paths')
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Input path escapes the explicitly supplied input directory')
    return path


def compare_npz(candidate, reference):
    import numpy as np
    mismatches, max_abs, arrays = [], 0., 0
    with np.load(candidate, allow_pickle=False) as new, np.load(reference, allow_pickle=False) as old:
        if set(new.files) != set(old.files):
            mismatches.append({'keys': [new.files, old.files]})
        for key in sorted(set(new.files) & set(old.files)):
            a, b = new[key], old[key]
            arrays += 1
            equal = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            error = None
            if a.shape == b.shape and a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
                error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
                max_abs = max(max_abs, error)
            if not equal:
                mismatches.append({'key': key, 'max_absolute_error': error,
                                   'new_shape': list(a.shape), 'old_shape': list(b.shape),
                                   'new_dtype': str(a.dtype), 'old_dtype': str(b.dtype)})
    return {'arrays': arrays, 'exact': not mismatches, 'max_absolute_error': max_abs,
            'mismatches': mismatches, 'archive_bytes_equal': sha(candidate) == sha(reference)}


def compare_outputs(out, inputs, config):
    ref = safe_child(inputs, config['verification_reference'])
    records = []
    for folder in ('initialization', 'models'):
        for old in sorted((ref / folder).glob('*.npz')):
            new = out / folder / old.name
            records.append({'path': str(new.relative_to(out)), **compare_npz(new, old)})
    records.append({'path': 'frozen_readouts.npz', **compare_npz(out / 'frozen_readouts.npz', ref / 'frozen_readouts.npz')})
    histories = []
    for old in sorted((ref / 'models').glob('*_history.json')):
        new = out / 'models' / old.name
        histories.append({'path': str(new.relative_to(out)), 'json_exact': read(new) == read(old),
                          'bytes_exact': sha(new) == sha(old)})
    csvs = []
    for old in sorted((ref / 'evaluation').glob('*.csv')):
        new = out / 'evaluation' / old.name
        with new.open(encoding='utf-8-sig', newline='') as f:
            nrows = list(csv.DictReader(f))
        with old.open(encoding='utf-8-sig', newline='') as f:
            orows = list(csv.DictReader(f))
        csvs.append({'path': str(new.relative_to(out)), 'rows': len(nrows),
                     'all_cells_exact': nrows == orows, 'bytes_exact': sha(new) == sha(old)})
    result, old_result = read(out / 'evaluation/results.json'), read(ref / 'evaluation/results.json')
    scientific_keys = [k for k in old_result if k not in ('utc', 'implementation_lock_sha256',
                                                        'weights_lock_sha256', 'readout_freeze_sha256')]
    scientific_result = {k: result[k] == old_result[k] for k in scientific_keys}
    # Recover the manuscript display from freshly scored numeric rows, then
    # compare exact rounded strings to the frozen v10 manuscript excerpt.
    with (out / 'evaluation/summary.csv').open(encoding='utf-8-sig', newline='') as f:
        summary = {(r['arm'], r['target']): r for r in csv.DictReader(f)}
    names = [('D DirectWith', 'D_Direct_WithEarlyBCE'), ('CV DirectWith', 'CV_Direct_WithEarlyBCE'),
             ('D MarginalWith', 'D_Marginal_WithEarlyBCE'), ('CV MarginalWith', 'CV_Marginal_WithEarlyBCE'),
             ('D q-first joint', 'D_QFirstJoint'), ('CV q-first joint', 'CV_QFirstJoint')]
    display = []
    for label, arm in names:
        r = summary[arm, 'q10']
        display.append('| ' + label + ' | ' + ' | '.join(
            f"{float(r[k + '_mean']):.9f} ± {float(r[k + '_seed_SD']):.9f}"
            for k in ('logloss', 'Brier')) + ' |')
    expected = read(safe_child(inputs, config['manuscript_display']))
    display_equal = display == expected['rows']
    with (out / 'manuscript_table7_rebuilt.md').open('x', encoding='utf-8') as f:
        f.write(expected['caption'] + '\n\n| Frontend and head | Logloss | Brier |\n| --- | ---: | ---: |\n')
        f.write('\n'.join(display) + '\n')
    report = {'status': 'EXACT_REPRODUCTION' if (all(r['exact'] for r in records)
              and all(r['json_exact'] for r in histories) and all(r['all_cells_exact'] for r in csvs)
              and all(scientific_result.values()) and display_equal) else 'MISMATCH',
              'npz': records, 'histories': histories, 'csv': csvs,
              'scientific_results_fields': scientific_result,
              'manuscript_table7_display_exact': display_equal, 'manuscript_table7_rows': display,
              'primary_negative_comparison': result['primary'],
              'array_count': sum(r['arrays'] for r in records),
              'max_absolute_array_error': max(r['max_absolute_error'] for r in records),
              'all_archive_bytes_equal': all(r['archive_bytes_equal'] for r in records),
              'scope': '60 new Qfirst heads; 240 first-stage arms reused only as frozen reference predictions; current manuscript Table7.',
              'excluded': ['manuscript Table8 native transfer/E-OOF scalar adaptation',
                           'frontend/world-model retraining', 'new data', 'new 50-epoch experiments', 'WC2018 tuning']}
    put(out / 'comparison.json', report)
    if report['status'] != 'EXACT_REPRODUCTION':
        raise AssertionError('Strict saved-artifact comparison failed; all differences retained in comparison.json')
    return report


def execute(args):
    code = Path(__file__).resolve().parent
    inputs, out = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError('Create-only output directory already exists: ' + str(out))
    for protected in (inputs, code):
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Code, input and output directories must be separate')
    manifest = read(inputs / 'input_manifest.json')
    config = read(inputs / 'portable_config.json')
    for record in manifest['files']:
        path = safe_child(inputs, record['path'])
        if sha(path) != record['sha256']:
            raise ValueError('Input hash mismatch: ' + record['path'])
    sources = read(code / 'source_manifest.json')
    for record in sources['files']:
        if sha(safe_child(code, record['path'])) != record['sha256']:
            raise ValueError('Vendor source hash mismatch: ' + record['path'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state = {'status': 'RUNNING', 'argv': sys.argv, 'input_dir': str(inputs), 'output_dir': str(out),
             'pid': os.getpid(), 'python': sys.executable, 'python_version': sys.version,
             'platform': platform.platform(), 'cwd': str(Path.cwd()),
             'input_manifest_sha256': sha(inputs / 'input_manifest.json'),
             'source_manifest_sha256': sha(code / 'source_manifest.json'),
             'runner_sha256': sha(__file__), 'threads': 1, 'device': 'cpu',
             'no_training_resume': True, 'pretrained_Qfirst_weights_used_for_training': False,
             'pretrained_Table7_weights_used_for_training': False}
    put(out / 'invocation.json', state)
    stage_name, accesses, durations = 'bootstrap', [], []
    try:
        sys.path.insert(0, str(code / 'vendor'))
        import numpy as np
        import torch
        import qfirst_io as io
        io.P = out
        original_selected = io.selected
        original_checkpoint = io.checkpoint
        table7_root = safe_child(inputs, config['binding']['reference_factorial']['study_path'])
        def selected(path, keys):
            resolved = Path(path).resolve()
            if resolved.is_relative_to(table7_root / 'models'):
                allowed = {'initial_hidden_weight', 'initial_hidden_bias', 'initial_output_weight', 'initial_output_bias',
                           'scaler_mean', 'scaler_scale', 'scaler_observed_weight_mass',
                           'scaler_completely_absent_training_feature', 'training_initial_rows', 'held_rows',
                           'training_initial_weights', 'reference_priors_q_e_l'}
                if not set(keys).issubset(allowed):
                    raise ValueError('Table7 references may expose only initialization audit members')
            if not (resolved.is_relative_to(inputs) or resolved.is_relative_to(out)):
                raise ValueError('Payload read escaped explicit input/output roots')
            accesses.append({'stage': stage_name, 'path': str(resolved), 'members': list(keys)})
            return original_selected(resolved, keys)
        def checkpoint(path):
            if not Path(path).resolve().is_relative_to(out / 'models'):
                raise ValueError('Whole checkpoint reads are restricted to newly trained output models')
            return original_checkpoint(path)
        io.selected, io.checkpoint = selected, checkpoint
        import qfirst_data as data
        import runtime
        import score_qfirst as scorer
        for module in (data, runtime, scorer):
            module.P = out
        binding = config['binding']
        for value in binding['assets'].values():
            value['path'] = str(safe_child(inputs, value['path']))
        reference = binding['reference_factorial']
        reference['study_path'] = str(safe_child(inputs, reference['study_path']))
        for key, value in reference.items():
            if isinstance(value, dict) and 'path' in value:
                value['path'] = str(safe_child(inputs, value['path']))
        for value in binding['initial_DirectWith']:
            value['path'] = str(safe_child(inputs, value['path']))
        io.put(out / 'data_binding.json', binding)
        with (out / 'experiment_plan.json').open('xb') as f:
            f.write(safe_child(inputs, config['experiment_plan']).read_bytes())
        lock = {'status': 'PORTABLE_REPRODUCTION_BYTE_IDENTICAL_SCIENTIFIC_SOURCE',
                'experiment_plan_sha256': sha(out / 'experiment_plan.json'),
                'data_binding_sha256': sha(out / 'data_binding.json'),
                'files': [{'path': str(code / r['path']), 'sha256': r['sha256']} for r in sources['files']],
                'runner_sha256': sha(__file__), 'no_mathematical_changes': True}
        io.put(out / 'implementation_lock.json', lock)
        expected = sha(out / 'implementation_lock.json')
        io.put(out / 'runtime_authorization.json', {
            'status': 'AUTHORIZED_FIXED_QFIRST_JOINT_PILOT', 'implementation_lock_sha256': expected,
            'allowed_stages': ['init_audit', 'fit', 'readout', 'score'],
            'scope': 'User-authorized original small-scale CPU reproduction, 60 heads and 12000 updates only'})
        io.put(out / 'environment.json', {'numpy': np.__version__, 'torch': torch.__version__,
                                         'torch_config': torch.__config__.show(), 'cuda_used': False})
        for stage_name, function in [('init_audit', runtime.init_audit), ('fit', runtime.fit_all),
                                     ('readout', runtime.readout), ('score', scorer.score)]:
            t0 = time.time()
            print('START_STAGE', stage_name, flush=True)
            with (out / (stage_name + '.stdout.log')).open('x', encoding='utf-8') as stdout, \
                 (out / (stage_name + '.stderr.log')).open('x', encoding='utf-8') as stderr:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    function(expected)
            durations.append({'stage': stage_name, 'seconds': time.time() - t0})
            print('COMPLETED_STAGE', stage_name, durations[-1]['seconds'], flush=True)
        stage_name = 'verify'
        report = compare_outputs(out, inputs, config)
        state.update(status='COMPLETED_EXACT', seconds=time.time() - started, stages=durations,
                     heads_trained=60, actual_optimizer_updates=12000,
                     max_absolute_array_error=report['max_absolute_array_error'])
    except BaseException as exc:
        state.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RESTART', seconds=time.time() - started,
                     failed_stage=stage_name, exception_type=type(exc).__name__, exception=str(exc),
                     traceback=traceback.format_exc(), completed_stages=durations)
        put(out / 'exception.json', state)
        raise
    finally:
        put(out / 'read_access.json', {'reads': accesses, 'new_head_initialization': 'fresh seeded tensors',
                                      'Table7_trained_parameter_members_allowed': False})
        put(out / 'completion.json', state)
        files = [{'path': p.relative_to(out).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(out.rglob('*')) if p.is_file()]
        put(out / 'output_manifest.json', {'files': files, 'status': state['status']})
    print(json.dumps({k: state[k] for k in ('status', 'heads_trained', 'actual_optimizer_updates', 'seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    execute(parser.parse_args())
