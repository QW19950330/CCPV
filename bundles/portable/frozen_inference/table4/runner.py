"""Portable frozen Table4 preparation, natural evaluation and bounded step check.

There is no full-training command. All input paths are explicit and hash-pinned.
The original CPU evaluation is retained; a GPU may be used only by an explicit
step-check invocation, which never replaces a frozen checkpoint.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

CODE = Path(__file__).resolve().parent
CELLS = ('C', 'D', 'E', 'F')
SEEDS = tuple(range(5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def put(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def record(path, base=None):
    path = Path(path)
    return dict(path=path.relative_to(base).as_posix() if base else path.name, sha256=digest(path), bytes=path.stat().st_size)


def source_manifest():
    return [record(p, CODE) for p in sorted(CODE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def load_inputs(args):
    manifest = json.loads((CODE / 'input_manifest.json').read_text(encoding='utf-8'))
    paths = {'data': args.data.resolve(), 'noise': args.noise.resolve()}
    cp_root = args.checkpoint_root.resolve()
    out = args.output.resolve()
    require(not out.exists(), 'Create-only output already exists; partial output is preserved.')
    for protected in (CODE, cp_root, paths['data'].parent, paths['noise'].parent):
        require(not out.is_relative_to(protected), 'Output cannot be inside code or supplied input/checkpoint folders.')
    for name, path in paths.items():
        pin = manifest[name]
        require(path.is_file() and path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'],
                'Frozen ' + name + ' identity mismatch.')
    checkpoints = {}
    for pin in manifest['checkpoints']:
        path = (cp_root / pin['relative_path']).resolve()
        require(path.is_relative_to(cp_root), 'Checkpoint path escaped explicit root.')
        require(path.is_file() and path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'],
                'Frozen checkpoint identity mismatch.')
        checkpoints[pin['cell'], pin['seed'], pin['epoch']] = path
    with np.load(paths['data'], allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    required = {'states', 'valid', 'position_valid', 'teams', 'entity_ids', 'match_ids', 'source_indices', 'train_indices',
                'validation_indices', 'source_frame_ids', 'frame_times_s', 'source_period', 'positions', 'split',
                'pitch_L', 'pitch_W', 'probe24', 'validation32'}
    require(set(a) == required, 'Frozen input member schema changed.')
    require(a['states'].shape == (1024, 30, 23, 4) and a['states'].dtype == np.float32, 'Original FP32 state shape changed.')
    require(a['valid'].shape == (1024, 30, 23) and a['valid'].dtype == bool, 'State-validity contract changed.')
    require(a['positions'].shape == (1024, 30, 23, 2) and a['position_valid'].shape == (1024, 30, 23), 'Position truth contract changed.')
    require(np.array_equal(a['teams'], [0] * 11 + [1] * 11 + [2]), 'Original23 entity-type order changed.')
    require(np.array_equal(a['train_indices'], np.arange(768)) and np.array_equal(a['validation_indices'], np.arange(768, 1024)),
            'Frozen split changed.')
    require(np.isfinite(a['states'][a['valid']]).all() and np.isfinite(a['positions'][a['position_valid']]).all()
            and not np.isfinite(a['positions'][~a['position_valid']]).any() and not np.any(a['valid'] & ~a['position_valid']),
            'Masked factual-state/position contract changed.')
    require(len(set(a['source_indices'].tolist())) == 1024 and (np.diff(a['source_frame_ids'], axis=1) == 1).all()
            and np.allclose(np.diff(a['frame_times_s'], axis=1), .1, rtol=0, atol=1e-10), 'Original identity/clock contract changed.')
    for split, count, fixtures in [('TRAIN', 768, 12), ('validation', 256, 4)]:
        selected = a['split'].astype(str) == split
        require(int(selected.sum()) == count and len(set(a['match_ids'][selected].astype(str))) == fixtures, 'Split denominator changed.')
    ix = a['validation32']
    _, counts = np.unique(a['match_ids'][ix], return_counts=True)
    require(len(ix) == len(set(ix.tolist())) == 32 and len(counts) == 4 and (counts == 8).all()
            and (a['split'][ix] == 'validation').all(), 'Original32-window selection changed.')
    with np.load(paths['noise'], allow_pickle=False) as z:
        require(set(z.files) == {'mixture_uniform', 'normal'}, 'Frozen noise schema changed.')
        noise = {k: z[k] for k in z.files}
    require(noise['mixture_uniform'].shape == (32, 32, 20, 23) and noise['normal'].shape == (32, 32, 20, 23, 2)
            and all(x.dtype == np.float32 for x in noise.values()), 'Fixed FP32 noise bank shape changed.')
    a['position_targets'] = a['positions']
    out.mkdir(parents=True, exist_ok=False)
    put(out / 'identity.json', dict(stage=args.stage, utc=datetime.now(timezone.utc).isoformat(), data=record(paths['data']),
                                  noise=record(paths['noise']), checkpoint_files_verified=40, code=source_manifest(),
                                  output_policy='create-only; no original checkpoint replacement', raw_source_reconstruction=False,
                                  full_CCPV_reproduction=False, public_release=False))
    return a, noise, checkpoints, out


def configure_torch(device):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    from model_source import api
    api.configure(device)
    require(device == 'cpu' or torch.cuda.is_available(), 'Explicit CUDA step check requested but CUDA unavailable.')
    return torch, api


def checkpoint_model(torch, api, path, cell, seed, epoch, device):
    # The exact bytes have been checked against the packaged hash allowlist.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require((payload['cell'], payload['seed'], payload['epoch']) == (cell, seed, epoch), 'Checkpoint metadata identity mismatch.')
    model = api.make_model('D_coupled_ar', seed)
    model.load_state_dict(payload['state_dict'], strict=True)
    model = model.to(device)
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Original model FP32 dtype required.')
    return model, payload


def close_output(out, completion):
    put(out / 'completion.json', completion)
    put(out / 'output_freeze.json', dict(files=[record(p, out) for p in sorted(out.rglob('*')) if p.is_file()]))


def prepare(a, noise, checkpoints, out):
    close_output(out, dict(status='PREPARED_EXACT_FROZEN_TABLE4_INPUTS', input_windows=1024, train_windows=768,
                          ADE_windows=256, Energy_windows=32, draws=32, final_checkpoints=20, initial_checkpoints=20,
                          primary_table_forward_budget=800, original_full_diagnostic_forward_budget=4400,
                          full_training_budget_not_executed=dict(models=20, epochs=4, updates_per_model=96, total_updates=1920),
                          new_forwards=0, new_training_updates=0, new_noise_generated=0, GPU_work=0,
                          evaluation_complete=False, training_reproduced=False, submission_ready=False))


def evaluate(a, noise, checkpoints, out):
    started = time.monotonic()
    torch, api = configure_torch('cpu')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from eval_steps import collect_block, sample_block
    from evaluation_core import grid
    from score_table4 import mean_rows, energy_rows, csv_write, summarize
    u, e = grid(noise)
    tg = (torch.from_numpy(u), torch.from_numpy(e))
    ix, small = a['validation_indices'], a['validation32']
    teams = torch.tensor(a['teams'], dtype=torch.long)
    mean_scores, path_scores, calls = [], [], 0
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        h, v = torch.tensor(a['states'][ix, :10]), torch.tensor(a['valid'][ix, :10])
        h, v, chosen = api.mask_history(h, v, teams, a['source_indices'][ix], condition='natural', seed=2026102417, epoch=0)
        cv = api.analytic_forecast('cv', h, v).numpy()
        save_npz(out / 'CV_natural.npz', states=cv, scene_indices=ix, hidden_players=chosen.numpy(),
                 support=np.broadcast_to(a['valid'][ix, :10].any(1)[:, None], (256, 20, 23)).copy())
        # Match original record order: seed first, then C/D/E/F.
        for seed in SEEDS:
            for cell in CELLS:
                require(not (out / 'PAUSE_REQUESTED').exists(), 'Pause requested; completed cells remain.')
                model, payload = checkpoint_model(torch, api, checkpoints[cell, seed, 4], cell, seed, 4, 'cpu')
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                before = api.state_hash(model.state_dict())
                count = {'value': 0}
                hook = model.register_forward_hook(lambda *_: count.__setitem__('value', count['value'] + 1))
                folder = out / f'{cell}_s{seed}'
                folder.mkdir()
                mean = collect_block(api, model, torch.tensor(a['states'][ix]), torch.tensor(a['valid'][ix]), teams,
                                     'original_t0_mean_feedback')['mean_state']
                save_npz(folder / 'mean256.npz', states=mean, scene_indices=ix)
                h, v = torch.tensor(a['states'][small, :10]), torch.tensor(a['valid'][small, :10])
                h, v, chosen = api.mask_history(h, v, teams, a['source_indices'][small], condition='natural', seed=2026102417, epoch=0)
                raw = sample_block(api, model, h.repeat_interleave(32, 0), v.repeat_interleave(32, 0), teams, tg)
                sample = dict(states=raw['states'].reshape(32, 32, 20, 23, 4), support=raw['support'].reshape(32, 32, 20, 23),
                              completed_steps=raw['completed_steps'].reshape(32, 32), scene_indices=small, hidden_players=chosen.numpy())
                save_npz(folder / 'sample32_M32.npz', **sample)
                require(count['value'] == 40 and api.state_hash(model.state_dict()) == before, 'Forward budget or frozen state changed.')
                hook.remove(); calls += count['value']
                mean_scores.extend(mean_rows(a, mean, cv, cell, seed))
                path_scores.extend(energy_rows(a, sample, mean, cv, cell, seed))
                put(folder / 'completion.json', dict(cell=cell, seed=seed, checkpoint=record(checkpoints[cell, seed, 4]),
                                                     forward_calls=40, weights_unchanged=True, natural_only=True))
                print('TABLE4_CELL_COMPLETE', cell, seed, flush=True)
                del model, payload, raw, sample, mean
    require(calls == 800, 'Complete Table4 primary evaluation budget mismatch.')
    csv_write(out / 'mean_ADE256_scores.csv', mean_scores)
    csv_write(out / 'path_EnergyU32_scores.csv', path_scores)
    summary = summarize(mean_scores, path_scores)
    put(out / 'table4_summary.json', summary)
    feedback = dict(C=('TF', 'TF'), D=('AR mean', 'AR mean'), E=('TF', 'AR mean'), F=('AR mean', 'TF'))
    lines = ['| Arm | NLL history | Position history | Player ADE256 (m) | Player path EnergyU32 (m) |',
             '|---|---|---|---:|---:|']
    for row in summary:
        ade, energy = row['metrics']
        lines.append(f'| {row["arm"]} | {feedback[row["arm"]][0]} | {feedback[row["arm"]][1]} | '
                     f'{ade["mean"]:.6f} ± {ade["sample_seed_SD"]:.6f} | '
                     f'{energy["mean"]:.6f} ± {energy["sample_seed_SD"]:.6f} |')
    with (out / 'table4.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    close_output(out, dict(status='COMPLETE_TABLE4_NATURAL_FROZEN_WEIGHT_EVALUATION', cells=20, forward_calls=calls,
                          CPU=True, threads=1, dtype='float32', attention_backend='MATH', torch=torch.__version__, numpy=np.__version__,
                          wall_seconds=time.monotonic() - started, ADE_windows=256, Energy_windows=32, draws=32,
                          new_training_updates=0, new_noise_generated=0, original_full4400_diagnostics_repeated=False,
                          training_reproduced=False, raw_data_rebuilt=False, full_CCPV_reproduction=False, public_release=False))


def same_tensors(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tensors(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(same_tensors(x, y, torch) for x, y in zip(a, b))
    return a == b


def step_check(a, noise, checkpoints, out, args):
    started = time.monotonic()
    torch, api = configure_torch(args.device)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from factorized_objective import factorized_loss
    from training_core import optimizer_for, first_batch, batch_to
    import legacy_objective
    legacy_objective.bind_api(api)
    cell, seed = args.cell, args.seed
    model, payload = checkpoint_model(torch, api, checkpoints[cell, seed, 0], cell, seed, 0, args.device)
    model.train()
    api.seed_all(100003003 + seed)
    raw = first_batch(api, a, seed)
    raw = {k: v[:args.batch_size] for k, v in raw.items()}
    batch = batch_to(raw, args.device)
    optimizer = optimizer_for(model)
    require(not payload['history'] and payload['epoch'] == 0, 'Single-step check must begin at original initialization.')
    before = api.state_hash(model.state_dict())
    reference = copy.deepcopy(model) if cell in ('C', 'D') else None
    reference_optimizer = optimizer_for(reference) if reference is not None else None
    count = {'value': 0}
    hook = model.register_forward_hook(lambda *_: count.__setitem__('value', count['value'] + 1))
    bridge = None
    with sdpa_kernel([SDPBackend.MATH]):
        optimizer.zero_grad(set_to_none=True)
        loss, metadata = factorized_loss(model, batch, cell)
        require(torch.isfinite(loss), 'Nonfinite fixed loss; no retry or repair.')
        loss.backward()
        if reference is not None:
            reference_optimizer.zero_grad(set_to_none=True)
            other, _ = legacy_objective.objective(reference, batch, 'C_coupled_tf' if cell == 'C' else 'D_coupled_ar')
            other.backward()
            bridge = dict(loss_bitwise_equal=torch.equal(loss, other),
                          gradients_bitwise_equal=all(same_tensors(p.grad, q.grad, torch) for p, q in zip(model.parameters(), reference.parameters())))
            require(all(bridge.values()), 'Exact original C/D loss/gradient bridge failed.')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 200., error_if_nonfinite=True)
        optimizer.step()
        if reference is not None:
            reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 200., error_if_nonfinite=True)
            reference_optimizer.step()
            bridge.update(clip_norm_bitwise_equal=torch.equal(norm, reference_norm),
                          post_Adam_parameters_bitwise_equal=same_tensors(model.state_dict(), reference.state_dict(), torch),
                          Adam_state_bitwise_equal=same_tensors(optimizer.state_dict(), reference_optimizer.state_dict(), torch))
            require(all(bridge.values()), 'Exact original clip/Adam bridge failed.')
    hook.remove()
    expected = 20 if cell in ('C', 'D') else 40
    require(count['value'] == expected, 'Original feedback branch budget changed.')
    require(before != api.state_hash(model.state_dict()), 'Single-step check did not update its disposable model.')
    with (out / 'disposable_single_step.pt').open('xb') as stream:
        torch.save(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), cell=cell, seed=seed,
                        updates=1, role='DISPOSABLE_INTERFACE_CHECK_NOT_A_TRAINED_TABLE4_MODEL'), stream)
    close_output(out, dict(status='PASS_BOUNDED_SINGLE_STEP_INTERFACE_CHECK', cell=cell, seed=seed,
                          batch_size=args.batch_size, original_batch_size=32, original_epoch=1,
                          selected_loader_item_indices=raw['item_index'].tolist(), device=args.device,
                          torch=torch.__version__, numpy=np.__version__, loss=float(loss.detach()),
                          preclip_gradient_norm=float(norm), clip_threshold=200., clip_activated=bool(norm > 200.),
                          nll_count=metadata['nll_count'], position_count=metadata['position_count'],
                          primary_forward_calls=count['value'], legacy_shadow_forward_calls=20 if reference is not None else 0,
                          disposable_Adam_steps=2 if reference is not None else 1, primary_training_steps=0,
                          C_D_original_bridge=bridge, initial_state_sha256=before, checkpoint_unchanged=digest(checkpoints[cell, seed, 0]) ==
                          next(r['sha256'] for r in json.loads((CODE / 'input_manifest.json').read_text())['checkpoints']
                               if (r['cell'], r['seed'], r['epoch']) == (cell, seed, 0)),
                          wall_seconds=time.monotonic() - started, full_training_reproduced=False,
                          Table4_evaluation_complete=False, full_CCPV_reproduction=False, public_release=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'evaluate', 'step-check'))
    for name in ('data', 'checkpoint-root', 'noise', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--cell', choices=CELLS, default='C')
    parser.add_argument('--seed', choices=SEEDS, type=int, default=0)
    parser.add_argument('--batch-size', choices=(2, 32), type=int, default=2)
    args = parser.parse_args()
    require(args.stage == 'step-check' or args.device == 'cpu', 'Original Table4 evaluation/prepare is CPU-only.')
    a, noise, checkpoints, out = load_inputs(args)
    if args.stage == 'prepare':
        prepare(a, noise, checkpoints, out)
    elif args.stage == 'evaluate':
        evaluate(a, noise, checkpoints, out)
    else:
        step_check(a, noise, checkpoints, out, args)


if __name__ == '__main__':
    main()
