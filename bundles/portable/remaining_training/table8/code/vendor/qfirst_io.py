"""Create-only study writes and explicit locked payload boundaries."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import sys
import numpy as np

P = Path(__file__).resolve().parent
PLAN = 'experiment_plan.json'
BINDING = 'data_binding.json'
STATE_KEYS = ('hidden_weight', 'hidden_bias', 'output_weight', 'output_bias')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(P):
        raise ValueError('Output escapes this create-only study directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def put(path, value):
    with output_path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save(path, **arrays):
    with output_path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with output_path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def guard(expected, stage):
    """Root must create and authorize the immutable implementation lock first."""
    lp = P / 'implementation_lock.json'
    if sha(lp) != expected:
        raise ValueError('Implementation-lock identity changed')
    lock = read(lp)
    if lock['experiment_plan_sha256'] != sha(P / PLAN) or lock['data_binding_sha256'] != sha(P / BINDING):
        raise ValueError('Frozen plan/binding changed')
    for r in lock['files']:
        if sha(P / r['path']) != r['sha256']:
            raise ValueError('Frozen implementation changed: ' + r['path'])
    auth = read(P / 'runtime_authorization.json')
    if (auth['status'] != 'AUTHORIZED_FIXED_QFIRST_JOINT_PILOT'
            or auth['implementation_lock_sha256'] != expected or stage not in auth['allowed_stages']):
        raise ValueError('Missing exact root runtime authorization')
    return lock


def selected(path, keys):
    """Only explicit members may be decoded; callers supply a fixed whitelist."""
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('An explicit nonempty unique member whitelist is required')
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in keys}


def checkpoint(path):
    with np.load(path, allow_pickle=False) as z:
        if not set(STATE_KEYS).issubset(z.files):
            raise ValueError('Whole reads are restricted to small fitted-head checkpoints')
        return {key: z[key] for key in z.files}


def process_identity(stage, expected):
    return dict(utc=utc(), pid=os.getpid(), parent_pid=os.getppid(),
                python_executable=sys.executable, argv=sys.argv, cwd=str(Path.cwd()),
                stage=stage, implementation_lock_sha256=expected)
