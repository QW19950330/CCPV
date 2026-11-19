"""Explicit-path, create-only I/O for deterministic saved-readout calculations."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save(path, **values):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **values)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def table(path, entries):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def output_record(path):
    return {'path': Path(path).name, 'sha256': sha(path)}
