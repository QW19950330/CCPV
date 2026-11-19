"""Create a verified input directory from an explicitly supplied authorized archive.

This utility neither downloads data nor claims that the historical source archive
is publicly available. The manifest supplies every source-relative mapping/hash.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, name):
    relative = Path(name)
    if relative.is_absolute():
        raise ValueError('Manifest requires relative paths')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path escaped its explicit root')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, help='Authorized original workspace/archive root; never inferred')
    parser.add_argument('--output-dir', required=True, help='New input directory, which must not exist')
    parser.add_argument('--metadata-dir', default=str(Path(__file__).resolve().parent.parent / 'metadata'))
    parser.add_argument('--verify-only', action='store_true', help='Verify all source hashes without copying or creating output')
    args = parser.parse_args()
    source, output, metadata = map(lambda p: Path(p).resolve(), (args.source_root, args.output_dir, args.metadata_dir))
    if output.exists():
        raise FileExistsError('Create-only input destination already exists')
    if output == source or source.is_relative_to(output) or output.is_relative_to(metadata):
        raise ValueError('Input destination conflicts with source or metadata directory')
    manifest_path = metadata / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    jobs = []
    for r in manifest['files']:
        src = inside(metadata, r['path']) if r['original'].startswith('generated') else inside(source, r['original'])
        dest = inside(output, r['path'])
        if not src.is_file():
            raise FileNotFoundError('Required fixed source absent: ' + r['original'])
        if sha(src) != r['sha256']:
            raise ValueError('Source hash mismatch: ' + r['original'])
        jobs.append((src, dest, r))
    if args.verify_only:
        print(json.dumps({'status': 'ALL_REQUIRED_SOURCES_VERIFIED', 'files': len(jobs), 'copies': 0}))
        return
    output.mkdir(parents=True, exist_ok=False)
    copied = 0
    try:
        for src, dest, r in jobs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open('rb') as inf, dest.open('xb') as outf:
                shutil.copyfileobj(inf, outf)
            if sha(dest) != r['sha256']:
                raise ValueError('Copied input hash mismatch: ' + r['path'])
            copied += 1
        with manifest_path.open('rb') as inf, (output / 'input_manifest.json').open('xb') as outf:
            shutil.copyfileobj(inf, outf)
    except BaseException:
        with (output / 'PREPARATION_FAILED.json').open('x', encoding='utf-8') as f:
            json.dump({'status': 'FAILED_PRESERVED', 'copied_files': copied,
                       'traceback': traceback.format_exc(), 'automatic_retry': False}, f, indent=2)
        raise
    print(json.dumps({'status': 'INPUTS_PREPARED_AND_VERIFIED', 'files': copied,
                      'bytes': sum(r['bytes'] for _, _, r in jobs), 'manifest_sha256': sha(manifest_path)}))


if __name__ == '__main__':
    main()
