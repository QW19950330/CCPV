"""Execution-only destination guard; leaves the executed mathematical modules unchanged."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent


def preflight(data_root, output):
    root = Path(data_root).resolve(strict=True)
    out = Path(output).resolve()
    contract = json.loads((PACKAGE / 'input_contract.json').read_text('utf-8'))
    relative = [r['relative_path'] for r in contract['inputs'] + contract['source_pins']]
    relative += [r['reference_relative_path'] for r in contract['reference_outputs']]
    relative.append(contract['current_manuscript']['relative_path'])
    protected = {(root / PurePosixPath(r).parts[0]).resolve() for r in relative}
    if out == root or any(out.is_relative_to(p) for p in protected):
        raise ValueError('Output must be outside every bound original source project tree')
    if out.is_relative_to(PACKAGE):
        raise ValueError('Private outputs must be outside the code package')
    if out.exists():
        raise FileExistsError('Output must be a new directory')
    return root, out, len(protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        root, out, count = preflight(args.data_root, args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps({'status': 'PASS_OUTPUT_SEPARATED_FROM_ORIGINAL_SOURCE_TREES',
                      'protected_project_trees': count, 'check_only': args.check_only,
                      'raw_or_readout_payload_read_by_preflight': False}), flush=True)
    if not args.check_only:
        subprocess.run([sys.executable, '-B', str(PACKAGE / 'pipeline.py'), 'run',
                        '--data-root', str(root), '--output', str(out)], check=True)


if __name__ == '__main__':
    main()
