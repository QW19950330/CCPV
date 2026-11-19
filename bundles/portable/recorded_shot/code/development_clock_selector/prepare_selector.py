"""Derive the already fixed clock-case selector from its pinned raw event XML."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from lxml import etree

sys.dont_write_bytecode = True
MATCH_ID = 'DFL-MAT-J03WOH'
FRAME_LITERAL = '158639'
EVENT_RELATIVE_PATH = 'CCPV_data_expansion_20261023_IDSSE/raw/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml'
EVENT_BYTES = 635747
EVENT_SHA256 = 'abc27e3268e6126fb3e6e90bff5cca8d4586a5e0cefe9bd59bf32871e89d8544'
SELECTOR_CANONICAL_SHA256 = 'f2453293527a32b89c2dc70d6ea80b63d3edbfa1892f8a1619ac17ec58814822'


def tag(element):
    return element.tag.rsplit('}', 1)[-1]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def prepare(data_root):
    source = (data_root / EVENT_RELATIVE_PATH).resolve(strict=True)
    if not source.is_relative_to(data_root):
        raise ValueError('Fixed raw event path escapes the explicit input root')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != EVENT_BYTES or digest != EVENT_SHA256:
        raise ValueError('Pinned raw event XML identity mismatch')
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    root = etree.fromstring(raw, parser=parser)
    final, versions = {}, 0
    for event in root:
        if tag(event) != 'Event':
            continue
        if event.get('MatchId') != MATCH_ID or not event.get('EventId'):
            raise ValueError('Raw source match or event identity invalid')
        final[event.get('EventId')] = event
        versions += 1
    hits = []
    literal_frame_records = 0
    for event in final.values():
        if event.get('CalculatedFrame') != FRAME_LITERAL:
            continue
        literal_frame_records += 1
        names = [tag(node) for node in event.iter()]
        if 'Delete' in names or names.count('ShotAtGoal') != 1:
            continue
        hits.append(event)
    if len(hits) != 1:
        raise ValueError('Fixed literal match/frame/final-shot rule is not uniquely identifying')
    event = hits[0]
    selector = {'match_id': event.get('MatchId'), 'event_id': event.get('EventId'),
                'calculated_frame': int(event.get('CalculatedFrame'))}
    selector_sha = hashlib.sha256(canonical(selector)).hexdigest()
    if selector_sha != SELECTOR_CANONICAL_SHA256:
        raise ValueError('Raw-derived selector differs from the previously fixed selector pin')
    evidence = {'source_relative_path': EVENT_RELATIVE_PATH, 'source_bytes': len(raw), 'source_sha256': digest,
                'document_event_versions': versions, 'final_event_ids': len(final),
                'final_literal_frame_records': literal_frame_records, 'unique_final_active_shot_hits': len(hits),
                'selector_canonical_sha256': selector_sha,
                'rule': 'Exact fixed match + literal CalculatedFrame; final EventId version; no Delete; exactly one ShotAtGoal.',
                'player_or_outcome_fields_used': False, 'timestamps_or_residuals_used_for_selection': False,
                'frame_or_time_nearest_search': False, 'alternate_cases_or_offsets_searched': False,
                'positions_bytes_opened': 0, 'model_forward': 0, 'new_fit': 0}
    return selector, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True, help='New private selector JSON outside the code package.')
    p.add_argument('--receipt', required=True, help='New execution receipt JSON outside the code package.')
    p.add_argument('--compare-selector', help='Optional existing private selector, used only after raw derivation.')
    args = p.parse_args()
    data_root = Path(args.data_root).resolve(strict=True)
    output, receipt = Path(args.output).resolve(), Path(args.receipt).resolve()
    if output == receipt:
        raise ValueError('Selector and receipt outputs must differ')
    for path in (output, receipt):
        if path.exists():
            raise FileExistsError('Refusing existing output: ' + str(path))
        if path.is_relative_to(Path(__file__).resolve().parent):
            raise ValueError('Private outputs must stay outside the code package')
        if not path.parent.is_dir():
            raise FileNotFoundError('Create the explicit output parent first: ' + str(path.parent))
    selector, evidence = prepare(data_root)
    if args.compare_selector:
        with Path(args.compare_selector).open(encoding='utf-8-sig') as f:
            existing = json.load(f)
        if canonical(existing) != canonical(selector):
            raise ValueError('Existing selector differs in fields, types or values')
        evidence['comparison_with_existing_selector'] = {'all_three_fields_and_types_exact': True,
                                                         'existing_selector_used_for_derivation': False}
    write_new(output, selector)
    write_new(receipt, {'status': 'PASS_FIXED_SELECTOR_DERIVED_FROM_RAW_EVENT_XML',
                        'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'python': sys.version,
                        'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'evidence': evidence})
    print(json.dumps({'status': 'PASS_FIXED_SELECTOR_DERIVED_FROM_RAW_EVENT_XML', 'pid': os.getpid(),
                      'source_bytes': EVENT_BYTES, 'unique_hits': evidence['unique_final_active_shot_hits'],
                      'selector_canonical_sha256': evidence['selector_canonical_sha256']}), flush=True)


if __name__ == '__main__':
    main()
