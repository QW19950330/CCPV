"""Reaggregate saved results into current v10 S3/S4/S5/S7. Standard library only."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import time

PACKAGE = Path(__file__).resolve().parent
SEEDS = list(range(5))
CHECKS = Counter()


def require(value, message):
    if not value:
        raise ValueError(message)
    CHECKS[message] += 1


def number(value):
    if value is None or str(value).strip().upper() in ('', 'NA', 'NAN', 'NONE'):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def avg(values):
    return math.fsum(values) / len(values) if values and all(x is not None for x in values) else None


def delta(left, right):
    return left - right if left is not None and right is not None else None


def stats(values):
    complete = bool(values) and all(x is not None for x in values)
    return {'mean': avg(values), 'sd': statistics.stdev(values) if complete and len(values) > 1 else None,
            'planned': len(values), 'defined': sum(x is not None for x in values),
            'negative': sum(x is not None and x < 0 for x in values),
            'positive': sum(x is not None and x > 0 for x in values)}


def close(a, b, message):
    a, b = number(a), number(b)
    require((a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=2e-14)), message)


def pm(s, digits, sign=False):
    if s['mean'] is None or s['sd'] is None:
        return 'NA'
    return format(s['mean'], ('+' if sign else '') + '.' + str(digits) + 'f') + ' ± ' + format(s['sd'], '.' + str(digits) + 'f')


def unique(rows, **selector):
    found = [r for r in rows if all(r[k] == str(v) for k, v in selector.items())]
    require(len(found) == 1, 'Unique selector ' + json.dumps(selector, sort_keys=True))
    return found[0]


def index(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    require(len(result) == len(rows), 'No duplicate saved keys: ' + ','.join(keys))
    return result


def read_inputs(root):
    contract = json.loads((PACKAGE / 'input_contract.json').read_text('utf-8'))
    result = {}
    for rec in contract['files']:
        rel = PurePosixPath(rec['relative_path'])
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in str(rel) and ':' not in str(rel), 'Safe input path ' + rec['role'])
        path = (root / Path(*rel.parts)).resolve(strict=True)
        require(path.is_relative_to(root), 'Input remains under explicit root ' + rec['role'])
        raw = path.read_bytes()
        require(len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256'], 'Pinned input identity ' + rec['role'])
        text = raw.decode('utf-8-sig')
        result[rec['role']] = list(csv.DictReader(io.StringIO(text, newline=''))) if rec['type'] == 'csv' else json.loads(text)
    return result, contract


def s3(data):
    rows = data['hm_cells']
    groups = defaultdict(list)
    fields = ('endpoint_level', 'group', 'horizon_s', 'metric', 'comparison')
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    full = []
    for key, rr in sorted(groups.items()):
        ix = index(rr, ('match_id', 'seed'))
        fixtures = sorted({r['match_id'] for r in rr})
        require(len(fixtures) == 12 and len(rr) == 60, 'S3 full planned 12 fixture x 5 seed endpoint')
        counts = {m: {int(r['windows']) for r in rr if r['match_id'] == m} for m in fixtures}
        require(all(len(x) == 1 for x in counts.values()), 'S3 same window count across seeds')
        seed_values = []
        for s in SEEDS:
            left = avg([number(ix[m, str(s)]['left_value']) for m in fixtures])
            right = avg([number(ix[m, str(s)]['reference_value']) for m in fixtures])
            seed_values.append(delta(left, right))
        fixture_values = [delta(avg([number(ix[m, str(s)]['left_value']) for s in SEEDS]),
                                avg([number(ix[m, str(s)]['reference_value']) for s in SEEDS])) for m in fixtures]
        st, ft = stats(seed_values), stats(fixture_values)
        r = dict(zip(fields, key)) | st | {'fixtures': len(fixtures), 'fixture_defined': ft['defined'],
             'fixture_negative': ft['negative'], 'fixture_positive': ft['positive'],
             'windows': sum(next(iter(x)) for x in counts.values()), 'seed_values': seed_values,
             'fixture_values': dict(zip(fixtures, fixture_values))}
        ref = unique(data['hm_summary'], **dict(zip(fields, key)))
        close(st['mean'], ref['paired_difference'], 'S3 saved endpoint mean agrees')
        close(st['sd'], ref['paired_seed_SD'], 'S3 paired ddof1 SD agrees')
        require(st['negative'] == int(ref['seeds_improved']) and ft['negative'] == int(ref['fixtures_improved']), 'S3 all adverse and improving counts retained')
        full.append(r)
    require(len(full) == 56, 'S3 all 56 saved endpoints reaggregated without selection')
    plan = data['hm_plan']
    require(plan['logical_truth_rows'] == 191 and plan['cohort_rule'] == 'strict_C_ALL191', 'S3 source plan uses all 191')
    require('original32 trajectories' in plan['metrics']['primary'] and 'draw-minor32' in plan['prediction_target'], 'S3 M32 is configured source scope, not new sampling')
    selected = []
    for group, contrast in [('all_known_history_players_including_targets', 'H_minus_M'), ('ball', 'H_minus_D')]:
        for metric in ('position_energy_U', 'velocity_energy_U'):
            rr = [r for r in full if r['group'] == group and r['comparison'] == contrast and r['horizon_s'] == '2.0' and r['metric'] == metric]
            require(len(rr) == 1 and rr[0]['windows'] == 191, 'S3 final retained endpoint selector')
            selected.append(rr[0])
    headers = ['Population / endpoint', 'Contrast', 'Difference ± paired seed SD', 'Units', 'Improving seeds', 'Improving fixture means']
    body = [[r['group'] + ';2s;' + r['metric'], r['comparison'], pm(r, 9, True),
             'native_m' if r['metric'] == 'position_energy_U' else 'native_m_per_s',
             f"{r['negative']}/{r['planned']}", f"{r['fixture_negative']}/{r['fixtures']}"] for r in selected]
    denominator = {'windows': 191, 'fixtures': 12, 'seeds': len(SEEDS), 'configured_draws': 32,
                   'configured_horizon_s': 2, 'input_endpoint_rows': len(rows), 'all_reaggregated_endpoints': len(full),
                   'selected_endpoints': selected, 'NA_policy': 'A missing required seed/fixture propagates NA; no survivor average.',
                   'scope': 'Reaggregation of saved Energy metrics; no trajectory generation or model inference.'}
    return {'headers': headers, 'rows': body}, denominator, full


def s4(data):
    source = data['matched_fixture']
    ix = index(source, ('cell', 'seed', 'target', 'match_id'))
    fixtures = sorted({r['match_id'] for r in source})
    cells = ('OldTrained_Mean', 'OldTrained_MC', 'MCTrained_Mean', 'MCTrained_MC')
    require(len(fixtures) == 6, 'S4 six equal fixtures')
    metrics = ('log_loss', 'brier')
    macro = {}
    selected_counts = [ix[cells[0], '0', 'Y10', m] for m in fixtures]
    counts = {'windows': sum(int(r['windows']) for r in selected_counts), 'positive_Y10': sum(int(r['positives']) for r in selected_counts)}
    lineage = data['matched_lineage']
    require(counts['windows'] == lineage['N'] and counts['positive_Y10'] == lineage['positive_Y10'], 'S4 windows and positives agree with label lineage')
    for cell in cells:
        for seed in SEEDS:
            for m, original in zip(fixtures, selected_counts):
                r = ix[cell, str(seed), 'Y10', m]
                require(r['windows'] == original['windows'] and r['positives'] == original['positives'], 'S4 unchanged denominator across cells/seeds')
            for metric in metrics:
                macro[cell, seed, metric] = avg([number(ix[cell, str(seed), 'Y10', m][metric]) for m in fixtures])
    output, detail = [], {}
    for cell in cells:
        st = stats([macro[cell, s, 'log_loss'] for s in SEEDS])
        ref = unique(data['matched_summary'], cell=cell, target='Y10', metric='log_loss')
        close(st['mean'], ref['mean'], 'S4 recomputed macro cell mean agrees')
        close(st['sd'], ref['seed_SD'], 'S4 recomputed macro cell SD agrees')
        output.append([cell, pm(st, 12), 'NA'])
        detail[cell] = st
    for contrast, reference in [('A_match', 'OldTrained_MC'), ('B_replace', 'OldTrained_Mean')]:
        ss = {}
        for metric in metrics:
            values = [delta(macro['MCTrained_MC', s, metric], macro[reference, s, metric]) for s in SEEDS]
            for s, x in zip(SEEDS, values):
                close(x, unique(data['matched_pairs'], contrast=contrast, seed=s, target='Y10')[metric], 'S4 paired seed agrees ' + contrast + metric)
            ss[metric] = stats(values)
            fv = [avg([delta(number(ix['MCTrained_MC', str(s), 'Y10', m][metric]), number(ix[reference, str(s), 'Y10', m][metric])) for s in SEEDS]) for m in fixtures]
            ss[metric]['fixture_values'] = dict(zip(fixtures, fv))
            ss[metric]['improving_fixture_means'] = sum(x is not None and x < 0 for x in fv)
        st = ss['log_loss']
        output.append([contrast, pm(st, 12), f"{st['negative']}/{st['planned']}"])
        detail[contrast] = ss
    cv = unique(data['matched_cv'], arm='CV_Direct_WithEarlyBCE', target='q10')
    cvstats = {'mean': number(cv['logloss_mean']), 'sd': number(cv['logloss_seed_SD'])}
    output.append(['CV_Direct_WithEarlyBCE', pm(cvstats, 12), 'NA'])
    detail['historical_CV'] = cvstats | {'scope': 'Unchanged historical aggregate row display only; original CV training/readouts not rerun.'}
    detail.update(counts | {'fixtures': len(fixtures), 'seeds': len(SEEDS), 'target': lineage['target'],
                           'clock': lineage['clock'], 'omitted_zero_eligible_match': lineage['omitted_zero_eligible_match'],
                           'readout_mean': 'OldGPUmean; CPUmean bridge is not substituted',
                           'scope': 'Saved per-fixture scores reaggregated; no head fitting or head readout inference.'})
    return {'headers': ['Training / readout or contrast', 'Y10 logloss mean / difference ± seed SD', 'Improving seeds for contrast'], 'rows': output}, detail


def s5(data):
    source, coverage = data['dep_phase_fixture'], data['dep_phase_coverage']
    ix = index(source, ('phase', 'held', 'arm', 'seed'))
    fixtures = sorted({r['held'] for r in source})
    require(len(fixtures) == 6, 'S5 six equal fixtures')
    metrics = ('logloss', 'brier', 'absolute_bias')
    phases, rows = [], []
    for phase in range(4):
        counts = {k: sum(int(r[k]) for r in coverage if r['phase'] == str(phase)) for k in ('candidate_anchors', 'eligible_anchors', 'positive_anchors', 'excluded_anchors')}
        ss = {}
        for metric in metrics:
            values = []
            for seed in SEEDS:
                arm_values = {a: avg([number(ix[str(phase), m, a, str(seed)][metric]) for m in fixtures]) for a in ('G', 'GM')}
                values.append(delta(arm_values['GM'], arm_values['G']))
            st = stats(values)
            ref = unique(data['dep_phase_summary'], phase=phase, arm_or_contrast='GM-G', metric=metric)
            close(st['mean'], ref['mean'], 'S5 phase mean agrees')
            close(st['sd'], ref['seed_SD'], 'S5 phase paired SD agrees')
            require(st['negative'] == int(ref['negative_values']), 'S5 phase signs retained')
            ss[metric] = st
        for seed in SEEDS:
            for arm in ('G', 'GM'):
                require(sum(int(ix[str(phase), m, arm, str(seed)]['n']) for m in fixtures) == counts['eligible_anchors'], 'S5 phase count matches all score arms/seeds')
                require(sum(int(ix[str(phase), m, arm, str(seed)]['positives']) for m in fixtures) == counts['positive_anchors'], 'S5 positive count matches all score arms/seeds')
        phases.append({'phase': phase} | counts | ss)
        rows.append([str(phase), str(counts['eligible_anchors']), str(counts['positive_anchors'])] + [pm(ss[m], 8) for m in metrics])
    support = data['dep_event_support']
    sup = index(support, ('match_id', 'native_event_id'))
    supported = {k for k, r in sup.items() if int(r['eligible_windows']) > 0}
    events = index(data['dep_event_metrics'], ('match_id', 'native_event_id', 'arm_or_contrast', 'seed'))
    event_stats = {}
    for metric in ('conditional_logloss', 'conditional_brier'):
        values = []
        for seed in SEEDS:
            arms = {}
            for arm in ('G', 'GM'):
                perfixture = []
                for mid in fixtures:
                    kk = sorted(k for k in supported if k[0] == mid)
                    perfixture.append(avg([number(events[mid, eid, arm, str(seed)][metric]) for _, eid in kk]))
                arms[arm] = avg(perfixture)
            values.append(delta(arms['GM'], arms['G']))
        st = stats(values)
        ref = unique(data['dep_event_summary'], arm_or_contrast='GM-G', metric=metric)
        close(st['mean'], ref['mean'], 'S5 supported-event within-fixture equal mean agrees')
        close(st['sd'], ref['seed_SD'], 'S5 supported-event paired SD agrees')
        st['unsupported_event_seed_NAs_retained'] = sum(number(r[metric]) is None for r in data['dep_event_metrics'] if r['arm_or_contrast'] == 'GM-G')
        event_stats[metric] = st
    for key in set(sup) - supported:
        require(all(number(events[key[0], key[1], 'GM-G', str(s)]['conditional_logloss']) is None for s in SEEDS), 'S5 unsupported events stay NA')
    full = index(data['dep_full_fixture'], ('held', 'arm', 'seed'))
    deletion = {}
    for omitted in fixtures:
        retained = [m for m in fixtures if m != omitted]
        values = []
        for s in SEEDS:
            x = delta(avg([number(full[m, 'GM', str(s)]['logloss']) for m in retained]), avg([number(full[m, 'G', str(s)]['logloss']) for m in retained]))
            ref = unique(data['dep_deletion_metrics'], omitted_fixture=omitted, arm_or_contrast='GM-G', seed=s, metric='logloss')
            close(x, ref['value'], 'S5 deletion reaggregation from original six fixture scores agrees')
            values.append(x)
        deletion[omitted] = stats(values)
    limits = [x['mean'] for x in deletion.values()]
    deletion_range = {'min': min(limits), 'max': max(limits), 'negative_fixture_deletions': sum(x < 0 for x in limits)}
    ref = unique(data['dep_deletion_summary'], arm_or_contrast='GM-G', metric='logloss')
    close(deletion_range['min'], ref['min_deletion_mean'], 'S5 all six deletion range minimum agrees')
    close(deletion_range['max'], ref['max_deletion_mean'], 'S5 all six deletion range maximum agrees')
    histogram = Counter(int(sup[k]['eligible_windows']) for k in supported)
    total = {k: sum(p[k] for p in phases) for k in ('candidate_anchors', 'eligible_anchors', 'positive_anchors', 'excluded_anchors')}
    multi = [r for r in data['dep_multi'] if r['eligible'] == 'True']
    require(sum(histogram[k] * k for k in histogram) == total['positive_anchors'] + sum(int(r['linked_event_count']) - 1 for r in multi), 'S5 event links and multi-event windows reconcile')
    detail = total | {'fixtures': len(fixtures), 'seeds': len(SEEDS), 'phase': phases,
        'candidate_event_keys': len(sup), 'supported_event_keys': len(supported), 'unsupported_event_keys': len(sup) - len(supported),
        'eligible_windows_per_supported_event': dict(sorted(histogram.items())), 'multi_event_eligible_windows': len(multi),
        'conditional_event_metrics': event_stats, 'all_six_deletions': deletion, 'deletion_logloss_range': deletion_range,
        'event_aggregation': 'Equal supported events within fixture, then six fixtures equally, then paired five-seed summaries. Unsupported events remain explicit NA.',
        'scope': 'Saved score reaggregation only; phase assignments/event links are inputs, not independently regenerated from raw; no refit.'}
    return {'headers': ['Phase', 'Eligible windows', 'Positive windows', 'Δ logloss ± seed SD', 'Δ Brier ± seed SD', 'Δ absolute fixture bias ± seed SD'], 'rows': rows}, detail


def s7(data):
    first, second = data['qc_first_summary'], data['qc_second_summary']
    measurement, mp = data['qc_measurement'], data['qc_measurement_plan']
    event, binding, decision = data['qc_event_summary'], data['qc_binding'], data['qc_decision']
    fc, sc = first['counts'], second['counts']
    frames = data['qc_first_frames']
    require(sum(int(r['matched_keys']) for r in frames) == fc['common_unique_finite_identity_keys'], 'S7 saved frame counts reconcile first-half common keys')
    unmatched_inside = unmatched_outside = 0
    for r in frames:
        missing = int(r['raw_finite_ids']) - int(r['matched_keys'])
        require(missing >= 0, 'S7 no negative unmatched saved frame count')
        if 1 <= int(r['g']) <= fc['images']:
            unmatched_inside += missing
        else:
            unmatched_outside += missing
    require(unmatched_inside + unmatched_outside == fc['raw_valid_without_valid_unique_production'], 'S7 all unmatched raw keys reconciled')
    require(fc['exact_float_equal_keys'] == fc['common_unique_finite_identity_keys'] and fc['production_valid_unique_without_valid_raw'] == 0, 'S7 first-half exact finite equality and no extra production keys')
    require(measurement['on_hypothesized_grid_count_xy'] == [measurement['unique_finite_player_rows']] * 2, 'S7 all first-half coordinates on fixed grid')
    require(max(measurement['grid_residual_max_m_xy']) <= mp['grid_residual_tolerance_m'], 'S7 coordinate grid residual within recorded tolerance')
    close(measurement['zero_displacement_fraction'], measurement['zero_displacement_pairs'] / measurement['lag5_pairs'], 'S7 displacement fraction preserves denominator')
    inv = data['qc_literal']
    labels = [json.loads(r['label_raw_json']) for r in inv]
    require(all(r['label_known_exact'] == 'True' and label == label.upper() for r, label in zip(inv, labels)), 'S7 accepted uppercase literals retained')
    require(sum(int(r['record_count']) for r in inv) == event['all_event_record_count'], 'S7 inventory total preserved')
    counts = {(h, label): sum(int(r['record_count']) for r in inv if int(r['half']) == h and json.loads(r['label_raw_json']) == label) for h in (1, 2) for label in ('SHOT', 'GOAL')}
    require(event['unknown_literal_record_count'] == event['invalid_clock_record_count'] == 0, 'S7 original no invalid literal/clock result')
    require(all(v['canonical_shot_goal_groups'] == 0 for v in event['same_time_diagnostics'].values()), 'S7 absence of same-frame/position Shot-Goal co-groups retained')
    target = event['literal_target_diagnostics']
    require(all(target[x]['position_vs_frame_binary_changed'] == target[x]['position_vs_frame_member_changed'] == 0 for x in ('Shot', 'Goal')), 'S7 only fixed-grid Shot/Goal clock sensitivity is zero')
    require(first['repository_revision'] == second['repository_revision'] == binding['BAS_provenance']['actual_previously_read_input']['fixed_revision'], 'S7 fixed release revision agrees across final source records')
    require(binding['BAS_provenance']['fixed_release_binding_matches_GSR'], 'S7 final released BAS clock binding is explicit')
    require(decision['scope_decision']['legacy_raw_adapter']['status'] == 'FAIL_UNCHANGED' and sc['above_serialization_tolerance_keys'] > 0, 'S7 original second-half legacy qualification remains failed')
    require(not decision['accepted_endpoint_unchanged']['goal_header_block_union'], 'S7 Goal/Header/Block not unioned into Shot')
    require(all(x['BAS']['frames_missing_production_images'] == x['BAS']['position_frame_outside_20ms'] == 0 for x in (first, second)), 'S7 both halves BAS rows join their released image clocks')
    bas_n = first['BAS']['rows'] + second['BAS']['rows']
    require(bas_n == event['all_event_record_count'], 'S7 no BAS records lost across halves')
    tolerance_ms = max(x['BAS']['max_abs_position_frame_residual_ms'] for x in (first, second))
    require(tolerance_ms == 20 and 1000 / first['info']['frame_rate'] == 40, 'S7 recorded clock convention retained')
    rows = [
        ['Production scope/version', f"QC half 1: {fc['images']:,} images; QC half 2: {sc['images']:,} images. Repository {first['repository_release']} at revision {first['repository_revision']}; payload info.version {first['info']['version']}", 'Version labels remain separate; legacy raw compatibility and released native-clock qualification are different checks'],
        ['First-half fixed player projection', f"g=N−250; {fc['common_unique_finite_identity_keys']:,} common finite identity–frame keys exactly coordinate-equal; no extra production keys", 'No BALL input or output; not a new outcome test'],
        ['First-half unmatched raw keys', f"{unmatched_inside + unmatched_outside}: {unmatched_outside} beyond image boundary, three internal absences" if unmatched_inside == 3 else f"{unmatched_inside + unmatched_outside}: {unmatched_outside} beyond image boundary, {unmatched_inside} internal absences", 'Retained without interpolation or assumed cause'],
        ['First-half measurement grid', f"All {measurement['unique_finite_player_rows']:,} coordinates within 10⁻⁹ m of {mp['coordinate_grids_m'][0]} m × {mp['coordinate_grids_m'][1]} m grid; {measurement['zero_displacement_pairs']:,}/{measurement['lag5_pairs']:,} valid 200 ms differences zero", 'Zero measured displacement does not mean physical stillness'],
        ['BAS inventory', f"{event['all_event_record_count']:,} records: first half {event['first_half_records']:,}, second half {event['second_half_records']:,}; all accepted uppercase spellings; no invalid clock", 'Source ordinal is a record key, not an authenticated independent physical event'],
        ['SHOT / GOAL', f"SHOT {counts[1, 'SHOT']}/{counts[2, 'SHOT']}; GOAL {counts[1, 'GOAL']}/{counts[2, 'GOAL']}; zero Shot–Goal co-groups under identical frame or position", 'No forced pair, invented label or Goal union; no inference that a goal lacked a shot'],
        ['Primary clock/target', 'position milliseconds from released half-video first frame; Shot/SHOT in (t0,t0+10 s], t0=40(g−1)', 'Not a kick-off certification or automatic physical-label equivalence to DFL'],
        ['First-half label-clock grid', f"{event['first_half_clock_only_candidates']} anchors: {event['complete_10s_candidates']} complete, {target['Shot']['position_positive_windows']} SHOT-positive windows, four tail NAs" if event['incomplete_tail_candidates'] == 4 else f"{event['first_half_clock_only_candidates']} anchors: {event['complete_10s_candidates']} complete, {target['Shot']['position_positive_windows']} SHOT-positive windows, {event['incomplete_tail_candidates']} tail NAs", 'Not final model-input eligibility; overlapping windows are not independent events'],
        ['Clock sensitivity', 'No Shot/Goal membership or label differences using 40(frame−1) on this grid', 'Fixed-grid sensitivity; ±20 ms field quantization is not a physical event-error bound'],
        ['Second-half legacy projection', f"Under g=N−250, {sc['above_serialization_tolerance_keys']:,} of {sc['common_unique_finite_identity_keys']:,} common finite keys exceeded 10⁻⁹ m coordinate tolerance; the source-compatibility review failed", 'The original failure is retained. A post hoc 27-frame raw/release relation does not authorize shifting native GSR or BAS clocks'],
        ['Both-half released clock binding', f"All {bas_n:,} QC BAS records map to released images; position/frame residuals are at most {tolerance_ms} ms in each half", 'Same-release record compatibility, not independent physical synchronization; native shift remains zero']]
    detail = {'first_half_images': fc['images'], 'second_half_images': sc['images'],
        'first_half_common_keys': fc['common_unique_finite_identity_keys'], 'raw_unmatched_outside_images': unmatched_outside,
        'raw_unmatched_inside_images': unmatched_inside, 'zero_displacement_pairs': measurement['zero_displacement_pairs'],
        'valid_lag5_pairs': measurement['lag5_pairs'], 'BAS_records': bas_n, 'halves': [first['BAS'], second['BAS']],
        'shot_records_by_half': [counts[1, 'SHOT'], counts[2, 'SHOT']], 'goal_records_by_half': [counts[1, 'GOAL'], counts[2, 'GOAL']],
        'clock_candidates': event['first_half_clock_only_candidates'], 'clock_complete': event['complete_10s_candidates'],
        'clock_tail_NA': event['incomplete_tail_candidates'], 'shot_positive_windows': target['Shot']['position_positive_windows'],
        'legacy_second_half_failed': True, 'legacy_second_half_common_keys': sc['common_unique_finite_identity_keys'],
        'legacy_second_half_tolerance_failures': sc['above_serialization_tolerance_keys'],
        'posthoc_shift_diagnostic_not_applied': binding['posthoc_raw_stream_diagnostic']['raw_g_minus_released_g'],
        'legacy_failure_status': decision['scope_decision']['legacy_raw_adapter']['status'],
        'native_record_clock_status': decision['scope_decision']['released_GSR_and_BAS_clock']['status'],
        'scope': 'Saved qualification-count aggregation/display only. No raw GSR/BAS/XML or SQLite reread; source checks themselves not regenerated.'}
    return {'headers': ['Check', 'Completed observation', 'Interpretation boundary'], 'rows': rows}, detail


def markdown(name, table):
    return '\n'.join(['## Table ' + name, '', '| ' + ' | '.join(table['headers']) + ' |',
                      '| ' + ' | '.join('---' for _ in table['headers']) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in table['rows']])


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def verify_current(path, tables, denominators, contract):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == contract['current_reference']['sha256'], 'Current v10 reference manuscript identity')
    text = raw.decode('utf-8-sig')
    comparisons = []
    for name, table in tables.items():
        match = re.search(r'^## Table ' + name + r'\..*?(?=^## Table |\Z)', text, re.M | re.S)
        require(match is not None, 'Current section exists ' + name)
        lines = [line for line in match.group().splitlines() if line.startswith('|')]
        parsed = [[x.strip() for x in line.strip().strip('|').split('|')] for line in lines]
        expected = [parsed[0]] + parsed[2:]
        actual = [table['headers']] + table['rows']
        require(len(expected) == len(actual), 'Current row count matches ' + name)
        for row_idx, (a, e) in enumerate(zip(actual, expected)):
            require(len(a) == len(e), 'Current column count matches ' + name)
            for col, (left, right) in enumerate(zip(a, e)):
                comparisons.append({'table': name, 'row': row_idx, 'column': col, 'exact': left == right})
                require(left == right, f'Current exact cell {name}[{row_idx},{col}]: {left!r} != {right!r}')
    s3d, s4d, s5d = (denominators[x] for x in ('S3', 'S4', 'S5'))
    a, event, deletion = s4d['A_match'], s5d['conditional_event_metrics'], s5d['deletion_logloss_range']
    prose = [f"All {s3d['windows']} strict-availability windows, twelve TRAIN fixtures, five seeds",
             f"These {s4d['windows']:,}-window values", f"only {a['log_loss']['improving_fixture_means']}/6 fixture means",
             f"Brier difference is adverse ({a['brier']['mean']:+.12f})",
             f"Among {s5d['candidate_event_keys']} candidate-linked event IDs, {s5d['supported_event_keys']} had eligible support and {s5d['unsupported_event_keys']} had none",
             f"{s5d['eligible_windows_per_supported_event'][3]} supported events appeared in three eligible windows and {s5d['eligible_windows_per_supported_event'][4]} in four",
             format(event['conditional_logloss']['mean'], '.8f').replace('-', '−') + ' ± ' + format(event['conditional_logloss']['sd'], '.8f'),
             format(event['conditional_brier']['mean'], '.8f').replace('-', '−') + ' ± ' + format(event['conditional_brier']['sd'], '.8f'),
             format(deletion['min'], '.8f').replace('-', '−') + ' to ' + format(deletion['max'], '.8f').replace('-', '−')]
    for fragment in prose:
        require(fragment in text, 'Current accompanying denominator/negative-result text: ' + fragment)
    return {'status': 'PASS_CURRENT_V10_EVERY_CELL_EXACT', 'cells_including_headers': len(comparisons),
            'data_cells': len(comparisons) - sum(len(t['headers']) for t in tables.values()),
            'cell_checks': comparisons, 'accompanying_numeric_text_checks': prose,
            'reference_sha256': hashlib.sha256(raw).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manuscript', help='Optional current v10 Markdown, for post-generation comparison only.')
    args = p.parse_args()
    root, out = Path(args.input_root).resolve(strict=True), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    if out.is_relative_to(PACKAGE):
        raise ValueError('Derived outputs must stay outside the code package')
    started = time.monotonic()
    data, contract = read_inputs(root)
    tables, denominators = {}, {}
    tables['S3'], denominators['S3'], full = s3(data)
    tables['S4'], denominators['S4'] = s4(data)
    tables['S5'], denominators['S5'] = s5(data)
    tables['S7'], denominators['S7'] = s7(data)
    verification = verify_current(Path(args.manuscript).resolve(strict=True), tables, denominators, contract) if args.manuscript else None
    out.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        with (out / (name + '.csv')).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(table['headers'])
            writer.writerows(table['rows'])
    with (out / 'tables.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n\n'.join(markdown(k, v) for k, v in tables.items()) + '\n')
    write_json(out / 'tables.json', tables)
    write_json(out / 'denominators.json', denominators)
    write_json(out / 'S3_all56_reaggregated.json', full)
    if verification:
        write_json(out / 'current_v10_verification.json', verification)
    write_json(out / 'completion.json', {'status': 'PASS_SAVED_RESULTS_REAGGREGATED_CURRENT_V10_VERIFIED' if verification else 'REAGGREGATED_NOT_COMPARED_TO_MANUSCRIPT',
        'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'python': sys.version,
        'elapsed_seconds': time.monotonic() - started, 'checks_passed': sum(CHECKS.values()), 'check_counts': CHECKS,
        'input_files': contract['files'], 'total_input_bytes': sum(x['bytes'] for x in contract['files']),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'new_fit': 0, 'model_forward': 0,
        'GPU': False, 'raw_source_or_model_arrays_opened': False,
        'scope': 'S3/S4/S5 saved-score reaggregation, S4 historical CV aggregate display, S7 saved-QC summary aggregation/display; not training/inference/raw-QC generation.'})
    print(json.dumps({'status': 'PASS', 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started,
                      'current_cells_verified': verification['cells_including_headers'] if verification else 0,
                      'checks_passed': sum(CHECKS.values()), 'new_fit': 0, 'model_forward': 0}), flush=True)


if __name__ == '__main__':
    main()
