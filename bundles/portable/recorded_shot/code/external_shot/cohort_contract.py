"""Complete candidate-cohort checks before any label-based exclusion.

Pure in-memory assembly. This is not a production acquisition/evaluation CLI.
All half contracts must be bound before parsing the independently held-out data.
"""
from copy import deepcopy
import re
import numpy as np
from gsr_features import validate_contract
from frozen_predict import make_row_key


def assemble_cohort(feature_batches, contracts, planned_fixtures):
    """Require both halves and every fixed-grid candidate for each fixture.

    Return all ledger rows and only input-eligible X/keys for frozen prediction.
    Future image support or eventual labels never affect prediction inclusion.
    A source summary/hash is checked here, but is not physical alignment proof.
    """
    planned = tuple(planned_fixtures)
    if not planned or len(set(planned)) != len(planned):
        raise ValueError('planned fixtures must be nonempty and unique')
    for fixture in planned:
        make_row_key(fixture, 1, 1)
    contracts, feature_batches = list(contracts), list(feature_batches)
    if len(contracts) != len(feature_batches):
        raise ValueError('one feature batch per half contract required')
    expected = {(f, h) for f in planned for h in (1, 2)}
    by_half = {}
    for contract, batch in zip(contracts, feature_batches):
        validate_contract(contract)
        half_key = contract['fixture'], contract['half']
        if half_key not in expected or half_key in by_half:
            raise ValueError('unplanned or duplicate fixture-half')
        summary, rows = batch['summary'], batch['ledger']
        first, last = contract['first_frame'], contract['last_frame']
        if (summary.get('fixture'), summary.get('half')) != half_key:
            raise ValueError('summary fixture-half mismatch')
        if (summary.get('input_sha256') != contract['input_sha256']
                or summary.get('input_bytes') != contract['input_bytes']
                or summary.get('declared_domain') != [first, last]
                or summary.get('info') != contract['expected_info']):
            raise ValueError('source summary not bound to half contract')
        missing_images = summary.get('missing_image_frames')
        if (not isinstance(missing_images, list)
                or any(type(g) is not int or not first <= g <= last for g in missing_images)
                or len(missing_images) != len(set(missing_images))):
            raise ValueError('missing image-frame support must be exact and unique')
        missing_images = set(missing_images)
        if summary.get('actual_image_count') != last - first + 1 - len(missing_images):
            raise ValueError('actual image count disagrees with declared support')
        expected_g = list(range(first + 24, last + 1, 75))
        keys = [make_row_key(*half_key, g) for g in expected_g]
        if len(rows) != len(keys) or [row.get('row_key') for row in rows] != keys:
            raise ValueError('full ordered candidate grid required, including excluded rows')
        x = np.asarray(batch['X'])
        if x.shape != (len(keys), 2, 16) or x.dtype != np.dtype('float64'):
            raise ValueError('feature shape/dtype does not match complete candidate ledger')
        for i, (g, row) in enumerate(zip(expected_g, rows)):
            if type(row.get('half')) is not int or type(row.get('g')) is not int:
                raise ValueError('row half/g must be exact integers')
            if (row.get('fixture'), row.get('half'), row.get('g')) != (*half_key, g):
                raise ValueError('row identity differs from full key')
            if type(row.get('candidate_index')) is not int or row['candidate_index'] != i:
                raise ValueError('candidate_index mismatch')
            if (type(row.get('t0_ms')) is not int or row['t0_ms'] != 40 * (g - 1)
                    or type(row.get('end_ms')) is not int or row['end_ms'] != 40 * (g - 1) + 10000):
                raise ValueError('row clock mismatch')
            eligible = bool(np.isfinite(x[i, :, :13]).all())
            if type(row.get('input_eligible')) is not bool or row['input_eligible'] != eligible:
                raise ValueError('input eligibility must match current shared features')
            if type(row.get('future_record_clock_complete')) is not bool:
                raise ValueError('future support must be explicit boolean')
            expected_missing = sum(f < first or f > last or f in missing_images
                                   for f in range(g, g + 251))
            if (row['future_record_clock_complete'] != (expected_missing == 0)
                    or type(row.get('future_missing_image_count')) is not int
                    or row['future_missing_image_count'] != expected_missing):
                raise ValueError('future clock flag disagrees with full image support')
            if eligible and not np.array_equal(x[i, 0, :13], x[i, 1, :13]):
                raise ValueError('shared geometry differs between variants')
        if summary.get('candidate_count') != len(keys) or summary.get('input_eligible_count') != sum(r['input_eligible'] for r in rows):
            raise ValueError('candidate counts disagree')
        if summary.get('future_complete_count') != sum(r['future_record_clock_complete'] for r in rows):
            raise ValueError('future complete count disagrees with full ledger')
        by_half[half_key] = (deepcopy(rows), np.array(x, copy=True))
    if set(by_half) != expected:
        raise ValueError('missing planned fixture-half; partial cohort cannot be full validation')
    all_rows, arrays = [], []
    for fixture in planned:
        for half in (1, 2):
            rows, x = by_half[fixture, half]
            all_rows.extend(rows); arrays.append(x)
    X_all = np.concatenate(arrays, axis=0)
    keep = np.array([row['input_eligible'] for row in all_rows], dtype=bool)
    X = X_all[keep].copy(); X.setflags(write=False)
    return dict(X=X, row_keys=tuple(row['row_key'] for row in all_rows if row['input_eligible']),
                all_ledger=all_rows, n_candidates=len(all_rows),
                n_input_ineligible=int((~keep).sum()), n_input_eligible=int(keep.sum()),
                planned_fixtures=list(planned), verified_fixture_half_count=len(by_half),
                qualification='COMPLETE_CANDIDATE_GRID_ONLY_NOT_PHYSICAL_OR_EXTERNAL_VALIDITY')


def aligned_scoring_labels(cohort, full_labels):
    """Check all candidate labels, then project the exact prediction key set."""
    full_labels = list(full_labels)
    keys = [row['row_key'] for row in full_labels]
    expected = {row['row_key'] for row in cohort['all_ledger']}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('full candidate label keys must match complete ledger exactly')
    by_key = {row['row_key']: row for row in full_labels}
    result = []
    for row in cohort['all_ledger']:
        label = by_key[row['row_key']]
        if type(label.get('y')) is not int or label['y'] not in (-1, 0, 1):
            raise ValueError('invalid y')
        if not row['input_eligible'] and label['y'] != -1:
            raise ValueError('ineligible input cannot enter scoring')
        if not row['future_record_clock_complete'] and label['y'] != -1:
            raise ValueError('incomplete future clock cannot become a known label')
        if row['input_eligible']:
            result.append({'row_key': row['row_key'], 'y': label['y']})
    if tuple(r['row_key'] for r in result) != cohort['row_keys']:
        raise ValueError('eligible prediction/label key mismatch')
    return result


class ProductionNotReady(RuntimeError):
    pass


def production_readiness(plan):
    """Inspect metadata only. No raw input path is opened, even on failure.

    This preparatory version intentionally supplies no production executor.
    A true return documents fields, not verification of the receipt contents.
    """
    missing = []
    if plan.get('status') != 'FROZEN_SOURCE_SPECIFIC_EXTERNAL_PROTOCOL':
        missing.append('source-specific protocol not frozen')
    for name in ('second_half_qc_receipt', 'source_file_manifest', 'code_seal',
                 'prediction_freeze_then_label_read_protocol', 'exposure_boundary_receipt'):
        item = plan.get(name)
        if (not isinstance(item, dict) or not isinstance(item.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
            missing.append(name + ' missing')
    if plan.get('all_planned_fixture_halves_bound') is not True:
        missing.append('all planned fixture-halves not bound')
    if missing:
        raise ProductionNotReady('; '.join(missing))
    return {'metadata_fields_present': True,
            'production_execution_authorized_by_this_function': False,
            'receipt_contents_and_source_qualifications_still_require_verification': True}
