"""Create prospective binding using JSON/source only; never open NPZ payloads."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'CCPV_direct_terminal_transport_20261112'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pin(p):
    p = Path(p).resolve()
    if p.suffix.lower() == '.npz':
        raise ValueError('No scientific payload before authorization')
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    p = read(OLD/'plan.json')
    f = read(OLD/'readout01/freeze.json')
    assert f['status'] == 'ALL_DIRECT_TRANSPORT_READOUTS_FROZEN_BEFORE_LABEL_ACCESS'
    assert f['plan_sha256'] == pin(OLD/'plan.json')['sha256']
    assert p['population']['N'] == 5344 and p['population']['K'] == 16
    assert len(p['heads']) == 30 and len(p['shards']) == 835
    native = ROOT.parent/'CCPV_native_neural_bridge_20261026/neural_bridge'
    sources = sorted(ROOT.glob('*.py'))+[ROOT/'prospective_contract.md']
    plan = dict(status='PREFIT_PROSPECTIVE_MATCHED_OBJECTIVE_SOURCE_ONLY', utc=datetime.now(timezone.utc).isoformat(),
        population=p['population'], heads=p['heads'], shards=p['shards'], labels=p['labels'],
        historical_cv_context=p['historical_cv_context'],
        old_readout=dict(path=str(OLD/'readout01/readouts.npz'), sha256=f['predictions_sha256']),
        metadata=p['metadata']+[pin(OLD/'plan.json'), pin(OLD/'readout01/freeze.json'),
            pin(ROOT.parent/'CCPV_direct_terminal_transport_20261112_root_seal.json')],
        source_evidence=p['source_evidence']+[pin(native/'head_models.py'), pin(native/'parent_heads.py')],
        sources=[pin(x) for x in sources],
        model=dict(raw=36, encoded=72, width=8, outputs=['Y10', 'Y2'], parameters=602,
                   initial='SAVED_INITIAL_PARAMETERS', scaler='EXACT_FROZEN_OLD_MEAN_TRAIN_FOLD', fresh_Adam=True),
        objective='SUM_Y10_Y2_WINDOW_NEGATIVE_LOG_MEAN16_PROBABILITY_PLUS_0.005_WEIGHT_L2',
        training=dict(heads=30, updates=200, fullbatch=True, lr=.003, betas=[.9,.999], eps=1e-8,
                      weight_decay=0., global_clip=10., dtype='float64', CPU_threads=1,
                      held_loss_access=False, no_warm_start=True, no_auto_retry=True),
        cells=['OldTrained_Mean','OldTrained_MC','MCTrained_Mean','MCTrained_MC'],
        co_primary={'A':['MCTrained_MC','OldTrained_MC'], 'B':['MCTrained_MC','OldTrained_Mean']},
        interaction='(MCTrained_MC-MCTrained_Mean)-(OldTrained_MC-OldTrained_Mean)',
        primary_metric='q10 equal-six-fixture log_loss',
        secondary=['Brier','absolute_fixture_bias','signed_bias','mean_probability','Y2','Early_gt_q','feature_OOD','MCSE'],
        limits=dict(wall_seconds=1200, output_bytes=512*1024**2),
        assertions=dict(old_readout_tolerance=1e-12, no_held_prediction_until_all30_frozen=True,
                        no_trajectory_forward=True, no_formal50=True, independent_validation=False),
        preparation_scientific_payload_reads=0)
    with (ROOT/'plan.json').open('x', encoding='utf-8') as out:
        json.dump(plan, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(pin(ROOT/'plan.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
