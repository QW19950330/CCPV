"""Raw PLAYER two-point features, future clock coverage, and original any-shot labels."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, time
from pathlib import Path
import common as c
import frozen_players as p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-root',required=True,type=Path)
    ap.add_argument('--candidate-grid',required=True,type=Path)
    ap.add_argument('--event-clocks',required=True,type=Path)
    ap.add_argument('--authorize-raw-xml',action='store_true')
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if not a.authorize_raw_xml: raise PermissionError('Explicit raw scan flag required')
    if 'prefix_grid' not in a.candidate_grid.name or 'event_clocks' not in a.event_clocks.name:
        raise ValueError('Use original artifact basenames for the exact original routing block')
    contract=c.read(c.HERE/'input_contract.json'); started=time.monotonic()
    c.checked_csv(a.candidate_grid,contract['candidate_grid'])
    c.checked_csv(a.event_clocks,contract['aggregate_outputs']['event_clocks.csv'])
    # Positions are hashed in the actual original streaming scan, avoiding a redundant read.
    sources=[c.resolve(a.raw_root,r,hash_now='positions' not in r['file']) for r in contract['raw_player_sources']]
    plan=dict(scope=dict(matches=contract['matches']),raw_sources=sources,
              cached_inputs=[dict(path=str(a.candidate_grid.resolve())),dict(path=str(a.event_clocks.resolve()))])
    out=c.output_dir(a.output,[a.raw_root]);c.identity('PLAYER_LABEL_LEDGER_RAW6',out)
    def guard():
        if (out/'PAUSE_REQUESTED').exists(): raise RuntimeError('User pause requested; no automatic restart')
        if time.monotonic()-started>1800: raise TimeoutError('1800 second fixed scan budget exceeded')
    summary=p.build_data(out,plan,guard)
    checks={}
    for name,pin in contract['player_outputs'].items():
        rr=c.checked_csv(out/name,pin);checks[name]=dict(rows=len(rr),exact_ordered_fields=True,**c.pin(out/name))
    arrays=c.check_arrays(out/'features.npz',contract['feature_arrays'])
    receipt=dict(status='PASS_EXACT_RAW_PLAYER_LABEL_LEDGER_AND_FEATURES',seconds=time.monotonic()-started,
        inputs=[c.pin(a.candidate_grid),c.pin(a.event_clocks)],summary=summary,outputs=checks,
        feature_arrays=arrays,feature_container=c.pin(out/'features.npz'),
        original_feature_container_sha256=contract['feature_file']['sha256'],
        feature_container_exact=c.sha(out/'features.npz')==contract['feature_file']['sha256'],
        new_training=False,new_independent_matches=0,physical_event_capture_validated=False,
        limits=['Original two-point support, not shared-six-history support.',
                'Future PLAYER timestamps only establish recorded-clock coverage, not physical event ascertainment.',
                'Candidate and event clock inputs require their own raw reconstruction receipts.'])
    c.put(out/'completion.json',receipt)
    print(receipt['status'],{k:v for k,v in summary.items() if k!='source_records'},flush=True)

if __name__=='__main__': main()
