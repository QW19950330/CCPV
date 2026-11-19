"""Rebuild Table4 input from pinned development arrays and raw-derived truth."""
import argparse,traceback
from pathlib import Path
import numpy as np
import io_core as io
DEV=('states','valid','position_valid','teams','entity_ids','match_ids','source_indices','train_indices','validation_indices','source_frame_ids','frame_times_s','source_period')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--development',required=True,type=Path);ap.add_argument('--truth',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    here=Path(__file__).resolve().parent;c=io.J(here/'contract.json');b=c['prepare']
    dev=args.development.resolve(strict=True);truth=args.truth.resolve(strict=True);out=args.output.resolve()
    for p,k in ((dev,'development'),(truth,'truth')):
        if io.sha(p)!=b[k]:raise ValueError('Raw-derived input differs: '+k)
    for root in (here,dev.parent,truth.parent):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Output overlaps input/code tree')
    if not out.parent.is_dir():raise FileNotFoundError('Output parent must already exist')
    out.mkdir(exist_ok=False);io.P=out
    try:
        with np.load(dev,allow_pickle=False) as z:d={k:z[k] for k in DEV}
        with np.load(truth,allow_pickle=False) as z:t={k:z[k] for k in z.files}
        for k in ('position_valid','source_indices','match_ids','source_frame_ids','entity_ids'):
            if not np.array_equal(d[k],t[k]):raise ValueError('Truth alignment differs: '+k)
        if d['states'].shape!=(1024,30,23,4) or d['states'].dtype!=np.float32 or not np.array_equal(d['train_indices'],np.arange(768)) or not np.array_equal(d['validation_indices'],np.arange(768,1024)):raise ValueError('Fixed split differs')
        if not np.isfinite(d['states'][d['valid']]).all() or not np.isfinite(t['positions'][d['position_valid']]).all() or np.isfinite(t['positions'][~d['position_valid']]).any() or np.any(d['valid']&~d['position_valid']):raise ValueError('Masked truth contract')
        probe=np.asarray([r['scene_index'] for r in b['probe24']],np.int64);val=np.asarray([r['scene_index'] for r in b['validation32']],np.int64)
        for ix,rows in ((probe,b['probe24']),(val,b['validation32'])):
            for w,r in zip(ix,rows):
                if int(d['source_indices'][w])!=r['source_index'] or str(d['match_ids'][w])!=r['match_id']:raise ValueError('Fixed selection differs')
        io.save(out/'input.npz',**d,positions=t['positions'],split=np.where(np.arange(1024)<768,'TRAIN','validation'),pitch_L=np.asarray([b['pitch'][str(m)]['L'] for m in d['match_ids']]),pitch_W=np.asarray([b['pitch'][str(m)]['W'] for m in d['match_ids']]),probe24=probe,validation32=val)
        if io.sha(out/'input.npz')!=c['data']['sha256']:raise AssertionError('Full reconstructed input differs from pinned original')
        io.put(out/'receipt.json',dict(status='EXACT_REBUILT_TABLE4_INPUT',output=io.record(out/'input.npz'),
            development=io.record(dev),truth=io.record(truth),selection='Fixed existing TRAIN24 and validation32; no reselection',
            validation_noise_read=False,models_loaded=0))
        print('EXACT_REBUILT_TABLE4_INPUT',flush=True)
    except BaseException:
        io.put(out/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
