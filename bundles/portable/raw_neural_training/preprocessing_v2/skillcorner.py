"""Rebuild the retained full cache and its fixed development selection."""
from pathlib import Path
import numpy as np
from common import dump, save_npz, sha, ah
import skillcorner_reader as reader

def prepare(raw: Path, contract: dict, out: Path):
    manifest=contract['skillcorner'];split=manifest['split']
    ids=manifest['matches'];rows={x['path']:x for x in manifest['files']}
    records=[];qc=[]
    for mid in ids:
        meta=reader.read_json(raw/str(mid)/f'{mid}_match.json')
        starts={int(p['period']):int(p['start_frame']) for p in meta['match_periods']}
        reader.prepare_one(raw, mid, rows, starts, records, qc)
    metadata=dict(source=manifest['source'],commit=manifest['commit'],match_qc=qc,
        tracking_hashes={str(x['match_id']):x['tracking_sha256'] for x in qc},
        metadata_hashes={str(x['match_id']):x['match_metadata_sha256'] for x in qc})
    arrays,meta=reader.assemble(records,ids,split,metadata)
    save_npz(out/'skillcorner_10hz.npz',**arrays)
    # This is a present-day reconstruction. Preserve historical source metadata,
    # but never describe already-exposed data as newly held out.
    dump(out/'skillcorner_historical_metadata.json',meta)
    del arrays,records
    select_development(out/'skillcorner_10hz.npz',split['groups'],out)
    dump(out/'cache_manifest.json',dict(status='RAW_CACHE_REBUILT',matches=ids,
        match_qc=qc,cache_sha256=sha(out/'skillcorner_10hz.npz'),
        model_forward_calls=0,optimizer_updates=0,
        scope='Raw reconstruction only; no test metrics, fitting, selection changes or new generalization claim.'))

def select_development(src: Path, split: dict, output_dir: Path):
    with np.load(src,allow_pickle=False) as archive:
        ids = archive['match_ids'].astype(str)
        selected, counts = [], {}
        for name, number in (('train',64),('validation',64)):
            group_rows = archive[name+'_indices']
            out = []
            for match in sorted(split[name]):
                eligible = np.sort(group_rows[ids[group_rows]==match])
                if len(eligible)<number:
                    raise ValueError('Insufficient locked rows: '+match)
                chosen = eligible[np.linspace(0,len(eligible)-1,number,dtype=np.int64)]
                if len(np.unique(chosen))!=number:
                    raise AssertionError('Duplicate source rows')
                out.extend(chosen.tolist())
            selected.extend(out)
            counts[name] = len(out)
        ix = np.array(selected,dtype=np.int64)
        if len(ix)!=1024 or len(np.unique(ix))!=1024:
            raise AssertionError('Expected 768 training plus 256 validation rows')
        keys = ('states','valid','position_valid','match_ids','clip_ids','frame_times_s',
                'source_frame_ids','source_period','entity_ids')
        data = {key:archive[key][ix] for key in keys if key in archive.files}
        data['teams'] = archive['teams']
    data['source_indices'] = ix
    data['train_indices'] = np.arange(768,dtype=np.int64)
    data['validation_indices'] = np.arange(768,1024,dtype=np.int64)
    if set(data['match_ids'][:768])&set(data['match_ids'][768:]):
        raise AssertionError('Match overlap')
    if set(data['match_ids'])&set(split['test']):
        raise AssertionError('Test match was selected')
    for key in ('valid','position_valid'):
        if data[key].dtype!=np.bool_:
            raise ValueError('Masks must be Boolean')
    if np.any(data['valid']&~data['position_valid']):
        raise AssertionError('Full-state validity cannot exceed position validity')
    save_npz(output_dir/'development.npz',**data)
    dump(output_dir/'selection_manifest.json',dict(status='FIXED_SELECTION_REBUILT',counts=counts,
        data_sha256=sha(output_dir/'development.npz'),array_hashes={k:ah(v) for k,v in data.items()},
        selected_test_matches=False,model_forward_calls=0))
