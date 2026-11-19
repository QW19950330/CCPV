"""Recover player detected-only future xy AFTER an explicit prediction freeze. No fitting."""
from pathlib import Path
import argparse,collections,hashlib,json,re
import numpy as np
from analyze_responses import sha,read,dump,frozen_run,MATCHES
FRAME=re.compile(rb'^\s*\{\s*"frame"\s*:\s*(\d+)')

def detected_players(frame,roster,L,W):
 points={};seen=set()
 for p in frame.get('player_data') or []:
  pid=str(int(p['player_id']))
  if pid in seen:raise ValueError('Duplicate player identity in selected raw frame')
  seen.add(pid)
  if pid not in roster:raise ValueError('Player absent from pinned roster')
  if p.get('is_detected') is not True:continue
  if p.get('x') is None or p.get('y') is None:continue
  xy=np.asarray([p['x'],p['y']],dtype=float)
  if not np.isfinite(xy).all():continue
  points[pid]=np.asarray((float(xy[0])*105./L,float(xy[1])*68./W),np.float32)
 return points

def recover_file(path,expected_sha,metadata,rows,entity_ids,period,frame_ids):
 L=float(metadata['pitch_length']);W=float(metadata['pitch_width'])
 if not np.isfinite([L,W]).all() or min(L,W)<=0:raise ValueError('Invalid source pitch')
 roster={str(int(p['id'])) for p in metadata['players'] if p.get('id') is not None and p.get('team_id') is not None}
 lookup=collections.defaultdict(list)
 for local,global_row in enumerate(rows):
  for t,fr in enumerate(frame_ids[global_row]):lookup[(int(period[global_row]),int(fr))].append((local,t))
 positions=np.full((len(rows),30,23,2),np.nan,np.float32);valid=np.zeros((len(rows),30,23),bool)
 frame_found=np.zeros((len(rows),30),bool);seen_frames=set();digest=hashlib.sha256();lines=selected=0
 wanted_frames={k[1] for k in lookup}
 with Path(path).open('rb') as f:
  for raw in f:
   digest.update(raw);lines+=1;header=FRAME.match(raw)
   if header is None:raise ValueError('Unexpected raw frame header')
   if int(header.group(1)) not in wanted_frames:continue
   d=json.loads(raw);per=d.get('period')
   if per not in (1,2):continue
   key=(int(per),int(d['frame']))
   if key not in lookup:continue
   if key in seen_frames:raise ValueError('Duplicate selected period/frame')
   seen_frames.add(key);selected+=1;points=detected_players(d,roster,L,W)
   for local,t in lookup[key]:
    frame_found[local,t]=True;ids=entity_ids[rows[local]]
    for slot,pid in enumerate(ids[:22]):
     if str(pid) in points:positions[local,t,slot]=points[str(pid)];valid[local,t,slot]=True
 if digest.hexdigest()!=expected_sha:raise ValueError('Raw tracking SHA changed')
 if not frame_found.all():raise ValueError('Requested continuous cache frame missing: technical identity/time failure, not missing player truth')
 return positions,valid,frame_found,dict(lines_streamed=lines,selected_frames=selected,raw_sha256=digest.hexdigest(),only_requested_frame_coordinates_deserialized=True,pitch_length=L,pitch_width=W)

def run(base,run_dir,freeze_sha,source_spec,spec_sha,out):
 frozen_run(run_dir,freeze_sha) # Must precede any prefix/future raw access.
 base=Path(base);source_spec=Path(source_spec)
 if sha(source_spec)!=spec_sha:raise ValueError('Explicit frozen source-spec SHA required')
 spec=read(source_spec)
 if [r['match_id'] for r in spec['matches']]!=list(MATCHES):raise ValueError('Only fixed four matches')
 for k in ('download_manifest','original_parser','prepared_manifest'):
  if sha(base/spec[k])!=spec[k+'_sha256']:raise ValueError('Pinned source metadata/code changed: '+k)
 pm=read(base/spec['prepared_manifest']);pp=base/pm['path']
 if sha(pp)!=pm['sha256']:raise ValueError('Prepared prefix changed')
 with np.load(pp,allow_pickle=False) as z:a={k:z[k] for k in ('source_indices','match_ids','entity_ids','source_period','source_frame_ids','history_states','history_valid','history_position_valid')}
 n=len(a['source_indices'])
 if n!=4992 or not np.all(np.diff(a['source_frame_ids'],axis=1)==1):raise ValueError('Unexpected prefix/time contract')
 future=a['source_frame_ids'][:,-1,None]+np.arange(1,21)[None,:];frames=np.concatenate((a['source_frame_ids'],future),axis=1)
 positions=np.full((n,20,23,2),np.nan,np.float32);valid=np.zeros((n,20,23),bool);found=np.zeros((n,20),bool);receipts=[]
 for source in spec['matches']:
  bykind={r['kind']:r for r in source['files']};md=bykind['metadata'];tr=bykind['tracking']
  if sha(base/md['path'])!=md['sha256']:raise ValueError('Match metadata SHA changed')
  meta=read(base/md['path']);ix=np.flatnonzero(a['match_ids']==source['match_id'])
  pos,vm,ff,rec=recover_file(base/tr['path'],tr['sha256'],meta,ix,a['entity_ids'],a['source_period'],frames)
  if not np.array_equal(vm[:,:10,:22],a['history_position_valid'][ix,:,:22]):raise ValueError('Recovered history detected mask differs from frozen cache')
  common=a['history_valid'][ix,:,:22]
  if not np.array_equal(pos[:,:10,:22][common],a['history_states'][ix,:,:22,:2][common]):raise ValueError('Recovered valid history xy differs from frozen cache')
  positions[ix]=pos[:,10:];valid[ix]=vm[:,10:];found[ix]=ff[:,10:]
  receipts.append(dict(match_id=source['match_id'],windows=len(ix),future_frames_requested=len(ix)*20,future_frames_found=int(ff[:,10:].sum()),detected_player_points=int(vm[:,10:,:22].sum()),history_mask_and_valid_xy_exact=True,**rec))
 out=Path(out);out.mkdir(parents=True,exist_ok=False);target=out/'detected_player_truth.npz'
 with target.open('xb') as f:np.savez_compressed(f,positions=positions,position_valid=valid,frame_found=found,source_indices=a['source_indices'],match_ids=a['match_ids'],source_period=a['source_period'],future_frame_ids=future,entity_ids=a['entity_ids'])
 dump(out/'completion.json',dict(status='COMPLETE_DETECTED_ONLY_PLAYER_TRUTH_AFTER_PREDICTION_FREEZE',path=target.name,sha256=sha(target),prediction_completion_sha256=freeze_sha,source_spec_sha256=spec_sha,prepared_prefix_sha256=pm['sha256'],prepared_prefix_path=str(pp.resolve()),code_sha256=sha(__file__),matches=receipts,ball_scored=False,raw_event_files_read=0,model_calls=0,training_updates=0,missing_values='NaN with false mask, not zero truth',units='provider xy scaled105/pitch_length and68/pitch_width; standardized metre-equivalents'))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--prediction-freeze-sha256',required=True);p.add_argument('--source-spec',required=True);p.add_argument('--source-spec-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.base,a.run_dir,a.prediction_freeze_sha256,a.source_spec,a.source_spec_sha256,a.out)
