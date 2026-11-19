"""Build a create-only map from explicit SUCCESSFUL composition receipts."""
from pathlib import Path
import argparse,json
from compose_inputs import long,digest,child
p=argparse.ArgumentParser();p.add_argument('--archive-root',required=True);p.add_argument('--output',required=True);p.add_argument('--receipts',nargs='+',required=True);a=p.parse_args();root=long(a.archive_root);out=long(a.output);archives={}
if out.exists():raise FileExistsError('New map only')
for name in a.receipts:
 receipt=long(name);r=json.loads(receipt.read_text(encoding='utf8'))
 if r['status']!='PASS_COMPOSED_PUBLIC_CANDIDATE' or not r['all_final_members_verified'] or not r['original_archive_unchanged']:raise ValueError('Unsuccessful composition receipt')
 f=child(receipt.parent,r['candidate_filename'])
 if not f.is_relative_to(root) or digest(f)!=r['sha256']:raise ValueError('Candidate root/SHA mismatch')
 key=r['original_archive']
 if key in archives:raise ValueError('Duplicate original source archive')
 archives[key]={'path':f.relative_to(root).as_posix(),'sha256':r['sha256']}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x',encoding='utf8') as f:json.dump({'archives':archives},f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','archives':len(archives)}))
