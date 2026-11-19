"""Reassemble local frozen inputs from an authorized source tree and metadata.

This copies specified cached inputs/references; it does not derive them from raw
tracking data and does not download or distribute the source data.
"""
from pathlib import Path
import argparse,hashlib,json,shutil

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.source_root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError('Create-only input destination must not exist')
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    controls={'portable_binding.json':a.binding,'reference.json':a.references}
    todo=[]
    for r in manifest['files']:
        rel=r.get('original_workspace_relative')
        src=(root/rel).resolve() if rel else controls[r['path']].resolve()
        if rel and not src.is_relative_to(root):raise ValueError('Original source path escape')
        dst=(out/r['path']).resolve()
        if not dst.is_relative_to(out):raise ValueError('Input destination escape')
        if not src.is_file() or sha(src)!=r['sha256']:raise ValueError('Missing/changed authorized source: '+str(rel or r['path']))
        todo.append((src,dst,r['sha256']))
    out.mkdir(parents=True,exist_ok=False)
    for src,dst,digest in todo:
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g)
        if sha(dst)!=digest:raise ValueError('Copy identity failure')
    with a.manifest.open('rb') as f,(out/'manifest.json').open('xb') as g:shutil.copyfileobj(f,g)
    print('FROZEN_LOCAL_INPUT_BUNDLE_REASSEMBLED',len(todo),flush=True)
if __name__=='__main__':main()
