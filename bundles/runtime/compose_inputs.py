"""Create-only SHA-verified input composition. Original archives are LOCAL composition sources, never upload them.
No training, no inference, no mutation of source archives or numerical payloads.
"""
from pathlib import Path
import argparse,json,hashlib,os,zipfile,shutil

def long(p):
 p=Path(p).resolve()
 return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def child(root,relative):
 r=Path(relative)
 if r.is_absolute() or '..' in r.parts or ':' in str(r):raise ValueError('Unsafe relative member')
 p=(root/r).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('Member escapes root')
 return p

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def compose(candidate,archive_root,output,components,local_exact_source_root=None,cache_root=None,public_archive_map=None):
 candidate=long(candidate);archive_root=long(archive_root);output=long(output)
 if output.exists():raise FileExistsError('Create-only output exists')
 if output.is_relative_to(candidate) or candidate.is_relative_to(output):raise ValueError('Separate output required')
 cat=json.loads((candidate/'runtime_catalog.json').read_text(encoding='utf8'));chosen=components or list(cat['components'])
 amap=json.loads(long(public_archive_map).read_text(encoding='utf8'))['archives'] if public_archive_map else {};dirty={x['original_archive'] for x in cat['metadata_derivatives']};verified_archives=set()
 for k in chosen:
  v=cat['components'][k]
  if v['missing'] or v['restricted_original_inputs']:raise ValueError('Incomplete component '+k)
 output.mkdir(parents=True);zips={};seen={};rows=[];links=0;copied=0;cache=long(cache_root) if cache_root else output/"_local_content_store"
 try:
  for k in chosen:
   v=cat['components'][k];root=output/k/v['input_dir'];root.mkdir(parents=True)
   targets=set()
   for j in v['jobs']:
    target=child(root,j['target']);key=target.as_posix()
    if key in targets:raise ValueError('Duplicate target '+key)
    targets.add(key);target.parent.mkdir(parents=True,exist_ok=True);data=None;source=None
    if 'public_file' in j:source=child(candidate,j['public_file'])
    elif local_exact_source_root is not None and j.get('source_relative'):
     p=child(long(local_exact_source_root),j['source_relative'])
     if p.is_file():
      stamp=(str(p),p.stat().st_size,p.stat().st_mtime_ns);h=seen.get(stamp)
      if h is None:h=digest(p);seen[stamp]=h
      if h==j['sha256'] and p.stat().st_size==j['bytes']:source=p
    if source is not None:
     stamp=(str(source),source.stat().st_size,source.stat().st_mtime_ns);h=seen.get(stamp)
     if h is None:h=digest(source);seen[stamp]=h
     if h!=j['sha256'] or source.stat().st_size!=j['bytes']:raise ValueError('Source SHA/size mismatch '+j['target'])
     if local_exact_source_root is not None:
      try:os.link(source,target)
      except OSError:
       cache.mkdir(exist_ok=True);cached=cache/j['sha256']
       if cached.exists() and digest(cached)!=j['sha256']:raise ValueError('Existing content cache SHA mismatch')
       if not cached.exists():
        with source.open('rb') as src,cached.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
        if digest(cached)!=j['sha256']:raise ValueError('Copied source changed')
        copied+=j['bytes']
       os.link(cached,target)
      links+=1
     else:
      with target.open('xb') as f:f.write(source.read_bytes())
      copied+=j['bytes']
    else:
     archive=j['archive']
     if archive in dirty and archive not in amap and local_exact_source_root is None:raise ValueError('Public composition requires clean archive map; do not upload/use dirty source ZIP')
     mapped=amap.get(archive);zp=child(archive_root,mapped['path'] if mapped else archive)
     if mapped and str(zp) not in verified_archives:
      if digest(zp)!=mapped['sha256']:raise ValueError('Clean public archive SHA mismatch')
      verified_archives.add(str(zp))
     if str(zp) not in zips:zips[str(zp)]=zipfile.ZipFile(zp)
     z=zips[str(zp)];info=z.getinfo(j['member'])
     if info.file_size!=j['bytes']:raise ValueError('ZIP member size mismatch')
     h=hashlib.sha256()
     with z.open(info) as src,target.open('xb') as dst:
      for b in iter(lambda:src.read(1<<20),b''):h.update(b);dst.write(b)
     if h.hexdigest()!=j['sha256']:raise ValueError('ZIP member SHA mismatch')
     copied+=j['bytes']
    rows.append({'component':k,'target':j['target'],'sha256':j['sha256'],'bytes':j['bytes']})
   print('MATERIALIZED',k,len(v['jobs']),flush=True)
  result={'status':'PASS_CREATE_ONLY_EXACT_INPUT_COMPOSITION','files':rows,'hardlinks_for_local_readonly_validation':links,'copied_bytes':copied,'catalog_sha256':digest(candidate/'runtime_catalog.json'),'training':0,'predictions':0,'original_archives_modified':False}
  (output/'composition_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result
 finally:
  for z in zips.values():z.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--archive-root',required=True);p.add_argument('--output',required=True);p.add_argument('--components',nargs='*');p.add_argument('--local-exact-source-root',help='Optional local validation only: hardlink exact-SHA original payloads, never publish this tree')
 p.add_argument('--cache-root');p.add_argument('--public-archive-map');a=p.parse_args();r=compose(a.candidate,a.archive_root,a.output,a.components,a.local_exact_source_root,a.cache_root,a.public_archive_map);print(json.dumps({k:v for k,v in r.items() if k!='files'}))
