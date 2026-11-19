"""Replace declared metadata members into a NEW public candidate ZIP.
Never upload the immutable source ZIPs. No license is granted by this composer.
"""
from pathlib import Path
import argparse,json,hashlib,zipfile,os,shutil,re
from compose_inputs import long,child,digest
PAT=re.compile(r'(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/(?:home|Users)/[^/\s]+/)|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY')

def plan(candidate,archive_root,archive):
 candidate=long(candidate);archive_root=long(archive_root);cat=json.loads((candidate/'runtime_catalog.json').read_text(encoding='utf8'));rows=[r for r in cat['metadata_derivatives']+cat.get('additional_text_derivatives',[]) if r['original_archive']==archive]
 if not rows:raise ValueError('Archive outside exact65 metadata scope')
 src=child(archive_root,archive);expected={r['original_archive_sha256_from_existing_receipt'] for r in rows}
 if len(expected)!=1 or digest(src)!=next(iter(expected)):raise ValueError('Original archive SHA mismatch')
 replace={};members=[]
 with zipfile.ZipFile(src) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate source ZIP member')
  for name in names:child(candidate,name)
  for r in rows:
   old=z.read(r['original_member']);new=child(candidate,r['runtime_public_id']).read_bytes()
   if hashlib.sha256(old).hexdigest()!=r['original_member_sha256']:raise ValueError('Original metadata identity mismatch')
   if len(new)!=r['runtime_bytes'] or hashlib.sha256(new).hexdigest()!=r['runtime_sha256']:raise ValueError('Public derivative SHA mismatch')
   if PAT.search(new.decode('utf8')):raise ValueError('Private/token pattern in replacement')
   replace[r['original_member']]=new
  size=sum(i.file_size for i in z.infolist())
 return src,replace,size,rows

def compose(candidate,archive_root,archive,output=None):
 src,replace,size,rows=plan(candidate,archive_root,archive)
 if output is None:return {'status':'PASS_EXACT_REPLACEMENT_PLAN','original_archive':archive,'replacement_members':len(replace),'uncompressed_upper_bound_bytes':size,'source_sha256':digest(src),'written':False}
 requested=long(output)
 if requested.exists():raise FileExistsError('Create-only public output already exists')
 if requested==src:raise ValueError('Never mutate source ZIP')
 output=requested.with_name(requested.name+'.BUILDING_LOCAL_ONLY')
 if output.exists():raise FileExistsError('Prior incomplete build preserved')
 output.parent.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(output.parent).free<size+64*1024*1024:raise OSError('Insufficient space for bounded new archive')
 records=[]
 with zipfile.ZipFile(src) as old,zipfile.ZipFile(output,'x',allowZip64=True) as new:
  for info in old.infolist():
   if info.is_dir():continue
   name=info.filename;child(output.parent,name)
   if name=='PUBLIC_CONTENT_MANIFEST.json':raise ValueError('Reserved composition manifest name exists')
   zi=zipfile.ZipInfo(name,date_time=(2026,11,19,0,0,0));zi.external_attr=0o100644<<16;zi.compress_type=zipfile.ZIP_DEFLATED if Path(name).suffix.lower() in ('.csv','.json','.md','.txt','.py') else zipfile.ZIP_STORED
   h=hashlib.sha256();n=0
   if name in replace:
    data=replace[name];new.writestr(zi,data);h.update(data);n=len(data)
   else:
    if Path(name).suffix.lower() in ('.csv','.json','.md','.txt','.py'):
     data=old.read(info)
     if PAT.search(data.decode('utf8','replace')):raise ValueError('Unlisted private/token text member: '+name)
     new.writestr(zi,data);h.update(data);n=len(data)
    else:
     with old.open(info) as a,new.open(zi,'w',force_zip64=True) as b:
      for chunk in iter(lambda:a.read(1<<20),b''):b.write(chunk);h.update(chunk);n+=len(chunk)
   records.append({'member':name,'sha256':h.hexdigest(),'bytes':n,'metadata_replaced':name in replace})
  notice={'status':'PUBLIC_CANDIDATE_NOT_PUBLISHED_LICENSE_PENDING','historical_source_archive':archive,'historical_source_sha256':rows[0]['original_archive_sha256_from_existing_receipt'],'original_manifest_semantics':'Pre-existing manifest remains historical provenance; THIS manifest is authoritative for current member identities.','files':records,'replacement_map':[{'member':r['original_member'],'old_sha256':r['original_member_sha256'],'new_sha256':r['runtime_sha256']} for r in rows],'numerical_payloads_changed':0,'provider_notices_preserved':True,'original_archive_must_not_be_uploaded':True}
  new.writestr('PUBLIC_CONTENT_MANIFEST.json',json.dumps(notice,ensure_ascii=False,indent=2)+'\n')
 # Actual final compressed stream CRC and every current member SHA, no blind write success.
 with zipfile.ZipFile(output) as z:
  for r in records:
   h=hashlib.sha256();n=0
   with z.open(r['member']) as f:
    for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b)
   if h.hexdigest()!=r['sha256'] or n!=r['bytes']:raise ValueError('Final archive member verification failed')
 if digest(src)!=rows[0]['original_archive_sha256_from_existing_receipt']:raise ValueError('Original source changed during composition')
 os.link(output,requested);output.unlink();output=requested
 result={'status':'PASS_COMPOSED_PUBLIC_CANDIDATE','original_archive':archive,'candidate_filename':output.name,'sha256':digest(output),'bytes':output.stat().st_size,'files':len(records)+1,'replaced_members':len(replace),'all_final_members_verified':True,'original_archive_unchanged':digest(src)==rows[0]['original_archive_sha256_from_existing_receipt'],'training':0,'published':False}
 output.with_suffix(output.suffix+'.receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--archive-root',required=True);p.add_argument('--archive',required=True);p.add_argument('--output',help='Omit for SHA-pinned verify-plan only, no numerical copy');a=p.parse_args();print(json.dumps(compose(a.candidate,a.archive_root,a.archive,a.output),ensure_ascii=False))
