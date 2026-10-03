"""NEW_PUBLIC_REANALYSIS: verify SHA blobs and optionally restore selected aliases into a new reader directory."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,stat,zipfile
import public_support as S

def relative(value):
 S.check(isinstance(value,str) and value.startswith('closure_final/') and not value.endswith('/') and '\\' not in value and ':' not in value and '//' not in value and not any(ord(c)<32 for c in value),'unsafe relative alias');p=PurePosixPath(value);S.check(not p.is_absolute() and all(x not in ['.','..',''] for x in value.split('/')),'alias traversal');return p
def digest(value):S.check(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'invalid SHA');return value
def ordinary(path):
 p=Path(path);S.check(p.is_file() and not p.is_symlink() and not any(q.is_symlink() for q in p.parents),'missing or symlink input');return p

def validate(index_path,index_sha,volume_dir,prefixes):
 index_path=ordinary(index_path);S.check(S.sha(index_path)==digest(index_sha),'index SHA differs');index=S.read(index_path);S.check(index.get('schema')=='a17.closure.evidence.alias.blob.index.v1','wrong asset index schema');S.check(prefixes,'explicit nonempty prefix selection required')
 prefixes=[str(relative(p.rstrip('/'))) for p in prefixes];aliases={};groups={};volumes={}
 for i,v in enumerate(index['volumes'],1):
  name=v['filename'];S.check(name==Path(name).name and name not in ['.','..'] and ':' not in name and '\\' not in name and name.endswith('.zip') and not any(ord(c)<32 for c in name),'unsafe volume filename');S.check(name not in {r['filename'] for r in volumes.values()},'duplicate volume filename');digest(v['sha256']);volumes[i]=v
 for row in index['aliases']:
  alias=str(relative(row['public_path']));h=digest(row['sha256']);S.check(alias not in aliases,'duplicate alias');S.check(isinstance(row['volume'],int) and not isinstance(row['volume'],bool) and row['volume'] in volumes,'unknown volume');S.check(isinstance(row['bytes'],int) and not isinstance(row['bytes'],bool) and row['bytes']>=0,'invalid byte size');S.check(row['member_path']=='sha256/'+h,'noncanonical blob member');aliases[alias]=row;key=(row['volume'],row['member_path']);old=groups.get(key);S.check(old is None or old['sha256']==h and old['bytes']==row['bytes'],'blob alias conflict');groups[key]=row
 seen_hash={}
 for row in aliases.values():
  old=seen_hash.get(row['sha256']);S.check(old is None or old==(row['volume'],row['member_path']),'duplicate canonical SHA blob across volumes');seen_hash[row['sha256']]=(row['volume'],row['member_path'])
  parts=PurePosixPath(row['public_path']).parts;S.check(not any('/'.join(parts[:i]) in aliases for i in range(1,len(parts))),'file alias used as parent directory')
 selected=[r for name,r in aliases.items() if any(name==p or name.startswith(p+'/') for p in prefixes)];S.check(selected,'selected prefixes match no aliases');used_volumes={r['volume'] for r in selected};verified=[]
 for i in sorted(used_volumes):
  expected={member:row for (v,member),row in groups.items() if v==i};v=volumes[i];p=ordinary(Path(volume_dir)/v['filename']);S.check(S.sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],'volume SHA/size differs')
  with zipfile.ZipFile(p) as z:
   S.check(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(expected),'volume exact member set differs');S.check(z.testzip() is None,'volume CRC failure');S.check(v['members']==len(expected),'volume member count differs')
   for member,row in expected.items():
    info=z.getinfo(member);mode=info.external_attr>>16;S.check(not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in [0,stat.S_IFREG] and not info.is_dir(),'ZIP symlink/special member');S.check(info.compress_type==zipfile.ZIP_STORED and info.file_size==row['bytes'] and not info.flag_bits&1,'ZIP storage/size/encryption differs');h=hashlib.sha256()
    with z.open(member) as f:
     for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    S.check(h.hexdigest()==row['sha256'],'member SHA differs')
  verified.append({'volume':i,'filename':v['filename'],'sha256':v['sha256'],'CRC':'PASS','all_member_SHA':'PASS'})
 return index,selected,verified

def run(index_path,index_sha,volume_dir,prefixes,out=None,materialize=False):
 index,selected,verified=validate(index_path,index_sha,volume_dir,prefixes);result={'status':'VERIFIED_DRY_RUN','identity':'NEW_PUBLIC_REANALYSIS_NOT_HISTORICAL_RUNTIME','index_sha256':index_sha,'selected_aliases':len(selected),'verified_volumes':verified,'materialized':False}
 if not materialize:return result
 S.check(out,'new output directory required');dest=Path(out).absolute();S.check(not dest.exists() and not dest.is_symlink() and dest.parent.is_dir() and not any(q.is_symlink() for q in dest.parents),'existing/symlink/missing parent output rejected');dest.mkdir(exist_ok=False)
 for row in selected:
  p=dest/relative(row['public_path']);p.parent.mkdir(parents=True,exist_ok=True);S.check(not any(q.is_symlink() for q in p.parents),'symlink output ancestor');volume=index['volumes'][row['volume']-1];source=ordinary(Path(volume_dir)/volume['filename']);S.check(S.sha(source)==volume['sha256'],'volume changed after verification')
  with zipfile.ZipFile(source) as z,z.open(row['member_path']) as src,p.open('xb') as target:
   for chunk in iter(lambda:src.read(1024*1024),b''):target.write(chunk)
  S.check(S.sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],'materialized alias bytes differ')
 result.update(status='MATERIALIZED_NEW_READER_DIRECTORY',materialized=True);(dest/'MATERIALIZATION_RECEIPT.json').write_bytes(S.canonical(result));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--index-sha256',required=True);p.add_argument('--volumes',required=True);p.add_argument('--prefix',action='append',required=True);p.add_argument('--materialize',action='store_true');p.add_argument('--out');a=p.parse_args();print(json.dumps(run(a.index,a.index_sha256,a.volumes,a.prefix,a.out,a.materialize)))
if __name__=='__main__':main()
