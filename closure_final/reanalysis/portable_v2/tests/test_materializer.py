import sys,json,tempfile,unittest,zipfile,hashlib,stat
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import materialize_evidence as M
import public_support as S
class Tests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.vol=self.root/'volumes';self.vol.mkdir();self.out=self.root/'restored';self.index=self.root/'index.json';self.aliases=[];self.blobs={}
 def tearDown(self):self.temp.cleanup()
 def fixture(self,extra=False,symlink=False):
  raw=b'full original bytes';h=hashlib.sha256(raw).hexdigest();name='part01.zip';zpath=self.vol/name
  with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_STORED) as z:
   info=zipfile.ZipInfo('sha256/'+h)
   if symlink:info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
   z.writestr(info,raw)
   if extra:z.writestr('unknown',b'x')
  self.aliases=[{'public_path':'closure_final/a/x.npz','sha256':h,'bytes':len(raw),'volume':1,'member_path':'sha256/'+h},{'public_path':'closure_final/b/y.npz','sha256':h,'bytes':len(raw),'volume':1,'member_path':'sha256/'+h}];self.data={'schema':'a17.closure.evidence.alias.blob.index.v1','volumes':[{'filename':name,'sha256':S.sha(zpath),'bytes':zpath.stat().st_size,'members':1}],'aliases':self.aliases};self.save()
 def save(self):self.index.write_bytes(S.canonical(self.data))
 def runcheck(self,write=False,prefix=None):return M.run(self.index,S.sha(self.index),self.vol,prefix or ['closure_final/a'],self.out,write)
 def test_dry_run_prefix_and_dedup(self):
  self.fixture();r=self.runcheck();self.assertEqual(r['selected_aliases'],1);self.assertFalse(self.out.exists());self.assertEqual(len(r['verified_volumes']),1)
 def test_materialize_exact_no_overwrite(self):
  self.fixture();self.runcheck(True,['closure_final/a','closure_final/b']);self.assertEqual((self.out/'closure_final/a/x.npz').read_bytes(),b'full original bytes');self.assertEqual((self.out/'closure_final/b/y.npz').read_bytes(),b'full original bytes')
  with self.assertRaises(ValueError):self.runcheck(True)
 def test_index_sha_stale(self):
  self.fixture();old=S.sha(self.index);self.index.write_text('{}')
  with self.assertRaises(ValueError):M.run(self.index,old,self.vol,['closure_final/a'])
 def test_volume_sha_drift(self):
  self.fixture();p=self.vol/'part01.zip';p.write_bytes(p.read_bytes()+b'drift')
  with self.assertRaises(ValueError):self.runcheck()
 def test_crc_failure_after_rebound_volume_sha(self):
  self.fixture();p=self.vol/'part01.zip';raw=p.read_bytes();at=raw.index(b'full original bytes');p.write_bytes(raw[:at]+b'X'+raw[at+1:]);self.data['volumes'][0]['sha256']=S.sha(p);self.save()
  with self.assertRaises((ValueError,zipfile.BadZipFile)):self.runcheck()
 def test_member_sha_wrong(self):
  self.fixture();h='a'*64;p=self.vol/'part01.zip'
  with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('sha256/'+h,b'full original bytes')
  for a in self.aliases:a.update(sha256=h,member_path='sha256/'+h)
  self.data['volumes'][0].update(sha256=S.sha(p),bytes=p.stat().st_size);self.save()
  with self.assertRaisesRegex(ValueError,'member SHA'):self.runcheck()
 def test_alias_traversal_absolute_windows_rejected(self):
  self.fixture()
  for alias in ['closure_final/../escape','/absolute/escape','closure_final/a//x','closure_final/a\\x','closure_final/a:escape']:
   self.aliases[0]['public_path']=alias;self.save()
   with self.assertRaises(ValueError):self.runcheck()
 def test_zip_symlink_and_unknown_member(self):
  self.fixture(symlink=True)
  with self.assertRaisesRegex(ValueError,'symlink'):self.runcheck()
  self.fixture(extra=True)
  with self.assertRaisesRegex(ValueError,'member set'):self.runcheck()
 def test_symlink_target_parent(self):
  self.fixture();real=self.root/'real';real.mkdir();link=self.root/'link';link.symlink_to(real,target_is_directory=True);self.out=link/'out'
  with self.assertRaises(ValueError):self.runcheck(True)
 def test_no_prefix_match_or_empty_selection(self):
  self.fixture()
  with self.assertRaises(ValueError):self.runcheck(prefix=['closure_final/notthere'])
  with self.assertRaises(ValueError):M.validate(self.index,S.sha(self.index),self.vol,[])
if __name__=='__main__':unittest.main()
