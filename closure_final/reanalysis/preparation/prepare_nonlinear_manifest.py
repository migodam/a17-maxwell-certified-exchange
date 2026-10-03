"""NEW_PUBLIC_REANALYSIS: split saved audited curves and bind explicit owned aliases. Default dry-run; no array loading."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,math
OBJECTS=(2002,2005,2009,2016);POLICIES=('receiver_only','verified_exchange')
REFERENCES={'collect_saved_nonlinear_v2.py':'f09fc95f40c7fb557ca43165d8a289efeaba3cd85a8c81f5f752b27a04f5aec4','build_figures.py':'0dcab78ada5aab167ccd8b67440232dfe1367161da28b7b31b0c017cdaf447b8'}
def check(v,m):
 if not v:raise ValueError(m)
def canonical(d):return (json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alias(s):
 check(isinstance(s,str) and s.startswith('closure_final/') and not any(c in s for c in '\\:') and not '//' in s and not s.endswith('/') and not any(ord(c)<32 for c in s) and all(p not in ('.','..') for p in s.split('/')),'unsafe alias');return s
def digest(s):check(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s),'bad SHA');return s
def prepare(spec,curves,curves_sha):
 check(spec.get('schema')=='a17.public.saved.nonlinear.alias.spec.v1','explicit alias spec required');digest(curves_sha)
 cases=spec['cases'];check(len(cases)==8 and {(c['object_id'],c['policy']) for c in cases}=={(o,p) for o in OBJECTS for p in POLICIES},'all-eight case scope required');check(isinstance(curves,list),'merged audited curve rows required');files={alias(k):digest(v) for k,v in spec['files'].items()};products={};outcases=[]
 for c in cases:
  oid,policy=c['object_id'],c['policy'];status='COMPLETED' if oid!=2016 else 'NOT_RUN' if policy=='receiver_only' else 'FAILED_PREFIX';n=18 if status=='COMPLETED' else 17 if status=='FAILED_PREFIX' else 0
  check(c['status']==status and c['saved_updates']==n,'outcome promotion/scope differs');out={k:c[k] for k in ('object_id','policy','status','saved_updates')};rows=[r for r in curves if (r.get('object_id'),r.get('policy'))==(oid,policy)];check(len(rows)==n,'audited curve count differs');previous=None
  for i,r in enumerate(rows):
   check(r.get('scope')==('COMPLETED_ENDPOINT' if status=='COMPLETED' else 'FAILED_PREFIX_ONLY'),'curve endpoint scope differs');check(r['iteration']==i,'curve iteration order differs')
   for k in ('objective_before','objective_after'):check(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>=0,'invalid saved objective')
   if previous is not None:check(math.isclose(r['objective_before'],previous,rel_tol=2e-12,abs_tol=2e-13),'saved curve discontinuity')
   previous=r['objective_after']
  if n:
   for k in ('common_file','state_file'):
    out[k]=alias(c[k]);check(out[k] in files,'missing exact array alias SHA')
   target=alias(c['curve_file']);check(target not in files and target not in products,'derived curve alias collision');raw=canonical({'objective_curve':rows,'derivation_identity':'NEW_PUBLIC_REANALYSIS_NOT_HISTORICAL_RUNTIME','source_merged_curves_sha256':curves_sha});products[target]=raw;files[target]=hashlib.sha256(raw).hexdigest();out['curve_file']=target
   if 'audited_metrics' in c:out['audited_metrics']=c['audited_metrics']
  else:check(not any(k in c for k in ('common_file','state_file','curve_file')),'notrun cannot claim arrays/curve')
  outcases.append(out)
 check(len(curves)==125 and all((r.get('object_id'),r.get('policy')) in {(o,p) for o in OBJECTS for p in POLICIES} for r in curves),'merged denominator differs')
 manifest={'schema':'a17.public.saved.nonlinear.inputs.v1','synthetic_fixture':False,'objective_regularization_lambda2':1e-5,'grid_shape':[12,12,12],'counts':{'planned':8,'completed':6,'failed_prefix':1,'not_run':1},'source_reference_sha256':REFERENCES,'files':files,'cases':outcases}
 target=alias(spec['manifest_alias']);check(target not in products and target not in files,'manifest alias collision');products[target]=canonical(manifest);return products

def main():
 p=argparse.ArgumentParser();p.add_argument('--alias-spec',required=True);p.add_argument('--alias-spec-sha256',required=True);p.add_argument('--merged-curves',required=True);p.add_argument('--merged-curves-sha256',required=True);p.add_argument('--write-derived',action='store_true');p.add_argument('--out');a=p.parse_args();check(sha(a.alias_spec)==digest(a.alias_spec_sha256) and sha(a.merged_curves)==digest(a.merged_curves_sha256),'input SHA differs');products=prepare(json.loads(Path(a.alias_spec).read_text()),json.loads(Path(a.merged_curves).read_text()),a.merged_curves_sha256)
 if a.write_derived:
  check(a.out,'new derived output required');root=Path(a.out).absolute();check(not root.exists() and not root.is_symlink() and root.parent.is_dir() and not any(q.is_symlink() for q in root.parents),'unsafe existing output');root.mkdir()
  for name,raw in products.items():
   target=root/PurePosixPath(name);target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(raw)
 print(json.dumps({'status':'DERIVED_CANDIDATES_NOT_PACKAGE_APPROVAL' if a.write_derived else 'DRY_PLAN_ONLY','array_bytes_read':False,'products':[{'public_path':n,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'rights_approved':False,'privacy_reviewed':False,'provenance_reviewed':False} for n,raw in products.items()]}))
if __name__=='__main__':main()
