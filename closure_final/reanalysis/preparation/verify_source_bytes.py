"""Read-only source byte/AST/MatMul regression against exact approved plan; never imports checked source."""
from pathlib import Path
import argparse,ast,hashlib,json
from prepare_nonlinear_manifest import check,alias,digest,sha

def profile(raw):
 tree=ast.parse(raw);return {'ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),'matrix_multiply_nodes':sum(isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult) for n in ast.walk(tree))}
def verify(plan,package_root,prefixes):
 check(prefixes,'explicit prefixes required');root=Path(package_root).absolute();check(root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.parents),'safe package parent required');prefixes=[alias(p) for p in prefixes];rows=[r for r in plan['files']+plan.get('release_asset_candidates',[]) if any(r['public_path']==p or r['public_path'].startswith(p+'/') for p in prefixes)];check(rows,'no selected source rows');checks=[]
 for r in rows:
  name=alias(r['public_path']);p=root/name;source=Path(r['source_path']);check(p.is_file() and source.is_file() and not p.is_symlink() and not source.is_symlink() and not any(q.is_symlink() for q in p.parents) and not any(q.is_symlink() for q in source.parents),'ordinary exact source/alias required');check(sha(p)==digest(r['sha256'])==sha(source),'source/public bytes drift');a=profile(source.read_bytes()) if p.suffix=='.py' else None;b=profile(p.read_bytes()) if p.suffix=='.py' else None;check(a==b,'source AST/MatMul drift');checks.append({'public_path':name,'sha256':r['sha256'],'ast_matmul':b,'byte_preserved':True})
 return {'status':'SELECTED_SOURCE_BYTES_AND_AST_PASS','source_imported_or_executed':False,'checks':checks}
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--package-root',required=True);p.add_argument('--prefix',action='append',required=True);a=p.parse_args();check(sha(a.plan)==digest(a.plan_sha256),'plan SHA differs');print(json.dumps(verify(json.loads(Path(a.plan).read_text()),a.package_root,a.prefix)))
if __name__=='__main__':main()
