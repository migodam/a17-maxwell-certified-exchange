import sys,tempfile,unittest,subprocess,shutil,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def test_isolated_public_imports_no_workspace_runtime(self):
  with tempfile.TemporaryDirectory() as tmp:
   for name in ['public_support.py','materialize_evidence.py','reanalyze_saved_nonlinear.py','validate_public_coverage.py']:shutil.copyfile(ROOT/name,Path(tmp)/name)
   code='import sys;sys.path.insert(0,sys.argv[1]);import materialize_evidence,reanalyze_saved_nonlinear,validate_public_coverage'
   subprocess.run([sys.executable,'-I','-c',code,tmp],check=True,capture_output=True)
 def test_no_kernel_transport_optimizer_imports(self):
  banned={'torch','scipy','paramiko','socket','run_nonlinear','closure_runtime','full_maxwell','collect_saved_nonlinear_v2','collect_coverage_saved','subprocess'}
  for name in ['materialize_evidence.py','reanalyze_saved_nonlinear.py','validate_public_coverage.py']:
   tree=ast.parse((ROOT/name).read_text());imports={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module};imports.update(a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names);self.assertFalse(imports & banned)
if __name__=='__main__':unittest.main()
