"""Load exact new-execution kernel copies before any legacy path injection.

This pins import resolution only. It changes no numerical function, class,
selection policy or parameter. Each actual imported file is rehashed.
"""
from pathlib import Path
import sys, importlib.abc, importlib.util, hashlib, json, platform, os

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
class PinnedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in MANIFEST['modules']:return None
        row=MANIFEST['modules'][fullname];p=ROOT/row['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ImportError('Pinned source drift: '+fullname)
        return importlib.util.spec_from_file_location(fullname,p)
sys.meta_path.insert(0,PinnedFinder())

def receipt():
    import numpy,scipy
    actual={}
    for name,row in MANIFEST['modules'].items():
        mod=sys.modules.get(name)
        if mod is None:continue
        p=Path(mod.__file__).resolve();expected=(ROOT/row['path']).resolve()
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if p!=expected or h!=row['sha256']:raise RuntimeError('Runtime source mismatch '+name)
        actual[name]=dict(bundle_path=row['path'],sha256=h,actual_path=str(p))
    output=dict(python=platform.python_version(),executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,modules=actual,numerical_monkeypatch=False,threads={k:os.getenv(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    try:
        import torch
        output.update(torch=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    except ImportError:pass
    own={}
    for p in (ROOT/'code').glob('*.py'):own[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    output['a17_sources']=own
    return output
