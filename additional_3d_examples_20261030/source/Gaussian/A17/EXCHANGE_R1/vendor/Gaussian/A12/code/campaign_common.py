"""A12 experiment contracts; reuse frozen A9 Maxwell/A10 inverse machinery."""
from pathlib import Path
import sys,os,time,json,hashlib,platform
HERE=Path(__file__).resolve().parent
for p in [HERE.parent/'deps'/'a10',HERE.parent/'deps'/'a9',HERE.parents[1]/'A10'/'code',HERE.parents[1]/'A9'/'code']:
    if p.exists():sys.path.insert(0,str(p))
import numpy as np
from scipy import linalg as la
from a10_common import DenseDDA,Ops,balanced,grid,Tangent,append_basis,current_columns,cost,costdiff,solve_step,receiver_operator
from experiment_support import dump,orth

def sync():
    try:
        import torch
        if torch.cuda.is_available():torch.cuda.synchronize()
    except ImportError:pass

def stamp():sync();return time.perf_counter()
def provenance(out,config):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',config); sources={}
    roots={'a12':HERE}
    for name in ['a9_engine','experiment_support','a10_common','feasible_step','reduced_normal']:
        mod=__import__(name);roots[name]=Path(mod.__file__)
    for prefix,root in roots.items():
        for p in ([root] if root.is_file() else sorted(root.glob('*.py'))):
            raw=p.read_bytes();rel=f'{prefix}/{p.name}';dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw);sources[rel]=hashlib.sha256(raw).hexdigest()
    for p in sorted((HERE.parent/'protocols').glob('*')):
        if p.is_file():
            raw=p.read_bytes();rel='protocols/'+p.name;dst=out/'source_snapshot'/rel;dst.parent.mkdir(exist_ok=True);dst.write_bytes(raw);sources[rel]=hashlib.sha256(raw).hexdigest()
    dump(out/'source_hashes.json',sources)
    env=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=platform.node(),python=sys.version,numpy=np.__version__,threads=os.getenv('OPENBLAS_NUM_THREADS'))
    try:
        import torch
        env.update(torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    except ImportError:pass
    dump(out/'environment.json',env)

def gpu_peak():
    try:
        import torch
        return dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
    except ImportError:return None

def physical_object(p,c):
    """Continuous material map evaluated identically on inversion/truth grids."""
    idx=c['object_id'];rng=np.random.default_rng(12000+idx)
    centers=np.stack(np.meshgrid(*([np.linspace(-.45,.45,3)]*3),indexing='ij'),-1).reshape(-1,3)
    use=rng.choice(27,4,replace=False); weights=rng.uniform(.4,1.,4)
    if c['family']=='contact':use=np.array([4,13,22,16]);weights=np.array([.9,1.1,.8,.45])*(.9+.2*rng.random(4))
    if c['family']=='curved':use=np.array([1,3,5,7,19,21,23,25]);weights=rng.uniform(.35,.75,8)
    blob=sum(w*np.exp(-np.sum((p-centers[i])**2,axis=1)/(2*.28**2)) for i,w in zip(use,weights))
    return .1+c['strength']*blob+1j*(.04+(.04+.01*(idx%3))*c['strength']*blob)

def geometry(c,rx=64):
    acq=balanced(rx,rotation=c['rotation'],partial=c.get('partial',False));d,p,r,o=acq
    a=c['rotation'];cz,sz=np.cos(a),np.sin(a);cy,sy=np.cos(.37*a),np.sin(.37*a)
    R=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    return d@R.T,p@R.T,r,o

def hnorm(op,z,lam):return float(np.sqrt(max(0.,np.linalg.norm(op.j(z))**2+lam*float(z@z))))
def quadratic(Js,s,r,lam,ell):return float(np.real(np.vdot(r,Js))+.5*np.linalg.norm(Js)**2+.5*lam*(s@s)+ell@s)
def make_H(op,lam):return lambda z:op.v(op.j(z))+lam*z

def build_current_basis(state,Vs,rank):
    # Actual nonlinear Galerkin interpolation at anchor: include full primal states.
    return append_basis(Vs[:,:rank],state.current.T)

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['vectors','Z','HZ','basis','x','step','s','d','Js','K','z','q']}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist() if x.size<5000 else {'shape':list(x.shape),'omitted_from_json':True}
    return x
