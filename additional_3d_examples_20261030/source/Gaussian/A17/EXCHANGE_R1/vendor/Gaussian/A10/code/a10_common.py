"""A10 shared contracts: real physical tangents, balanced acquisition, provenance."""
import sys,time,json,hashlib,platform
from pathlib import Path
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator,cg
HERE=Path(__file__).resolve().parent
for p in [HERE.parents[1]/'A9'/'code',Path('D:/AI/A9_STRONG_TAP/code')]:
    if p.exists():sys.path.insert(0,str(p))
from a9_engine import DenseDDA,receiver_operator,dyadic_green
from experiment_support import grid,Tangent,orth,acquisition,dump,angle

def provenance(out,args):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',args)
    dump(out/'environment.json',dict(host=platform.node(),python=sys.version,numpy=np.__version__,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    dest=out/'source_snapshot';dest.mkdir();hashes={}
    for p in HERE.glob('*.py'):
        raw=p.read_bytes();(dest/p.name).write_bytes(raw);hashes[p.name]=hashlib.sha256(raw).hexdigest()
    dump(out/'source_hashes.json',hashes)

def balanced(rx=64,rotation=0.,partial=False):
    _,_,rr,obs=acquisition(6,rx,rotation)
    if partial:return acquisition(6,rx,rotation)
    dirs=[];pol=[]
    for i in range(3):
        for j in range(3):
            if i!=j:dirs.append(np.eye(3)[i]);pol.append(np.eye(3)[j])
    return np.array(dirs),np.array(pol),rr,obs

def scene(p,regime='smooth',strength=1.):
    p=np.asarray(p)
    if regime=='smooth':
        a=np.exp(-np.sum(((p-[-.18,.02,0])/[.28,.32,.25])**2,axis=1));a+=.65*np.exp(-np.sum(((p-[.27,-.12,.08])/[.20,.22,.3])**2,axis=1))
    elif regime=='curved':
        rho=np.sqrt(p[:,0]**2+p[:,1]**2);a=np.exp(-((rho-.30)/.11)**2-(p[:,2]/.25)**2)*(1+.3*np.sin(4*np.arctan2(p[:,1],p[:,0])))
    elif regime=='contact':
        a=np.exp(-np.sum(((p-[-.2,0,0])/[.19,.28,.27])**2,axis=1))+1.1*np.exp(-np.sum(((p-[.2,0,0])/[.19,.25,.29])**2,axis=1))
    elif regime=='mixed':
        a=.8*(np.sum(((p-[-.2,0,.1])/[.22,.35,.3])**2,axis=1)<1)+.6*np.exp(-np.sum(((p-[.25,.1,-.1])/.22)**2,axis=1))
    else:raise ValueError(regime)
    return .1+strength*a+1j*(.04+.06*strength*a)

class Ops:
    def __init__(self,state,tangent,scale):self.state=state;self.t=tangent;self.scale=scale;self.d=tangent.d
    def j(self,z):return self.state.jvp(self.t.expand(z))/self.scale
    def v(self,w):return self.t.adjoint(self.state.vjp(w/self.scale))
    def k(self,z):return self.state.current_jvp(self.t.expand(z))

def solve_step(op,r,lam,h=None,rtol=1e-7,maxiter=100,x0=None,save_vectors=False):
    h=np.zeros(op.d) if h is None else h
    f=op.v(r)+h;hist=[];vectors=[]
    def mv(z):return op.v(op.j(z))+lam*z
    H=LinearOperator((op.d,op.d),matvec=mv,dtype=float)
    def cb(x):
        hist.append(float(np.linalg.norm(x)))
        if save_vectors and len(vectors)<12:vectors.append(x.copy())
    s,info=cg(H,-f,x0=x0,rtol=rtol,atol=0,maxiter=maxiter,callback=cb)
    d=mv(s)+f
    return s,dict(iterations=len(hist),info=int(info),normal_residual=float(la.norm(d)),gradient_norm=float(la.norm(f)),vectors=vectors)

def certify(fullop,s,r,lam,h):
    t=time.perf_counter();K=fullop.k(s);Js=np.einsum('mn,pn->pm',fullop.state.model.GS,K)/fullop.scale
    d=fullop.v(r+Js)+lam*s+h
    return dict(s=s.copy(),d=d,Js=Js,K=K,eta=float(la.norm(d)/lam),q=float(.5*la.norm(r+Js)**2+h@s+.5*lam*(s@s)),wall_s=time.perf_counter()-t)

def summary_cert(c):return {k:v for k,v in c.items() if k not in ['s','d','Js','K']}
def cost(model):return dict(model.counters.totals)
def costdiff(a,b):return {k:b.get(k,0)-a.get(k,0) for k in set(a)|set(b)}

def append_basis(V,U,maxadd=None):
    U=np.asarray(U).reshape(V.shape[0],-1);U=U-V@(V.conj().T@U);U=U-V@(V.conj().T@U)
    Q=orth(U)
    if maxadd is not None:Q=Q[:,:maxadd]
    return np.column_stack([V,Q])

def current_columns(K):return K.T if K.ndim==2 else K.transpose(1,0,2).reshape(K.shape[1],-1)

def certified_full_cg(op,r,lam,h,theta=.05,maxiter=180):
    start=time.perf_counter();before=cost(op.state.model);f=op.v(r)+h;s=np.zeros(op.d);d=f.copy();p=-d;hist=[]
    for it in range(maxiter):
        Jp=op.j(p);Hp=op.v(Jp)+lam*p;alpha=-(d@p)/(p@Hp);s+=alpha*p;old=d;d=d+alpha*Hp
        eta=la.norm(d)/lam;hist.append(float(eta))
        if eta<=theta/(1+theta)*la.norm(s)+1e-10:break
        beta=(d@d)/max(old@old,1e-300);p=-d+beta*p
    cert=certify(op,s,r,lam,h)
    return s,dict(wall_s=time.perf_counter()-start,cost=costdiff(before,cost(op.state.model)),iterations=it+1,eta=cert['eta'],passed=bool(cert['eta']<=theta/(1+theta)*la.norm(s)+1e-10),history=hist)
