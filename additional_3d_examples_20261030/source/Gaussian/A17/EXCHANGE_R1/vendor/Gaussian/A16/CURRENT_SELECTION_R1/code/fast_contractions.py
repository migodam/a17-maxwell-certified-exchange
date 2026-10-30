"""Exact BLAS contractions for frozen compressed models; no solver or score.

Projection layout may add one paid persistent PB-sized copy per workspace.
Model F layout normally shares contiguous F storage. Adjoint products conjugate
small inputs/output instead of retaining a second full F array. Layout metadata
reports actual persistent copies; transient BLAS buffers are backend-dependent.
"""
from dataclasses import dataclass
import time
import numpy as np


def _complex(X):return np.asarray(X,dtype=np.complex128)


@dataclass(frozen=True)
class ProjectionLayout:
    flat:np.ndarray
    P:int
    n:int
    q:int
    metadata:dict


def prepare_projection_layout(PB):
    start=time.perf_counter();PB=_complex(PB)
    if PB.ndim!=3:raise ValueError('PB must have shape(P,n,q)')
    P,n,q=PB.shape;flat=np.ascontiguousarray(PB.transpose(0,2,1)).reshape(P*q,n)
    shares=np.shares_memory(flat,PB)
    return ProjectionLayout(flat,P,n,q,{'wall_s':time.perf_counter()-start,
        'persistent_copy_bytes':0 if shares else flat.nbytes,'layout_bytes':flat.nbytes,
        'shares_memory_with_PB':bool(shares),'layout':'(P*q,n) C-contiguous; Q*B, not full-J',
        'additional_layout_copy':not shares})


def project_B(layout,C):
    C=_complex(C)
    if C.ndim!=2 or C.shape[0]!=layout.n:raise ValueError('C must have shape(n,b)')
    return (layout.flat@C.conj()).reshape(layout.P,layout.q,C.shape[1]).transpose(0,2,1)


def prepare_reduced_layout(F):
    start=time.perf_counter();F=_complex(F)
    if F.ndim!=3:raise ValueError('F must have shape(P,k,q)')
    P,k,q=F.shape;flat=np.ascontiguousarray(F).reshape(P*k,q)
    shares=np.shares_memory(flat,F)
    return flat,{'wall_s':time.perf_counter()-start,'persistent_copy_bytes':0 if shares else flat.nbytes,
        'layout_bytes':flat.nbytes,'shares_memory_with_F':bool(shares),
        'layout':'(P*k,q); no persistent conjugate/adjoint F copy',
        'adjoint_transient_rule':'conjugate compressed input then material output; no full F conjugation'}


def _material(Z,q):
    if np.iscomplexobj(Z):raise ValueError('material coordinates must be real')
    Z=np.asarray(Z,dtype=np.float64)
    if Z.ndim not in (1,2) or Z.shape[0]!=2*q:raise ValueError('real material shape mismatch')
    one=Z.ndim==1;c=Z[:q]+1j*Z[q:]
    return c[:,None] if one else c,one


def _flat(F,F_flat):
    F=_complex(F)
    if F.ndim!=3:raise ValueError('F must have shape(P,k,q)')
    flat=prepare_reduced_layout(F)[0] if F_flat is None else _complex(F_flat)
    if flat.shape!=(F.shape[0]*F.shape[1],F.shape[2]):raise ValueError('F layout mismatch')
    return F,flat


def _pack_data(Y):
    return np.concatenate([Y.real,Y.imag],axis=1).reshape(Y.shape[0]*2*Y.shape[1],Y.shape[2])


def reduced_j(F,CV,Z,*,F_flat=None):
    F,flat=_flat(F,F_flat);P,k,q=F.shape;CV=_complex(CV)
    if CV.ndim!=2 or CV.shape[1]!=k:raise ValueError('CV must have shape(m,k)')
    c,one=_material(Z,q);x=(flat@c).reshape(P,k,c.shape[1]);out=_pack_data(CV@x)
    return out[:,0] if one else out


def reduced_current_jvp(F,U,Z,*,F_flat=None):
    F,flat=_flat(F,F_flat);P,k,q=F.shape;U=_complex(U)
    if U.ndim!=2 or U.shape[1]!=k:raise ValueError('U must have shape(n,k)')
    c,one=_material(Z,q);out=U@(flat@c).reshape(P,k,c.shape[1])
    return out[:,:,0] if one else out


def reduced_jt(F,CV,W,*,F_flat=None):
    F,flat=_flat(F,F_flat);P,k,q=F.shape;CV=_complex(CV)
    if CV.ndim!=2 or CV.shape[1]!=k:raise ValueError('CV must have shape(m,k)')
    if np.iscomplexobj(W):raise ValueError('data coordinates must be real')
    W=np.asarray(W,dtype=np.float64);m=CV.shape[0]
    if W.ndim not in (1,2) or W.shape[0]!=P*2*m:raise ValueError('real data shape mismatch')
    one=W.ndim==1;w=W.reshape(P,2*m,-1);w=w[:,:m]+1j*w[:,m:]
    a=(CV.conj().T@w).reshape(P*k,w.shape[2])
    # F* a=(a* F)*; avoid persistent or per-call full F conjugation.
    c=(a.conj().T@flat).conj().T
    out=np.concatenate([c.real,c.imag],axis=0)
    return out[:,0] if one else out


def s5_frobenius(G,F):
    """2 real vdot(F,G@F), summed per illumination to bound temporaries."""
    G=_complex(G);F=_complex(F)
    if F.ndim!=3 or G.shape!=(F.shape[1],F.shape[1]):raise ValueError('Gram/F dimensions mismatch')
    return float(2*sum(np.vdot(fp,G@fp).real for fp in F))
