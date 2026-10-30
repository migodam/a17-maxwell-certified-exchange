"""Frozen current selection adapter. No selector access to full references.

Current coordinates are existing A9 normalized dipoles (Euclidean metric).
Material and per-illumination data coordinates are real [Re; Im].
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator, cg
from fast_contractions import (prepare_reduced_layout, reduced_j, reduced_jt, reduced_current_jvp)

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT/'A14/src', ROOT/'A12/code', ROOT/'A10/code', ROOT/'A9/code'):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from pairtsom_adapter import (PhaseLedger, stack_real_illuminations,
    unstack_real_illuminations, tangent_adjoint, project_orth)
from galerkin_rom import GalerkinROM
from reduced_normal import ReducedNormalInverse
from cair import domain_bank

TOL = 1e-10

def orth(X, against=None, rank=None):
    X=np.asarray(X,dtype=np.complex128)
    # Projection roundoff is measured against the input signature scale, never
    # an absolute unit floor that would erase physically small currents.
    atol=TOL*la.norm(X) if against is not None and against.shape[1] else 0.
    return project_orth(X, against, rank=rank, tol=TOL, atol=atol)

def _real(X):
    if np.iscomplexobj(X): raise ValueError('material/data coordinates must be real')
    return np.asarray(X, dtype=np.float64)

def _complex_material(Z, q):
    Z = _real(Z)
    if Z.shape[0] != 2*q: raise ValueError('material dimension mismatch')
    return Z[:q] + 1j*Z[q:]

def _stability(A):
    sv = la.svdvals(A)
    if not len(sv): return dict(sigma_min=None, sigma_max=None, relative=None, condition=1.)
    ratio = float(sv[-1]/sv[0]) if sv[0] else 0.
    if ratio <= TOL: raise ValueError('compressed current matrix near-singular; no jitter')
    return dict(sigma_min=float(sv[-1]), sigma_max=float(sv[0]), relative=ratio,
                condition=float(sv[0]/sv[-1]))

def _timed_compressed(ctx, label, callback, *, rhs=None, dimension=None):
    """Charge actual compressed kernels, including failed attempts; no full RHS."""
    ledger=ctx.ledger
    ledger.add(label+'_calls',1)
    if rhs is not None: ledger.add(label+'_rhs',int(rhs))
    if dimension is not None:
        ledger.add(label+'_dimension_sum',int(dimension))
        record=ledger._records.setdefault(ledger.current,{})
        key=label+'_dimension_peak'
        record[key]=max(record.get(key,0),int(dimension))
    started=time.perf_counter()
    try: return callback()
    finally: ledger.add(label+'_wall_s',time.perf_counter()-started)


class SelectorContext:
    __slots__ = ('n_current','dim_material','P','m','q','lam','ell','r','tangent','material_gauge',
                 'points','current_forward','current_metric','ledger','_L','_LH','_S','_B','_receiver','_internal','_Bfull','_Badj','_Sadj')
    def phase(self, name): return self.ledger.phase(name)
    def records(self): return self.ledger.as_dict()
    def apply_L(self, C): return self._L(np.asarray(C, dtype=np.complex128))
    def apply_L_adjoint(self, C): return self._LH(np.asarray(C, dtype=np.complex128))
    def apply_S(self, C): return self._S(np.asarray(C, dtype=np.complex128))
    def project_B(self, C): return self._B(np.asarray(C, dtype=np.complex128))
    def receiver_modes(self, count): return self._receiver(int(count))
    def internal_modes(self, count, seed): return self._internal(int(count), int(seed))
    def B(self, Z): return self._Bfull(_real(Z))
    def B_adjoint(self, K): return self._Badj(np.asarray(K,dtype=complex))
    observe = apply_S
    def observe_adjoint(self, W): return self._Sadj(np.asarray(W,dtype=complex))
    def transfer(self, U, C): return build_transfer(self,U,C)
    def make_model(self, U): return ReducedModel(self, U)

class ReducedModel:
    def __init__(self, ctx, U):
        self.ctx = ctx
        self.U = np.asarray(U, dtype=np.complex128).copy()
        if self.U.ndim != 2 or self.U.shape[0] != ctx.n_current: raise ValueError('invalid U')
        if la.norm(self.U.conj().T@self.U-np.eye(self.U.shape[1])) > TOL:
            raise ValueError('U must be current-metric orthonormal')
        self.LU = ctx.apply_L(self.U)
        self.A = self.U.conj().T@self.LU
        self.stability = _stability(self.A)
        self.CV = ctx.apply_S(self.U)
        PB = ctx.project_B(self.U)
        if self.U.shape[1]:
            self._fac = _timed_compressed(ctx,'reduced_current_factorization',
                lambda:la.lu_factor(self.A),dimension=self.U.shape[1])
            self.F = np.stack([self._solve_current(p) for p in PB])
            # Existing exact compressed normal inverse, in physical tangent coordinates.
            tan = SimpleNamespace(d=ctx.dim_material, Q=None, volume=1.)
            self.normal = _timed_compressed(ctx,'reduced_normal_construction',
                lambda:ReducedNormalInverse(self,tan,1.,ctx.lam))
            ctx.ledger.add('reduced_normal_factorizations',1)
            ctx.ledger.add('reduced_normal_'+self.normal.branch+'_factorizations',1)
            ctx.ledger.add('reduced_normal_factor_dimension_sum',self.normal.factor_dimension)
            record=ctx.ledger._records.setdefault(ctx.ledger.current,{})
            record['reduced_normal_factor_dimension_peak']=max(
                record.get('reduced_normal_factor_dimension_peak',0),self.normal.factor_dimension)
            ctx.ledger.add('reduced_normal_compressed_rows_sum',self.normal.rows)
            ctx.ledger.add('reduced_normal_material_complex_dimension_sum',self.normal.q)
        else:
            self.F = np.zeros((ctx.P,0,ctx.q), complex)
            self.normal = None
        self._F_flat,self.fast_layout=_timed_compressed(ctx,'reduced_F_layout',
            lambda:prepare_reduced_layout(self.F))
        ctx.ledger.add('reduced_F_layout_copy_bytes',self.fast_layout['persistent_copy_bytes'])
    def _solve_current(self,rhs,adjoint=False):
        rhs=np.asarray(rhs,dtype=np.complex128)
        cols=1 if rhs.ndim==1 else rhs.shape[1]
        return _timed_compressed(self.ctx,
            'reduced_current_adjoint_solve' if adjoint else 'reduced_current_solve',
            lambda:la.lu_solve(self._fac,rhs,trans=2 if adjoint else 0),
            rhs=cols,dimension=self.U.shape[1])
    def apply_R(self, rhs):
        rhs=np.asarray(rhs,dtype=complex)
        if not self.U.shape[1]: return np.zeros_like(rhs)
        return self.U@self._solve_current(self.U.conj().T@rhs)
    def apply_R_adjoint(self, rhs):
        rhs=np.asarray(rhs,dtype=complex)
        if not self.U.shape[1]: return np.zeros_like(rhs)
        return self.U@self._solve_current(self.U.conj().T@rhs,True)
    R=apply_R
    R_adjoint=apply_R_adjoint
    def current_jvp(self,Z):
        return reduced_current_jvp(self.F,self.U,Z,F_flat=self._F_flat)
    def j(self, Z):
        return reduced_j(self.F,self.CV,Z,F_flat=self._F_flat)
    def jt(self, W):
        return reduced_jt(self.F,self.CV,W,F_flat=self._F_flat)
    def H(self, Z): return self.jt(self.j(Z)) + self.ctx.lam*_real(Z)
    def solve(self, rhs):
        rhs=_real(rhs);cols=1 if rhs.ndim==1 else rhs.shape[1]
        if self.normal is None:
            return _timed_compressed(self.ctx,'lambda_only_solve',lambda:rhs/self.ctx.lam,rhs=cols)
        return _timed_compressed(self.ctx,'reduced_normal_solve',
            lambda:self.normal.solve(rhs),rhs=cols,dimension=self.normal.factor_dimension)
    solve_H=solve
    def step(self): return -self.solve(self.jt(self.ctx.r)+self.ctx.ell)
    def phi(self, s):
        s = _real(s); e = self.ctx.r+self.j(s)
        return float(.5*e@e+self.ctx.ell@s+.5*self.ctx.lam*(s@s))
    def compressed_stability(self): return dict(self.stability)

class Transfer:
    def __init__(self,ctx,U,C):
        self.ctx=ctx
        base = U if isinstance(U, ReducedModel) else ctx.make_model(U)
        self.C=np.asarray(C,dtype=np.complex128).copy()
        if self.C.ndim!=2 or self.C.shape[0]!=ctx.n_current: raise ValueError('invalid C')
        LC=ctx.apply_L(self.C)
        F=self.C.conj().T@base.LU
        E=base.U.conj().T@LC
        AE=base._solve_current(E) if base.U.shape[1] else E
        self.Y=ctx.apply_S(self.C-base.U@AE)
        self.Gamma=self.C.conj().T@LC-F@AE
        self.stability=_stability(self.Gamma)
        self.N=ctx.project_B(self.C)-np.einsum('br,prq->pbq',F,base.F)
        # Existing implementation factors Gamma once per illumination via solve.
        # Split the same LU kernels only to measure factor and solve time separately;
        # do not introduce cross-illumination factor reuse or alter the algorithm.
        factors=[]
        for p in self.N:
            fac=_timed_compressed(ctx,'transfer_Gamma_factorization',
                lambda:la.lu_factor(self.Gamma),dimension=self.Gamma.shape[0])
            factors.append(_timed_compressed(ctx,'transfer_Gamma_solve',
                lambda:la.lu_solve(fac,p),rhs=p.shape[1],dimension=self.Gamma.shape[0]))
        self.T=np.stack(factors)
        ctx.ledger.add('transfer_builds',1)
        ctx.ledger.add('transfer_current_columns',self.C.shape[1])
    def delta_j(self,Z):
        c=_complex_material(Z,self.ctx.q);one=c.ndim==1
        a=np.einsum('pbq,qk->pbk',self.T,c[:,None] if one else c)
        out=stack_real_illuminations(np.einsum('mb,pbk->pmk',self.Y,a))
        return out[:,0] if one else out
    def pullback(self,W):
        w=unstack_real_illuminations(_real(W),self.ctx.P,self.ctx.m);one=w.ndim==2
        a=np.einsum('mb,pmk->pbk',self.Y.conj(),w[:,:,None] if one else w)
        c=np.einsum('pbq,pbk->qk',self.T.conj(),a)
        out=np.concatenate([c.real,c.imag],axis=0)
        return out[:,0] if one else out
    j=delta_j
    jt=pullback
    def __getitem__(self,key): return getattr(self,key)

def build_transfer(ctx,U,C): return Transfer(ctx,U,C)

class EvaluationContext:
    def __init__(self,state,tangent,ctx,scale):
        self._state=state; self._ctx=ctx;self._scale=scale;self._tangent=tangent
        self.last_step_record=None
    def j(self,Z):
        Z=_real(Z);b=1 if Z.ndim==1 else Z.shape[1]
        self._ctx.ledger.add('full_tangent_rhs',self._ctx.P*b)
        return stack_real_illuminations(self._state.jvp(self._tangent.expand(Z))/self._scale)
    def jt(self,W):
        ctx=self._ctx;state=self._state;model=state.model
        w=unstack_real_illuminations(_real(W),ctx.P,ctx.m);one=w.ndim==2
        w=w[:,:,None] if one else w; b=w.shape[2]
        ctx.ledger.add('full_adjoint_rhs',ctx.P*b)
        y=np.einsum('oc,pob->pcb',model.GS.conj(),w/self._scale)
        rhs=y.transpose(1,0,2).reshape(model.n,ctx.P*b)
        solved=model.solver.solve(state._L_factor,rhs,trans=2,matrix=state.L,label='full_adjoint')
        u=solved.reshape(model.n,ctx.P,b).transpose(1,0,2)
        inner=np.einsum('pnc,pncb->nb',state.exciting.reshape(ctx.P,model.N,3).conj(),u.reshape(ctx.P,model.N,3,b))
        grad=state.da.conj()[:,None]*inner/state.sqrt_v
        out=tangent_adjoint(self._tangent,grad)
        return out[:,0] if one else out
    def H(self,Z): return self.jt(self.j(Z))+self._ctx.lam*_real(Z)
    def phi(self,s):
        ctx=self._ctx;s=_real(s);e=ctx.r+self.j(s)
        return float(.5*e@e+ctx.ell@s+.5*ctx.lam*(s@s))
    def dense_j(self):
        if self._ctx.dim_material>128: raise ValueError('offline dense J restricted to d<=128')
        return self.j(np.eye(self._ctx.dim_material))
    def step(self):
        ctx=self._ctx;before=ctx.records()
        with ctx.phase('full_reference'):
            h=self.jt(ctx.r)+ctx.ell
            if ctx.dim_material<=128:
                J=self.dense_j();H=J.T@J+ctx.lam*np.eye(ctx.dim_material)
                s=la.solve(H,-h,assume_a='pos');info=0
            else:
                H=LinearOperator((ctx.dim_material,)*2,matvec=self.H,dtype=float)
                iterations=[0]
                def callback(_): iterations[0]+=1
                s,info=cg(H,-h,rtol=1e-12,atol=0.,maxiter=150,callback=callback)
            residual=float(la.norm(self.H(s)+h)/max(la.norm(h),1e-300))
        self.last_step_record=dict(relative_true_residual=residual,info=int(info),
            method='dense' if ctx.dim_material<=128 else 'scipy_cg',
            iterations=0 if ctx.dim_material<=128 else iterations[0],
            maxiter=None if ctx.dim_material<=128 else 150,before=before,after=ctx.records())
        if info!=0 or residual>1e-10: raise ValueError('full reference solve failed true-residual gate')
        return s
    def current_j(self,Z):
        ctx=self._ctx;Z=_real(Z);ctx.ledger.add('full_tangent_rhs',ctx.P*(1 if Z.ndim==1 else Z.shape[1]))
        return self._state.current_jvp(self._tangent.expand(Z))
    def witness(self,s=None):
        ctx=self._ctx;s=self.step() if s is None else _real(s)
        primal=self.current_j(s).T
        w=unstack_real_illuminations(ctx.r+self.j(s),ctx.P,ctx.m)
        rhs=self._state.model.GS.conj().T@w.T/self._scale
        ctx.ledger.add('full_adjoint_rhs',ctx.P)
        dual=self._state.model.solver.solve(self._state._L_factor,rhs,trans=2,matrix=self._state.L,label='full_adjoint')
        U=orth(np.column_stack([primal,dual]));model=ctx.make_model(U)
        return U,dict(actual_rank=int(U.shape[1]),num_illuminations=ctx.P,stability=model.stability,
                      relative_step_error=float(la.norm(model.step()-s)/max(la.norm(s),1e-300)))

def make_problem(state,tangent,data,scale,lam,ell):
    if state.reduced: raise ValueError('full frozen anchor required')
    if scale<=0 or lam<=0 or not np.isfinite(scale+lam): raise ValueError('scale and lam must be positive finite')
    Q=getattr(tangent,'Q',None)
    if Q is not None and la.norm(tangent.volume*np.asarray(Q).T@Q-np.eye(Q.shape[1]))>TOL:
        raise ValueError('physical material basis must be volume-orthonormal')
    ctx=SelectorContext();model=state.model
    ctx.n_current=model.n;ctx.P=model.P;ctx.m=model.m;ctx.dim_material=tangent.d;ctx.q=tangent.d//2
    ctx.points=model.points.copy();ctx.current_forward=state.current.copy()
    ctx.lam=float(lam);ctx.ell=_real(ell).copy();ctx.tangent=tangent
    if ctx.ell.shape!=(ctx.dim_material,): raise ValueError('ell shape mismatch')
    ctx.r=stack_real_illuminations((state.field-np.asarray(data))/scale)
    ctx.current_metric='A9 normalized dipole Euclidean, physical volume already absorbed'
    ctx.ledger=PhaseLedger();ctx.material_gauge={}
    def counted(name,fn,C):
        ctx.ledger.add(name+'_rhs',1 if C.ndim==1 else C.shape[1]);return fn(C)
    # Reuse the exact validated A12 state-local GPU cache and FP64 matrix action.
    action_owner=SimpleNamespace(model=model,state=state,counters=state.counters)
    def apply_l(C,adjoint=False):
        cols=1 if C.ndim==1 else C.shape[1]
        ctx.ledger.add(('L_adjoint' if adjoint else 'L')+'_rhs',cols)
        started=time.perf_counter()
        if model.device=='cuda':
            if getattr(state,'_a12_L_cuda',None) is None:
                ctx.ledger.add('L_cuda_uploads',1)
                ctx.ledger.add('L_cuda_upload_bytes',state.L.nbytes)
            ctx.ledger.add('L_cuda_input_transfers',1)
            ctx.ledger.add('L_cuda_input_bytes',C.nbytes)
            ctx.ledger.add('L_cuda_output_transfers',1)
            ctx.ledger.add('L_cuda_output_bytes',C.nbytes)
        result=GalerkinROM._apply_L(action_owner,C,adjoint=adjoint)
        ctx.ledger.add(('L_adjoint' if adjoint else 'L')+'_wall_s',time.perf_counter()-started)
        return result
    ctx._L=lambda C:apply_l(C)
    ctx._LH=lambda C:apply_l(C,True)
    ctx._S=lambda C:counted('S',lambda x:model.GS@x/scale,C)
    field=state.exciting.reshape(model.P,model.N,3)*state.da[None,:,None]/state.sqrt_v
    def PB(C):
        ctx.ledger.add('B_projection_current_columns',C.shape[1])
        N=np.einsum('bnc,pnc->pbn',C.conj().T.reshape(C.shape[1],model.N,3),field)
        return N/np.sqrt(tangent.volume) if Q is None else N@Q
    ctx._B=PB
    def bfull(Z):
        c=_complex_material(Z,ctx.q);one=c.ndim==1
        dc=tangent.expand(Z);dc=dc[:,None] if one else dc
        ctx.ledger.add('B_full_illumination_rhs',ctx.P*dc.shape[1])
        out=np.einsum('pnc,nb->pncb',field,dc).reshape(ctx.P,ctx.n_current,-1)
        return out[:,:,0] if one else out
    def badj(K):
        one=K.ndim==2;K=K[:,:,None] if one else K
        ctx.ledger.add('B_adjoint_illumination_rhs',ctx.P*K.shape[2])
        g=np.einsum('pnc,pncb->nb',field.conj(),K.reshape(ctx.P,model.N,3,-1))
        out=tangent_adjoint(tangent,g);return out[:,0] if one else out
    ctx._Bfull=bfull;ctx._Badj=badj
    ctx._Sadj=lambda W:counted('S_adjoint',lambda x:model.GS.conj().T@x/scale,W)
    def receiver(count):
        ctx.ledger.add('receiver_mode_accesses',1)
        build=model._gs_svd is None
        started=time.perf_counter()
        V=model.gs_modes()['V'][:,:count].copy()
        ctx.ledger.add('receiver_svd_builds' if build else 'receiver_svd_cache_hits',1)
        ctx.ledger.add('receiver_mode_wall_s',time.perf_counter()-started)
        return V
    ctx._receiver=receiver
    internal_cache={}
    def internal(count,seed):
        if count<0: raise ValueError('mode count must be nonnegative')
        key=(count,seed)
        ctx.ledger.add('internal_mode_accesses',1)
        if key in internal_cache:
            ctx.ledger.add('internal_mode_cache_hits',1)
            V,meta=internal_cache[key]
            return V.copy(),dict(meta)
        started=time.perf_counter()
        V,meta=domain_bank(model,count,seed)
        q=min(model.n,count+12);b=V.shape[1]
        # Count existing helper's actual products, including Q^H GD and the
        # two independent GD^H AV residual diagnostic products.
        ctx.ledger.add('GD_rhs',3*q+b)
        ctx.ledger.add('GD_adjoint_rhs',3*q+2*b)
        ctx.ledger.add('internal_mode_builds',1)
        ctx.ledger.add('internal_mode_wall_s',time.perf_counter()-started)
        meta['label']='existing randomized GD proxy; not reproduced original TSOM'
        meta['backend']='numpy_cpu_complex128'
        internal_cache[key]=(V.copy(),dict(meta))
        return V,dict(meta)
    ctx._internal=internal
    return ctx,EvaluationContext(state,tangent,ctx,float(scale))
