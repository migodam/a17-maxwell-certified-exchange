"""A10 anchored actual minimum-residual ROM. Parent-reviewed implementation.
For frozen V, min_a ||b(chi)-L(chi)(u0+Va)||. At anchor residual=0.
No full J: cached reduced injection map has shape P x r x N.
"""
import time
import numpy as np
from scipy import linalg as la
from a9_engine import polarizability

class AnchoredROM:
    def __init__(self,state,V,previous=None):
        t=time.perf_counter();self.state=state;self.model=state.model;self.counters=state.counters
        if state.reduced:raise ValueError('full anchor required')
        self.basis=np.array(V,complex,copy=True);self.rstate=V.shape[1]
        if la.norm(V.conj().T@V-np.eye(self.rstate))>1e-7:raise ValueError('basis is not orthonormal')
        old=previous.rstate if previous is not None and previous.state is state and V.shape[1]>=previous.rstate and np.array_equal(V[:,:previous.rstate],previous.basis) else 0
        def Lapply(U):
            if self.model.device=='cuda':
                import torch
                if not hasattr(state,'_a10_L_cuda'):state._a10_L_cuda=torch.as_tensor(state.L,device='cuda')
                return (state._a10_L_cuda@torch.as_tensor(U,device='cuda')).cpu().numpy()
            return state.L@U
        if old and old<V.shape[1]:
            W=Lapply(V[:,old:]);cross=previous.Q.conj().T@W;W-=previous.Q@cross
            correction=previous.Q.conj().T@W;W-=previous.Q@correction;cross+=correction
            Qnew,Rnew=la.qr(W,mode='economic');self.Q=np.column_stack([previous.Q,Qnew])
            self.R=np.block([[previous.R,cross],[np.zeros((Rnew.shape[0],old),complex),Rnew]])
        elif old==V.shape[1]:self.Q,self.R=previous.Q,previous.R
        else:self.Q,self.R=la.qr(Lapply(V),mode='economic')
        if np.min(abs(np.diag(self.R)))<1e-12:raise ValueError('rank deficient L V')
        T=la.solve_triangular(self.R,self.Q.conj().T)
        E=state.exciting.reshape(self.model.P,self.model.N,3)*state.da[None,:,None]/state.sqrt_v
        self.F=np.einsum('rnc,pnc->prn',T.reshape(self.rstate,self.model.N,3),E,optimize=True)
        self.CV=self.model.GS@V
        self.counters.bump('a10_rom_build');self.counters.bump('a10_L_matvec_rhs',self.rstate-old)
        self.counters.add_time('a10_rom_build_s',time.perf_counter()-t)
    @property
    def field(self):return self.state.field
    @property
    def current(self):return self.state.current
    def coeff(self,dc):
        self.counters.bump('a10_reduced_tangent_rhs',self.model.P*(1 if dc.ndim==1 else dc.shape[1]))
        return np.einsum('prn,nb->prb',self.F,dc[:,None] if dc.ndim==1 else dc,optimize=True)
    def current_jvp(self,dc):
        z=self.coeff(np.asarray(dc));k=np.einsum('nr,prb->pnb',self.basis,z,optimize=True)
        return k[:,:,0] if dc.ndim==1 else k
    def jvp(self,dc):
        z=self.coeff(np.asarray(dc));y=np.einsum('mr,prb->pmb',self.CV,z,optimize=True)
        return y[:,:,0] if dc.ndim==1 else y
    def vjp(self,w):
        self.counters.bump('a10_reduced_adjoint_rhs',self.model.P)
        c=np.einsum('mr,pm->pr',self.CV.conj(),w,optimize=True)
        return np.einsum('prn,pr->n',self.F.conj(),c,optimize=True)
    def tangent_rhs(self,dc):
        rhs,_,single=self.state._tangent_rhs(dc);return rhs[:,:,0] if single else rhs
    def tangent_residual(self,dc):
        k=self.current_jvp(dc)
        return self.tangent_rhs(dc)-(self.state.L@k.T).T if k.ndim==2 else self.tangent_rhs(dc)-np.einsum('nm,pmb->pnb',self.state.L,k,optimize=True)
    def nonlinear_field(self,chi):
        L=self.model.L(chi);a,_=polarizability(chi,self.model.volume,self.model.k)
        b=np.repeat(a,3)[:,None]*self.model.incident.T/self.state.sqrt_v
        rhs=b-L@self.state.current.T
        coeff=la.lstsq(L@self.basis,rhs)[0]
        return self.field+(self.CV@coeff).T
