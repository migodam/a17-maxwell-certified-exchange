"""Exact reusable removal core; standalone endpoint API remains unchanged.

CoreEndpointBatch extends EndpointBatch with prepare_removal_core(V) and
evaluate_incoming(core,C). Core/schur factors are non-Hermitian pivoted LU;
compressed material normal remains the same SPD direct/Woodbury solve.
"""
import time
from dataclasses import dataclass
import numpy as np
from scipy import linalg as la
from endpoint_batch import EndpointBatch,TOL

@dataclass
class RemovalCore:
    owner: object
    V: np.ndarray
    status: str
    A: object
    LV: object
    CV: object
    Fv: object
    factor: object
    stability: dict
    statistics: dict

class CoreEndpointBatch(EndpointBatch):
    def _host(self,x,stats):
        if self.device=='cpu':return x
        y=x.detach().cpu().numpy()
        if self.device=='cuda':stats['host_download_bytes']+=y.nbytes
        return y

    def prepare_removal_core(self,V):
        V=np.asarray(V,dtype=np.complex128).copy();r=V.shape[1] if V.ndim==2 else -1
        if V.ndim!=2 or V.shape[0]!=self.qc or not np.isfinite(V).all() or np.linalg.norm(V.conj().T@V-np.eye(r))>TOL:raise ValueError('core V must be finite current-orthonormal')
        V.setflags(write=False)
        stats={'core_factorizations':0,'core_factor_solve_calls':0,'core_solve_rhs':0,'host_upload_bytes':0,'host_download_bytes':0,'core_rank':r,'core_prepare_wall_s':0.,'charge_once':'prepare fields belong to this core acquisition, not each incoming evaluation'}
        started=time.perf_counter();gpu=self.device!='cpu';t=getattr(self,'torch',None)
        v=t.as_tensor(V.copy(),device=self.tdevice,dtype=t.complex128) if gpu else V
        if self.device=='cuda':stats['host_upload_bytes']+=V.nbytes
        LV=self.L@v;CV=self.S@v;A=v.conj().swapaxes(-1,-2)@LV
        aa=self._host(A,stats)
        sv=np.linalg.svd(aa,compute_uv=False) if r else np.empty(0)
        ratio=float(sv[-1]/sv[0]) if r and sv[0] else (0. if r else None)
        st={'sigma_min':float(sv[-1]) if r else None,'sigma_max':float(sv[0]) if r else None,'relative':ratio}
        factor=None;Fv=None;status='STABLE_CORE'
        if not r:status='EMPTY_CORE_DIRECT_ENDPOINT'
        elif ratio<=TOL:status='UNSTABLE_CORE_DIRECT_ENDPOINT_FALLBACK'
        else:
            PBv=(self.PBflat@v.conj()).reshape(self.P,self.q,r)
            PBv=PBv.transpose(1,2) if gpu else PBv.transpose(0,2,1)
            rhs=PBv.permute(1,0,2).reshape(r,self.P*self.q) if gpu else PBv.transpose(1,0,2).reshape(r,self.P*self.q)
            stats['core_factorizations']=1;stats['core_factor_solve_calls']=1;stats['core_solve_rhs']=self.P*self.q
            if gpu:
                lu,piv,info=t.linalg.lu_factor_ex(A,check_errors=False);factor=(lu,piv)
                if int(self._host(info,stats))!=0:status='CORE_FACTOR_FAILED_DIRECT_ENDPOINT_FALLBACK'
                else:Fv=t.linalg.lu_solve(lu,piv,rhs).reshape(r,self.P,self.q).transpose(0,1)
            else:
                try:factor=la.lu_factor(aa);Fv=la.lu_solve(factor,rhs).reshape(r,self.P,self.q).transpose(1,0,2)
                except la.LinAlgError:status='CORE_FACTOR_FAILED_DIRECT_ENDPOINT_FALLBACK'
        self._sync();stats['core_prepare_wall_s']=time.perf_counter()-started
        stats['core_persistent_bytes']=sum(x.numel()*x.element_size() if gpu else x.nbytes for x in [LV,CV,A])+(Fv.numel()*Fv.element_size() if gpu and Fv is not None else Fv.nbytes if Fv is not None else 0)
        stats['core_persistent_bytes']+=V.nbytes+sum(x.numel()*x.element_size() if gpu else x.nbytes for x in (factor or ()))
        return RemovalCore(self,V,status,A,LV,CV,Fv,factor,st,stats)

    def evaluate_incoming(self,core,incoming,*,return_coefficients=False):
        if not isinstance(core,RemovalCore) or core.owner is not self:raise ValueError('core belongs to another frozen endpoint engine')
        C=np.asarray(incoming,dtype=np.complex128)
        if C.ndim!=3 or C.shape[1]!=self.qc or C.shape[2]<1 or core.V.shape[1]+C.shape[2]>self.qc:raise ValueError('incoming=(batch,qc,d) positive feasible d')
        b,_,d=C.shape;r=core.V.shape[1];statuses=['PENDING']*b;basis=np.full_like(C,np.nan);prep=time.perf_counter()
        # Twice residual-project against declared core, then retain exact rank d.
        # Reject dependent banks; never silently drop an atom or jitter.
        for i,c in enumerate(C):
            if not np.isfinite(c).all():statuses[i]='NONFINITE_INCOMING';continue
            residual=c-core.V@(core.V.conj().T@c);residual-=core.V@(core.V.conj().T@residual)
            sv=np.linalg.svd(residual,compute_uv=False)
            if len(sv)<d or sv[-1]<=TOL*np.linalg.norm(c):statuses[i]='INCOMING_RANK_DEFICIENT';continue
            q,_=np.linalg.qr(residual,mode='reduced');basis[i]=q
        prepwall=time.perf_counter()-prep;valid=[i for i,s in enumerate(statuses) if s=='PENDING']
        steps=np.full((b,2*self.q),np.nan);js=np.full((b,self.P*2*self.m),np.nan);resids=np.full(b,np.nan);info_trace=np.full(b,np.nan);stability=[None]*b
        stats={'candidate_count':b,'chunks':0,'public_projection_batches':0,'normal_matrix_batches':0,'current_factorizations':0,'current_solve_rhs':0,'normal_factorizations':0,'normal_solve_rhs':0,'normal_factor_dimension_sum':0,'info_trace_evaluations':0,'factorization_failures':0,'successful_units':0,'host_upload_bytes':0,'host_download_bytes':0,'stage_wall_s':{'incoming_residual_orthogonalization':prepwall},'explicit_workspace_bytes_peak':0,'branch':None,'device':self.device,'incoming_schur_factorizations':0,'core_reuse_solve_calls':0,'core_reuse_solve_rhs':0,'core_factorizations_during_evaluate':0,'normalization_policy':'declared incoming span residualized against V; rank d preserved or failed','core_prepare_statistics_nonadditive_snapshot':core.statistics,'full_J_materialized':False,'real_material_H_materialized':False,'physical_full_rhs':0}
        fullF=np.full((b,self.P,r+d,self.q),np.nan+0j) if return_coefficients else None
        start=time.perf_counter()
        if core.status!='STABLE_CORE':
            if valid:
                U=np.stack([np.concatenate([core.V,basis[i]],axis=1) for i in valid]);out=self.evaluate(U)
                steps[valid]=out['steps'];js[valid]=out['j_steps'];resids[valid]=out['normal_relative_residual'];info_trace[valid]=out['info_trace']
                for j,i in enumerate(valid):statuses[i]=out['status'][j];stability[i]=out['stability'][j]
                stats['direct_endpoint_fallback_statistics']=out['statistics'];stats['fallback_reason']=core.status;stats['core_factorizations_during_evaluate']=0
                stats['direct_endpoint_fallback_statistics_scope']='audit duplicate of totals/stages below; never charge both'
                for key in ['chunks','public_projection_batches','normal_matrix_batches','current_factorizations','current_solve_rhs','normal_factorizations','normal_solve_rhs','normal_factor_dimension_sum','info_trace_evaluations','factorization_failures','host_upload_bytes','host_download_bytes']:
                    stats[key]+=out['statistics'][key]
                stats['branch']=out['statistics']['branch'];stats['explicit_workspace_bytes_peak']=out['statistics']['explicit_workspace_bytes_peak']
                for key,value in out['statistics']['stage_wall_s'].items():stats['stage_wall_s'][key]=stats['stage_wall_s'].get(key,0.)+value
                # Optional F inspection can be reconstructed independently by tests;
                # fallback does not copy/download F just for this diagnostic switch.
            stats['path']='DIRECT_ENDPOINT_FALLBACK_CORE_UNSTABLE_OR_EMPTY'
        else:
            stats['path']='SHARED_CORE_EXACT_SCHUR'
            for first in range(0,len(valid),self.batch_size):
                ids=valid[first:first+self.batch_size];stats['chunks']+=1
                self._incoming_chunk(core,basis[ids],ids,steps,js,statuses,stability,resids,stats,fullF,info_trace)
        self._sync();stats['total_evaluate_wall_s']=time.perf_counter()-start+prepwall;stats['successful_units']=statuses.count('OK')
        return {'steps':steps,'j_steps':js,'status':statuses,'stability':stability,'normal_relative_residual':resids,'info_trace':info_trace,'statistics':stats,'incoming_basis':basis,'coefficients_F':fullF,'coefficient_diagnostic':'F returned only if explicitly requested; fallback F unavailable/NaN','admission':'parent contract review; no scientific gain certificate'}

    def _incoming_chunk(self,core,C,ids,steps,js,statuses,stability,resids,stats,fullF,info_trace):
        n,_,d=C.shape;r=core.V.shape[1];gpu=self.device!='cpu';t=getattr(self,'torch',None)
        def mark(label,start):
            self._sync();stats['stage_wall_s'][label]=stats['stage_wall_s'].get(label,0.)+time.perf_counter()-start
        start=time.perf_counter();c=t.as_tensor(C,device=self.tdevice,dtype=t.complex128) if gpu else C
        v=t.as_tensor(core.V.copy(),device=self.tdevice,dtype=t.complex128) if gpu else core.V
        if self.device=='cuda':stats['host_upload_bytes']+=C.nbytes+core.V.nbytes
        LC=self.L@c;CVc=self.S@c;E=v.conj().swapaxes(-1,-2)@LC;B=c.conj().swapaxes(-1,-2)@core.LV;D=c.conj().swapaxes(-1,-2)@LC
        packed_c=c.transpose(0,1).reshape(self.qc,n*d) if gpu else c.transpose(1,0,2).reshape(self.qc,n*d)
        PBc=(self.PBflat@packed_c.conj()).reshape(self.P,self.q,n,d)
        PBc=PBc.permute(2,0,3,1) if gpu else PBc.transpose(2,0,3,1)
        stats['public_projection_batches']+=1;mark('incoming_only_projections',start)
        start=time.perf_counter();rhs=E.transpose(0,1).reshape(r,n*d) if gpu else E.transpose(1,0,2).reshape(r,n*d)
        if gpu:X=t.linalg.lu_solve(*core.factor,rhs).reshape(r,n,d).transpose(0,1)
        else:X=la.lu_solve(core.factor,rhs).reshape(r,n,d).transpose(1,0,2)
        stats['core_reuse_solve_calls']+=1;stats['core_reuse_solve_rhs']+=n*d
        Gamma=D-B@X;N=PBc-B[:,None]@core.Fv[None]
        # Both Gamma and the full child block must be stable: do not infer one.
        if gpu:
            upper=t.cat([core.A.expand(n,-1,-1),E],dim=2);lower=t.cat([B,D],dim=2);A=t.cat([upper,lower],dim=1)
        else:
            upper=np.concatenate([np.broadcast_to(core.A,(n,r,r)),E],axis=2);lower=np.concatenate([B,D],axis=2);A=np.concatenate([upper,lower],axis=1)
        aa=self._host(A,stats);gg=self._host(Gamma,stats);keep=[];gamma_fallback=[]
        for j,i in enumerate(ids):
            sv=np.linalg.svd(aa[j],compute_uv=False);sg=np.linalg.svd(gg[j],compute_uv=False)
            rel=float(sv[-1]/sv[0]) if sv[0] else 0.;grel=float(sg[-1]/sg[0]) if sg[0] else 0.
            stability[i]={'sigma_min':float(sv[-1]),'sigma_max':float(sv[0]),'relative':rel,'condition':float(sv[0]/sv[-1]) if sv[-1] else None,'Gamma_relative':grel,'core_relative':core.stability['relative']}
            if rel<=TOL:statuses[i]='CURRENT_NEAR_SINGULAR'
            elif grel<=TOL:gamma_fallback.append(j)
            else:keep.append(j)
        mark('shared_core_solve_and_full_Gamma_stability',start)
        # A problematic Schur representation does not make a stable endpoint
        # infeasible. Preserve its charged screening, then use exact endpoint.
        if gamma_fallback:
            U=np.stack([np.concatenate([core.V,C[j]],axis=1) for j in gamma_fallback])
            fallback=self.evaluate(U)
            for jj,j in enumerate(gamma_fallback):
                i=ids[j];steps[i]=fallback['steps'][jj];js[i]=fallback['j_steps'][jj];resids[i]=fallback['normal_relative_residual'][jj];info_trace[i]=fallback['info_trace'][jj];statuses[i]=fallback['status'][jj]
                stability[i]['Schur_representation_fallback']='near-singular Gamma; complete endpoint solve'
            stats.setdefault('Gamma_fallback_audit_views',[]).append({'scope':'overlapping audit view; counters merged below, never charge twice','statistics':fallback['statistics']})
            for key in ['chunks','public_projection_batches','normal_matrix_batches','current_factorizations','current_solve_rhs','normal_factorizations','normal_solve_rhs','normal_factor_dimension_sum','info_trace_evaluations','factorization_failures','host_upload_bytes','host_download_bytes']:stats[key]+=fallback['statistics'][key]
            for key,value in fallback['statistics']['stage_wall_s'].items():stats['stage_wall_s'][key]=stats['stage_wall_s'].get(key,0.)+value
            stats['explicit_workspace_bytes_peak']=max(stats['explicit_workspace_bytes_peak'],fallback['statistics']['explicit_workspace_bytes_peak']);stats['branch']=fallback['statistics']['branch']
        if not keep:return
        ix=t.as_tensor(keep,device=self.tdevice,dtype=t.int64) if gpu else keep;X,CVc,Gamma,N=X[ix],CVc[ix],Gamma[ix],N[ix];accepted=[ids[j] for j in keep];n=len(accepted)
        start=time.perf_counter();stats['incoming_schur_factorizations']+=n;stats['current_factorizations']+=n;stats['current_solve_rhs']+=n*self.P*self.q
        rhs=N.permute(0,2,1,3).reshape(n,d,self.P*self.q) if gpu else N.transpose(0,2,1,3).reshape(n,d,self.P*self.q)
        if gpu:
            Fc,info=t.linalg.solve_ex(Gamma,rhs,check_errors=False);ok=self._host(info,stats)==0;Fc=Fc.reshape(n,d,self.P,self.q).transpose(1,2)
        else:
            Fc=np.full((n,self.P,d,self.q),np.nan+0j);ok=np.ones(n,bool)
            for j in range(n):
                try:Fc[j]=np.linalg.solve(Gamma[j],rhs[j]).reshape(d,self.P,self.q).transpose(1,0,2)
                except np.linalg.LinAlgError:ok[j]=False
        for j in np.where(~ok)[0]:statuses[accepted[j]]='SCHUR_SOLVE_FAILED';stats['factorization_failures']+=1
        good=np.where(ok)[0].tolist()
        if not good:return
        ix=t.as_tensor(good,device=self.tdevice,dtype=t.int64) if gpu else good;Fc,X,CVc=Fc[ix],X[ix],CVc[ix];accepted=[accepted[j] for j in good];n=len(accepted)
        top=core.Fv[None]-X[:,None]@Fc
        F=t.cat([top,Fc],dim=2) if gpu else np.concatenate([top,Fc],axis=2)
        CV=t.cat([core.CV.expand(n,-1,-1),CVc],dim=2) if gpu else np.concatenate([np.broadcast_to(core.CV,(n,*core.CV.shape)),CVc],axis=2)
        if fullF is not None:fullF[accepted]=self._host(F,stats)
        mark('batched_Schur_child_coefficients',start)
        bytes_=sum(a.numel()*a.element_size() if gpu else a.nbytes for a in [LC,PBc,X,Gamma,N,Fc,A])
        self._material_finish(F,CV,accepted,steps,js,statuses,resids,stats,info_trace=info_trace,base_workspace_bytes=bytes_)
