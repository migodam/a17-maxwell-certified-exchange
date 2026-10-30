"""Original A17 CUDA batches with same-system CUDA FP64 refinement.

New named runtime adapter. The original CPU adapter and historical kernels are
untouched. Only finite original NORMAL_RESIDUAL_FAILED rows receive at most
three corrections using the same compressed SPD factor, lambda and 1e-10 gate.
No jitter, CPU fallback, new selector or additional validation is introduced.
"""
import time
import numpy as np
from refined_endpoint_batch import RefinedCoreEndpointBatch,CoreEndpointBatch,TOL


class GpuRefinedCoreEndpointBatch(RefinedCoreEndpointBatch):
    """CUDA complex128 numerical path; inherit the existing audited APIs."""
    def __init__(self,*args,device='cuda',**kwargs):
        if device!='cuda':raise ValueError('GpuRefinedCoreEndpointBatch requires cuda')
        CoreEndpointBatch.__init__(self,*args,device='cuda',**kwargs)
        self._refinement_audit_log=[]
        self._refinement_depth=0

    def _audited_evaluate(self,operation,*args,**kwargs):
        result=super()._audited_evaluate(operation,*args,**kwargs)
        result['statistics']['refinement']['implementation']='CUDA complex128 / float64, same SPD inverse; maximum 3 corrections'
        return result

    def _material_finish(self,F,CV,accepted,steps,jsteps,statuses,residuals,stats,*,info_trace=None,base_workspace_bytes=0):
        gpu=self.device!='cpu';t=getattr(self,'torch',None);k=F.shape[-2];n=len(accepted)
        def mark(name,start):
            self._sync();stats['stage_wall_s'][name]=stats['stage_wall_s'].get(name,0.)+time.perf_counter()-start
        def host(x):
            if gpu:
                y=x.detach().cpu().numpy();stats['host_download_bytes']+=y.nbytes if self.device=='cuda' else 0;return y
            return x
        started=time.perf_counter()
        Rc=t.linalg.qr(CV,mode='reduced')[1] if gpu else np.linalg.qr(CV,mode='reduced')[1]
        rr=Rc.shape[1];Z=(Rc[:,None]@F).reshape(n,self.P*rr,self.q)
        # Trace of the real data normal operator; excludes lambda regularization.
        trace=2*(Z.abs().square().sum(dim=(-2,-1)) if gpu else np.sum(np.abs(Z)**2,axis=(-2,-1)))
        stats['info_trace_evaluations']=stats.get('info_trace_evaluations',0)+n
        zh=Z.conj().swapaxes(-1,-2)
        branch=self.branch
        if branch=='auto':branch='direct' if self.q<=256 else 'woodbury'
        # Zero-current endpoint has lambda-only material solve.
        if k==0:branch='lambda_only'
        stats['branch']=branch
        if branch!='lambda_only':
            H=zh@Z if branch=='direct' else Z@zh
            d=H.shape[-1]
            if gpu:H.diagonal(dim1=-2,dim2=-1).add_(self.lam)
            else:H[:,np.arange(d),np.arange(d)]+=self.lam
            stats['normal_matrix_batches']+=1;stats['normal_factorizations']+=n
            stats['normal_factor_dimension_sum']+=n*d
            if gpu:chol,info=t.linalg.cholesky_ex(H,check_errors=False);ok=host(info)==0
            else:
                chol=np.empty_like(H);ok=np.ones(n,bool)
                for j in range(n):
                    try:chol[j]=np.linalg.cholesky(H[j])
                    except np.linalg.LinAlgError:ok[j]=False
            for j in np.where(~ok)[0]:statuses[accepted[j]]='NORMAL_FACTORIZATION_FAILED';stats['factorization_failures']+=1
        else:ok=np.ones(n,bool)
        mark('compressed_normal_batch',started)
        started=time.perf_counter();valid3=np.where(ok)[0].tolist()
        if not valid3:return
        ix=t.as_tensor(valid3,device=self.tdevice,dtype=t.int64) if gpu else valid3
        F,CV,Z,zh,trace=F[ix],CV[ix],Z[ix],zh[ix],trace[ix];accepted=[accepted[j] for j in valid3];n=len(accepted)
        wc=self.r.reshape(self.P,2*self.m);wc=wc[:,:self.m]+1j*wc[:,self.m:]
        e=self.ell[:self.q]+1j*self.ell[self.q:]
        projected=CV.conj().swapaxes(-1,-2)[:,None]@wc[None,:,:,None]
        g=(F.conj().swapaxes(-1,-2)@projected).sum(axis=1).reshape(n,self.q)+e
        if branch=='lambda_only':step=-g/self.lam
        else:
            chol=chol[ix]
            rhs=g if branch=='direct' else (Z@g[:,:,None]).reshape(n,-1)
            stats['normal_solve_rhs']+=n
            if gpu:sol=t.cholesky_solve(rhs[:,:,None],chol).squeeze(-1)
            else:sol=np.stack([np.linalg.solve(c.conj().T,np.linalg.solve(c,v)) for c,v in zip(chol,rhs)])
            step=-sol if branch=='direct' else -(g-(zh@sol[:,:,None]).squeeze(-1))/self.lam
        normal=(zh@(Z@step[:,:,None])).squeeze(-1)+self.lam*step+g
        if gpu:rel=t.linalg.vector_norm(normal,dim=1)/t.linalg.vector_norm(g,dim=1).clamp_min(1e-300)
        else:rel=np.linalg.norm(normal,axis=1)/np.maximum(np.linalg.norm(g,axis=1),1e-300)
        y=(CV[:,None]@(F@step[:,None,:,None])).squeeze(-1)
        # One whole batch transfer for all final material/data arrays, never F/Z.
        packed_y=t.cat([y.real,y.imag],dim=2).reshape(n,-1) if gpu else np.concatenate([y.real,y.imag],axis=2).reshape(n,-1)
        combined=t.cat([step.real,step.imag,packed_y,rel[:,None],trace[:,None]],dim=1) if gpu else np.concatenate([step.real,step.imag,packed_y,rel[:,None],trace[:,None]],axis=1)
        # Only small eligibility flags and raw residuals leave CUDA here.
        raw_rel=host(rel).copy()
        eligible=host(t.isfinite(combined).all(dim=1)&(rel>TOL))
        iterations=np.zeros(n,dtype=int)
        active=np.flatnonzero(eligible)
        for _ in range(3):
            if not len(active):break
            ax=t.as_tensor(active,device=self.tdevice,dtype=t.int64)
            defect=(zh[ax]@(Z[ax]@step[ax,:,None])).squeeze(-1)+self.lam*step[ax]+g[ax]
            rhs=defect if branch=='direct' else (Z[ax]@defect[:,:,None]).reshape(len(active),-1)
            if branch=='lambda_only':correction=-defect/self.lam
            else:
                sol=t.cholesky_solve(rhs[:,:,None],chol[ax]).squeeze(-1)
                correction=-sol if branch=='direct' else -(defect-(zh[ax]@sol[:,:,None]).squeeze(-1))/self.lam
                stats['normal_solve_rhs']+=len(active)
            step[ax]=step[ax]+correction
            iterations[active]+=1
            defect=(zh[ax]@(Z[ax]@step[ax,:,None])).squeeze(-1)+self.lam*step[ax]+g[ax]
            rel[ax]=t.linalg.vector_norm(defect,dim=1)/t.linalg.vector_norm(g[ax],dim=1).clamp_min(1e-300)
            active=active[host(t.isfinite(rel[ax])&(rel[ax]>TOL))]
        if np.any(eligible):
            repair=np.flatnonzero(eligible)
            rx=t.as_tensor(repair,device=self.tdevice,dtype=t.int64)
            y[rx]=(CV[rx,None]@(F[rx]@step[rx,None,:,None])).squeeze(-1)
            trace[rx]=2*Z[rx].abs().square().sum(dim=(-2,-1))
            stats['info_trace_evaluations']+=len(repair)
        # Final material/data output is packed on CUDA, then transferred once.
        packed_y=t.cat([y.real,y.imag],dim=2).reshape(n,-1)
        combined=t.cat([step.real,step.imag,packed_y,rel[:,None],trace[:,None]],dim=1)
        output=host(combined)
        audit={'accepted_indices':list(accepted),'index_scope':'local evaluate or evaluate_incoming invocation; nested fallback calls have local indices',
               'branch':branch,'raw_normal_relative_residual':raw_rel.tolist(),
               'final_normal_relative_residual':output[:,-2].tolist(),
               'raw_residual_failed_indices':[accepted[j] for j in np.flatnonzero(eligible)],
               'refinement_iterations':iterations.tolist(),
               'attempted_units':int(np.count_nonzero(iterations)),
               'correction_steps':int(iterations.sum()),
               'recovered_units':int(np.sum(eligible&np.isfinite(output).all(axis=1)&(output[:,-2]<=TOL))),
               'remaining_residual_failed_units':int(np.sum(eligible&np.isfinite(output).all(axis=1)&(output[:,-2]>TOL))),
               'refinement_nonfinite_units':int(np.sum(eligible&~np.isfinite(output).all(axis=1)))}
        self._refinement_audit_log.append(audit)
        for j,i in enumerate(accepted):
            residuals[i]=output[j,-2]
            if not np.isfinite(output[j]).all():statuses[i]='NONFINITE_STEP';continue
            if output[j,-2]>TOL:statuses[i]='NORMAL_RESIDUAL_FAILED';continue
            steps[i]=output[j,:2*self.q];jsteps[i]=output[j,2*self.q:-2];statuses[i]='OK'
            if info_trace is not None:info_trace[i]=output[j,-1]
        sizes=[base_workspace_bytes]+[x.numel()*x.element_size() if gpu else x.nbytes for x in [CV,F,Z,zh,step,y,trace]]
        if branch!='lambda_only':sizes.extend([H.numel()*H.element_size()+chol.numel()*chol.element_size() if gpu else H.nbytes+chol.nbytes])
        stats['explicit_workspace_bytes_peak']=max(stats['explicit_workspace_bytes_peak'],sum(sizes))
        mark('material_step_and_batched_return',started)
