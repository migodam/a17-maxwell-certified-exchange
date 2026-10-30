"""Exact batched endpoints on an already-paid coefficient current space.

No Maxwell actions, full data Jacobian, real material Hessian, candidate search,
Schur downdate, or gain certificate. PB must already be in physical material
coordinates and data-whitened consistently with S/r. Shared material c=x+i*y;
data layout per illumination is [Re(m), Im(m)]. Outputs for failed units are
NaN and unusable; status explicitly reports the failure (never adds jitter).
"""
import time
import numpy as np

TOL = 1e-10


class EndpointBatch:
    def __init__(self, L, S, PB, r, ell, lam, *, device='cpu',
                 branch='auto', batch_size=16):
        if device not in ('cpu', 'cuda', 'torch_cpu'):
            raise ValueError('device must be cpu, cuda or CPU-only torch_cpu validation')
        if branch not in ('auto', 'direct', 'woodbury'):
            raise ValueError('invalid normal branch')
        if not isinstance(batch_size,int) or batch_size<1: raise ValueError('batch_size positive integer')
        if np.iscomplexobj(r) or np.iscomplexobj(ell): raise ValueError('r/ell must be real packed')
        arrays = [np.asarray(a,dtype=np.complex128) for a in (L,S,PB)]
        L,S,PB=arrays; r=np.asarray(r,dtype=np.float64);ell=np.asarray(ell,dtype=np.float64)
        if L.ndim!=2 or L.shape[0]!=L.shape[1] or S.ndim!=2 or S.shape[1]!=L.shape[0]:raise ValueError('L/S shape')
        if PB.ndim!=3 or PB.shape[1]!=L.shape[0]:raise ValueError('PB=(P,qc,qmaterial)')
        self.P,self.qc,self.q=PB.shape;self.m=S.shape[0]
        if branch=='direct' and self.q>256:
            raise ValueError('direct complex normal is restricted to small material spaces(q<=256); large voxel uses Woodbury')
        if r.shape!=(self.P*2*self.m,) or ell.shape!=(2*self.q,):raise ValueError('r/ell packing shape')
        if min(self.P,self.q,self.qc,self.m)<1 or not all(np.isfinite(a).all() for a in [*arrays,r,ell]):raise ValueError('nonfinite/empty common inputs')
        if not np.isscalar(lam) or not np.isfinite(lam) or lam<=0:raise ValueError('positive finite lambda required')
        self.lam=float(lam);self.branch=branch;self.batch_size=batch_size;self.device=device
        self.common_bytes=sum(a.nbytes for a in [*arrays,r,ell]);self.setup_wall_s=0.;self.setup_upload_bytes=0
        started=time.perf_counter()
        # One packed PB copy, shared by all candidates and batches.
        layout=np.ascontiguousarray(PB.transpose(0,2,1)).reshape(self.P*self.q,self.qc)
        self.layout_copy_bytes=0 if np.shares_memory(layout,PB) else layout.nbytes
        if device=='cpu':self.L,self.S,self.PBflat,self.r,self.ell=L.copy(),S.copy(),layout.copy(),r.copy(),ell.copy()
        else:
            import torch
            self.torch=torch;self.tdevice='cuda' if device=='cuda' else 'cpu'
            if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
            self.L,self.S,self.PBflat=[torch.as_tensor(a.copy(),device=self.tdevice,dtype=torch.complex128) for a in (L,S,layout)]
            self.r=torch.as_tensor(r.copy(),device=self.tdevice,dtype=torch.float64);self.ell=torch.as_tensor(ell.copy(),device=self.tdevice,dtype=torch.float64)
            self.setup_upload_bytes=self.common_bytes if device=='cuda' else 0
            self._sync()
        self.setup_wall_s=time.perf_counter()-started

    def _sync(self):
        if self.device=='cuda':self.torch.cuda.synchronize()

    def evaluate(self, coefficients):
        U=np.asarray(coefficients,dtype=np.complex128)
        if U.ndim!=3 or U.shape[1]!=self.qc:raise ValueError('coefficients=(batch,qc,k)')
        b,_,k=U.shape
        if k>self.qc:raise ValueError('k exceeds coefficient dimension')
        steps=np.full((b,2*self.q),np.nan);jsteps=np.full((b,self.P*2*self.m),np.nan)
        statuses=['PENDING']*b;stability=[None]*b;residuals=np.full(b,np.nan)
        info_trace=np.full(b,np.nan)
        stats={'candidate_count':b,'chunks':0,'public_projection_batches':0,'normal_matrix_batches':0,
               'current_factorizations':0,'current_solve_rhs':0,'normal_factorizations':0,
               'normal_solve_rhs':0,'normal_factor_dimension_sum':0,'info_trace_evaluations':0,
               'factorization_failures':0,'successful_units':0,'host_upload_bytes':0,'host_download_bytes':0,
               'setup_upload_bytes':self.setup_upload_bytes,'setup_wall_s':self.setup_wall_s,
               'paid_common_input_bytes':self.common_bytes,'PB_layout_copy_bytes':self.layout_copy_bytes,
               'stage_wall_s':{},'explicit_workspace_bytes_peak':0,
               'bytes_scope':'explicit array estimates; not allocator peak; overlap not summed as job occupation',
               'physical_full_rhs':0,'full_J_materialized':False,'real_material_H_materialized':False,
               'branch':None,'device':self.device}
        begin=time.perf_counter()
        for first in range(0,b,self.batch_size):
            ids=list(range(first,min(first+self.batch_size,b)));keep=[]
            for i in ids:
                if not np.isfinite(U[i]).all():statuses[i]='NONFINITE_BASIS';continue
                if np.linalg.norm(U[i].conj().T@U[i]-np.eye(k))>TOL:statuses[i]='NONORTHONORMAL_BASIS';continue
                keep.append(i)
            if not keep:continue
            stats['chunks']+=1
            self._chunk(U[keep],keep,steps,jsteps,statuses,stability,residuals,stats,info_trace)
        self._sync();stats['total_evaluate_wall_s']=time.perf_counter()-begin
        stats['successful_units']=statuses.count('OK')
        return {'steps':steps,'j_steps':jsteps,'status':statuses,'stability':stability,
                'normal_relative_residual':residuals,'info_trace':info_trace,'statistics':stats,
                'packing':'material [Re(q),Im(q)], data per P [Re(m),Im(m)]',
                'admission':'implementation results only; not full-quadratic gain certificate or scientific gate'}

    def _chunk(self,U,ids,steps,jsteps,statuses,stability,residuals,stats,info_trace):
        gpu=self.device!='cpu'; t=getattr(self,'torch',None);k=U.shape[-1];n=len(ids)
        def mark(name,start):
            self._sync();stats['stage_wall_s'][name]=stats['stage_wall_s'].get(name,0.)+time.perf_counter()-start
        def host(x):
            if gpu:
                y=x.detach().cpu().numpy();stats['host_download_bytes']+=y.nbytes if self.device=='cuda' else 0;return y
            return x
        started=time.perf_counter()
        if gpu:
            u=t.as_tensor(U,device=self.tdevice,dtype=t.complex128);stats['host_upload_bytes']+=U.nbytes if self.device=='cuda' else 0
        else:u=U
        uh=u.conj().swapaxes(-1,-2)
        A=uh@(self.L@u);CV=self.S@u
        packed_u=u.transpose(0,1).reshape(self.qc,n*k) if gpu else u.transpose(1,0,2).reshape(self.qc,n*k)
        raw=self.PBflat@packed_u.conj()
        PB=raw.reshape(self.P,self.q,n,k).permute(2,0,3,1) if gpu else raw.reshape(self.P,self.q,n,k).transpose(2,0,3,1)
        stats['public_projection_batches']+=1;mark('shared_public_projections',started)
        started=time.perf_counter();AA=host(A);valid=[]
        for j,i in enumerate(ids):
            try:
                sv=np.linalg.svd(AA[j],compute_uv=False) if k else np.empty(0)
                ratio=float(sv[-1]/sv[0]) if k and sv[0] else (0. if k else None)
                stability[i]={'sigma_min':float(sv[-1]) if k else None,'sigma_max':float(sv[0]) if k else None,'relative':ratio,'condition':float(sv[0]/sv[-1]) if k and sv[-1] else (1. if not k else None)}
                if k and ratio<=TOL:statuses[i]='CURRENT_NEAR_SINGULAR';continue
                valid.append(j)
            except np.linalg.LinAlgError:statuses[i]='CURRENT_SVD_FAILED'
        mark('current_stability',started)
        if not valid:return
        ix=t.as_tensor(valid,device=self.tdevice,dtype=t.int64) if gpu else valid
        A,CV,PB=A[ix],CV[ix],PB[ix];accepted=[ids[j] for j in valid];n=len(accepted)
        started=time.perf_counter();stats['current_factorizations']+=n if k else 0;stats['current_solve_rhs']+=n*self.P*self.q if k else 0
        if k:
            packed=PB.permute(0,2,1,3).reshape(n,k,self.P*self.q) if gpu else PB.transpose(0,2,1,3).reshape(n,k,self.P*self.q)
            if gpu:
                F,info=t.linalg.solve_ex(A,packed,check_errors=False);good=host(info)==0
                F=F.reshape(n,k,self.P,self.q).transpose(1,2)
            else:
                F=np.full((n,self.P,k,self.q),np.nan+0j);good=np.ones(n,bool)
                for j in range(n):
                    try:F[j]=np.linalg.solve(A[j],packed[j]).reshape(k,self.P,self.q).transpose(1,0,2)
                    except np.linalg.LinAlgError:good[j]=False
            failed=np.where(~good)[0]
            for j in failed:statuses[accepted[j]]='CURRENT_SOLVE_FAILED';stats['factorization_failures']+=1
            valid2=np.where(good)[0].tolist()
            if not valid2:return
            ix=t.as_tensor(valid2,device=self.tdevice,dtype=t.int64) if gpu else valid2
            F,CV=F[ix],CV[ix];accepted=[accepted[j] for j in valid2];n=len(accepted)
        else:F=t.zeros((n,self.P,0,self.q),device=self.tdevice,dtype=t.complex128) if gpu else np.zeros((n,self.P,0,self.q),complex)
        mark('current_full_coupling_solve',started)
        self._material_finish(F,CV,accepted,steps,jsteps,statuses,residuals,stats,
                              info_trace=info_trace,base_workspace_bytes=U.nbytes+AA.nbytes+(PB.numel()*PB.element_size() if gpu else PB.nbytes))

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
        output=host(combined)
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
