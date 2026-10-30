"""Exact isometric workspace for existing current selectors, not a solver.

Only paid public L/S/B projections enter. Physical material coordinates remain
unchanged. Never build a new Fourier bank from the coefficient context.
"""
from dataclasses import dataclass
import hashlib,time
import numpy as np
from scipy import linalg as la
from operators import SelectorContext
from fast_contractions import prepare_projection_layout, project_B as fast_project_B
from selectors_current import CandidatePool,SelectionResult,orth

CURRENT_SNAPSHOT_KEYS=('raw_primal_snapshots','raw_dual_snapshots','balanced_PD_snapshots')


def _charge(ledger,label,X,callback,columns=None):
    ledger.add('coordinate_'+label+'_calls')
    ledger.add('coordinate_'+label+'_columns',columns if columns is not None else (1 if X.ndim==1 else X.shape[-1]))
    start=time.perf_counter()
    try:return callback()
    finally:ledger.add('coordinate_'+label+'_wall_s',time.perf_counter()-start)


def _unit_columns(X):
    X=np.asarray(X,complex)
    if X.ndim!=2 or not np.isfinite(X).all():raise ValueError('finite current matrix required')
    norms=la.norm(X,axis=0);keep=norms>0
    return X[:,keep]/norms[keep]


@dataclass
class SelectorWorkspace:
    ctx:SelectorContext
    pool:CandidatePool
    anchor:object
    Q:np.ndarray
    metadata:dict

    def lift_result(self,result):
        if not isinstance(result,SelectionResult):raise TypeError('SelectionResult required')
        C=np.asarray(result.U,complex)
        if C.ndim!=2 or C.shape[0]!=self.Q.shape[1]:raise ValueError('coefficient result dimension mismatch')
        start=time.perf_counter();physical=self.Q@C
        self.ctx.ledger.add('coordinate_lift_calls');self.ctx.ledger.add('coordinate_lift_columns',C.shape[1])
        elapsed=time.perf_counter()-start;self.ctx.ledger.add('coordinate_lift_wall_s',elapsed)
        metadata=dict(result.metadata)
        metadata.update(coordinate_U=C.copy(),physical_embedding_key=self.metadata['physical_embedding_key'],
            coordinate_storage='lossless complex Q and Ucoeff; NPZ only; physical U=Q@Ucoeff',
            coordinate_workspace_cost=self.metadata,standalone_coordinate_setup_wall_charge_s=self.metadata['setup_wall_s'],
            physical_current_rows=self.Q.shape[0],coordinate_current_rows=self.Q.shape[1],physical_lift_wall_s=elapsed)
        return SelectionResult(physical,None if result.step is None else np.asarray(result.step).copy(),
                               result.status,result.selector_id,result.k,metadata)


def prepare_selector_workspace(ctx,pool,anchor,*,chunk_size=16,embedding_key=None):
    """Return workspace(ctx,pool,anchor,Q,metadata) using original public actions.

    Setup must be paid once in shared totals, and in full for EVERY standalone
    selector. Use already constructed pool/anchor; do not regenerate candidates
    in coefficient coordinates. Random seeds now refer to this implementation
    version; the uniform complex random-space distribution is invariant.
    """
    if chunk_size<1:raise ValueError('positive block size required')
    if getattr(anchor,'ctx',None) is not ctx:raise ValueError('same physical frozen context anchor required')
    n=ctx.n_current
    if pool.vectors.shape[0]!=n:raise ValueError('physical pool dimension mismatch')
    before=ctx.records();start=time.perf_counter()
    banks={'pool':pool.vectors,'anchor':anchor.U,'current_forward':np.asarray(ctx.current_forward).T}
    for key in CURRENT_SNAPSHOT_KEYS:
        if key in pool.metadata:banks[key]=np.asarray(pool.metadata[key],complex)
    if any(X.ndim!=2 or X.shape[0]!=n for X in banks.values()):raise ValueError('physical current bank shape mismatch')
    with ctx.phase('selector_coordinate_workspace_setup'):
        qstart=time.perf_counter();Q=orth(np.column_stack([_unit_columns(X) for X in banks.values()]))
        ctx.ledger.add('coordinate_embedding_builds');ctx.ledger.add('coordinate_embedding_rank',Q.shape[1])
        ctx.ledger.add('coordinate_embedding_wall_s',time.perf_counter()-qstart)
        errors={}
        for key,X in banks.items():
            C=Q.conj().T@X;norms=la.norm(X,axis=0);nonzero=norms>0
            errors[key]=float(np.max(la.norm(X-Q@C,axis=0)[nonzero]/norms[nonzero])) if np.any(nonzero) else 0.
        if max(errors.values())>1e-9:raise ValueError('workspace lost an exposed current direction')
        compressed_L=[];compressed_S=[];compressed_B=[]
        for offset in range(0,Q.shape[1],chunk_size):
            block=Q[:,offset:offset+chunk_size]
            compressed_L.append(Q.conj().T@ctx.apply_L(block))
            compressed_S.append(ctx.apply_S(block));compressed_B.append(ctx.project_B(block))
            ctx.ledger.add('coordinate_setup_paid_L_columns',block.shape[1])
            ctx.ledger.add('coordinate_setup_paid_S_columns',block.shape[1])
            ctx.ledger.add('coordinate_setup_paid_B_projection_columns',block.shape[1])
            ctx.ledger.add('coordinate_setup_physical_blocks')
        r=Q.shape[1]
        L=np.column_stack(compressed_L) if r else np.empty((0,0),complex)
        S=np.column_stack(compressed_S) if r else np.empty((ctx.m,0),complex)
        PB=np.concatenate(compressed_B,axis=1) if r else np.empty((ctx.P,0,ctx.q),complex)
        PB_layout=prepare_projection_layout(PB)
        ctx.ledger.add('coordinate_PB_layout_builds')
        ctx.ledger.add('coordinate_PB_layout_wall_s',PB_layout.metadata['wall_s'])
        ctx.ledger.add('coordinate_PB_layout_copy_bytes',PB_layout.metadata['persistent_copy_bytes'])
    cctx=SelectorContext()
    for key in ('dim_material','P','m','q','lam','ell','r','tangent','ledger'):
        setattr(cctx,key,getattr(ctx,key))
    cctx.n_current=r;cctx.current_metric='Euclidean complex coefficients of isometric physical Q'
    # No physical coordinate points exist for coefficient axes. Labels remain in
    # the existing pool; this empty sentinel prevents accidental Fourier rebuild.
    cctx.points=np.empty((0,3),float);cctx.points.setflags(write=False)
    cctx.current_forward=(Q.conj().T@np.asarray(ctx.current_forward).T).T
    cctx._L=lambda C:_charge(cctx.ledger,'L',C,lambda:L@C)
    cctx._LH=lambda C:_charge(cctx.ledger,'L_adjoint',C,lambda:L.conj().T@C)
    cctx._S=lambda C:_charge(cctx.ledger,'S',C,lambda:S@C)
    cctx._B=lambda C:_charge(cctx.ledger,'B_projection',C,lambda:fast_project_B(PB_layout,C))
    def injection(Z):
        z=Z[:cctx.q]+1j*Z[cctx.q:];one=z.ndim==1
        K=np.einsum('pnq,qb->pnb',PB,z[:,None] if one else z)
        return K[:,:,0] if one else K
    def injection_adjoint(K):
        one=K.ndim==2;z=np.einsum('pnq,pnb->qb',PB.conj(),K[:,:,None] if one else K)
        Z=np.vstack([z.real,z.imag]);return Z[:,0] if one else Z
    cctx._Bfull=lambda Z:_charge(cctx.ledger,'B',Z,lambda:injection(Z))
    cctx._Badj=lambda K:_charge(cctx.ledger,'B_adjoint',K,lambda:injection_adjoint(K),columns=1 if K.ndim==2 else K.shape[2])
    cctx._Sadj=lambda W:_charge(cctx.ledger,'S_adjoint',W,lambda:S.conj().T@W)
    V=Q.conj().T@pool.vectors;V.setflags(write=False)
    metadata=dict(pool.metadata)
    for key in CURRENT_SNAPSHOT_KEYS:
        if key in metadata:metadata[key]=Q.conj().T@np.asarray(metadata[key])
    metadata.update(current_array_coordinates='isometric Q coefficients; material scenarios unchanged',
                    physical_current_rows=n,coordinate_current_rows=r,points_nonphysical=True)
    cpool=CandidatePool(V,pool.labels,pool.families,metadata)
    receiver=V[:,np.asarray(pool.families)=='receiver'];internal=V[:,np.asarray(pool.families)=='internal']
    cctx._receiver=lambda count:receiver[:,:count]
    cctx._internal=lambda count,seed:(internal[:,:count],{'label':'existing projected physical internal bank; no regeneration','seed':seed})
    with cctx.phase('selector_coordinate_anchor_remake'):
        cctx.ledger.add('coordinate_anchor_remakes');canchor=cctx.make_model(Q.conj().T@anchor.U)
    key=embedding_key or 'Q_sha256_'+hashlib.sha256(np.ascontiguousarray(Q).tobytes()).hexdigest()
    meta={'physical_embedding_key':key,'physical_current_rows':n,'coordinate_current_rows':r,
          'setup_wall_s':time.perf_counter()-start,'before':before,'after':ctx.records(),
          'block_size':chunk_size,'bank_roundtrip_relative_errors':errors,
          'snapshot_keys_mapped':[key for key in CURRENT_SNAPSHOT_KEYS if key in banks],'physical_points_count':len(ctx.points),
          'coefficient_points':'empty nonphysical sentinel; NEVER generate Fourier/candidates after conversion',
          'standalone_charge_rule':'EVERY from-scratch selector pays full anchor/pool/workspace setup; shared total pays once',
          'random_seed_rule':'seeded draw basis changed before official results; uniform complex subspace law unchanged',
          'scope':'exact Range(Q) Galerkin/Schur actions; Q*LQ is a projected operator, not full inverse',
          'PB_cached_layout':PB_layout.metadata,
          'new_solver':False,'full_J_H_truth_access':False}
    Q.setflags(write=False)
    return SelectorWorkspace(cctx,cpool,canchor,Q,meta)


# Explicit alternate noun matching caller preference; both return the same object.
build_selector_workspace=prepare_selector_workspace
