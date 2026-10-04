"""Physical previous-space atoms; fresh current-state public workspace, v1."""
import time
import numpy as np

VERSION='a17.efficiency.previous_space_workspace.v1'

def public_workspace_with_previous(ctx,U_previous,*,origin_iteration,k=8):
    """Original anchor/pool; append history without deleting/reordering atoms.

    Invalid rank/layout/orthogonality is a recorded receiver reset. Anchor/pool
    acquisition, history inspection, failed representation and reset work are
    charged by their measured outer acquisition interval. No stale maps/actions.
    """
    from selectors_current import SelectorConfig,CandidatePool,build_independent_anchor,build_candidates
    from selector_coordinates import prepare_selector_workspace
    from maxwell_state import array_hash
    started=time.perf_counter();cfg=SelectorConfig(seed=20260930,pool_count=32,residual_up_count=16,random_pool_count=10)
    anchor,anchor_meta=build_independent_anchor(ctx,64,seed=cfg.seed)
    t=time.perf_counter();original_pool=build_candidates(ctx,anchor,cfg);pool_wall=time.perf_counter()-t
    nbase=original_pool.vectors.shape[1];t=time.perf_counter()
    history=dict(version=VERSION,origin_iteration=int(origin_iteration),requested_rank=k,base_atom_count=nbase,
      base_physical_pool_hash=array_hash(original_pool.vectors),
      history_appended=False,receiver_reset=False,reset_reason=None,stale_Jx_Jd_or_coeff_reuse=False,
      endpoint_and_Jx_reacquisition_required=True,failed_workspace_wall_s=0.)
    usable=True
    if not isinstance(U_previous,np.ndarray) or U_previous.dtype!=np.complex128 or U_previous.shape!=(ctx.n_current,k) or not np.isfinite(U_previous).all():
        usable=False;history['reset_reason']='INFEASIBLE_PREVIOUS_PHYSICAL_LAYOUT_OR_DTYPE'
    else:
        sv=np.linalg.svd(U_previous,compute_uv=False)
        history['previous_singular_values']=sv.tolist()
        history['previous_orthogonality_defect']=float(np.linalg.norm(U_previous.conj().T@U_previous-np.eye(k)))
        if sv[-1]<=1e-10*sv[0] or history['previous_orthogonality_defect']>1e-9:
            usable=False;history['reset_reason']='INFEASIBLE_PREVIOUS_RANK_OR_ORTHOGONALITY'
        history['physical_previous_space_hash']=array_hash(U_previous)
    history['validation_wall_s']=time.perf_counter()-t
    selected=None;pool=original_pool
    if usable:
        vectors=np.column_stack([original_pool.vectors,U_previous])
        labels=list(original_pool.labels)+[f'previous_verified_outer_{origin_iteration}_column_{j}' for j in range(k)]
        families=list(original_pool.families)+['previous_verified']*k
        pool=CandidatePool(vectors,labels,families,dict(original_pool.metadata))
        selected=list(range(nbase,nbase+k));history['history_appended']=True
    t=time.perf_counter()
    try:w=prepare_selector_workspace(ctx,pool,anchor,chunk_size=16)
    except ValueError as exc:
        if not usable:raise
        history['failed_workspace_wall_s']=time.perf_counter()-t
        history['failed_workspace_error']=str(exc);history['reset_reason']='INFEASIBLE_PREVIOUS_WORKSPACE_REPRESENTATION'
        # The failed helper's original action/ledger work stays charged; reset
        # rebuilds current maps from the unchanged original anchor/pool.
        w=prepare_selector_workspace(ctx,original_pool,anchor,chunk_size=16)
        selected=None;pool=original_pool;history['history_appended']=False
    if selected is None:history['receiver_reset']=True
    cctx=w.ctx;eye=np.eye(cctx.n_current,dtype=np.complex128)
    L=cctx.apply_L(eye);S=cctx.apply_S(eye);PB=cctx.project_B(eye)
    history.update(actual_atom_count=w.pool.vectors.shape[1],initial_ids=selected,
      current_embedding_rank=w.Q.shape[1],fresh_current_maps=True,
      full_original_pool_preserved=True,anchor_rank=64,incoming_rule='original balanced_ids count12')
    return dict(workspace=w,L=L,S=S,PB=PB,anchor_meta=anchor_meta,pool_wall_s=pool_wall,
      workspace_wall_s=time.perf_counter()-started,pool_hash=array_hash(pool.vectors),
      full_J_H_truth_access=False,initial_ids=selected,reuse_history=history)
