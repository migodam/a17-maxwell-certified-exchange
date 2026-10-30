"""Capability-separated current selectors on frozen real-material ROM operators.

No full material step, true material, held-out data, or full current inverse is
read by cheap selectors. Complete references enter only explicit probe/oracle
interfaces. Models preserve exact complex current rank and all illuminations
share the same spatial basis. This module adds selectors, not a solver.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from contextlib import nullcontext
from math import comb
import time

import numpy as np
from scipy import linalg as la
from fast_contractions import s5_frobenius

BUDGETS=(4,8,12,16,24,32)
TOL=1e-10


@dataclass(frozen=True)
class SelectorConfig:
    seed:int=20260930
    pool_count:int=32
    residual_up_count:int=16
    random_pool_count:int=10
    residual_probe_count:int=4
    task_probe_count:int=16
    pair_top:int=8
    pair_always:bool=False
    swaps_max_pairs:int=16
    score_zero_tol:float=1e-12
    random_attempts:int=8
    unit_test_allow_equal_anchor:bool=False


@dataclass(frozen=True)
class CandidatePool:
    vectors:np.ndarray
    labels:tuple[str,...]
    families:tuple[str,...]
    metadata:dict=field(default_factory=dict)


@dataclass
class SelectionResult:
    U:np.ndarray
    step:np.ndarray|None
    status:str
    selector_id:str
    k:int
    metadata:dict


@dataclass(frozen=True)
class LimitedProbeContext:
    """Explicit paid full actions only; no full step/truth capability."""
    j:object
    jt:object
    max_rounds:int=2
    max_candidates:int=2
    cadence:int=1
    provenance:str="declared limited matrix-free full probes"
    def __post_init__(self):
        if not 0<=self.max_rounds<=2 or not 1<=self.max_candidates<=2 or self.cadence<1:
            raise ValueError('limited probes: 0..2 rounds, 1..2 candidates, positive cadence')


@dataclass(frozen=True)
class OracleContext:
    """Explicit full quadratic privilege, separate from SelectorContext."""
    phi:object
    provenance:str
    domain:str="realizable_current_dictionary_best_found"


def _phase(ctx,name):return ctx.phase(name) if hasattr(ctx,'phase') else nullcontext()
def _record(ctx):return ctx.records() if hasattr(ctx,'records') else {}


def _model(ctx,U):
    """Charge every selector-owned compressed build, including failed trials."""
    start=time.perf_counter();ctx.ledger.add('selector_Galerkin_build_attempts')
    ctx.ledger.add('selector_Galerkin_build_columns',U.shape[1])
    try:return ctx.make_model(U)
    except (ValueError,la.LinAlgError):
        ctx.ledger.add('selector_Galerkin_build_failures');raise
    finally:ctx.ledger.add('selector_Galerkin_build_wall_s',time.perf_counter()-start)


class LinearActionCache:
    """Allowed public linear actions on the common pool span, in paid blocks."""
    def __init__(self,ctx,pool,chunk_size=16):
        if chunk_size<1:raise ValueError('positive cache block size required')
        start=time.perf_counter();before=_record(ctx);self.ctx=ctx
        self.Q=orth(pool.vectors);L=[];S=[];PB=[]
        with _phase(ctx,'selector_shared_linear_cache_build'):
            for index in range(0,self.Q.shape[1],chunk_size):
                block=self.Q[:,index:index+chunk_size]
                L.append(ctx.apply_L(block));S.append(ctx.apply_S(block));PB.append(ctx.project_B(block))
            self.L=np.column_stack(L) if L else np.empty((ctx.n_current,0),complex)
            self.S=np.column_stack(S) if S else np.empty((ctx.m,0),complex)
            self.PB=np.concatenate(PB,axis=1) if PB else np.empty((ctx.P,0,ctx.q),complex)
            ctx.ledger.add('linear_cache_builds',1);ctx.ledger.add('linear_cache_basis_columns',self.Q.shape[1])
        self.metadata={'cache_basis_columns':self.Q.shape[1],'block_size':chunk_size,
            'build_wall_s':time.perf_counter()-start,'before':before,'after':_record(ctx),
            'scope':'L/S/B projections on declared common current span; no full inverse',
            'standalone_charge_rule':'add this complete cache-build cost to EACH from-scratch selector; shared total pays once'}

    def action(self,X,kind):
        X=np.asarray(X,complex);one=X.ndim==1;M=X[:,None] if one else X
        A=self.Q.conj().T@M
        error=la.norm(M-self.Q@A);norm=la.norm(M)
        if error>1e-12*norm:
            self.ctx.ledger.add('linear_cache_misses',1)
            return {'L':self.ctx.apply_L,'S':self.ctx.apply_S,'PB':self.ctx.project_B}[kind](X)
        self.ctx.ledger.add('linear_cache_'+kind+'_hits',1)
        self.ctx.ledger.add('linear_cache_'+kind+'_columns',M.shape[1])
        if kind=='PB':return np.einsum('rb,prq->pbq',A.conj(),self.PB)
        result=(self.L if kind=='L' else self.S)@A
        return result[:,0] if one else result


class CachedSelectorContext:
    """Public operator proxy; small Galerkin/Gamma models remain exact."""
    __slots__=('_ctx','linear_action_cache')
    def __init__(self,ctx,cache):self._ctx=ctx;self.linear_action_cache=cache
    def __getattr__(self,name):return getattr(self._ctx,name)
    def apply_L(self,C):return self.linear_action_cache.action(C,'L')
    def apply_S(self,C):return self.linear_action_cache.action(C,'S')
    observe=apply_S
    def project_B(self,C):return self.linear_action_cache.action(C,'PB')
    def make_model(self,U):
        from operators import ReducedModel
        return ReducedModel(self,U)
    def transfer(self,U,C):
        from operators import build_transfer
        return build_transfer(self,U,C)


def prepare_linear_action_cache(ctx,pool,chunk_size=16):
    cache=LinearActionCache(ctx,pool,chunk_size)
    return CachedSelectorContext(ctx,cache),cache.metadata


def _unwrap(ctx):return ctx._ctx if isinstance(ctx,CachedSelectorContext) else ctx


def orth(X,against=None,rank=None):
    X=np.asarray(X,complex)
    if X.ndim==1:X=X[:,None]
    if X.ndim!=2 or not np.isfinite(X).all():raise ValueError('finite 2-D current bank required')
    reference=float(la.norm(X,2)) if X.shape[1] else 0.
    if against is not None and against.shape[1]:
        for _ in range(2):X=X-against@(against.conj().T@X)
    if not X.shape[1]:return X.copy()
    U,s,_=la.svd(X,full_matrices=False)
    keep=int(np.sum(s>TOL*reference))
    if rank is not None:keep=min(keep,int(rank))
    return U[:,:keep].copy()


def _append(U,C):
    C=orth(C,U)
    return np.column_stack([U,C]),C


def normalized_pool(vectors,labels=None,families=None,metadata=None):
    vectors=np.asarray(vectors,complex)
    if vectors.ndim!=2:raise ValueError('pool vectors must be a matrix')
    labels=list(labels if labels is not None else (f'candidate_{i}' for i in range(vectors.shape[1])))
    families=list(families if families is not None else ['declared']*vectors.shape[1])
    if len(labels)!=vectors.shape[1] or len(families)!=vectors.shape[1]:raise ValueError('pool labels mismatch')
    norms=la.norm(vectors,axis=0)
    keep=np.isfinite(norms)&(norms>0)
    V=(vectors[:,keep]/norms[keep]).copy();V.setflags(write=False)
    meta=dict(metadata or {});meta.update(individually_normalized=True,
        original_column_norms=norms.tolist(),removed_zero_columns=int(np.sum(~keep)),
        candidate_bank_dim=int(V.shape[1]),effective_dictionary_rank=orth(V).shape[1])
    return CandidatePool(V,tuple(np.asarray(labels,dtype=object)[keep]),
                         tuple(np.asarray(families,dtype=object)[keep]),meta)


def fourier_modes(points,n_current,count):
    """Low frequency (l2,lex xyz) periodic Fourier; xyz vector components."""
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or n_current!=3*len(points):
        raise ValueError('Fourier requires n_current=3*N and actual voxel centres')
    lengths=[]
    for axis in range(3):
        x=np.unique(points[:,axis]);h=np.min(np.diff(x)) if len(x)>1 else 1.
        lengths.append(float(np.ptp(x)+h))
    radius=0;freqs=[]
    while 3*len(freqs)<count:
        radius+=1
        freqs=sorted(((i,j,k) for i in range(-radius,radius+1) for j in range(-radius,radius+1)
                      for k in range(-radius,radius+1)),key=lambda f:(sum(x*x for x in f),f))
    columns=[];labels=[]
    for freq in freqs:
        wave=np.exp(2j*np.pi*(points/np.asarray(lengths))@np.asarray(freq))
        for axis in range(3):
            column=np.zeros((len(points),3),complex);column[:,axis]=wave
            columns.append(column.reshape(-1));labels.append(f'fourier_{freq}_component_{axis}')
            if len(columns)==count:return np.column_stack(columns),labels
    raise RuntimeError('Fourier frequency construction exhausted')


def _call_internal(ctx,count,seed):
    result=ctx.internal_modes(count,seed)
    return result if isinstance(result,tuple) else (result,{'label':'declared internal proxy'})


def build_independent_anchor(ctx,rank=64,*,oracle_headroom_passed=False,unit_test=False,seed=20260930):
    """Geometry/primal-only fixed mix; no own selected model or full step."""
    if rank!=64 and not(unit_test or (rank in (48,96) and oracle_headroom_passed)):
        raise ValueError('nondefault anchor48/96 only after explicit oracle headroom gate')
    if rank>ctx.n_current:raise ValueError('infeasible reference rank')
    start=time.perf_counter();before=_record(ctx)
    with _phase(ctx,'selector_anchor_build'):
        receiver=ctx.receiver_modes(32);internal,imeta=_call_internal(ctx,32,seed)
        fourier,_=fourier_modes(ctx.points,ctx.n_current,32)
        forward=np.asarray(ctx.current_forward,complex).T
        banks=[receiver,internal,fourier,forward];columns=[]
        for index in range(max(b.shape[1] for b in banks)):
            for b in banks:
                if index<b.shape[1]:columns.append(b[:,index])
        raw=np.column_stack(columns);U=np.empty((ctx.n_current,0),complex)
        for column in raw.T:
            U,_=_append(U,column)
            if U.shape[1]>=rank:break
        if U.shape[1]!=rank:raise ValueError('infeasible independent anchor rank from fixed mix')
        anchor=_model(ctx,U)
    return anchor,{'rank':rank,'default_reference_rank':64,'unit_test':unit_test,
        'oracle_headroom_passed':oracle_headroom_passed,'generator':'fixed interleaved receiver/internal/Fourier/paid primal',
        'internal_metadata':imeta,'wall_s':time.perf_counter()-start,'before':before,'after':_record(ctx)}


def _complex_data(ctx,w):
    w=np.asarray(w,float).reshape(ctx.P,2*ctx.m)
    return w[:,:ctx.m]+1j*w[:,ctx.m:]


def residual_scenarios(ctx,cfg):
    """Actual residual plus four same-norm standalone complex probes."""
    rng=np.random.default_rng(cfg.seed);result=[np.asarray(ctx.r,float).copy()]
    norm=float(la.norm(ctx.r))
    for _ in range(cfg.residual_probe_count):
        w=rng.normal(size=(ctx.P,ctx.m))+1j*rng.normal(size=(ctx.P,ctx.m))
        w*=norm/la.norm(w)
        result.append(np.concatenate([w.real,w.imag],axis=1).reshape(-1))
    return result


def anchor_snapshots(ctx,anchor,cfg=SelectorConfig()):
    """Strong primal+dual snapshots, all actions through independent ROM."""
    if not hasattr(anchor,'U'):raise TypeError('independent reduced current model required')
    primal=[];dual=[];steps=[]
    for r in residual_scenarios(ctx,cfg):
        s=-anchor.solve_H(anchor.jt(r)+ctx.ell);steps.append(s)
        primal.append(np.asarray(anchor.current_jvp(s)).T)
        e=r+anchor.j(s)
        dual.append(anchor.R_adjoint(ctx.observe_adjoint(_complex_data(ctx,e).T)))
    X=np.column_stack(primal);Z=np.column_stack(dual)
    nx,nz=float(la.norm(X)),float(la.norm(Z))
    balanced=np.column_stack([X/(nx if nx>0 else 1.),Z/(nz if nz>0 else 1.)])
    return {'primal':X,'dual':Z,'balanced_PD':balanced,'steps':np.column_stack(steps),
            'metadata':{'num_primal_snapshots':X.shape[1],'num_dual_snapshots':Z.shape[1],
                'primal_block_norm':nx,'dual_block_norm':nz,'PD_block_weights':[1.,1.],
                'primal_covariance_weight':1/(nx*nx) if nx>0 else 0.,
                'dual_covariance_weight':1/(nz*nz) if nz>0 else 0.,
                'weight_rule':'predeclared equal-trace primal and dual covariance',
                'residual_scenario_norms':[float(la.norm(r)) for r in residual_scenarios(ctx,cfg)],
                'residual_scenarios':'actual residual plus four same-norm standalone complex probes; zero residual gives zero probes; ell retained',
                'snapshot_inverse':'reference ROM R/R_adjoint only','scenario_seed':cfg.seed}}


def build_candidates(ctx,anchor,cfg=SelectorConfig()):
    start=time.perf_counter();before=_record(ctx)
    with _phase(ctx,'selector_common_pool_build'):
        receiver=ctx.receiver_modes(cfg.pool_count);internal,imeta=_call_internal(ctx,cfg.pool_count,cfg.seed)
        fourier,flabel=fourier_modes(ctx.points,ctx.n_current,cfg.pool_count)
        snapshots=anchor_snapshots(ctx,anchor,cfg)
        pd=orth(snapshots['balanced_PD'],rank=cfg.pool_count)
        tasks=snapshots['steps'];currents=anchor.current_jvp(tasks)
        n,p,t=ctx.n_current,ctx.P,tasks.shape[1]
        Lcurrents=ctx.apply_L(currents.transpose(1,0,2).reshape(n,p*t)).reshape(n,p,t).transpose(1,0,2)
        rhs=ctx.B(tasks);tangent_residual=rhs-Lcurrents
        raw_residual=tangent_residual.transpose(1,0,2).reshape(n,p*t)
        rhs_scale=la.norm(rhs.transpose(1,0,2).reshape(n,p*t),axis=0)
        numerical_zero=la.norm(raw_residual,axis=0)<=TOL*rhs_scale
        raw_residual[:,numerical_zero]=0.
        residual_up=orth(raw_residual,rank=cfg.residual_up_count)
        rng=np.random.default_rng(cfg.seed+1)
        random=rng.normal(size=(ctx.n_current,cfg.random_pool_count))+1j*rng.normal(size=(ctx.n_current,cfg.random_pool_count))
        banks=[('receiver',receiver),('internal',internal),('fourier',fourier),('anchor_PD',pd),('residual_up',residual_up),('random',random)]
        labels=[];families=[]
        for name,bank in banks:
            labels.extend(flabel[:bank.shape[1]] if name=='fourier' else [f'{name}_{i}' for i in range(bank.shape[1])])
            families.extend([name]*bank.shape[1])
        pool=normalized_pool(np.column_stack([b for _,b in banks]),labels,families,
            {'generator_counts':{name:b.shape[1] for name,b in banks},'internal_metadata':imeta,
             'snapshot_metadata':snapshots['metadata'],'reference_rank':anchor.U.shape[1],
             'residual_up_definition':'B_p s_hat - L R_ref B_p s_hat; anchor task scenarios; residual-current proxy, not native TSOM',
             'raw_primal_snapshots':snapshots['primal'],'raw_dual_snapshots':snapshots['dual'],
             'balanced_PD_snapshots':snapshots['balanced_PD'],'anchor_step_scenarios':snapshots['steps']})
    pool.metadata.update(build_wall_s=time.perf_counter()-start,before=before,after=_record(ctx))
    return pool


def _require_anchor(ctx,anchor,k,cfg):
    if anchor is None or not hasattr(anchor,'U') or _unwrap(getattr(anchor,'ctx',ctx)) is not _unwrap(ctx):
        raise ValueError('same frozen context independent reduced anchor required')
    dim=anchor.U.shape[1]
    if dim<k or (dim==k and not cfg.unit_test_allow_equal_anchor):
        raise ValueError('independent anchor must be richer than retained current rank')


def _check_not_self(base,anchor):
    if base is anchor:raise ValueError('own-model reference prohibited (zero-gradient trap)')
    if base.U.shape[1]==anchor.U.shape[1] and la.norm(base.U-anchor.U@(anchor.U.conj().T@base.U))<TOL:
        raise ValueError('own current subspace cannot certify itself')


def finite_gain(base,child,anchor):
    _check_not_self(base,anchor)
    s=base.step();d=child.step()-s;ctx=anchor.ctx
    g=anchor.jt(ctx.r+anchor.j(s))+ctx.lam*s+ctx.ell
    q=anchor.j(d)
    return float(-g@d-.5*(q@q+ctx.lam*(d@d))),d


def first_order_gain(ctx,base,child,anchor):
    _check_not_self(base,anchor)
    s=base.step();e=ctx.r+base.j(s)
    # Delta is the ACTUAL Galerkin difference, including full joint feedback.
    Ds=child.j(s)-base.j(s)
    pull=child.jt(e)-base.jt(e)
    v=-base.solve_H(base.jt(Ds)+pull)
    g=anchor.jt(ctx.r+anchor.j(s))+ctx.lam*s+ctx.ell
    return float(-g@v),v


def _score(ctx,base,child,kind,anchor,prepared=None):
    if prepared is not None:
        from signed_round_reuse import score_signed_round
        return score_signed_round(ctx,base,child,kind,anchor,prepared)
    if kind=='finite':return finite_gain(base,child,anchor)[0]
    if kind=='first_order':return first_order_gain(ctx,base,child,anchor)[0]
    if kind=='magnitude':
        # Exact realified Frobenius norm through small output/input Grams.
        # No d-by-d identity or full data-by-material Jacobian is allocated.
        C=np.column_stack([child.CV,base.CV]);G=C.conj().T@C
        F=np.concatenate([child.F,-base.F],axis=1)
        return float(max(0.,s5_frobenius(G,F)))
    raise ValueError('unknown conditional scoring law')


def _result(ctx,U,k,selector,start,before,meta):
    if isinstance(ctx,CachedSelectorContext):
        meta.update(shared_linear_cache_cost=ctx.linear_action_cache.metadata,
            standalone_cache_wall_charge_s=ctx.linear_action_cache.metadata['build_wall_s'])
    if U.shape[1]!=k:
        return SelectionResult(U,None,'infeasible_at_budget',selector,k,{**meta,'actual_rank':U.shape[1],
            'wall_s':time.perf_counter()-start,'before':before,'after':_record(ctx)})
    try:
        with _phase(ctx,'selector_final_Galerkin_step'):
            model_start=time.perf_counter();model=_model(ctx,U);build_s=time.perf_counter()-model_start
            step_start=time.perf_counter();step=model.step();step_s=time.perf_counter()-step_start
    except (ValueError,la.LinAlgError) as error:
        return SelectionResult(U,None,'infeasible_at_budget',selector,k,{**meta,'actual_rank':U.shape[1],
            'failure':str(error),'wall_s':time.perf_counter()-start,'before':before,'after':_record(ctx)})
    return SelectionResult(U,step,'ok',selector,k,{**meta,'actual_rank':k,'retained_current_dim_complex':k,
        'retained_current_dim_realified':2*k,'shared_over_illuminations':True,'num_illuminations':ctx.P,
        'stability':model.stability,'fast_F_layout':model.fast_layout,
        'final_model_build_s':build_s,'final_step_wall_s':step_s,
        'wall_s':time.perf_counter()-start,'before':before,'after':_record(ctx)})


def _exact_prefix(ctx,bank,k):
    U=np.empty((ctx.n_current,0),complex)
    for column in bank.T:
        U,_=_append(U,column)
        if U.shape[1]==k:break
    return U if U.shape[1]==k else np.empty((ctx.n_current,0),complex)


def _pool_family(pool,family):
    return pool.vectors[:,np.asarray(pool.families)==family]


def _fixed_completion(ctx,U,pool,k):
    """Declared rank-shortfall completion receiver->internal->Fourier only."""
    added=[];failures=[];start=time.perf_counter();before=_record(ctx)
    with _phase(ctx,'selector_rank_shortfall_completion'):
        for family in ('receiver','internal','fourier'):
            indices=[i for i,name in enumerate(pool.families) if name==family]
            for index in indices:
                if U.shape[1]>=k:break
                proposed,C=_append(U,pool.vectors[:,index])
                if C.shape[1]!=1:continue
                try:_model(ctx,proposed)
                except (ValueError,la.LinAlgError) as error:
                    failures.append({'index':index,'failure':str(error)});continue
                U=proposed;added.append(index)
            if U.shape[1]>=k:break
    return U,{'completion_used':bool(added),'completion_indices':added,
        'completion_rule':'fixed receiver -> internal -> Fourier, charged, no risk tuning',
        'completion_failures':failures,'completion_wall_s':time.perf_counter()-start,
        'completion_before':before,'completion_after':_record(ctx)}


def _single_candidates(ctx,U,pool):
    for i in range(pool.vectors.shape[1]):
        proposed,C=_append(U,pool.vectors[:,i])
        if C.shape[1]==1:yield (i,),proposed


def _probe_scores(ctx,base,entries,probe,used_rounds):
    if not isinstance(probe,LimitedProbeContext):raise TypeError('explicit LimitedProbeContext required')
    if used_rounds>=probe.max_rounds:return entries,None
    shortlist=sorted(entries,key=lambda e:(-e['score'],e['indices']))[:probe.max_candidates]
    s=base.step();g=probe.jt(ctx.r+probe.j(s))+ctx.lam*s+ctx.ell
    costs={'full_tangent_rhs':ctx.P,'full_adjoint_rhs':ctx.P,'probe_candidates':len(shortlist)}
    for entry in shortlist:
        d=entry['model'].step()-s;q=probe.j(d)
        entry['score']=float(-g@d-.5*(q@q+ctx.lam*(d@d)))
        entry['full_probe_scored']=True;costs['full_tangent_rhs']+=ctx.P
    # Once an exact shortlist is calibrated, choice is within that declared
    # shortlist, never mixed with incomparable uncalibrated scores.
    return shortlist,costs


def _swap_pass(ctx,U,pool,reference,max_pairs,*,oracle=False):
    best=_model(ctx,U);best_phi=float(reference.phi(best.step()));attempts=[]
    candidates=[]
    outside=[add for add in range(pool.vectors.shape[1]) if orth(pool.vectors[:,add],U).shape[1]]
    # Fixed family-balanced proposals, independent of any objective values.
    family_order=('receiver','internal','fourier','anchor_PD','residual_up','random')
    families=list(family_order)+sorted(set(pool.families)-set(family_order))
    queues={family:[i for i in outside if pool.families[i]==family] for family in families}
    selected=[]
    while len(selected)<8 and any(queues.values()):
        for family in families:
            if queues[family]:selected.append(queues[family].pop(0))
            if len(selected)==8:break
    # Spread removals across the retained basis instead of its first columns.
    removal_order=np.random.default_rng(20260930).permutation(U.shape[1]).tolist()
    for round_index in range(U.shape[1]):
        for candidate_index,add in enumerate(selected):
            remove=removal_order[(round_index+candidate_index)%len(removal_order)]
            reduced=np.delete(U,remove,axis=1)
            proposed,C=_append(reduced,pool.vectors[:,add])
            if C.shape[1]==1:candidates.append((remove,add,proposed))
            if len(candidates)>=max_pairs:break
        if len(candidates)>=max_pairs:break
    for remove,add,proposed in candidates:
        try:
            model=_model(ctx,proposed);value=float(reference.phi(model.step()))
            gain=best_phi-value if oracle else finite_gain(best,model,reference)[0]
            attempts.append({'remove_column':remove,'add_candidate':add,'add_label':pool.labels[add],
                'add_family':pool.families[add],'finite_gain':gain,'admissible':True,
                'proposal_rule':'family round-robin top8; fixed seeded spread removal; one pass'})
            if gain>0:best,best_phi=model,value
        except (ValueError,la.LinAlgError) as error:
            attempts.append({'remove_column':remove,'add_candidate':add,'add_label':pool.labels[add],
                'add_family':pool.families[add],'admissible':False,'failure':str(error)})
    return best.U,attempts


def _greedy(ctx,pool,k,kind,anchor,cfg,probe=None):
    U=np.empty((ctx.n_current,0),complex);trace=[];chosen=[];probe_rounds=0;failures=[]
    while U.shape[1]<k:
        base=_model(ctx,U);entries=[]
        from signed_round_reuse import prepare_signed_round
        prepared=prepare_signed_round(ctx,base,anchor,kind) if kind in ('finite','first_order') else None
        for indices,proposed in _single_candidates(ctx,U,pool):
            try:
                model=_model(ctx,proposed)
                entries.append({'indices':indices,'U':proposed,'model':model,'score':_score(ctx,base,model,kind,anchor,prepared=prepared)})
            except (ValueError,la.LinAlgError) as error:failures.append({'rank':U.shape[1],'indices':indices,'failure':str(error)})
        if not entries:break
        allzero=all(abs(entry['score'])<=cfg.score_zero_tol for entry in entries);pair_proposals=[]
        if k-U.shape[1]>=2 and (allzero or cfg.pair_always):
            shortlist=sorted(entries,key=lambda e:(-e['score'],e['indices']))[:cfg.pair_top]
            for pair_index,(first,second) in enumerate(combinations(shortlist,2)):
                if pair_index>=8:break
                indices=(first['indices'][0],second['indices'][0]);proposed,C=_append(U,pool.vectors[:,indices])
                pair_proposals.append(indices)
                if C.shape[1]!=2:continue
                try:
                    model=_model(ctx,proposed)
                    entries.append({'indices':indices,'U':proposed,'model':model,'score':_score(ctx,base,model,kind,anchor,prepared=prepared)})
                except (ValueError,la.LinAlgError) as error:failures.append({'rank':U.shape[1],'indices':indices,'failure':str(error)})
        probe_cost=None
        if probe is not None and len(trace)%max(probe.cadence,1)==0:
            entries,probe_cost=_probe_scores(ctx,base,entries,probe,probe_rounds)
            if probe_cost is not None:probe_rounds+=1
        winner=min(entries,key=lambda e:(-e['score'],e['indices']))
        trace.append({'base_rank':U.shape[1],'chosen_indices':winner['indices'],'score':winner['score'],
            'tested_paircount':len(pair_proposals),'predeclared_pairproposal':pair_proposals,
            'singleton_all_zero':allzero,'tested_candidates':len(entries),'probe':probe_cost,
            'all_candidate_scores':[{'indices':e['indices'],'score':e['score']} for e in entries]})
        U=winner['U'];chosen.extend(winner['indices'])
    swaps=[]
    if U.shape[1]==k and kind=='finite' and cfg.swaps_max_pairs:
        U,swaps=_swap_pass(ctx,U,pool,anchor,min(cfg.swaps_max_pairs,16))
    return U,{'selected_dictionary_indices_before_swaps':chosen,'trace':trace,'swaps':swaps,
              'candidate_failures':failures,'probe_rounds':probe_rounds,'score_kind':kind,
              'peak_selection_current_dim':min(k,ctx.n_current),'peak_reference_current_dim':anchor.U.shape[1] if anchor is not None else 0}


def _joint_transfer_lift(ctx,pool,k,Z):
    """REAL task SVD, complex current coefficient lifting, actual reevaluation."""
    C=orth(pool.vectors);base=_model(ctx,np.empty((ctx.n_current,0),complex))
    transfer=ctx.transfer(base,C)
    Z=np.asarray(Z,float)
    task_map=transfer.delta_j(Z)
    _,sv,Vh=la.svd(task_map,full_matrices=False)
    V=Vh.T[:,sv>TOL*(sv[0] if len(sv) else 0.)]
    complex_Z=Z[:ctx.q]+1j*Z[ctx.q:]
    # transfer.T=Gamma^-1 N retains the FULL joint Gamma, per illumination.
    coefficients=np.einsum('pbq,qt->pbt',transfer.T,complex_Z@V)
    lifted=C@coefficients.transpose(1,0,2).reshape(C.shape[1],-1)
    U=orth(lifted,rank=k)
    return U,{'task_singular_values':sv.tolist(),'task_rank':V.shape[1],
        'lifted_current_snapshot_columns':lifted.shape[1],'raw_joint_bank_rank':C.shape[1],
        'Gamma_condition':transfer.stability,'lifting':'C Gamma^-1 N Z v (per illumination); current POD; actual Galerkin',
        'peak_temporary_joint_current_dim':C.shape[1],
        'material_vectors_used_as_current':False,'full_Gamma_coupling':True}


def select_current(ctx,pool,k,selector_id,*,anchor=None,variant=None,cfg=SelectorConfig(),probe=None,seed=0):
    if isinstance(ctx,OracleContext):raise TypeError('oracle privileges cannot enter cheap SelectorContext')
    k=int(k)
    if k<1 or k>ctx.n_current:raise ValueError('invalid exact current budget')
    if pool.vectors.shape[0]!=ctx.n_current:raise ValueError('pool/context current mismatch')
    start=time.perf_counter();before=_record(ctx);meta={'candidate_bank_dim':pool.vectors.shape[1],
       'candidate_metadata':{key:value for key,value in pool.metadata.items() if not isinstance(value,np.ndarray)},
       'variant':variant,'anchor_reference_rank':anchor.U.shape[1] if anchor is not None else 0,
       'reference_cost_not_amortized_away':True,'peak_reference_current_dim':anchor.U.shape[1] if anchor is not None else 0}
    U=np.empty((ctx.n_current,0),complex)
    try:
        with _phase(ctx,f'selector_{selector_id}_{variant or "default"}'):
            if selector_id=='S1':
                U=_exact_prefix(ctx,_pool_family(pool,'receiver'),k);meta['label']='receiver-SVD SOM-like; not native SOM'
            elif selector_id=='S2':
                receiver=_pool_family(pool,'receiver');internal=_pool_family(pool,'internal');half=k//2
                U=_exact_prefix(ctx,receiver,half);old=U.shape[1]
                for column in internal.T:
                    U,_=_append(U,column)
                    if U.shape[1]>=k:break
                meta.update(label='book-inspired projected receiver/internal half; not native TSOM',receiver_rank=half,internal_added_rank=U.shape[1]-old)
            elif selector_id=='S3a':
                # Ordered columns, not an SVD that could reorder degenerate Fourier modes.
                for column in _pool_family(pool,'fourier').T:
                    U,_=_append(U,column)
                    if U.shape[1]==k:break
                meta['label']='fixed low frequency l2/xyz/component Fourier'
            elif selector_id=='S3b':
                X=pool.metadata.get('raw_primal_snapshots')
                if X is None:raise ValueError('current-energy requires declared anchor primal snapshots')
                U=orth(np.column_stack([np.asarray(ctx.current_forward).T,X]),rank=k)
                meta.update(label='raw current-energy POD of paid primal plus same-anchor primal snapshots',dual_used=False)
            elif selector_id=='S4':
                domain=orth(pool.vectors);rng=np.random.default_rng(seed);attempts=[]
                for attempt in range(cfg.random_attempts):
                    U=orth(domain@(rng.normal(size=(domain.shape[1],k))+1j*rng.normal(size=(domain.shape[1],k))),rank=k)
                    try:
                        if U.shape[1]!=k:raise ValueError('dictionary domain rank below k')
                        _model(ctx,U);attempts.append({'attempt':attempt,'stable':True});break
                    except (ValueError,la.LinAlgError) as error:attempts.append({'attempt':attempt,'stable':False,'failure':str(error)})
                meta.update(seed=seed,random_attempts=attempts,label='complex random k-space inside common dictionary span')
            elif selector_id in ('S5','S7','S8'):
                kind={'S5':'magnitude','S7':'first_order','S8':'finite'}[selector_id]
                if selector_id=='S8':
                    if variant in (None,'no_probes') and probe is None:meta['variant']='no_probes'
                    elif variant in (None,'limited_probes') and probe is not None:meta['variant']='limited_probes'
                    else:raise ValueError('S8 probe capability and explicit variant disagree')
                if selector_id!='S5':_require_anchor(ctx,anchor,k,cfg)
                if selector_id=='S5' and variant in (None,'singleton'):
                    base=_model(ctx,U);scores=[]
                    for indices,proposed in _single_candidates(ctx,U,pool):
                        try:scores.append((indices[0],_score(ctx,base,_model(ctx,proposed),kind,None)))
                        except (ValueError,la.LinAlgError):continue
                    for index,score in sorted(scores,key=lambda x:(-x[1],x[0])):
                        proposed,C=_append(U,pool.vectors[:,index])
                        if C.shape[1]!=1:continue
                        try:_model(ctx,proposed)
                        except (ValueError,la.LinAlgError):continue
                        U=proposed
                        if U.shape[1]==k:break
                    meta['singleton_scores']=scores;meta['score_kind']='fixed base singleton magnitude'
                else:
                    if selector_id!='S8' and probe is not None:raise ValueError('full probes only in declared S8 variant')
                    U,extra=_greedy(ctx,pool,k,kind,anchor,cfg,probe);meta.update(extra)
            elif selector_id=='S6':
                _require_anchor(ctx,anchor,k,cfg)
                if variant in (None,'anchor_steps'):
                    Z=pool.metadata.get('anchor_step_scenarios')
                    if Z is None:raise ValueError('anchor task scenarios unavailable')
                elif variant=='prior_probes':
                    Z=np.random.default_rng(cfg.seed+2).normal(size=(ctx.dim_material,cfg.task_probe_count))/np.sqrt(ctx.lam)
                else:raise ValueError('S6 variant must be anchor_steps or prior_probes')
                U,extra=_joint_transfer_lift(ctx,pool,k,Z);meta.update(extra)
            elif selector_id=='S11':
                _require_anchor(ctx,anchor,k,cfg)
                X=pool.metadata.get('balanced_PD_snapshots')
                if X is None:raise ValueError('strong same-anchor primal+dual snapshots required')
                U=orth(X,rank=k);meta.update(label='generic equal-block primal-dual goal-oriented current POD',
                    primal_and_dual_total_rank=k,PD_block_weights=[1.,1.])
            else:raise ValueError('S9 requires explicit oracle function; S10 is non-current control')
            if U.shape[1]<k and selector_id in ('S1','S2','S3a','S3b','S6','S11'):
                U,completion=_fixed_completion(ctx,U,pool,k);meta.update(completion)
    except (ValueError,la.LinAlgError) as error:
        meta['failure']=str(error)
        return SelectionResult(U,None,'infeasible_at_budget',selector_id,k,{**meta,'wall_s':time.perf_counter()-start,'before':before,'after':_record(ctx)})
    return _result(ctx,U,k,selector_id,start,before,meta)


def select_random_seeds(ctx,pool,k,cfg=SelectorConfig()):
    return [select_current(ctx,pool,k,'S4',seed=seed,cfg=cfg) for seed in range(10)]


def certified_dictionary_oracle(ctx,pool,k,oracle:OracleContext,*,enumeration_limit=10000):
    if not isinstance(oracle,OracleContext):raise TypeError('explicit OracleContext required')
    total=comb(pool.vectors.shape[1],k)
    if total>enumeration_limit:raise ValueError('certified enumeration exceeds predeclared limit')
    start=time.perf_counter();before=_record(ctx);best=None;value=np.inf;tested=[]
    with _phase(ctx,'S9_certified_enumeration'):
        for indices in combinations(range(pool.vectors.shape[1]),k):
            U=orth(pool.vectors[:,indices])
            if U.shape[1]!=k:tested.append({'indices':indices,'admissible':False,'failure':'dependent'});continue
            try:
                model=_model(ctx,U);phi=float(oracle.phi(model.step()))
                tested.append({'indices':indices,'admissible':True,'objective':phi})
                if phi<value:best,value=U,phi
            except (ValueError,la.LinAlgError) as error:tested.append({'indices':indices,'admissible':False,'failure':str(error)})
    U=best if best is not None else np.empty((ctx.n_current,0),complex)
    return _result(ctx,U,k,'S9a',start,before,{'oracle_domain':'certified finite dictionary k-subsets',
        'certified_in_declared_dictionary':True,'enumerated_subsets':total,'tests':tested,
        'best_objective':None if best is None else value,'oracle_privilege_provenance':oracle.provenance})


def best_found_oracle(ctx,pool,k,oracle:OracleContext,traditional_results,*,cfg=SelectorConfig(),starts=4):
    if not isinstance(oracle,OracleContext):raise TypeError('explicit OracleContext required')
    if starts!=4:raise ValueError('oracle locked to four starts')
    if not traditional_results:raise ValueError('all tested traditional spaces must be supplied')
    start=time.perf_counter();before=_record(ctx);evaluated=[];admissible=[]
    with _phase(ctx,'S9_best_found_dictionary'):
        for index,result in enumerate(traditional_results):
            if result.status!='ok' or result.U.shape[1]!=k:
                evaluated.append({'traditional_index':index,'selector':result.selector_id,'admissible':False});continue
            model=_model(ctx,result.U);objective=float(oracle.phi(model.step()))
            admissible.append((objective,result.U.copy(),index))
            evaluated.append({'traditional_index':index,'selector':result.selector_id,'admissible':True,'objective':objective})
        if not admissible:
            return _result(ctx,np.empty((ctx.n_current,0),complex),k,'S9b',start,before,
                           {'failure':'no admissible traditional spaces supplied','traditional_evaluations':evaluated})
        admissible.sort(key=lambda item:(item[0],item[2]));best_value,best_U,_=admissible[0];restarts=[]
        for index in range(4):
            # Starts include strongest exact tested traditional spaces. If fewer
            # than four are stable, repeat deterministically and record it.
            initial=admissible[index%len(admissible)]
            U,swaps=_swap_pass(ctx,initial[1],pool,oracle,min(cfg.swaps_max_pairs,16),oracle=True)
            value=float(oracle.phi(_model(ctx,U).step()))
            restarts.append({'start':index,'traditional_index':initial[2],'start_objective':initial[0],
                             'final_objective':value,'one_swap_pass':swaps})
            if value<best_value:best_value,best_U=value,U
    return _result(ctx,best_U,k,'S9b',start,before,{'oracle_domain':'union_dictionary_and_baseline_spaces',
        'certified_global':False,'traditional_evaluations':evaluated,'starts':restarts,
        'num_starts':4,'max_swap_pairs_per_start':min(cfg.swaps_max_pairs,16),'best_objective':best_value,
        'oracle_privilege_provenance':oracle.provenance,'includes_every_supplied_tested_traditional_space':True,
        'comparison_domain':'union_dictionary_and_baseline_spaces; includes baselines outside dictionary span'})


def noncurrent_material_control(H,steps,k):
    """S10 offline H-weighted material task POD, explicitly different budget."""
    H=np.asarray(H,float);steps=np.asarray(steps,float)
    if steps.ndim==1:steps=steps[:,None]
    values,vectors=la.eigh(H)
    if np.min(values)<=0:raise ValueError('material control H must be SPD')
    root=(vectors*np.sqrt(values))@vectors.T
    invroot=(vectors/np.sqrt(values))@vectors.T
    U,s,_=la.svd(root@steps,full_matrices=False);rank=min(int(k),U.shape[1])
    return {'selector_id':'S10','material_basis':invroot@U[:,:rank],'material_rank_real':rank,
            'current_rank':None,'same_current_budget':False,'offline_only':True,
            'control_type':'H-weighted material task POD; not current-realizable oracle',
            'singular_values':s.tolist()}
