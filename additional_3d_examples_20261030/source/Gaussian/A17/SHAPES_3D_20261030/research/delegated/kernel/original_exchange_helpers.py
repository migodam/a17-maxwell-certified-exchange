"""Original A17 exchange helpers, extracted without executing the old runner.

Every helper definition below is unchanged original source. Only the module
imports and CONFIG binding are new. CONFIG reads a per-invocation batch size;
no historical config, source lock, budget, input cache, or runtime is imported.
"""
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import time
import numpy as np
from isolated_imports import install
install()
from operators import orth

@dataclass
class _Invocation:
    candidate_batch_size: int
    scope: str
    array_observations: int = 0

_INVOCATION = ContextVar("a17_original_exchange_invocation", default=None)

class _InvocationConfig:
    def __getitem__(self, key):
        if key != "cuda_candidate_batch":
            raise KeyError(key)
        state = _INVOCATION.get()
        if state is None:
            raise RuntimeError("Use run_original_exchange to bind isolated batch configuration")
        return state.candidate_batch_size

CONFIG = _InvocationConfig()

@contextmanager
def invocation(candidate_batch_size, scope):
    if isinstance(candidate_batch_size, bool) or not isinstance(candidate_batch_size, int) or candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be a positive integer")
    token = _INVOCATION.set(_Invocation(candidate_batch_size, str(scope)))
    try:
        yield
    finally:
        _INVOCATION.reset(token)

def array_identity(value):
    """Logging-only observation label; neither content fingerprint nor equality claim.

    Legacy controller field names end in *_hash. Their values here are explicit
    descriptors, not hashes. Existing endpoint/finalist NPZ files retain actual
    arrays. No data reduction, equality comparison, or digest is performed.
    """
    state = _INVOCATION.get()
    if state is None:
        raise RuntimeError("Array observation requires run_original_exchange invocation")
    state.array_observations += 1
    array = np.asarray(value)
    return dict(kind="NON_HASH_ARRAY_OBSERVATION", scope=state.scope,
                observation_index=state.array_observations,
                shape=list(array.shape), dtype=str(array.dtype),
                content_fingerprint=False, equality_asserted=False)

# The following definitions are verbatim from original run_exchange.py.
def jswrite(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def balanced_ids(pool,selected,count):
    order=list(dict.fromkeys(pool.families));by={f:[i for i,x in enumerate(pool.families) if x==f and i not in selected] for f in order}
    result=[];pos=0
    while len(result)<count:
        any_added=False
        for family in order:
            if pos<len(by[family]):result.append(by[family][pos]);any_added=True
            if len(result)==count:break
        if not any_added:break
        pos+=1
    return result

def moves(ids,incoming,menu,k):
    result=[]
    if menu in ('swap','adaptive'):
        for drop in ids:
            for add in incoming:result.append(dict(kind='swap',drop=[drop],add=[add],ids=[i for i in ids if i!=drop]+[add]))
    if menu=='adaptive':
        for drop in ids:result.append(dict(kind='delete',drop=[drop],add=[],ids=[i for i in ids if i!=drop]))
        if len(ids)<k:
            for add in incoming:result.append(dict(kind='add',drop=[],add=[add],ids=list(ids)+[add]))
    return result

def basis(atoms,ids):return orth(atoms[:,ids]) if ids else np.empty((atoms.shape[0],0),complex)

def score_from_outputs(u,z,x,d,ell,lam,quadratic=True):
    linear=-np.einsum('i,ij->j',u,z)-(ell+lam*x)@d
    curvature=.5*(np.sum(z*z,axis=0)+lam*np.sum(d*d,axis=0))
    return linear-curvature if quadratic else linear

def gains_on_anchor(anchor,x,d,ctx,quadratic=True):
    return score_from_outputs(ctx.r+anchor.j(x),anchor.j(d),x,d,ctx.ell,ctx.lam,quadratic)

class Neighborhood:
    def __init__(self,w,engine,out,identity):
        self.w,self.engine,self.out,self.identity=w,engine,Path(out),identity
        self.timings=[];self.count=0
    def endpoints(self,move_list):
        """Shared core Schur API when available; direct fallback is recorded."""
        atoms=self.w.pool.vectors;steps=[];valid=[];rows=[];started=time.perf_counter()
        # Group by rank, then removal core. Preserve the declared logical IDs.
        groups={}
        for idx,move in enumerate(move_list):
            U=basis(atoms,move['ids']);rank=U.shape[1]
            if rank!=len(move['ids']):
                rows.append((idx,dict(status='INFEASIBLE_DEPENDENT_ATOMS',actual_rank=rank,requested_rank=len(move['ids']))));continue
            key=(rank,tuple(i for i in move['ids'] if i not in move['add']))
            groups.setdefault(key,[]).append((idx,U,move))
        for (rank,retained),group in groups.items():
            # Declared atom removal defines V; QR only represents that same span.
            has_incoming=bool(group[0][2]['add'])
            core=self.engine.prepare_removal_core(basis(atoms,list(retained))) if has_incoming else None
            if core is not None:self.timings.append(dict(kind='shared_core_prepare',statistics=core.statistics))
            for first in range(0,len(group),CONFIG['cuda_candidate_batch']):
                block=group[first:first+CONFIG['cuda_candidate_batch']]
                if has_incoming:
                    incoming=np.stack([atoms[:,x[2]['add']] for x in block])
                    evaluated=self.engine.evaluate_incoming(core,incoming)
                else:evaluated=self.engine.evaluate(np.stack([x[1] for x in block]))
                self.timings.append(evaluated['statistics'])
                for j,(idx,_,move) in enumerate(block):
                    stat=evaluated['status'][j];stability=evaluated['stability'][j]
                    fallback=(core is not None and core.status!='STABLE_CORE') or bool(isinstance(stability,dict) and stability.get('Schur_representation_fallback'))
                    row=dict(status=stat,actual_rank=rank,requested_rank=len(move['ids']),child_condition=stability,child_normal_relative_residual=float(evaluated['normal_relative_residual'][j]) if np.isfinite(evaluated['normal_relative_residual'][j]) else None,shared_core_fast_path=bool(core is not None and not fallback),direct_endpoint_fallback=bool(fallback),common_core_status=core.status if core is not None else 'DIRECT_DELETE_ENDPOINT',information_trace=float(evaluated['info_trace'][j]) if 'info_trace' in evaluated and np.isfinite(evaluated['info_trace'][j]) else None)
                    rows.append((idx,row))
                    if stat=='OK':valid.append(idx);steps.append(evaluated['steps'][j])
        result={i:row for i,row in rows};valid=np.asarray(valid,int)
        self.count+=len(move_list)
        return valid,np.asarray(steps).T if steps else np.empty((self.engine.q*2,0)),result,time.perf_counter()-started

def append_rows(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n')
