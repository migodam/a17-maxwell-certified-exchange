"""Algorithm-matched receiver controls on already-paid public coefficient maps.
No physical model construction. CUDA is opt-in for root; never silently used.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,time,json,hashlib,os,copy
import numpy as np
HERE=Path(__file__).resolve().parent
if not os.environ.get('A17_MATCHED_R1_ROOT'):raise RuntimeError('Portable v2 requires explicit A17_MATCHED_R1_ROOT, set by --r1-root')
R1=Path(os.environ['A17_MATCHED_R1_ROOT']).expanduser().resolve()
if not (R1/'A17_RUN_CONFIG.json').is_file():raise ValueError('invalid explicit R1 root')
sys.path.insert(0,str(R1/'code'))
import pinned_runtime
from operators import SelectorContext,ReducedModel,orth
from pairtsom_adapter import PhaseLedger
from selectors_current import CandidatePool,_append,_check_not_self
from endpoint_batch import EndpointBatch
from removal_core_batch import CoreEndpointBatch
from run_exchange import Neighborhood,balanced_ids,moves,score_from_outputs,CONFIG,gain_tolerance,phi_from_output

TOL=1e-10
METHODS=('cached_receiver_legacy16_onepass','same_shortlist12_directed_one_decision','a17_conditional12_directed3_matched')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sanitize(x):
    if isinstance(x,np.ndarray):return sanitize(x.tolist())
    if isinstance(x,np.generic):return sanitize(x.item())
    if isinstance(x,complex):return {'real':sanitize(x.real),'imag':sanitize(x.imag)}
    if isinstance(x,dict):return {str(k):sanitize(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [sanitize(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def make_context(bank):
    c=SelectorContext();c.P,c.n_current,c.q=bank['PB'].shape;c.m=bank['S'].shape[0];c.dim_material=2*c.q;c.lam=float(bank['lambda_total']);c.r=bank['r'];c.ell=bank['ell'];c.ledger=PhaseLedger()
    c._L=lambda U:bank['L']@U;c._S=lambda U:bank['S']@U;c._B=lambda U:np.stack([U.conj().T@p for p in bank['PB']]);return c

class DirectionAction:
    """Only j capability. Cached labels not available to the ranking function.
    New physical action can be provided by root with kind='PHYSICAL_CALLBACK'.
    """
    def __init__(self,*,P,fullJ=None,V=None,fullJV=None,action=None,kind=None,synchronize=None,counter_supplier=None,backend='cpu'):
        self._synchronize=synchronize or (lambda:None);self._counter_supplier=counter_supplier;self.backend=backend
        self.P=P;self._J=fullJ;self._V=V;self._JV=fullJV;self._action=action
        self.kind=kind or ('PAID_FULLJ_CACHE' if fullJ is not None else 'PAID_PROJECTED_CACHE')
        if sum([fullJ is not None,V is not None,action is not None])!=1:raise ValueError('one direction capability required')
        if V is not None and fullJV is None:raise ValueError('V requires already-paid fullJV')
        if action is not None and self.kind!='PHYSICAL_CALLBACK':raise ValueError('callback must declare physical charges')
        for a in [fullJ,V,fullJV]:
            if a is not None and (np.iscomplexobj(a) or a.dtype!=np.float64 or not np.isfinite(a).all()):raise ValueError('real FP64 finite direction cache required')
        self.calls=0;self.rhs_equivalent=0;self.new_physical_rhs=0;self.wall_s=0.;self.coverage_failures=0
    def snapshot(self):return dict(calls=self.calls,rhs_equivalent=self.rhs_equivalent,new_physical_rhs=self.new_physical_rhs,wall_s=self.wall_s,coverage_failures=self.coverage_failures)
    def sync(self):self._synchronize()
    def physical_snapshot(self):return copy.deepcopy(self._counter_supplier()) if self._counter_supplier else None
    def j(self,Z):
        Z=np.asarray(Z)
        if Z.dtype!=np.float64 or not np.isfinite(Z).all():raise ValueError('directions must be finite real FP64')
        self.sync();t=time.perf_counter();one=Z.ndim==1;Z2=Z[:,None] if one else Z
        self.calls+=1;self.rhs_equivalent+=self.P*Z2.shape[1]
        try:
            if self._J is not None:out=self._J@Z2
            elif self._V is not None:
                coeff=np.linalg.lstsq(self._V,Z2,rcond=None)[0]
                err=np.linalg.norm(self._V@coeff-Z2,axis=0)/np.maximum(np.linalg.norm(Z2,axis=0),1e-300)
                if np.any(err>TOL):self.coverage_failures+=1;raise ValueError('PROJECTED_CACHE_DOES_NOT_COVER_DIRECTION')
                out=self._JV@coeff
            else:
                self.new_physical_rhs+=self.P*Z2.shape[1];out=np.asarray(self._action(Z2))
            if out.dtype!=np.float64 or not np.isfinite(out).all():raise ValueError('invalid paid action output')
            return out[:,0] if one else out
        finally:self.sync();self.wall_s+=time.perf_counter()-t

class CachedControls:
    def __init__(self,bank,*,device='cpu',direction=None,reference_floor=0.,source=None):
        t=time.perf_counter();self.source=source;self.bank=bank
        for k in ['L','S','PB','pool','anchor']:
            if bank[k].dtype!=np.complex128 or not np.isfinite(bank[k]).all():raise ValueError(k+' requires complex128 finite')
        for k in ['r','ell']:
            if bank[k].dtype!=np.float64 or not np.isfinite(bank[k]).all():raise ValueError(k+' requires float64 finite')
        if len(bank['families'])!=bank['pool'].shape[1] or len(bank['labels'])!=bank['pool'].shape[1]:raise ValueError('atom metadata/order mismatch')
        self.ctx=make_context(bank);tanchor=time.perf_counter();self.anchor=ReducedModel(self.ctx,bank['anchor']);self.anchor_setup_wall_s=time.perf_counter()-tanchor
        self.pool=CandidatePool(bank['pool'],list(bank['labels']),list(bank['families']))
        self.w=SimpleNamespace(ctx=self.ctx,pool=self.pool,anchor=self.anchor)
        if direction is not None:
            if direction._J is not None and direction._J.shape!=(self.ctx.P*2*self.ctx.m,self.ctx.dim_material):raise ValueError('fullJ does not share physical packed data/material rows')
            if direction._V is not None and (direction._V.shape[0]!=self.ctx.dim_material or direction._JV.shape!=(len(self.ctx.r),direction._V.shape[1])):raise ValueError('projected cache row/column mismatch')
        self.engine=CoreEndpointBatch(bank['L'],bank['S'],bank['PB'],bank['r'],bank['ell'],float(bank['lambda_total']),device=device,batch_size=16)
        self.direction=direction;self.floor=float(reference_floor)
        self.setup_context_cost=self.ctx.records()
        self.setup=dict(anchor_context_setup_cost=self.setup_context_cost,cache_adapter_setup_wall_s=time.perf_counter()-t,anchor_model_setup_wall_s=self.anchor_setup_wall_s,endpoint_engine_setup_wall_s=self.engine.setup_wall_s,endpoint_engine_setup_upload_bytes=self.engine.setup_upload_bytes,direction_cache_bytes=sum(a.nbytes for a in [direction._J,direction._V,direction._JV] if a is not None) if direction else 0,direction_device=direction.backend if direction else None,loaded_cache_bytes=sum(v.nbytes for v in bank.values() if isinstance(v,np.ndarray)),physical_state_cold_wall_s=None,standalone_cold_wall_s=None,cold_speedup=None,cold_reason='Physical object/anchor/pool generation not rerun; persisted cache load/setup is warm adapter setup, not cold Maxwell timing.',history_setup_receipt=None)
    def receiver(self,k):
        ids=[i for i,f in enumerate(self.pool.families) if f=='receiver'][:k]
        U=orth(self.pool.vectors[:,ids])
        if len(ids)!=k or U.shape[1]!=k:raise ValueError('ACTUAL_RECEIVER_RANK_MISMATCH')
        if U.shape[1]==self.anchor.U.shape[1] and np.linalg.norm(U-self.anchor.U@(self.anchor.U.conj().T@U))<TOL:raise ValueError('own current subspace cannot certify itself')
        return ids,U
    def anchor_gain(self,x,D):
        g=self.anchor.jt(self.ctx.r+self.anchor.j(x))+self.ctx.lam*x+self.ctx.ell
        z=self.anchor.j(D)
        return -g@D-.5*(np.sum(z*z,axis=0)+self.ctx.lam*np.sum(D*D,axis=0))
    def proposals_legacy(self,U,max_pairs=16):
        if not 1<=max_pairs<=16:raise ValueError('legacy at-most16 proposals')
        outside=[i for i in range(self.pool.vectors.shape[1]) if orth(self.pool.vectors[:,i],U).shape[1]]
        families=list(('receiver','internal','fourier','anchor_PD','residual_up','random'))+sorted(set(self.pool.families)-set(('receiver','internal','fourier','anchor_PD','residual_up','random')))
        queues={f:[i for i in outside if self.pool.families[i]==f] for f in families};selected=[]
        while len(selected)<8 and any(queues.values()):
            for f in families:
                if queues[f]:selected.append(queues[f].pop(0))
                if len(selected)==8:break
        order=np.random.default_rng(20260930).permutation(U.shape[1]).tolist();proposals=[]
        for r in range(U.shape[1]):
            for j,add in enumerate(selected):
                remove=order[(r+j)%len(order)];proposed,C=_append(np.delete(U,remove,axis=1),self.pool.vectors[:,add])
                if C.shape[1]==1:proposals.append((remove,add,proposed))
                if len(proposals)>=max_pairs:break
            if len(proposals)>=max_pairs:break
        return proposals
    def run(self,name,k):
        if name not in METHODS:raise ValueError('unknown matched method')
        if self.direction:self.direction.sync()
        physical_before=self.direction.physical_snapshot() if self.direction else None
        started=time.perf_counter();stage={};stats=[];context_before=self.ctx.records();calls_before=self.direction.snapshot() if self.direction else None
        self.last_attempt=dict(method=name,k=k,started_perf_counter=started,stage_wall_s=stage,endpoint_statistics=stats,direction_before=calls_before,physical_before=physical_before)
        t=time.perf_counter();ids,U=self.receiver(k);stage['receiver_basis_wall_s']=time.perf_counter()-t
        t=time.perf_counter();out=self.engine.evaluate(U[None]);stage['receiver_endpoint_wall_s']=time.perf_counter()-t;stats.append(out['statistics'])
        if out['status']!=['OK']:raise ValueError('RECEIVER_ENDPOINT_'+out['status'][0])
        x=out['steps'][0];initial=x.copy();trace=[];basis_current=U;stop=None
        if name==METHODS[0]:
            t=time.perf_counter();proposals=self.proposals_legacy(U);stage['proposal_basis_wall_s']=time.perf_counter()-t
            t=time.perf_counter();out=self.engine.evaluate(np.stack([p[2] for p in proposals])) if proposals else None;stage['candidate_endpoint_wall_s']=time.perf_counter()-t
            if out is not None:stats.append(out['statistics'])
            t=time.perf_counter()
            for j,(remove,add,childU) in enumerate(proposals):
                row=dict(index=j,remove_column=remove,add_candidate=add,add_label=self.pool.labels[add],add_family=self.pool.families[add],status=out['status'][j],accepted=False)
                if out['status'][j]=='OK':
                    child=out['steps'][j];gain=float(self.anchor_gain(x,(child-x)[:,None])[0]);row['finite_gain']=gain
                    if gain>0:x=child;basis_current=childU;row['accepted']=True
                trace.append(row)
            stage['anchor_screen_wall_s']=time.perf_counter()-t;stop='FIXED_LEGACY_ONEPASS_COMPLETE'
        else:
            if self.direction is None:raise ValueError('legal full directed action required')
            t=time.perf_counter();Jx=self.direction.j(x);stage['Jx_wall_s']=time.perf_counter()-t
            stage.update(Jd_wall_s=0.,proposal_basis_endpoint_wall_s=0.)
            max_rounds=1 if name==METHODS[1] else 3
            for rnd in range(max_rounds):
                incoming=balanced_ids(self.pool,ids,12);ml=moves(ids,incoming,'swap',k)
                neigh=Neighborhood(self.w,self.engine,HERE,{});t=time.perf_counter();vi,steps,info,_=neigh.endpoints(ml);endpoint_total=time.perf_counter()-t;stage['proposal_basis_endpoint_wall_s']+=endpoint_total;stats.extend(neigh.timings)
                core_seconds=sum(z.get('statistics',{}).get('core_prepare_wall_s',0.) for z in neigh.timings if z.get('kind')=='shared_core_prepare')
                solve_seconds=sum(z.get('total_evaluate_wall_s',0.) for z in neigh.timings if z.get('kind')!='shared_core_prepare')
                stage['candidate_endpoint_wall_s']=stage.get('candidate_endpoint_wall_s',0.)+core_seconds+solve_seconds
                stage['proposal_basis_and_controller_overhead_wall_s']=stage.get('proposal_basis_and_controller_overhead_wall_s',0.)+max(0.,endpoint_total-core_seconds-solve_seconds)
                D=steps-x[:,None];t=time.perf_counter();q=self.anchor_gain(x,D) if len(vi) else np.empty(0);stage['anchor_screen_wall_s']=stage.get('anchor_screen_wall_s',0.)+time.perf_counter()-t
                finalists=np.argsort(-q,kind='stable')[:2]
                t=time.perf_counter();z=self.direction.j(D[:,finalists]) if len(finalists) else np.empty((len(self.ctx.r),0));stage['Jd_wall_s']+=time.perf_counter()-t
                qv=score_from_outputs(self.ctx.r+Jx,z,x,D[:,finalists],self.ctx.ell,self.ctx.lam)
                taus=gain_tolerance(self.ctx,SimpleNamespace(floor=self.floor),phi_from_output(self.ctx,x,Jx),qv)
                eligible=[j for j in range(len(finalists)) if qv[j]>taus[j]];chosen=max(eligible,key=lambda j:qv[j]) if eligible else None
                trace.append(dict(round=rnd,incoming=incoming,endpoint_count=len(ml),feasible=len(vi),failed=[dict(move_index=i,**info[i]) for i in range(len(ml)) if info[i]['status']!='OK'],finalists=[dict(move_index=int(vi[pos]),Qhat=float(q[pos]),Q_true=float(qv[j]),tau=float(taus[j])) for j,pos in enumerate(finalists)],accepted=chosen is not None))
                if chosen is None:stop='ABSTAIN_UNRESOLVED';break
                pos=int(finalists[chosen]);ids=list(ml[int(vi[pos])]['ids']);x=steps[:,pos];basis_current=orth(self.pool.vectors[:,ids]);Jx=Jx+z[:,chosen]
            if stop is None:stop='ONE_DECISION_COMPLETE' if max_rounds==1 else 'MOVE_BUDGET_EXHAUSTED'
        self.engine._sync()
        if self.direction:self.direction.sync()
        elapsed=time.perf_counter()-started
        physical_after_selection=self.direction.physical_snapshot() if self.direction else None
        # Offline endpoint evaluation occurs AFTER measured selection. No labels rank legacy.
        evaluation={};eval_before=self.direction.snapshot() if self.direction else None;te=time.perf_counter()
        if self.direction:
            js=self.direction.j(np.column_stack([initial,x]));phis=.5*np.sum((self.ctx.r[:,None]+js)**2,axis=0)+self.ctx.ell@np.column_stack([initial,x])+.5*self.ctx.lam*np.sum(np.column_stack([initial,x])**2,axis=0)
            evaluation=dict(initial_full_objective=float(phis[0]),final_full_objective=float(phis[1]),true_gain=float(phis[0]-phis[1]))
        eval_cost=self._delta(self.direction.snapshot(),eval_before) if self.direction else None
        selection_cost=self._delta(eval_before,calls_before) if self.direction else None
        physical_after_evaluation=self.direction.physical_snapshot() if self.direction else None
        return dict(method=name,actual_rank=basis_current.shape[1],requested_k=k,step=x,U=basis_current,step_sha256=arrhash(x),basis_sha256=arrhash(basis_current),trace=trace,stop=stop,warm_selection_wall_s=elapsed,stage_wall_s=stage,endpoint_statistics=stats,endpoint_fee_totals=endpoint_fee_totals(stats),physical_selection_counters_before=physical_before,physical_selection_counters_after=physical_after_selection,physical_selection_counter_delta=counter_delta(physical_after_selection,physical_before),physical_offline_evaluation_counter_delta=counter_delta(physical_after_evaluation,physical_after_selection),physical_all_counter_delta=counter_delta(physical_after_evaluation,physical_before),direction_selection=selection_cost,direction_offline_evaluation=eval_cost,offline_evaluation_wall_s=time.perf_counter()-te,evaluation=evaluation,device=self.engine.device,anchor_device='cpu',direction_device=self.setup['direction_device'],dtype='complex128/float64',cold_wall_s=None,cold_speedup=None,context_cost_before=context_before,context_cost_after=self.ctx.records(),scope='Receiver controls only; not complete S8. No scientific verdict.')
    @staticmethod
    def _delta(a,b):return {k:a[k]-b[k] for k in a}

def endpoint_fee_totals(records):
    # Only immediate paid records; nested snapshots/audit views overlap these.
    keys=['host_upload_bytes','host_download_bytes','current_factorizations','normal_factorizations','normal_solve_rhs','current_solve_rhs','public_projection_batches','core_factorizations','core_solve_rhs','core_reuse_solve_calls','core_reuse_solve_rhs']
    totals={k:0 for k in keys}
    for record in records:
        r=record['statistics'] if record.get('kind')=='shared_core_prepare' else record
        for k in keys:totals[k]+=r.get(k,0)
    return totals

def counter_delta(after,before):
    if after is None or before is None:return None
    out={}
    for key in ('totals','timings_s'):
        out[key]={k:after.get(key,{}).get(k,0)-before.get(key,{}).get(k,0) for k in sorted(set(after.get(key,{}))|set(before.get(key,{})))}
    out['event_count']=after.get('event_count',0)-before.get('event_count',0)
    out['events']=copy.deepcopy(after.get('events',[])[before.get('event_count',0):])
    return out
