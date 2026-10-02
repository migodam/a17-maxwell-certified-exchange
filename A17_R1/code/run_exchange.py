"""A17 frozen receiver-start pilot. Online capabilities exclude full labels.

Every recorded teacher neighborhood is explicit. Full gain checks use Jx/Jd,
never a full optimum or adjoint to rank an online proposal. No outer inverse
trajectory or new solver is run. Unavailable deterministic errors remain null.
"""
from pathlib import Path
from itertools import combinations
import argparse,csv,hashlib,json,sys,time,traceback
import numpy as np
from scipy import linalg as la
import pinned_runtime as runtime
from maxwell_state import frozen,public_workspace,EvaluatorContext,ProbeContext,TeacherContext,array_hash
from removal_core_batch import CoreEndpointBatch
from operators import orth
from info_logger import InfoLogger
from information_diagnostics import common_material_probes

ROOT=runtime.ROOT
CONFIG=json.loads((ROOT/'A17_RUN_CONFIG.json').read_text())
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

def tolerance(ctx,ev,x):
    # Baseline compatibility view only; exact acceptance uses gain_tolerance.
    return max(10*ev.floor,256*np.finfo(float).eps*max(float(ctx.r@ctx.r),1e-300))
def gain_tolerance(ctx,ev,base_phi,gain):
    scale=np.maximum.reduce([np.full_like(np.asarray(gain,dtype=float),float(ctx.r@ctx.r)),np.full_like(np.asarray(gain,dtype=float),abs(float(base_phi))),np.abs(base_phi-np.asarray(gain,dtype=float))])
    return np.maximum(10*ev.floor,256*np.finfo(float).eps*np.maximum(scale,1e-300))
def phi_from_output(ctx,x,Jx):
    u=ctx.r+Jx;return float(.5*(u@u)+ctx.ell@x+.5*ctx.lam*(x@x))

def evaluate_neighborhood(neigh,ids,incoming,menu,k,anchor,teacher=None,x=None,Jx=None):
    move_list=moves(ids,incoming,menu,k);valid,steps,info,build_wall=neigh.endpoints(move_list)
    ctx=neigh.w.ctx
    D=steps-x[:,None];start=time.perf_counter()
    qhat=gains_on_anchor(anchor,x,D,ctx) if len(valid) else np.empty(0)
    anchor_wall=time.perf_counter()-start
    qfull=None;Jd=None;teacher_wall_before=teacher.wall_s if teacher is not None else 0.;teacher_rhs_before=teacher.counts if teacher is not None else 0
    if teacher is not None:
        columns=[]
        for first in range(0,D.shape[1],CONFIG['cuda_candidate_batch']):columns.append(teacher.j(D[:,first:first+CONFIG['cuda_candidate_batch']]))
        Jd=np.column_stack(columns) if columns else np.empty((len(ctx.r),0))
        qfull=score_from_outputs(ctx.r+Jx,Jd,x,D,ctx.ell,ctx.lam)
    rows=[];valid_position={int(i):j for j,i in enumerate(valid)}
    for i,move in enumerate(move_list):
        row=dict(move_index=i,**move,**info[i],**neigh.identity)
        if i in valid_position:
            j=valid_position[i];row.update(Qhat=float(qhat[j]),Q_true=float(qfull[j]) if qfull is not None else None,step_hash=array_hash(steps[:,j]),step_difference_norm=float(la.norm(D[:,j])))
        else:row.update(Qhat=None,Q_true=None,step_hash=None,step_difference_norm=None)
        row['receiver_score']=float(sum(la.norm(neigh.engine.S@neigh.w.pool.vectors[:,a])**2 for a in move['add'])-sum(la.norm(neigh.engine.S@neigh.w.pool.vectors[:,a])**2 for a in move['drop'])) if neigh.engine.device=='cpu' else None
        row.update(eps_det=None,sigma_model=None,lower_gain=None,upper_gain=None,certificate_type='OFFLINE_HIGH_FIDELITY_LABEL' if teacher else 'ANCHOR_SURROGATE_ONLY')
        rows.append(row)
    return dict(rows=rows,valid=valid,steps=steps,D=D,qhat=qhat,qfull=qfull,Jd=Jd,build_wall_s=build_wall,anchor_wall_s=anchor_wall,move_list=move_list,full_label_wall_s=teacher.wall_s-teacher_wall_before if teacher is not None else 0.,full_label_rhs=teacher.counts-teacher_rhs_before if teacher is not None else 0)

def append_rows(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n')

def policy_run(name,seed_ids,k,w,engine,anchor,evaluator,teacher,probe,base_step,base_Jx,out,identity,menu='swap',first_neighborhood=None):
    ids=list(seed_ids);x=base_step.copy();Jx=base_Jx.copy();ctx=w.ctx
    is_teacher=name=='exact_score_teacher';guarded=name in ('directed_verified','adaptive_leq_k')
    accepted=[];rounds=[];screen_wall=0.;online_probe_before=probe.counts;teacher_before=teacher.counts
    teacher_wall_before=teacher.wall_s;probe_wall_before=probe.wall_s
    start=time.perf_counter();neigh=Neighborhood(w,engine,out,identity);stop='MOVE_BUDGET_EXHAUSTED'
    for roundno in range(CONFIG['max_actions']):
        incoming=[i for i in range(w.pool.vectors.shape[1]) if i not in ids] if is_teacher else balanced_ids(w.pool,ids,CONFIG['online_incoming_count'])
        tau=tolerance(ctx,evaluator,x)
        if is_teacher and roundno==0 and first_neighborhood is not None:
            nb=dict(first_neighborhood);nb['rows']=[dict(row) for row in first_neighborhood['rows']]
        else:nb=evaluate_neighborhood(neigh,ids,incoming,menu,k,anchor,teacher if is_teacher else None,x=x,Jx=Jx)
        screen_wall+=nb['build_wall_s']+nb['anchor_wall_s']
        for row in nb['rows']:row.update(policy_id=name,round=roundno,base_U_hash=array_hash(basis(w.pool.vectors,ids)),base_step_hash=array_hash(x),tau=tau)
        q=nb['qfull'] if is_teacher else nb['qhat']
        order=np.argsort(-q,kind='stable') if len(q) else np.empty(0,int)
        phi_base=phi_from_output(ctx,x,Jx) if is_teacher else float(anchor.phi(x))
        taus=gain_tolerance(ctx,evaluator,phi_base,q)
        for j,i in enumerate(nb['valid']):nb['rows'][int(i)]['tau']=float(taus[j])
        chosen=None;checks=[];zchosen=None
        if is_teacher and len(order) and q[order[0]]>taus[order[0]]:
            chosen=int(order[0]);zchosen=nb['Jd'][:,chosen]
        elif guarded:
            finalists=order[:CONFIG['max_finalists_per_round']]
            # Verify even a nonpositive anchor score: no zero-gradient shortcut.
            if len(finalists):
                z=probe.j(nb['D'][:,finalists]);qverify=score_from_outputs(ctx.r+Jx,z,x,nb['D'][:,finalists],ctx.ell,ctx.lam)
                verifytaus=gain_tolerance(ctx,evaluator,phi_from_output(ctx,x,Jx),qverify)
                for j,position in enumerate(finalists):
                    original=int(nb['valid'][position]);row=nb['rows'][original]
                    row.update(Q_true=float(qverify[j]),certificate_type='HIGH_FIDELITY_CHECK',probe_tangent_rhs=ctx.P,tau=float(verifytaus[j]))
                    checks.append(dict(move_index=original,Qhat=float(nb['qhat'][position]),Q_true=float(qverify[j]),eps_det=None,accepted_candidate=bool(qverify[j]>verifytaus[j])))
                feasible=[j for j in range(len(finalists)) if qverify[j]>verifytaus[j]]
                if feasible:
                    best=max(feasible,key=lambda j:qverify[j]);chosen=int(finalists[best]);zchosen=z[:,best]
        elif len(order) and q[order[0]]>taus[order[0]]:chosen=int(order[0])
        if chosen is None:
            stop='ABSTAIN_UNRESOLVED' if not is_teacher else 'NUMERICAL_1EXCHANGE_STOP_NO_DETERMINISTIC_BOUND'
            rounds.append(dict(round=roundno,neighborhood_total=len(nb['move_list']),neighborhood_feasible=len(nb['valid']),max_Q_true=float(np.max(q)) if is_teacher and len(q) else None,max_Qhat=float(np.max(nb['qhat'])) if len(q) else None,finalists=checks,accepted=False,stop=stop))
            append_rows(out/'moves.jsonl',nb['rows']);break
        idx=int(nb['valid'][chosen]);move=nb['move_list'][idx];tau=float(nb['rows'][idx]['tau'])
        d=nb['D'][:,chosen]
        if zchosen is None:zchosen=teacher.j(d)
        actual=float(score_from_outputs(ctx.r+Jx,zchosen[:,None],x,d[:,None],ctx.ell,ctx.lam)[0])
        # Surrogate path is intentionally ungated, even when full gain is negative.
        ids=list(move['ids']);x=nb['steps'][:,chosen];Jx=Jx+zchosen
        nb['rows'][idx].update(accepted=True,Q_true=actual,certificate_type='HIGH_FIDELITY_CHECK' if guarded else ('OFFLINE_HIGH_FIDELITY_LABEL' if is_teacher else 'UNGATED_SURROGATE_MOVE'))
        accepted.append(dict(round=roundno,drop=move['drop'],add=move['add'],kind=move['kind'],Qhat=float(nb['qhat'][chosen]),Q_true=actual,actual_rank=len(ids),tau=tau))
        rounds.append(dict(round=roundno,neighborhood_total=len(nb['move_list']),neighborhood_feasible=len(nb['valid']),max_Q_true=float(np.max(q)) if is_teacher and len(q) else None,max_Qhat=float(np.max(nb['qhat'])) if len(q) else None,finalists=checks,accepted=True,selected_move=move))
        append_rows(out/'moves.jsonl',nb['rows']);np.savez_compressed(out/f'{name}_round_{roundno}.npz',step=x,Js=Jx,selected_ids=ids,U_coeff=basis(w.pool.vectors,ids))
        jswrite(out/f'{name}_checkpoint.json',dict(policy=name,completed_round=roundno,accepted=accepted,selected_ids=ids))
    risk=evaluator.risk(x,Jx)
    initial_cached_wall=0. if first_neighborhood is None else sum(first_neighborhood.get(n,0.) for n in ('build_wall_s','anchor_wall_s','full_label_wall_s'))
    initial_cached_rhs=0 if first_neighborhood is None else first_neighborhood.get('full_label_rhs',0)
    result=dict(policy_id=name,actual_rank=len(ids),requested_k=k,selected_ids=ids,accepted=accepted,rounds=rounds,stop_reason=stop,certificate_type='HIGH_FIDELITY_CHECK' if guarded else ('OFFLINE_HIGH_FIDELITY_LABEL' if is_teacher else 'ANCHOR_SURROGATE_ONLY'),eps_det=None,risk=risk,first_cached_neighborhood_wall_s=initial_cached_wall,first_cached_neighborhood_tangent_rhs=initial_cached_rhs,standalone_path_wall_s=time.perf_counter()-start+initial_cached_wall,wall_offline_label_s=teacher.wall_s-teacher_wall_before,wall_online_probe_s=probe.wall_s-probe_wall_before,initial_teacher_neighborhood_cached=bool(first_neighborhood is not None),wall_total_s=time.perf_counter()-start,wall_screen_s=screen_wall,probe_tangent_rhs=probe.counts-online_probe_before,teacher_tangent_rhs=teacher.counts-teacher_before,adjoint_rhs=0,endpoint_statistics=neigh.timings,**identity)
    np.savez_compressed(out/f'{name}_endpoint.npz',step=x,Js=Jx,selected_ids=ids,U_coeff=basis(w.pool.vectors,ids))
    jswrite(out/f'{name}.json',clean(result))
    return result

def first_checkpoint_diagnostics(seed_ids,k,w,engine,anchor,evaluator,teacher,x,Jx,out,identity,information=None,raw_observation=None):
    ctx=w.ctx;neigh=Neighborhood(w,engine,out,identity)
    incoming=list(i for i in range(w.pool.vectors.shape[1]) if i not in seed_ids)
    nb=evaluate_neighborhood(neigh,seed_ids,incoming,'swap',k,anchor,teacher,x=x,Jx=Jx)
    tau=tolerance(ctx,evaluator,x)
    for row in nb['rows']:row.update(policy_id='receiver_neighborhood_teacher',round=0,tau=tau,base_U_hash=array_hash(basis(w.pool.vectors,seed_ids)))
    if raw_observation is not None:
        for row in nb['rows']:row['receiver_score']=float(sum(raw_observation[a] for a in row['add'])-sum(raw_observation[a] for a in row['drop']))
    append_rows(out/'teacher_receiver_moves.jsonl',nb['rows'])
    order=np.argsort(-nb['qhat'],kind='stable');selected=int(order[0]) if len(order) else None
    maximum=max(0.,float(np.max(nb['qfull']))) if len(nb['qfull']) else 0.
    chosen=float(nb['qfull'][selected]) if selected is not None and nb['qhat'][selected]>tau else 0.
    best=int(np.argmax(nb['qfull'])) if len(nb['qfull']) else None
    online=balanced_ids(w.pool,seed_ids,CONFIG['online_incoming_count'])
    subset=[j for j,i in enumerate(nb['valid']) if nb['move_list'][int(i)]['add'][0] in online]
    shortmax=max(0.,float(np.max(nb['qfull'][subset]))) if subset else 0.
    rows=nb['rows']
    if information is not None:
        incoming_fixed=set(online);diagnostic_rows=[]
        todo=[j for j,i in enumerate(nb['valid']) if nb['move_list'][int(i)]['add'][0] in incoming_fixed]
        if best is not None and best not in todo:todo.append(best)
        for j in todo:
            i=int(nb['valid'][j]);move=nb['move_list'][i];rec=information.candidate_summary(move['ids'],basis(w.pool.vectors,move['ids']),nb['steps'][:,j],spectrum=True,weak=True,evaluator=True)
            diagnostic_rows.append(dict(move_index=i,**{**move,**rec},Qhat=float(nb['qhat'][j]),Q_true=float(nb['qfull'][j]),receiver_score=rows[i]['receiver_score']))
        append_rows(out/'information_rankings_by_move.jsonl',diagnostic_rows)
        jswrite(out/'information_coverage.json',dict(full_move_count=len(nb['move_list']),spectrum_move_count=len(todo),fixed_incoming_ids=online,teacher_best_added_for_evaluation_only=best is not None and best not in [j for j,i in enumerate(nb['valid']) if nb['move_list'][int(i)]['add'][0] in incoming_fixed],selection_uses_information=False,scope='Fixed twelve incoming per removal + teacherbest; remaining moves trace only, no full spectrum coverage claim.'))
    diag=dict(**identity,declared_neighborhood_count=len(nb['move_list']),feasible_count=len(nb['valid']),teacher_best_gain=maximum,shortlist_best_gain=shortmax,omitted_opportunity=maximum-shortmax,surrogate_selected_true_gain=chosen,same_checkpoint_score_regret=maximum-chosen,no_op_included=True,tau=tau,nearzero_teacher=bool(maximum<=tau),full_domain_1exchange_coverage=True,deterministic_certificate=False,core_optimization='endpoint batching; Schur proof audited separately',wall_build_s=nb['build_wall_s'],wall_anchor_s=nb['anchor_wall_s'],teacher_tangent_rhs=teacher.counts)
    if best is not None:diag['teacher_best_move']=nb['move_list'][int(nb['valid'][best])]
    jswrite(out/'receiver_checkpoint.json',clean(diag))
    # 2x2: SAME feasible incoming choices and SAME actual endpoints.
    small=[j for j,i in enumerate(nb['valid']) if nb['move_list'][int(i)]['add'][0] in online]
    base=w.ctx.make_model(basis(w.pool.vectors,seed_ids));e=ctx.r+base.j(x);vx=[];dx=[];ident=[]
    for j in small:
        i=int(nb['valid'][j]);child=ctx.make_model(basis(w.pool.vectors,nb['move_list'][i]['ids']))
        Ds=child.j(x)-base.j(x);de=child.jt(e)-base.jt(e)
        v=-base.solve_H(base.jt(Ds)+de);vx.append(v);dx.append(nb['D'][:,j]);ident.append(nb['move_list'][i])
    if vx:
        V=np.column_stack(vx);D=np.column_stack(dx);jV=[]
        for j in range(0,V.shape[1],16):jV.append(teacher.j(V[:,j:j+16]))
        JV=np.column_stack(jV);JD=np.column_stack([nb['Jd'][:,j] for j in small]);U=ctx.r+Jx
        scores={}
        for nm,W,JW in (('v',V,JV),('d',D,JD)):
            scores[nm+'_full_linear']=score_from_outputs(U,JW,x,W,ctx.ell,ctx.lam,False)
            scores[nm+'_full_quadratic']=score_from_outputs(U,JW,x,W,ctx.ell,ctx.lam,True)
            scores[nm+'_anchor_linear']=gains_on_anchor(anchor,x,W,ctx,False)
            scores[nm+'_anchor_quadratic']=gains_on_anchor(anchor,x,W,ctx,True)
        table=[]
        for j,move in enumerate(ident):table.append(dict(move=move,actual_full_gain=float(scores['d_full_quadratic'][j]),direction_error=float(la.norm(V[:,j]-D[:,j])),**{nm:float(a[j]) for nm,a in scores.items()}))
        rankings={nm:dict(selected_index=int(np.argmax(a)),executed_actual_d_gain=float(scores['d_full_quadratic'][np.argmax(a)])) for nm,a in scores.items()}
        jswrite(out/'direction_curvature_2x2.json',clean(dict(rows=table,rankings=rankings,scope='Same checkpoint and candidate endpoints. v only a score direction; all choices execute actual d. No path-level causal claim.')))
    # Pair/forced-add diagnostic on a fixed <=k core and six family-balanced atoms.
    keep=seed_ids[:-2];inc=balanced_ids(w.pool,seed_ids,6)
    coreU=basis(w.pool.vectors,keep);core=engine.evaluate(coreU[None]);sx=core['steps'][0]
    if core['status'][0]=='OK':
        jsx=teacher.j(sx);proposal=[]
        for add in inc:proposal.append(dict(kind='singleton',drop=[],add=[add],ids=keep+[add]))
        for adds in combinations(inc,2):proposal.append(dict(kind='pair',drop=[],add=list(adds),ids=keep+list(adds)))
        vi,ss,inf,ww=neigh.endpoints(proposal);DD=ss-sx[:,None];jdd=teacher.j(DD);qq=score_from_outputs(ctx.r+jsx,jdd,sx,DD,ctx.ell,ctx.lam)
        qq_by={tuple(proposal[int(i)]['add']):float(qq[j]) for j,i in enumerate(vi)}
        pairs=[];witness=None
        for adds in combinations(inc,2):
            qa,qb,qp=qq_by.get((adds[0],)),qq_by.get((adds[1],)),qq_by.get(tuple(adds))
            interaction=qp-qa-qb if all(z is not None for z in (qa,qb,qp)) else None
            row=dict(add_ids=adds,Q_a=qa,Q_b=qb,Q_pair=qp,interaction=interaction,base_rank=len(keep),pair_rank=len(keep)+2)
            if qa is not None and qb is not None and qp is not None and qa<=tau and qb<=tau and qp>tau and witness is None:
                witness=row
                tr=ctx.transfer(coreU,w.pool.vectors[:,list(adds)])
                np.savez_compressed(out/'Maxwell_pair_witness.npz',core_ids=keep,added_ids=adds,Gamma=tr.Gamma,Y=tr.Y,N=tr.N,core_step=sx,child_step=ss[:,list(vi).index(next(i for i,mm in enumerate(proposal) if tuple(mm['add'])==tuple(adds)))],residual=ctx.r,ell=ctx.ell,lambda_total=ctx.lam)
            pairs.append(row)
        jswrite(out/'pair_forced_dimension.json',clean(dict(incoming_ids=inc,core_ids=keep,singleton_gains=[dict(add=i,Q=qq_by.get((i,))) for i in inc],pairs=pairs,Maxwell_singletons_nonpositive_pair_positive_witness=witness,scope='Fixed physical Maxwell core, <=k additions; a positive pair is not automatically a 1-swap trap.')))
    # Limited two swaps around the original receiver: fixed 4-out x 6-in.
    trial=[]
    for drops in combinations(seed_ids[:4],2):
        for adds in combinations(inc,2):trial.append(dict(kind='two_swap',drop=list(drops),add=list(adds),ids=[i for i in seed_ids if i not in drops]+list(adds)))
    vi,ss,inf,ww=neigh.endpoints(trial)
    if len(vi):
        DD=ss-x[:,None];columns=[]
        for first in range(0,DD.shape[1],16):columns.append(teacher.j(DD[:,first:first+16]))
        qq=score_from_outputs(ctx.r+Jx,np.column_stack(columns),x,DD,ctx.ell,ctx.lam)
        jswrite(out/'restricted_two_swap.json',clean(dict(endpoint_count=len(trial),feasible=len(vi),best_gain=float(np.max(qq)),best_move=trial[int(vi[np.argmax(qq)])],all_moves=[dict(**trial[int(i)],Q_true=float(qq[j]),status=inf[int(i)]['status']) for j,i in enumerate(vi)],comparison_one_swap=maximum,one_swap_trap_witness=bool(maximum<=tau and np.max(qq)>tau),scope='Fixed restricted 2-swap domain; not arbitrary pair/Grassmann optimum.')))
    return diag,nb

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    jswrite(out/'status.json',dict(status='RUNNING',phase='setup',object_id=args.object,state=args.phase))
    bundle=frozen(args.object,args.phase,args.device);ctx=bundle['ctx'];model=bundle['state'].model
    work=public_workspace(bundle);w=work['workspace'];cctx=w.ctx
    evaluator=EvaluatorContext(bundle);teacher=TeacherContext(bundle['ev'].j,ctx.P,evaluator.offline_gaussian_J())
    probe=ProbeContext(bundle['ev'].j,ctx.P)
    V=fullJV=probe_meta=None;info_reference_wall=0.;info_reference_rhs=0
    if ctx.dim_material>128:
        V,probe_meta=common_material_probes(ctx.dim_material,args.object,precision=bundle['information_metadata']['prior_precision'],count=16,seed=CONFIG['seed'])
        t=time.perf_counter()
        with ctx.phase('offline_information_public_probes'):fullJV=teacher.j(V)
        info_reference_wall=time.perf_counter()-t;info_reference_rhs=ctx.P*V.shape[1]
    if teacher.J is not None:np.savez_compressed(out/'offline_information_reference.npz',fullJ=teacher.J,scope=np.asarray('EVALUATOR_ONLY_ALREADY_PAID'))
    else:np.savez_compressed(out/'offline_information_reference.npz',V=V,fullJV=fullJV,probe_metadata=np.asarray(json.dumps(probe_meta)),scope=np.asarray('PROJECTED_ONLY_ALREADY_PAID'))
    raw_observation=np.sum(np.abs(work['S']@w.pool.vectors)**2,axis=0)
    engine=CoreEndpointBatch(work['L'],work['S'],work['PB'],ctx.r,ctx.ell,ctx.lam,device='cuda' if args.device=='cuda' and ctx.q>=256 else 'cpu',batch_size=CONFIG['cuda_candidate_batch'])
    np.savez_compressed(out/'public_workspace.npz',Q=w.Q,pool=w.pool.vectors,anchor=w.anchor.U,L=work['L'],S=work['S'],PB=work['PB'],labels=np.asarray(w.pool.labels),families=np.asarray(w.pool.families),r=ctx.r,ell=ctx.ell,lambda_total=ctx.lam)
    jswrite(out/'runtime_receipt.json',clean(dict(runtime=runtime.receipt(),inputs=bundle['inputs'],state_hash=bundle['state_hash'],pool_hash=work['pool_hash'],config_sha256=hashlib.sha256((ROOT/'A17_RUN_CONFIG.json').read_bytes()).hexdigest(),actual_dtype='complex128/float64',full_rhs_layout='all illumination share one real material direction',device=args.device,numerical_monkeypatch=False,material_gauge=ctx.material_gauge,information_metadata=bundle['information_metadata'])))
    setup=dict(physical_state_wall_s=bundle['physical_setup_wall_s'],public_workspace_wall_s=work['workspace_wall_s'],anchor_wall_s=work['anchor_meta']['wall_s'],pool_wall_s=work['pool_wall_s'],workspace_projection_wall_s=w.metadata['setup_wall_s'],endpoint_engine_setup_wall_s=engine.setup_wall_s,offline_reference_verification_wall_s=evaluator.wall_s,offline_Gaussian_full_J_acquisition_wall_s=evaluator.dense_J_wall_s,offline_Gaussian_full_J_acquisition_rhs=evaluator.dense_J_tangent_rhs,offline_information_public_probe_wall_s=info_reference_wall,offline_information_public_probe_rhs=info_reference_rhs,information_metadata=bundle['information_metadata'],public_embedding_dimension=w.Q.shape[1],current_rows=ctx.n_current,material_dimension=ctx.dim_material,cold_charge_rule='state and workspace are outer nonoverlapping intervals; anchor/pool/projection detail is nested, do not double sum.')
    jswrite(out/'setup_cost.json',clean(setup))
    records=[]
    for k in args.k:
        target=out/f'k{k}';target.mkdir()
        jswrite(out/'status.json',dict(status='RUNNING',phase='cell',object_id=args.object,state=args.phase,k=k,finished_k=[r['k'] for r in records]))
        receiver_ids=[i for i,f in enumerate(w.pool.families) if f=='receiver'][:k]
        if len(receiver_ids)!=k or basis(w.pool.vectors,receiver_ids).shape[1]!=k:raise ValueError('requested receiver rank infeasible')
        x0_result=engine.evaluate(basis(w.pool.vectors,receiver_ids)[None])
        if x0_result['status'][0]!='OK':raise ValueError('receiver infeasible '+x0_result['status'][0])
        x=x0_result['steps'][0];initwall=probe.wall_s;jx=probe.j(x);Jx_initialization_wall_s=probe.wall_s-initwall
        identity=dict(object_id=args.object,family=bundle['scene']['family'],material_basis=bundle['scene']['representation'],phase=args.phase,k=k,state_hash=bundle['state_hash'],pool_hash=work['pool_hash'])
        receiver=dict(policy_id='receiver_no_swap',risk=evaluator.risk(x,jx),actual_rank=basis(w.pool.vectors,receiver_ids).shape[1],selected_ids=receiver_ids,wall_total_s=x0_result['statistics']['total_evaluate_wall_s'],extra_online_full_rhs=0,scope='receiver baseline uses no full probe online; Jx above paid validation/shared verification cache',**identity)
        jswrite(target/'receiver_no_swap.json',clean(receiver))
        metadata=bundle['information_metadata']
        information=InfoLogger(cctx,precision=metadata['prior_precision'],fullJ=teacher.J,V=V,fullJV=fullJV,probe_metadata=probe_meta,full_tangent_rhs=info_reference_rhs,metric_hash=metadata['physical_metric_hash'],whitening_scope=metadata['data_whitening_scope'],whitening_hash=metadata['data_whitening_hash'],material_basis_hash=metadata['physical_metric_hash'],output_dir=target/'information_rhs')
        receiver['information']=information.base_summary(receiver_ids,basis(w.pool.vectors,receiver_ids),x,evaluator=True)
        jswrite(target/'receiver_no_swap.json',clean(receiver))
        diag,initial_neighborhood=first_checkpoint_diagnostics(receiver_ids,k,w,engine,w.anchor,evaluator,teacher,x,jx,target,identity,information,raw_observation)
        policies=[]
        for name,menu in (('surrogate_exchange','swap'),('exact_score_teacher','swap'),('directed_verified','swap'),('adaptive_leq_k','adaptive')):
            result=policy_run(name,receiver_ids,k,w,engine,w.anchor,evaluator,teacher,probe,x,jx,target,identity,menu,initial_neighborhood if name=='exact_score_teacher' else None)
            guarded=name in ('directed_verified','adaptive_leq_k')
            result['Jx_initialization_wall_s']=Jx_initialization_wall_s if guarded else 0.
            result['Jx_initialization_tangent_rhs']=ctx.P if guarded else 0
            result['standalone_extra_online_tangent_rhs']=result['probe_tangent_rhs']+(ctx.P if guarded else 0)
            result['deployment_wall_excluding_teacher_s']=None if name=='exact_score_teacher' else result['wall_total_s']-result['wall_offline_label_s']+(Jx_initialization_wall_s if guarded else 0.)
            result['cold_setup_wall_s']=bundle['physical_setup_wall_s']+work['workspace_wall_s']+engine.setup_wall_s
            result['standalone_cold_wall_s']=None if name=='exact_score_teacher' else result['cold_setup_wall_s']+result['deployment_wall_excluding_teacher_s']+receiver['wall_total_s']
            # Offline truth only, no clipping/line-search: explicitly Level 2 diagnostic.
            final=np.load(target/f'{name}_endpoint.npz');s=final['step']
            dc=bundle['tangent'].expand(s);chi=bundle['anchor']['chi']+dc;truth=bundle['common']['truth'];initial=bundle['anchor']['chi']
            result['information']=information.final_summary(result['selected_ids'],basis(w.pool.vectors,result['selected_ids']),s,evaluator=True)
            result['one_step_truth']=dict(relative_material_error=float(la.norm(chi-truth)/la.norm(truth)),initial_relative_material_error=float(la.norm(initial-truth)/la.norm(truth)),constraints_violated=bool(chi.imag.min()<-1e-8 or chi.real.min()<-.50000001),scope='Unconstrained alpha=1 diagnostic; not accepted nonlinear reconstruction, not guaranteed by Level-1 gain.')
            result['final_nonlinear_reconstruction']=dict(status='NOT_RUN',reason='No new outer trajectory authorized in A17 plan.')
            jswrite(target/f'{name}.json',clean(result));policies.append(result)
        jswrite(target/'information_cost.json',dict(total_logged_wall_s=sum(r['wall_total_s'] for r in information.records),logged_endpoints=len(information.records),full_actions_called_by_logger=0,setup_probe_cost_shared_once=True,scope='Offline diagnostics separate from policy timing; reference views repeated but acquisition charged once.'))
        rec=dict(k=k,receiver=receiver,checkpoint=diag,policies=policies)
        records.append(rec);jswrite(out/'cells.json',clean(records))
        print(json.dumps(dict(event='cell_completed',object=args.object,phase=args.phase,k=k,elapsed_s=time.perf_counter()-started,receiver_risk=receiver['risk']['full_gap'],teacher_risk=policies[1]['risk']['full_gap'],verified_risk=policies[2]['risk']['full_gap']),allow_nan=False),flush=True)
    peak={}
    if args.device=='cuda':
        import torch
        torch.cuda.synchronize();peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    result=dict(status='COMPLETED',object_id=args.object,phase=args.phase,records=len(records),completed_k=args.k,wall_total_s=time.perf_counter()-started,setup=setup,ctx_cost=ctx.records(),full_model_counters=model.counters.as_dict(),teacher_rhs=teacher.counts,teacher_full_action_wall_s=teacher.wall_s,online_probe_rhs=probe.counts,online_probe_wall_s=probe.wall_s,peak_memory=peak,cells=records,final_runtime=runtime.receipt())
    jswrite(out/'result.json',clean(result));jswrite(out/'status.json',dict(status='COMPLETED',object_id=args.object,phase=args.phase,completed_k=args.k,wall_total_s=result['wall_total_s']))
    print(json.dumps(dict(event='terminal',object=args.object,phase=args.phase,wall_s=result['wall_total_s'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--object',type=int,required=True);p.add_argument('--phase',choices=['early','middle','late'],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--k',type=int,nargs='+',default=[4,8,16]);p.add_argument('--out',required=True)
    args=p.parse_args()
    if any(k not in CONFIG['k'] for k in args.k):p.error('k outside frozen contract')
    try:run(args)
    except Exception as exc:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        jswrite(out/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),object_id=args.object,phase=args.phase))
        jswrite(out/'status.json',dict(status='FAILED',object_id=args.object,state=args.phase,error=repr(exc),result_file_present=(out/'result.json').exists()))
        raise
