"""Bounded new 3D comparison; old physical kernels/data remain read-only."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='4'
import argparse, copy, gc, importlib.util, json, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'SHAPES_3D_20261030'
_spec=importlib.util.spec_from_file_location('readonly_previous_shapes',OLD/'run_shapes.py')
old=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=old;_spec.loader.exec_module(old)
sys.path.insert(0,str(ROOT/'research/delegated/endpoint'))
sys.path.insert(0,str(ROOT/'research/delegated/som'))
from refined_endpoint_batch import RefinedCoreEndpointBatch
from shapes import truth_at, TARGETS
M=old.M;op=old.op;sel=old.sel
CASES=['digit3','letter_e','torus','double_torus','cross3d','offset_shell']
POLICIES=['full_gn','receiver','projected_internal','fourier','random','exchange']
CFG=dict(n=12,truth_n=12,edge=1.,wavenumber=float(4*np.pi),receiver_count=64,
         heldout_count=80,noise=.01,k=16,prior=1e-5,lm0=.01,lm_decay=.3,
         outer_cap=18,armijo=1e-4,max_trials=24,residual_stop=.011,
         init=[.1,.01],threads=4,precision='complex128/float64',
         wall_cap_s=7200.,state_cap=4096,rhs_cap=150000,trajectory_cap_s=1200.,
         cases=CASES,policies=POLICIES,seeds=list(range(2026103101,2026103107)),
         candidate_seed=20260930,random_basis_seed=2026103100,
         same_grid=True,native_SOM_or_TSOM=False,
         endpoint_implementation='FP64_original_plus_max3_true_residual_refinement',
         source='original_3D_vector_Maxwell_DDA',no_training=True)
write=old.write

class BudgetStop(RuntimeError):pass

class Budget:
    def __init__(self,phase):
        self.path=ROOT/'COST_STATE.json';self.log=ROOT/'COST_LEDGER.jsonl'
        self.data=json.loads(self.path.read_text()) if self.path.exists() else dict(states=0,rhs=0,wall_s=0.,sessions=[])
        if any(s['status']=='RUNNING' for s in self.data['sessions']):raise RuntimeError('UNRESOLVED_ACTIVE_SESSION')
        self.start=time.perf_counter();self.phase=phase;self.stage=phase
        self.data['sessions'].append(dict(phase=phase,status='RUNNING',pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        self.flush()
    def check(self):
        if self.data['wall_s']+time.perf_counter()-self.start>=CFG['wall_cap_s']:raise BudgetStop('WALL_CAP')
    def charge(self,kind,count=1):
        self.check();key='states' if kind=='state' else 'rhs';cap=CFG['state_cap' if kind=='state' else 'rhs_cap']
        if self.data[key]+count>cap:raise BudgetStop(kind.upper()+'_CAP')
        self.data[key]+=count
        with self.log.open('a') as f:f.write(json.dumps(dict(kind=kind,count=int(count),phase=self.phase,stage=self.stage,elapsed_s=time.perf_counter()-self.start,total=self.data[key]))+'\n')
        self.flush()
    def flush(self):
        d=copy.deepcopy(self.data);d['active_wall_s']=time.perf_counter()-self.start;write(self.path,d)
    def close(self,status):
        elapsed=time.perf_counter()-self.start;self.data['wall_s']+=elapsed
        self.data['sessions'][-1].update(status=status,wall_s=elapsed,ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write(self.path,self.data)

def new_model(budget,n=None):
    budget.check();p,v=old.grid(n or CFG['n'],CFG['edge'])
    model=old.DenseDDA(p,v,CFG['wavenumber'],*old.acquisition(),device='cpu')
    model.solver=old.CountedSolver(model.solver,budget)
    return model,old.Tangent(p,v,'voxel')

def full_normal(J,ctx):
    """Same real GN problem, complex-linear realization halves factor dimension."""
    q=ctx.q;packed=J[:,:q].reshape(ctx.P,2*ctx.m,q)
    T=(packed[:,:ctx.m]+1j*packed[:,ctx.m:]).reshape(ctx.P*ctx.m,q)
    r=ctx.r.reshape(ctx.P,2*ctx.m);r=(r[:,:ctx.m]+1j*r[:,ctx.m:]).reshape(-1)
    ell=ctx.ell[:q]+1j*ctx.ell[q:]
    H=T.conj().T@T;H.flat[::q+1]+=ctx.lam
    g=T.conj().T@r+ell
    fac=la.cho_factor(H,lower=True,check_finite=False)
    c=la.cho_solve(fac,-g,check_finite=False);s=np.r_[c.real,c.imag]
    h=J.T@ctx.r+ctx.ell
    rr=float(la.norm(J.T@(J@s)+ctx.lam*s+h)/max(la.norm(h),1e-300))
    if rr>1e-9:raise RuntimeError('FULL_NORMAL_RESIDUAL_'+str(rr))
    return s,h,rr

def baseline_basis(ctx,policy):
    k=CFG['k'];meta={}
    if policy=='receiver':U=op.orth(ctx.receiver_modes(k),rank=k)
    elif policy=='projected_internal':
        U=op.orth(ctx.receiver_modes(k//2),rank=k//2)
        bank,meta=ctx.internal_modes(32,CFG['candidate_seed'])
        for col in bank.T:
            U=op.orth(np.column_stack([U,col]),rank=min(k,U.shape[1]+1))
            if U.shape[1]==k:break
    elif policy=='fourier':
        bank,labels=sel.fourier_modes(ctx.points,ctx.n_current,32)
        U=op.orth(bank,rank=k);meta=dict(fourier_labels=labels[:k])
    elif policy=='random':
        rng=np.random.default_rng(CFG['random_basis_seed'])
        U=op.orth(rng.normal(size=(ctx.n_current,k))+1j*rng.normal(size=(ctx.n_current,k)),rank=k)
    else:raise ValueError(policy)
    if U.shape[1]!=k:raise RuntimeError('INFEASIBLE_EXACT_CURRENT_RANK')
    return U,meta

def endpoint(ctx,U):
    k=U.shape[1];engine=RefinedCoreEndpointBatch(U.conj().T@ctx.apply_L(U),ctx.apply_S(U),ctx.project_B(U),ctx.r,ctx.ell,ctx.lam,device='cpu',batch_size=16)
    e=engine.evaluate(np.eye(k,dtype=complex)[None])
    if e['status']!=['OK']:raise RuntimeError('ENDPOINT_'+str(e['status']))
    return e['steps'][0],e

def validate(b):
    """One substantive physical check for the new numerical adapter."""
    b.stage='new_adapter_validation';model,tan=new_model(b,5)
    p=model.points;chi=.12+.01j+.3*np.exp(-np.sum(p*p,axis=1)/.12)
    state=model.state(chi);scale=float(la.norm(state.field));rng=np.random.default_rng(2026103100)
    ctx,ev=op.make_problem(state,tan,state.field*.9,scale,3.43e-5,np.full(tan.d,1e-5))
    J,jmeta=old.assemble_full_material_jacobian(state,tan,scale)
    s,h,res=full_normal(J,ctx);dense,_,_=old.normal_reference(J,ctx)
    Z=rng.normal(size=(tan.d,2));W=rng.normal(size=(J.shape[0],2))
    errors=dict(full_complex_vs_real=float(la.norm(s-dense)/la.norm(dense)),
        full_normal=res,jvp=float(la.norm(J@Z-ev.j(Z))/la.norm(J@Z)),
        vjp=float(la.norm(J.T@W-ev.jt(W))/la.norm(J.T@W)))
    U,_=baseline_basis(ctx,'receiver');step,e=endpoint(ctx,U)
    rm=ctx.make_model(U);jr=rm.j(np.eye(tan.d));ds,_,_=old.normal_reference(jr,ctx)
    errors['reduced_endpoint_vs_dense']=float(la.norm(step-ds)/max(la.norm(ds),1e-300))
    w,L,S,PB,_=old.get_workspace(ctx)
    engine=RefinedCoreEndpointBatch(L,S,PB,ctx.r,ctx.ell,ctx.lam,device='cpu',batch_size=16)
    direction=old.original.DirectionAction(ev.j,ctx.P,counter_supplier=model.counters.as_dict)
    result,ee=old.original.run_original_exchange(w.ctx,w,engine,w.anchor,direction,16,ROOT/'verification/exchange',identity={'new_numeric_version':'refinement_v1'})
    x=ee['initial_step'];d=ee['step']-x;g=J.T@(ctx.r+J@x)+ctx.ell+ctx.lam*x
    Q=float(-g@d-.5*(la.norm(J@d)**2+ctx.lam*(d@d)))
    errors['directional_Q_abs']=abs(Q-sum(a['Q_true'] for a in result['accepted']))
    errors['Jx_relative']=float(la.norm(J@ee['step']-ee['Js'])/max(la.norm(ee['Js']),1e-300))
    ok=max(errors.values())<1e-8
    write(ROOT/'verification/IMPLEMENTATION.json',dict(status='PASS' if ok else 'FAIL',errors=errors,counters=model.counters.as_dict(),full_J_to_selector=False,module_paths={k:str(getattr(v,'__file__','')) for k,v in M.items()},endpoint=e['statistics'],jacobian=jmeta))
    print(json.dumps(dict(event='VALIDATION',status='PASS' if ok else 'FAIL',errors=errors)),flush=True)
    if not ok:raise RuntimeError('IMPLEMENTATION_FAIL')

def inputs(b):
    model,tan=new_model(b);p=model.points
    for i,name in enumerate(CASES):
        path=ROOT/'inputs'/f'{name}.npz'
        if path.exists():continue
        b.stage='synthesis/'+name;started=time.perf_counter();truth=truth_at(name,p)
        state=model.state(truth);exact=state.field.copy();held=old.held_observations(state)
        rng=np.random.default_rng(CFG['seeds'][i]);noise=rng.normal(size=exact.shape)+1j*rng.normal(size=exact.shape)
        noise*=CFG['noise']*la.norm(exact)/la.norm(noise);data=exact+noise
        path.parent.mkdir(exist_ok=True);np.savez_compressed(path,points=p,truth=truth,init=np.full(len(p),complex(*CFG['init'])),data=data,data_clean=exact,noise=noise,scale=float(la.norm(data)),held_truth=held)
        write(path.with_suffix('.json'),dict(case=name,definition=TARGETS[name],seed=CFG['seeds'][i],same_grid=True,truth_n=12,inverse_n=12,truth_voxels=int(np.sum(truth.real>=.5)),synthesis_wall_s=time.perf_counter()-started))
        print(json.dumps(dict(event='INPUT',case=name,voxels=int(np.sum(truth.real>=.5)))),flush=True)
        del state;gc.collect()

def run_case(name,policy,b):
    out=ROOT/'results'/name/policy
    if out.exists():raise RuntimeError('REFUSE_EXISTING_RESULT_'+str(out))
    out.mkdir(parents=True);start=time.perf_counter();records=[];state=None;model=None;status='RUNNING';U=None
    inp=dict(np.load(ROOT/'inputs'/f'{name}.npz'));chi=inp['init'].copy();init=chi.copy();truth=inp['truth'];scale=float(inp['scale'])
    try:
        t=time.perf_counter();model,tan=new_model(b);setup=time.perf_counter()-t
        for it in range(CFG['outer_cap']):
            b.check()
            if time.perf_counter()-start>=CFG['trajectory_cap_s']:raise BudgetStop('TRAJECTORY_WALL_CAP')
            b.stage=f'{name}/{policy}/update_{it}';begin=time.perf_counter();ph={};t=time.perf_counter()
            if state is None:state=model.state(chi)
            reg=tan.project(chi-init);lam=CFG['prior']+CFG['lm0']*CFG['lm_decay']**(it//3);ell=CFG['prior']*reg
            ctx,ev=op.make_problem(state,tan,inp['data'],scale,lam,ell)
            objective=float(.5*(ctx.r@ctx.r)+.5*CFG['prior']*(reg@reg))
            before=old.metrics(chi,truth,state,inp['data'],scale,inp['held_truth'])
            ph['linearization_s']=time.perf_counter()-t
            if before['measured_relative_residual']<=CFG['residual_stop']:status='DISCREPANCY_REACHED';break
            t=time.perf_counter();info={};J=None
            if policy=='full_gn':
                J,jmeta=old.assemble_full_material_jacobian(state,tan,scale)
                proposal,h,rr=full_normal(J,ctx);info=dict(normal_residual=rr,actual_rank=model.n,jacobian=jmeta)
            else:
                h=ev.jt(ctx.r)+ell
                if policy=='exchange':
                    w,L,S,PB,ameta=old.get_workspace(ctx)
                    engine=RefinedCoreEndpointBatch(L,S,PB,ctx.r,ell,lam,device='cpu',batch_size=16)
                    direction=old.original.DirectionAction(ev.j,ctx.P,counter_supplier=model.counters.as_dict)
                    result,ee=old.original.run_original_exchange(w.ctx,w,engine,w.anchor,direction,CFG['k'],out/f'exchange_{it:02d}',identity={'case':name,'iteration':it,'numerical_version':'refinement_v1'})
                    proposal=ee['step'].copy();U=w.Q@op.orth(w.pool.vectors[:,result['selected_ids']])
                    info=dict(actual_rank=U.shape[1],accepted_exchanges=len(result['accepted']),direction_cost=result['direction_cost'],endpoint_statistics=result['endpoint_statistics'],stop_reason=result['stop_reason'])
                    write(out/f'policy_{it:02d}.json',info)
                    del w,L,S,PB,engine,direction,ee,result
                else:
                    U,meta=baseline_basis(ctx,policy);proposal,e=endpoint(ctx,U)
                    info=dict(actual_rank=U.shape[1],normal_residual=float(e['normal_relative_residual'][0]),endpoint_statistics=e['statistics'],basis_metadata=meta)
            ph['policy_s']=time.perf_counter()-t
            t=time.perf_counter();step,projection=M['feasible_step'].project_step(tan,chi,proposal)
            fallback=False
            if h@step>=0:
                step,projection=M['feasible_step'].project_step(tan,chi,-h/max(lam,la.norm(h),1e-8));fallback=True
            slope=float(h@step);dc=tan.expand(step);ph['projection_s']=time.perf_counter()-t
            alpha=1.;trials=[];accepted=False;t=time.perf_counter()
            for trialno in range(CFG['max_trials']):
                if slope>=0 or la.norm(step)<=1e-15:break
                candidate=chi+alpha*dc
                if candidate.imag.min()<-1e-9 or candidate.real.min()<-.500000001:raise RuntimeError('FIXED_RAY_FEASIBILITY')
                trial=model.state(candidate);rg=tan.project(candidate-init);rrr=(trial.field-inp['data'])/scale
                value=float(.5*la.norm(rrr)**2+.5*CFG['prior']*(rg@rg));ok=value<=objective+CFG['armijo']*alpha*slope
                trials.append(dict(alpha=alpha,objective=value,accepted=bool(ok),source_residual=trial.source_residual()))
                if ok:chi,state=candidate,trial;accepted=True;break
                del trial;alpha*=.5
            ph['line_search_s']=time.perf_counter()-t
            after=old.metrics(chi,truth,state,inp['data'],scale,inp['held_truth'])
            row=dict(iteration=it,accepted=accepted,accepted_update_count=sum(r['accepted'] for r in records)+int(accepted),alpha=alpha,gradient_fallback=fallback,slope=slope,objective_before=objective,objective_after=trials[-1]['objective'] if accepted else objective,trials=trials,metrics=after,before_metrics=before,step_norm=float(la.norm(step)),executed_step_norm=float(la.norm(alpha*step)) if accepted else 0.,phases=ph,wall_s=time.perf_counter()-begin,**info)
            records.append(row);write(out/'trajectory.json',records)
            np.savez_compressed(out/f'chi_{it+1:02d}.npz',chi=chi)
            write(out/'progress.json',dict(case=name,policy=policy,attempts=len(records),updates=sum(r['accepted'] for r in records),metrics=after,elapsed_s=time.perf_counter()-start,status='RUNNING'))
            print(json.dumps(dict(event='UPDATE',case=name,policy=policy,iteration=it+1,nrmse=after['complex_nrmse'],residual=after['measured_relative_residual'],iou=after['support_iou'],wall_s=row['wall_s'],accepted=accepted,exchanges=info.get('accepted_exchanges',0))),flush=True)
            del ctx,ev,J,proposal;gc.collect();b.flush()
            if not accepted:status='NO_ARMIJO_ACCEPT';break
            if row['executed_step_norm']<1e-6 and it>=9:status='SMALL_STEP';break
        else:status='MAX_UPDATES'
    except BudgetStop as exc:status='BUDGET_STOP_'+str(exc)
    except Exception as exc:
        status='ERROR';write(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
    finally:
        fm=None if state is None else old.metrics(chi,truth,state,inp['data'],scale,inp['held_truth'],held=True)
        np.savez_compressed(out/'final.npz',chi=chi,truth=truth,points=inp['points'])
        if U is not None:np.savez_compressed(out/'final_current_basis.npz',U=U)
        final=dict(case=name,policy=policy,status=status,attempted_updates=len(records),updates=sum(r['accepted'] for r in records),metrics=fm,wall_s=time.perf_counter()-start,model_setup_s=locals().get('setup'),counters=None if model is None else model.counters.as_dict(),same_grid=True,native_SOM_or_TSOM=False)
        write(out/'final.json',final);print(json.dumps(dict(event='FINAL',**{k:final[k] for k in ('case','policy','status','updates','wall_s')},metrics=fm)),flush=True)
        del model,state;gc.collect()
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validate','run'],required=True);args=p.parse_args()
    config=ROOT/'CONFIG.json'
    if config.exists() and json.loads(config.read_text())!=CFG:raise RuntimeError('FROZEN_CONFIG_DIFFERENT')
    write(config,CFG);b=Budget(args.phase);status='ERROR';outcomes=[]
    try:
        if args.phase=='validate':validate(b)
        else:
            if json.loads((ROOT/'verification/IMPLEMENTATION.json').read_text())['status']!='PASS':raise RuntimeError('NO_IMPLEMENTATION_PASS')
            inputs(b)
            for i,name in enumerate(CASES):
                order=POLICIES[i:]+POLICIES[:i]
                for policy in order:
                    b.check();s=run_case(name,policy,b);outcomes.append(dict(case=name,policy=policy,status=s))
                    write(ROOT/'OUTCOMES.json',outcomes)
                    if s.startswith('BUDGET_STOP'):raise BudgetStop(s)
                print(json.dumps(dict(event='OBJECT_COMPLETE',case=name,wall_s=b.data['wall_s']+time.perf_counter()-b.start)),flush=True)
        status='ALL_PLANNED_TERMINAL' if args.phase=='run' else 'COMPLETE'
    except BudgetStop as exc:
        status='BUDGET_STOP';write(ROOT/'BUDGET_STOP.json',dict(reason=str(exc),outcomes=outcomes))
    finally:
        b.close(status)
        if args.phase=='run':write(ROOT/'RUN_TERMINAL.json',dict(status=status,outcomes=outcomes,new_physical_work_stopped=True))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
