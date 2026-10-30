"""New bounded, real 3D vector-Maxwell experiment using original A17 kernels.

All truth/reference access is confined to input synthesis and offline evaluation.
The original selector receives public contexts and charged directional actions.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse, copy, gc, json, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research/delegated/kernel'))
sys.path.insert(0,str(ROOT/'research/delegated/evidence'))
from isolated_imports import load_kernels
from full_material_jacobian import assemble_full_material_jacobian
M=load_kernels()
import importlib
for _module in ('feasible_step','pairtsom_adapter'):
    M[_module]=importlib.import_module(_module)
DenseDDA=M['a9_engine'].DenseDDA
Tangent=M['experiment_support'].Tangent
grid=M['experiment_support'].grid
op=M['operators'];sel=M['selectors_current'];coords=M['selector_coordinates']
CoreEndpointBatch=M['removal_core_batch'].CoreEndpointBatch
original=M['original_exchange_controller']

CFG=dict(n=12,truth_n=14,edge=1.,wavenumber=float(4*np.pi),receiver_count=64,
         heldout_count=80,noise=.01,k=16,prior=1e-5,lm0=.01,lm_decay=.3,
         outer_cap=18,armijo=1e-4,max_trials=24,residual_stop=.011,
         init=[.1,.01],threads=4,precision='complex128/float64',
         wall_cap_s=3600.,state_cap=768,rhs_cap=50000,trajectory_cap_s=1200.,
         cases=['sphere','digit8','depth_bridge'],
         order=['full_gn','receiver','exchange'],
         seeds=[2026103001,2026103002,2026103003],
         source='original finite 3D vector Maxwell DDA; no reduced forward',
         bootstrap_or_generalization_claim=False)

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x

def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False));tmp.replace(path)

class BudgetStop(RuntimeError):pass

class Budget:
    def __init__(self,phase):
        self.path=ROOT/'COST_STATE.json';self.log=ROOT/'COST_LEDGER.jsonl'
        if self.path.exists():self.data=json.loads(self.path.read_text())
        else:
            pre=json.loads((ROOT/'preflight/receipt.json').read_text())
            self.data=dict(states=2,rhs=36,wall_s=pre['wall_s'],sessions=[],preflight_included=True)
        self.start=time.perf_counter();self.phase=phase;self.stage=phase
        self.data['sessions'].append(dict(phase=phase,status='RUNNING',pid=os.getpid(),started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        self.flush()
    def check(self):
        if self.data['wall_s']+time.perf_counter()-self.start>=CFG['wall_cap_s']:
            raise BudgetStop('WALL_CAP')
    def charge(self,kind,count=1):
        self.check();key='states' if kind=='state' else 'rhs';cap=CFG['state_cap' if kind=='state' else 'rhs_cap']
        if self.data[key]+count>cap:raise BudgetStop(kind.upper()+'_CAP')
        self.data[key]+=count
        with self.log.open('a') as f:f.write(json.dumps(dict(kind=kind,count=int(count),phase=self.phase,stage=self.stage,elapsed_s=time.perf_counter()-self.start,totals={k:self.data[k] for k in ('states','rhs')}))+'\n')
        self.flush()
    def flush(self):
        d=copy.deepcopy(self.data);d['active_wall_s']=time.perf_counter()-self.start;write(self.path,d)
    def close(self,status):
        elapsed=time.perf_counter()-self.start;self.data['wall_s']+=elapsed
        self.data['sessions'][-1].update(status=status,wall_s=elapsed,ended_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        self.start=time.perf_counter();write(self.path,self.data)

class CountedSolver:
    """Admission/accounting proxy; calls unaltered original kernels."""
    def __init__(self,base,budget):self.base=base;self.budget=budget
    def __getattr__(self,key):return getattr(self.base,key)
    def factorize(self,A,label='full'):
        if label=='full':self.budget.charge('state')
        return self.base.factorize(A,label=label)
    def solve(self,fac,B,trans=0,matrix=None,label='full'):
        if str(label).startswith('full'):
            self.budget.charge('rhs',1 if np.ndim(B)==1 else np.shape(B)[1])
        return self.base.solve(fac,B,trans=trans,matrix=matrix,label=label)

def acquisition(held=False):
    # Twelve transverse vector plane waves: +/- axes, two polarizations each.
    dirs=[];pols=[]
    for axis in range(3):
        for sign in (-1,1):
            for pol in range(3):
                if pol!=axis:
                    dirs.append(sign*np.eye(3)[axis]);pols.append(np.eye(3)[pol])
    _,_,rx,obs=M['experiment_support'].acquisition(6,CFG['heldout_count'] if held else CFG['receiver_count'],.37 if held else .13)
    return np.array(dirs),np.array(pols),rx,obs

def new_model(n,budget):
    budget.check();p,v=grid(n,CFG['edge']);model=DenseDDA(p,v,CFG['wavenumber'],*acquisition(),device='cpu')
    model.solver=CountedSolver(model.solver,budget)
    return model,Tangent(p,v,'voxel')

def truth_at(name,p):
    x,y,z=p.T
    if name=='sphere':mask=x*x+y*y+z*z<=.27**2
    elif name=='digit8':
        outer=((x/.255)**2+((y-.175)/.235)**2<=1)|((x/.255)**2+((y+.175)/.235)**2<=1)
        holes=((x/.115)**2+((y-.175)/.115)**2<1)|((x/.115)**2+((y+.175)/.115)**2<1)
        mask=outer&(~holes)&(np.abs(z)<=.17)
    elif name=='depth_bridge':
        # Front arch plus a rear pillar and cross-depth connecting beam.
        left=(np.abs(x+.25)<=.085)&(np.abs(y)<=.32)&(np.abs(z+.20)<=.085)
        right=(np.abs(x-.25)<=.085)&(np.abs(y)<=.32)&(np.abs(z+.20)<=.085)
        lintel=(np.abs(x)<=.335)&(np.abs(y-.25)<=.085)&(np.abs(z+.20)<=.085)
        rear=(np.abs(x)<=.09)&(np.abs(y)<=.32)&(np.abs(z-.23)<=.09)
        cross=(np.abs(x)<=.09)&(np.abs(y-.25)<=.085)&(z>=-.28)&(z<=.32)
        mask=left|right|lintel|rear|cross
    else:raise ValueError(name)
    truth=np.where(mask,1.+.08j,0j)
    if name=='depth_bridge':truth[(np.abs(x)<=.09)&(np.abs(y)<=.32)&(np.abs(z-.23)<=.09)]=1.4+.12j
    return truth

def held_observations(state):
    _,_,rx,obs=acquisition(True)
    S=np.sqrt(state.model.volume)*M['a9_engine'].receiver_operator(state.model.points,CFG['wavenumber'],rx,obs)
    return state.current@S.T

def metrics(chi,truth,state,data,scale,held_truth,held=False):
    a=np.abs(chi-truth)
    out=dict(complex_nrmse=float(la.norm(chi-truth)/la.norm(truth)),
             real_nrmse=float(la.norm(chi.real-truth.real)/la.norm(truth.real)),
             imag_nrmse=float(la.norm(chi.imag-truth.imag)/la.norm(truth.imag)),
             complex_mae=float(a.mean()),real_mae=float(np.abs(chi.real-truth.real).mean()),
             imag_mae=float(np.abs(chi.imag-truth.imag).mean()),
             measured_relative_residual=float(la.norm(state.field-data)/scale),
             min_real=float(chi.real.min()),min_imag=float(chi.imag.min()),max_real=float(chi.real.max()))
    tr=truth.real>=.5;est=chi.real>=.5
    out['support_iou']=float(np.sum(tr&est)/max(1,np.sum(tr|est)))
    out['support_voxels']=int(est.sum());out['truth_voxels']=int(tr.sum())
    if held:out['heldout_relative']=float(la.norm(held_observations(state)-held_truth)/la.norm(held_truth))
    return out

def get_workspace(ctx):
    cfg=sel.SelectorConfig(seed=20260930,pool_count=32,residual_up_count=16,random_pool_count=10)
    anchor,meta=sel.build_independent_anchor(ctx,64,seed=cfg.seed)
    pool=sel.build_candidates(ctx,anchor,cfg)
    w=coords.prepare_selector_workspace(ctx,pool,anchor,chunk_size=16,embedding_key='new_run_explicit_coordinate_embedding_no_hash')
    eye=np.eye(w.ctx.n_current,dtype=complex)
    return w,w.ctx.apply_L(eye),w.ctx.apply_S(eye),w.ctx.project_B(eye),meta

def receiver_endpoint(ctx):
    k=CFG['k'];U=op.orth(ctx.receiver_modes(k),rank=k)
    if U.shape[1]!=k:raise RuntimeError('ACTUAL_RANK')
    L=U.conj().T@ctx.apply_L(U);S=ctx.apply_S(U);PB=ctx.project_B(U)
    engine=CoreEndpointBatch(L,S,PB,ctx.r,ctx.ell,ctx.lam,device='cpu',batch_size=16)
    endpoint=engine.evaluate(np.eye(k,dtype=complex)[None])
    if endpoint['status']!=['OK']:raise RuntimeError('RECEIVER_ENDPOINT_'+str(endpoint['status']))
    return endpoint['steps'][0],U

def normal_reference(J,ctx):
    H=J.T@J;H.flat[::len(H)+1]+=ctx.lam
    h=J.T@ctx.r+ctx.ell;s=la.solve(H,-h,assume_a='pos',check_finite=False)
    rr=float(la.norm(H@s+h)/max(la.norm(h),1e-300))
    if rr>1e-9:raise RuntimeError('FULL_NORMAL_RESIDUAL_'+str(rr))
    return s,h,rr

def validate(budget):
    budget.stage='required_physical_adapter_verification'
    n=5;model,tan=new_model(n,budget);p=model.points
    chi=.12+.01j+.3*np.exp(-np.sum(p*p,axis=1)/.12)
    state=model.state(chi);scale=float(la.norm(state.field));rng=np.random.default_rng(20261030)
    ctx,ev=op.make_problem(state,tan,state.field*.9,scale,.01,np.full(tan.d,1e-5))
    J,meta=assemble_full_material_jacobian(state,tan,scale)
    Z=rng.normal(size=(tan.d,3));Z/=la.norm(Z,axis=0)
    W=rng.normal(size=(J.shape[0],2))
    jp=ev.j(Z);jt=ev.jt(W)
    errors=dict(jvp=float(la.norm(J@Z-jp)/la.norm(jp)),vjp=float(la.norm(J.T@W-jt)/la.norm(jt)))
    dc=tan.expand(Z[:,0]);step=1e-6/max(1.,float(np.max(np.abs(dc))))
    plus=model.state(chi+step*dc);minus=model.state(chi-step*dc)
    fd=M['pairtsom_adapter'].stack_real_illuminations((plus.field-minus.field)/(2*step*scale))
    errors['centered_derivative']=float(la.norm(fd-J@Z[:,0])/la.norm(J@Z[:,0]))
    del plus,minus
    w,L,S,PB,ameta=get_workspace(ctx)
    engine=CoreEndpointBatch(L,S,PB,w.ctx.r,ctx.ell,ctx.lam,device='cpu',batch_size=16)
    action=original.DirectionAction(ev.j,ctx.P,counter_supplier=model.counters.as_dict)
    result,endpoint=original.run_original_exchange(w.ctx,w,engine,w.anchor,action,16,ROOT/'verification/exchange',identity={'case':'adapter_verification'},candidate_batch_size=16)
    U=w.Q@op.orth(w.pool.vectors[:,result['selected_ids']])
    direct=ctx.make_model(U).step()
    errors['endpoint_step']=float(la.norm(direct-endpoint['step'])/max(la.norm(direct),1e-300))
    actual=J@endpoint['step'];errors['accumulated_Jx']=float(la.norm(actual-endpoint['Js'])/max(la.norm(actual),1e-300))
    x=endpoint['initial_step'];d=endpoint['step']-x;g=J.T@(ctx.r+J@x)+ctx.ell+ctx.lam*x
    Q=float(-g@d-.5*(la.norm(J@d)**2+ctx.lam*(d@d)))
    recorded=sum(m['Q_true'] for m in result['accepted'])
    errors['directional_gain_abs']=float(abs(Q-recorded))
    status='PASS' if max(errors[k] for k in ('jvp','vjp','endpoint_step','accumulated_Jx'))<1e-9 and errors['centered_derivative']<1e-5 and errors['directional_gain_abs']<1e-10 else 'FAIL'
    write(ROOT/'verification/IMPLEMENTATION.json',dict(status=status,errors=errors,jacobian_metadata=meta,accepted_moves=len(result['accepted']),original_module_paths={k:str(getattr(v,'__file__','')) for k,v in M.items()},physical_counters=model.counters.as_dict(),selector_has_full_J=False))
    print('VERIFICATION '+json.dumps(dict(status=status,errors=errors)),flush=True)
    if status!='PASS':raise RuntimeError('IMPLEMENTATION_GATE_FAIL')

def prepare_inputs(budget):
    p,v=grid(CFG['n'],CFG['edge']);pf,vf=grid(CFG['truth_n'],CFG['edge'])
    for index,name in enumerate(CFG['cases']):
        path=ROOT/'inputs'/f'{name}.npz'
        if path.exists():continue
        budget.stage='truth_'+name;t=time.perf_counter();truth=truth_at(name,p);fine=truth_at(name,pf)
        model,_=new_model(CFG['truth_n'],budget);state=model.state(fine)
        exact=state.field.copy();held=held_observations(state)
        rng=np.random.default_rng(CFG['seeds'][index]);noise=rng.normal(size=exact.shape)+1j*rng.normal(size=exact.shape)
        noise*=CFG['noise']*la.norm(exact)/la.norm(noise);data=exact+noise;scale=float(la.norm(data))
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,points=p,truth=truth,n=CFG['n'],truth_points=pf,truth_fine=fine,
            init=np.full(len(p),complex(*CFG['init'])),data=data,data_clean=exact,noise=noise,scale=scale,held_truth=held)
        write(ROOT/'inputs'/f'{name}.json',dict(case=name,seed=CFG['seeds'][index],truth_grid=CFG['truth_n'],inverse_grid=CFG['n'],contrast_source='continuous geometry sampled independently',same_grid=False,truth_voxels=int(np.sum(truth.real>.5)),fine_voxels=int(np.sum(fine.real>.5)),input_wall_s=time.perf_counter()-t,counters=model.counters.as_dict()))
        print(json.dumps(dict(event='INPUT',case=name,wall_s=time.perf_counter()-t)),flush=True)
        del state,model;gc.collect()

def run_case(name,policy,budget):
    out=ROOT/'results'/name/policy
    if out.exists():raise RuntimeError('REFUSE_EXISTING_OUTPUT_'+str(out))
    out.mkdir(parents=True);started=time.perf_counter();records=[];status='RUNNING'
    budget.stage=name+'/'+policy
    inp=dict(np.load(ROOT/'inputs'/f'{name}.npz'));chi=inp['init'].copy();init=chi.copy();truth=inp['truth'];data=inp['data'];scale=float(inp['scale'])
    model=None;state=None;last_local=None
    try:
        t=time.perf_counter();model,tan=new_model(CFG['n'],budget);model_setup=time.perf_counter()-t
        for it in range(CFG['outer_cap']):
            budget.check()
            if time.perf_counter()-started>CFG['trajectory_cap_s']:raise BudgetStop('TRAJECTORY_WALL_CAP')
            budget.stage=f'{name}/{policy}/iteration_{it:02d}'
            begin=time.perf_counter();phase={};t=time.perf_counter()
            state=model.state(chi) if state is None else state
            reg=tan.project(chi-init);lam=CFG['prior']+CFG['lm0']*CFG['lm_decay']**(it//3);ell=CFG['prior']*reg
            ctx,ev=op.make_problem(state,tan,data,scale,lam,ell)
            obj=float(.5*(ctx.r@ctx.r)+.5*CFG['prior']*(reg@reg));phase['linearization_s']=time.perf_counter()-t
            before=metrics(chi,truth,state,data,scale,inp['held_truth'])
            if before['measured_relative_residual']<=CFG['residual_stop']:
                status='DISCREPANCY_REACHED';break
            J=None;proposal=None;policy_result={};t=time.perf_counter()
            if policy=='full_gn':
                J,jmeta=assemble_full_material_jacobian(state,tan,scale)
                proposal,h,normal_res=normal_reference(J,ctx)
                policy_result=dict(normal_residual=normal_res,jacobian_metadata=jmeta,actual_rank=model.n)
            else:
                # This gradient is shared full-physics outer optimization cost.
                h=ev.jt(ctx.r)+ell
                if policy=='receiver':
                    proposal,U=receiver_endpoint(ctx);policy_result=dict(actual_rank=CFG['k'],anchor_pool=False)
                elif policy=='exchange':
                    acquire=time.perf_counter();w,L,S,PB,ameta=get_workspace(ctx)
                    phase['anchor_pool_workspace_s']=time.perf_counter()-acquire
                    engine=CoreEndpointBatch(L,S,PB,ctx.r,ell,lam,device='cpu',batch_size=16)
                    direction=original.DirectionAction(ev.j,ctx.P,counter_supplier=model.counters.as_dict)
                    policy_result,endpoint=original.run_original_exchange(w.ctx,w,engine,w.anchor,direction,CFG['k'],out/f'exchange_{it:02d}',identity={'case':name,'iteration':it},candidate_batch_size=16)
                    proposal=endpoint['step'].copy();U=w.Q@op.orth(w.pool.vectors[:,policy_result['selected_ids']])
                    # Offline-only same-state comparator; never passed into controller.
                    if it in (0,3,17):
                        diagstart=time.perf_counter();J,jmeta=assemble_full_material_jacobian(state,tan,scale)
                        star,_,res=normal_reference(J,ctx)
                        def gap(s):
                            d=s-star;return float(.5*(la.norm(J@d)**2+lam*(d@d)))
                        local=dict(iteration=it,receiver_gap=gap(endpoint['initial_step']),exchange_gap=gap(proposal),full_normal_residual=res,receiver_step_truth_error=float(la.norm(chi+tan.expand(endpoint['initial_step'])-truth)/la.norm(truth)),exchange_step_truth_error=float(la.norm(chi+tan.expand(proposal)-truth)/la.norm(truth)),truth_diagnostic_scope='unconstrained alpha=1; NOT nonlinear reconstruction outcome',offline_wall_s=time.perf_counter()-diagstart,jacobian_metadata=jmeta)
                        last_local=local;write(out/f'local_fidelity_{it:02d}.json',local)
                        phase['offline_diagnostic_s']=local['offline_wall_s']
                        del star
                    np.savez_compressed(out/f'current_basis_{it:02d}.npz',U=U,chi=chi,proposal=proposal)
                else:raise ValueError(policy)
            phase['policy_with_diagnostics_s']=time.perf_counter()-t
            t=time.perf_counter();step,proj=M['feasible_step'].project_step(tan,chi,proposal)
            fallback=False
            if h@step>=0:
                step,proj=M['feasible_step'].project_step(tan,chi,-h/max(lam,la.norm(h),1e-8));fallback=True
            slope=float(h@step);dc=tan.expand(step);phase['projection_s']=time.perf_counter()-t
            alpha=1.;trials=[];accepted=False;t=time.perf_counter()
            for trialno in range(CFG['max_trials']):
                if slope>=0 or la.norm(step)<=1e-15:break
                candidate=chi+alpha*dc
                if candidate.imag.min() < -1e-9 or candidate.real.min()<-.500000001:raise RuntimeError('FIXED_RAY_FEASIBILITY')
                trial=model.state(candidate);rg=tan.project(candidate-init);rr=(trial.field-data)/scale
                val=float(.5*la.norm(rr)**2+.5*CFG['prior']*(rg@rg))
                ok=val<=obj+CFG['armijo']*alpha*slope
                trials.append(dict(alpha=alpha,objective=val,accepted=bool(ok),source_residual=trial.source_residual()))
                if ok:
                    chi,state=candidate,trial;accepted=True;break
                del trial;alpha*=.5
            phase['line_search_s']=time.perf_counter()-t
            after=metrics(chi,truth,state,data,scale,inp['held_truth'])
            diagwall=phase.get('offline_diagnostic_s',0.)
            rec=dict(iteration=it,accepted=accepted,alpha=alpha,gradient_fallback=fallback,slope=slope,
                     objective_before=obj,objective_after=trials[-1]['objective'] if trials else obj,
                     projected_step_norm=float(la.norm(step)),executed_step_norm=float(la.norm(alpha*step)) if accepted else 0.,
                     metrics=after,before_metrics=before,trials=trials,phases=phase,
                     wall_s=time.perf_counter()-begin,offline_diagnostic_s=diagwall,
                     accepted_exchanges=len(policy_result.get('accepted',[])) if policy=='exchange' else 0,
                     direction_cost=policy_result.get('direction_cost'),actual_rank=policy_result.get('actual_rank'),
                     normal_residual=policy_result.get('normal_residual'))
            records.append(rec);write(out/'trajectory.json',records)
            np.savez_compressed(out/'checkpoint.npz',chi=chi,iteration=it,proposal=proposal,projected_step=step,alpha=alpha)
            write(out/'progress.json',dict(case=name,policy=policy,iteration=it+1,metrics=after,elapsed_s=time.perf_counter()-started,status='RUNNING'))
            print(json.dumps(dict(event='UPDATE',case=name,policy=policy,iteration=it+1,nrmse=after['complex_nrmse'],residual=after['measured_relative_residual'],iou=after['support_iou'],wall_s=rec['wall_s'],exchanges=rec['accepted_exchanges'])),flush=True)
            del ctx,ev,J,proposal
            if policy=='exchange':del w,L,S,PB,engine,direction,endpoint,U
            gc.collect();budget.flush()
            if not accepted:status='NO_ARMIJO_ACCEPT';break
            if la.norm(alpha*step)<1e-6 and it>=9:status='SMALL_STEP';break
        else:status='MAX_UPDATES'
    except BudgetStop as exc:status='BUDGET_STOP_'+str(exc)
    except Exception as exc:
        status='ERROR';write(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
    finally:
        finalmetrics=None
        if state is not None:
            finalmetrics=metrics(chi,truth,state,data,scale,inp['held_truth'],held=True)
        wall=time.perf_counter()-started;offline=sum(r.get('offline_diagnostic_s',0.) for r in records)
        np.savez_compressed(out/'final.npz',chi=chi,truth=truth,points=inp['points'])
        write(out/'final.json',dict(case=name,policy=policy,status=status,updates=len(records),metrics=finalmetrics,wall_s=wall,offline_diagnostic_wall_s=offline,policy_wall_excluding_diagnostics_s=wall-offline,model_setup_s=locals().get('model_setup'),counters=None if model is None else model.counters.as_dict(),last_local_fidelity=last_local))
        print(json.dumps(dict(event='FINAL',case=name,policy=policy,status=status,metrics=finalmetrics,wall_s=wall)),flush=True)
        state=None;model=None;gc.collect()
    return status

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['validate','run'],required=True);args=parser.parse_args()
    configpath=ROOT/'CONFIG.json'
    if configpath.exists() and json.loads(configpath.read_text())!=CFG:raise RuntimeError('CONFIG_CHANGED')
    write(configpath,CFG);budget=Budget(args.phase);status='ERROR'
    try:
        if args.phase=='validate':validate(budget)
        else:
            if json.loads((ROOT/'verification/IMPLEMENTATION.json').read_text())['status']!='PASS':raise RuntimeError('IMPLEMENTATION_NOT_PASSED')
            prepare_inputs(budget)
            for policy in CFG['order']:
                for name in CFG['cases']:
                    budget.check();s=run_case(name,policy,budget)
                    if s.startswith('BUDGET_STOP'):raise BudgetStop(s)
        status='COMPLETE'
    except BudgetStop as exc:
        status='BUDGET_STOP';write(ROOT/'BUDGET_STOP.json',dict(reason=str(exc),phase=args.phase));print('BUDGET_STOP '+str(exc),flush=True)
    finally:budget.close(status)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
