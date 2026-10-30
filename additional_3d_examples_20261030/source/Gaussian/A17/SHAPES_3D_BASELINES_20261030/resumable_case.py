"""Resume an interrupted original trajectory from its committed chi prefix.

The original run_case iteration body is retained. Only trajectory input/output,
absolute iteration start, accumulated case wall and counter scope differ.
Existing prefix snapshots and a partial exchange directory are archived by the
parent before entry; this function does not move or recompute committed updates.
resume_meta requires next_iteration and prior_wall_s, including the parent's
abandoned-tail upper bound. Global ledger costs are retained by the caller.
"""
import gc
import json
import time
import traceback
import numpy as np
from scipy import linalg as la
import run_comparison as r


def continue_case(name,policy,b,resume_meta):
    out=r.ROOT/'results'/name/policy
    start=time.perf_counter();state=None;model=None;status='RUNNING';U=None
    next_iteration=int(resume_meta['next_iteration'])
    prior_wall_s=float(resume_meta['prior_wall_s'])
    records=json.loads((out/'trajectory.json').read_text())[:next_iteration]
    inp=dict(np.load(r.ROOT/'inputs'/f'{name}.npz'));init=inp['init'].copy();truth=inp['truth'];scale=float(inp['scale'])
    chi=np.load(out/f'chi_{next_iteration:02d}.npz')['chi'].copy()
    try:
        t=time.perf_counter();model,tan=r.new_model(b);setup=time.perf_counter()-t
        for it in range(next_iteration,r.CFG['outer_cap']):
            b.check()
            if prior_wall_s+time.perf_counter()-start>=r.CFG['trajectory_cap_s']:raise r.BudgetStop('TRAJECTORY_WALL_CAP')
            b.stage=f'{name}/{policy}/update_{it}';begin=time.perf_counter();ph={};t=time.perf_counter()
            if state is None:state=model.state(chi)
            reg=tan.project(chi-init);lam=r.CFG['prior']+r.CFG['lm0']*r.CFG['lm_decay']**(it//3);ell=r.CFG['prior']*reg
            ctx,ev=r.op.make_problem(state,tan,inp['data'],scale,lam,ell)
            objective=float(.5*(ctx.r@ctx.r)+.5*r.CFG['prior']*(reg@reg))
            before=r.old.metrics(chi,truth,state,inp['data'],scale,inp['held_truth'])
            ph['linearization_s']=time.perf_counter()-t
            if before['measured_relative_residual']<=r.CFG['residual_stop']:status='DISCREPANCY_REACHED';break
            t=time.perf_counter();info={};J=None
            if policy=='full_gn':
                J,jmeta=r.old.assemble_full_material_jacobian(state,tan,scale)
                proposal,h,rr=r.full_normal(J,ctx);info=dict(normal_residual=rr,actual_rank=model.n,jacobian=jmeta)
            else:
                h=ev.jt(ctx.r)+ell
                if policy=='exchange':
                    w,L,S,PB,ameta=r.old.get_workspace(ctx)
                    engine=r.RefinedCoreEndpointBatch(L,S,PB,ctx.r,ell,lam,device='cpu',batch_size=16)
                    direction=r.old.original.DirectionAction(ev.j,ctx.P,counter_supplier=model.counters.as_dict)
                    result,ee=r.old.original.run_original_exchange(w.ctx,w,engine,w.anchor,direction,r.CFG['k'],out/f'exchange_{it:02d}',identity={'case':name,'iteration':it,'numerical_version':'refinement_v1'})
                    proposal=ee['step'].copy();U=w.Q@r.op.orth(w.pool.vectors[:,result['selected_ids']])
                    info=dict(actual_rank=U.shape[1],accepted_exchanges=len(result['accepted']),direction_cost=result['direction_cost'],endpoint_statistics=result['endpoint_statistics'],stop_reason=result['stop_reason'])
                    r.write(out/f'policy_{it:02d}.json',info)
                    del w,L,S,PB,engine,direction,ee,result
                else:
                    U,meta=r.baseline_basis(ctx,policy);proposal,e=r.endpoint(ctx,U)
                    info=dict(actual_rank=U.shape[1],normal_residual=float(e['normal_relative_residual'][0]),endpoint_statistics=e['statistics'],basis_metadata=meta)
            ph['policy_s']=time.perf_counter()-t
            t=time.perf_counter();step,projection=r.M['feasible_step'].project_step(tan,chi,proposal)
            fallback=False
            if h@step>=0:
                step,projection=r.M['feasible_step'].project_step(tan,chi,-h/max(lam,la.norm(h),1e-8));fallback=True
            slope=float(h@step);dc=tan.expand(step);ph['projection_s']=time.perf_counter()-t
            alpha=1.;trials=[];accepted=False;t=time.perf_counter()
            for trialno in range(r.CFG['max_trials']):
                if slope>=0 or la.norm(step)<=1e-15:break
                candidate=chi+alpha*dc
                if candidate.imag.min()<-1e-9 or candidate.real.min()<-.500000001:raise RuntimeError('FIXED_RAY_FEASIBILITY')
                trial=model.state(candidate);rg=tan.project(candidate-init);rrr=(trial.field-inp['data'])/scale
                value=float(.5*la.norm(rrr)**2+.5*r.CFG['prior']*(rg@rg));ok=value<=objective+r.CFG['armijo']*alpha*slope
                trials.append(dict(alpha=alpha,objective=value,accepted=bool(ok),source_residual=trial.source_residual()))
                if ok:chi,state=candidate,trial;accepted=True;break
                del trial;alpha*=.5
            ph['line_search_s']=time.perf_counter()-t
            after=r.old.metrics(chi,truth,state,inp['data'],scale,inp['held_truth'])
            row=dict(iteration=it,accepted=accepted,accepted_update_count=sum(r['accepted'] for r in records)+int(accepted),alpha=alpha,gradient_fallback=fallback,slope=slope,objective_before=objective,objective_after=trials[-1]['objective'] if accepted else objective,trials=trials,metrics=after,before_metrics=before,step_norm=float(la.norm(step)),executed_step_norm=float(la.norm(alpha*step)) if accepted else 0.,phases=ph,wall_s=time.perf_counter()-begin,**info)
            records.append(row);r.write(out/'trajectory.json',records)
            np.savez_compressed(out/f'chi_{it+1:02d}.npz',chi=chi)
            r.write(out/'progress.json',dict(case=name,policy=policy,attempts=len(records),updates=sum(r['accepted'] for r in records),metrics=after,elapsed_s=prior_wall_s+time.perf_counter()-start,status='RUNNING'))
            print(json.dumps(dict(event='UPDATE',case=name,policy=policy,iteration=it+1,nrmse=after['complex_nrmse'],residual=after['measured_relative_residual'],iou=after['support_iou'],wall_s=row['wall_s'],accepted=accepted,exchanges=info.get('accepted_exchanges',0))),flush=True)
            del ctx,ev,J,proposal;gc.collect();b.flush()
            if not accepted:status='NO_ARMIJO_ACCEPT';break
            if row['executed_step_norm']<1e-6 and it>=9:status='SMALL_STEP';break
        else:status='MAX_UPDATES'
    except r.BudgetStop as exc:status='BUDGET_STOP_'+str(exc)
    except Exception as exc:
        status='ERROR';r.write(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
    finally:
        fm=None if state is None else r.old.metrics(chi,truth,state,inp['data'],scale,inp['held_truth'],held=True)
        np.savez_compressed(out/'final.npz',chi=chi,truth=truth,points=inp['points'])
        if U is not None:np.savez_compressed(out/'final_current_basis.npz',U=U)
        final=dict(case=name,policy=policy,status=status,attempted_updates=len(records),updates=sum(r['accepted'] for r in records),metrics=fm,wall_s=prior_wall_s+time.perf_counter()-start,model_setup_s=locals().get('setup'),counters=None if model is None else model.counters.as_dict(),same_grid=True,native_SOM_or_TSOM=False,counters_scope='continuation_only',model_setup_scope='continuation_only',continuation=dict(next_iteration=next_iteration,prior_wall_s=prior_wall_s,resume_meta=resume_meta,prior_center='original inputs.init',prior_global_ledger_preserved=True,abandoned_incomplete_update_recomputed_and_charged=True))
        r.write(out/'final.json',final);print(json.dumps(dict(event='FINAL',**{k:final[k] for k in ('case','policy','status','updates','wall_s')},metrics=fm)),flush=True)
        del model,state;gc.collect()
    return status
