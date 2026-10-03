"""V2 explicit OFFLINE coefficient-bound full-dictionary1swap coverage; no oracle feedback to online.

Local exact1swap neighborhood teacher, never a global oracle. Teacher trajectory
comparisons combine candidate coverage, ranking and search; not pure score loss.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,time,traceback
import numpy as np
import closure_runtime as runtime
from online_tolerance import directional_check


class OfflineBudgetExpired(RuntimeError):pass


def assess_checkpoint(workspace,engine,teacher,ids,k,x,Jx,out,deadline):
    from run_exchange import Neighborhood,balanced_ids,moves,gains_on_anchor,clean,jswrite,append_rows
    from maxwell_state import array_hash
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ctx=workspace.ctx
    incoming=[i for i in range(workspace.pool.vectors.shape[1]) if i not in ids]
    proposals=moves(ids,incoming,'swap',k);neigh=Neighborhood(workspace,engine,out,{})
    t=time.perf_counter();valid,steps,info,build=neigh.endpoints(proposals);buildwall=time.perf_counter()-t
    D=steps-x[:,None];t=time.perf_counter()
    qhat=gains_on_anchor(workspace.anchor,x,D,ctx) if len(valid) else np.empty(0)
    anchorwall=time.perf_counter()-t;position={int(i):j for j,i in enumerate(valid)}
    rows=[]
    for i,move in enumerate(proposals):
        row=dict(move_index=i,**move,**info[i],Qhat=float(qhat[position[i]]) if i in position else None,
                 Q_true=None,tau_online=None,label_status='PENDING' if i in position else 'INVALID_ENDPOINT',
                 add_labels=[str(workspace.pool.labels[a]) for a in move['add']],
                 drop_labels=[str(workspace.pool.labels[a]) for a in move['drop']])
        rows.append(row)
    q=np.full(len(valid),np.nan);taus=np.full(len(valid),np.nan);complete=True;best_z=None;best_j=None
    before_count=teacher.counts;before_wall=teacher.wall_s
    for first in range(0,len(valid),16):
        if time.perf_counter()>=deadline:complete=False;break
        last=min(first+16,len(valid));z=teacher.j(D[:,first:last])
        check=directional_check(ctx.r,ctx.r+Jx,z,x,D[:,first:last],ctx.ell,ctx.lam)
        q[first:last]=check['gain'];taus[first:last]=check['tau']
        for j in range(first,last):
            row=rows[int(valid[j])];row.update(Q_true=float(q[j]),tau_online=float(taus[j]),label_status='OFFLINE_FULL_DIRECTION_LABEL',
                      true_positive=bool(q[j]>0),significant_positive=bool(q[j]>taus[j]),
                      direction_hash=array_hash(D[:,j]),output_hash=array_hash(z[:,j-first]))
            if best_j is None or q[j]>q[best_j]:best_j=j;best_z=z[:,j-first].copy()
        append_rows(out/'label_block_receipts.jsonl',[dict(first=first,last=last,tangent_rhs=teacher.counts-before_count,
                            cumulative_teacher_wall_s=teacher.wall_s-before_wall)])
    append_rows(out/'moves.jsonl',rows)
    short=balanced_ids(workspace.pool,ids,12)
    subset=[j for j,i in enumerate(valid) if proposals[int(i)]['add'][0] in short]
    if complete:
        maximum=max(0.,float(np.max(q))) if len(q) else 0.
        eligible=np.where(q>taus)[0];significant=max(0.,float(np.max(q[eligible]))) if len(eligible) else 0.
        shortmax=max(0.,float(np.max(q[subset]))) if subset else 0.
        short_eligible=[j for j in subset if q[j]>taus[j]]
        short_significant=max(0.,float(np.max(q[short_eligible]))) if short_eligible else 0.
        top=int(np.argsort(-qhat,kind='stable')[0]) if len(qhat) else None
        signed=float(q[top]) if top is not None else 0.
        top_verified=signed if top is not None and signed>taus[top] else 0.
        eligible_best=int(eligible[np.argmax(q[eligible])]) if len(eligible) else None
        chosen=eligible_best
        # Highest Q significance eligibility; positive no-op is part of domain.
        if chosen is not None and chosen!=best_j:best_z=teacher.j(D[:,chosen])
        summary=dict(status='COMPLETE',declared_move_count=len(proposals),feasible_count=len(valid),invalid_count=len(proposals)-len(valid),
              labeled_count=len(valid),positive_count=int(np.sum(q>0)),significant_positive_count=len(eligible),
              teacher_best_raw_positive_gain=maximum,teacher_best_significant_gain=significant,
              balanced12_ids=short,balanced12_feasible_count=len(subset),balanced12_best_raw_positive_gain=shortmax,
              balanced12_opportunity_capture=shortmax/maximum if maximum>0 else None,omitted_opportunity=maximum-shortmax,
              balanced12_best_significant_gain=short_significant,
              balanced12_significant_opportunity_capture=short_significant/significant if significant>0 else None,
              full_dictionary_anchor_top1_signed_true_gain=signed,full_dictionary_anchor_top1_verified_noop_gain=top_verified,
              full_dictionary_anchor_top1_signed_regret=maximum-signed,full_dictionary_anchor_top1_noop_regret=maximum-max(0.,signed),
              full_dictionary_anchor_top1_signed_capture=signed/maximum if maximum>0 else None,
              full_dictionary_anchor_top1_noop_capture=max(0.,signed)/maximum if maximum>0 else None,
              full_dictionary_anchor_top1_significance_regret=significant-top_verified,
              best_raw_move=rows[int(valid[best_j])] if best_j is not None else None,
              best_significant_move=rows[int(valid[chosen])] if chosen is not None else None,
              anchor_top1_move=rows[int(valid[top])] if top is not None else None)
    else:
        chosen=None;summary=dict(status='PARTIAL_OFFLINE_BUDGET',declared_move_count=len(proposals),feasible_count=len(valid),
              invalid_count=len(proposals)-len(valid),labeled_count=int(np.isfinite(q).sum()),complete_domain=False,
              observed_best_positive_gain=max(0.,float(np.nanmax(q))) if np.isfinite(q).any() else None,
              teacher_best_raw_positive_gain=None,balanced12_opportunity_capture=None,
              reason='Unlabeled candidates cannot support whole-neighborhood maximum or stopping')
    summary.update(scope='Local exact1swap neighborhood teacher, not global oracle',deterministic_certificate=False,
          teacher_tangent_rhs=teacher.counts-before_count,teacher_wall_s=teacher.wall_s-before_wall,
          endpoint_wall_s=buildwall,anchor_wall_s=anchorwall,endpoint_statistics=neigh.timings,
          base_step_hash=array_hash(x),selected_ids=list(ids),no_op_included=True)
    jswrite(out/'coverage.json',clean(summary))
    np.savez_compressed(out/'scores.npz',valid_move_indices=valid,Qhat=qhat,Q_true=q,tau_online=taus)
    child=None if chosen is None else dict(ids=proposals[int(valid[chosen])]['ids'],step=steps[:,chosen],
                            Js=Jx+best_z,gain=float(q[chosen]),tau=float(taus[chosen]))
    return summary,child


def verify_coefficient_receipt(persisted,rec):
    from maxwell_state import array_hash
    with np.load(Path(persisted)/'public_workspace.npz',allow_pickle=False) as bank:
        coefficient_hash=array_hash(bank['pool'])
    if rec.get('coefficient_pool_hash')!=coefficient_hash:
        raise ValueError('completed v2 coefficient pool receipt missing/drifted')
    # The historical physical pool_hash is a separate representation and is NOT
    # compared to coefficient arrays; its value remains preserved in receipt.
    return coefficient_hash


def verify_completed_source_receipt(rec):
    actual=rec.get('runtime',{}).get('actual_imports',{})
    for name in ['closure_state_v2.py','run_closure_v2.py']:
        expected=runtime.sha(runtime.ROOT/'code'/name)
        matches=[row for row in actual.values() if row.get('actual_path','').replace('\\','/').rsplit('/',1)[-1]==name]
        if not matches or any(row.get('sha256')!=expected for row in matches):
            raise ValueError('completed state lacks matching actual v2 import receipt: '+name)


def check_coverage_lock(path,config_path):
    lock=json.loads(Path(path).read_text())
    if lock.get('schema')!='a17.closure.offline_coverage.v2.lock':raise ValueError('explicit owner coverage-v2 addendum required')
    if lock.get('rule_config_sha256')!=runtime.sha(config_path):raise ValueError('coverage addendum does not bind unchanged rule config')
    required={'code/'+name for name in ['run_coverage_v2.py','closure_state_v2.py','run_closure_v2.py','closure_runtime.py','online_tolerance.py']}
    sources=lock.get('deployment_sources',{})
    if not required.issubset(sources):raise ValueError('coverage addendum missing actual v2 source pins')
    for name,value in sources.items():
        source=(runtime.ROOT/name).resolve()
        expected=value if isinstance(value,str) else value['sha256']
        if not source.is_relative_to(runtime.ROOT) or runtime.sha(source)!=expected:raise ValueError('coverage addendum source drift: '+name)
    if lock.get('k')!=[4,8,16] or lock.get('dictionary_atoms')!=154 or lock.get('allowed_path_actions')!=[0,3]:
        raise ValueError('coverage math/domain lock differs')
    return dict(path=str(Path(path).resolve()),sha256=runtime.sha(path),rule_config_sha256=runtime.sha(config_path),deployment_sources=sources)


def verify_actual_imports():
    import closure_state_v2,run_closure_v2
    for module in [closure_state_v2,run_closure_v2]:
        source=Path(module.__file__).resolve();expected=runtime.ROOT/'code'/f'{module.__name__}.py'
        if source!=expected.resolve() or runtime.sha(source)!=runtime.sha(expected):raise ValueError('actual v2 import path/hash differs')


def run(args):
    runtime.initialize(args.baseline_root,args.baseline_lock)
    coverage_lock_receipt=check_coverage_lock(args.coverage_lock,args.config)
    from run_closure_v2 import validate_locked_config,synchronize
    from closure_state_v2 import load_state,load_evaluation_reference,persisted_workspace
    from maxwell_state import frozen,EvaluatorContext,TeacherContext,array_hash
    from removal_core_batch import CoreEndpointBatch
    from run_exchange import clean,jswrite
    persisted=Path(args.completed_state).resolve();status=json.loads((persisted/'status.json').read_text())
    result=json.loads((persisted/'result.json').read_text());rec=json.loads((persisted/'runtime_receipt.json').read_text())
    if status['status']!='COMPLETED' or result['status']!='COMPLETED' or result['completed_k']!=[4,8,16]:
        raise ValueError('coverage requires complete online/evaluation state at locked k4/8/16')
    if any(not (persisted/f'k{k}/online_result.json').exists() for k in [4,8,16]):raise ValueError('missing prior online result')
    verify_coefficient_receipt(persisted,rec)
    verify_completed_source_receipt(rec)
    cli=rec['config']['cli'];bound=SimpleNamespace(**cli)
    bound.baseline_root=args.baseline_root;bound.config=args.config;bound.input_manifest=args.input_manifest or cli.get('input_manifest')
    config,config_receipt=validate_locked_config(bound)
    verify_actual_imports()
    if args.path_actions not in (0,3) or not 0<args.offline_wall_limit_s<=7200:raise ValueError('bounded initial/3action diagnostic only')
    # No teacher object is created until completed online outputs are verified.
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+args.offline_wall_limit_s
    jswrite(out/'status.json',dict(status='RUNNING',scope='OFFLINE_ONLY'))
    if bound.mode=='replay':bundle=frozen(bound.object,bound.phase,args.device)
    else:
        bundle=load_state(args.inputs_root or cli['inputs_root'],bound.object,bound.phase,args.device,bound.input_manifest,
                    phases=config['phases'],noise_basis_points=bound.noise_basis_points,realization_index=bound.realization_index)
        load_evaluation_reference(bundle)
        if bound.mode=='noise':
            noise_ref=persisted/'offline_reference.npz'
            with np.load(noise_ref,allow_pickle=False) as saved:bundle['full_step']=saved['step'].copy()
    model=bundle['state'].model;physical_before=copy.deepcopy(model.counters.as_dict())
    if bundle['state_hash']!=rec['state_hash']:raise ValueError('completed state hash differs')
    work=persisted_workspace(bundle,persisted);ctx=bundle['ctx'];w=work['workspace']
    evaluator=EvaluatorContext(bundle);acquisition_wall=0.;acquisition_rhs=0;J=None
    if ctx.dim_material<=128:
        t=time.perf_counter();J=bundle['ev'].dense_j();synchronize(args.device)
        acquisition_wall=time.perf_counter()-t;acquisition_rhs=ctx.P*ctx.dim_material
        np.savez_compressed(out/'offline_fullJ.npz',J=J,scope=np.asarray('PHYSICAL_DERIVATIVE_ACQUIRED_AFTER_ONLINE'))
    teacher=TeacherContext(bundle['ev'].j,ctx.P,J)
    engine=CoreEndpointBatch(work['L'],work['S'],work['PB'],ctx.r,ctx.ell,ctx.lam,
                    device='cuda' if args.device=='cuda' and ctx.q>=256 else 'cpu',batch_size=16)
    if w.pool.vectors.shape[1]!=154:raise ValueError('locked154atom dictionary required')
    receipt=dict(runtime=runtime.receipt(),driver_path=str(Path(__file__).resolve()),driver_sha256=runtime.sha(__file__),
        completed_inputs={str(p):runtime.sha(p) for p in [persisted/'status.json',persisted/'result.json',persisted/'runtime_receipt.json',persisted/'public_workspace.npz']},
        coefficient_pool_hash=rec['coefficient_pool_hash'],recorded_legacy_pool_hash=rec.get('pool_hash'),
        coverage_lock=coverage_lock_receipt,config=config_receipt,mode='OFFLINE_COVERAGE_V2',source_mode=bound.mode,path_actions=args.path_actions,
        offline_wall_limit_s=args.offline_wall_limit_s,physical_setup_wall_s=bundle['physical_setup_wall_s'],
        physical_setup_counters=physical_before,
        dense_J_acquisition_wall_s=acquisition_wall,dense_J_acquisition_tangent_rhs=acquisition_rhs,
        teacher_backend='already_paid_after_online_physical_denseJ' if J is not None else 'physical_directional_blocks16',
        reference_wall_s=evaluator.wall_s,engine_setup_wall_s=engine.setup_wall_s,no_online_feedback=True,
        scope='Teacher labels are never inputs to previously completed online policy')
    jswrite(out/'runtime_receipt.json',clean(receipt));cells=[]
    for k in [4,8,16]:
        cell=out/f'k{k}';cell.mkdir();ids=[i for i,f in enumerate(w.pool.families) if f=='receiver'][:k]
        with np.load(persisted/f'k{k}/receiver_endpoint.npz',allow_pickle=False) as f:x=f['step'].copy();Jx=f['Js'].copy()
        with np.load(persisted/f'k{k}/endpoint.npz',allow_pickle=False) as f:actual=f['step'].copy();actualJ=f['Js'].copy()
        if len(ids)!=k:raise ValueError('receiver rank infeasible')
        summary,child=assess_checkpoint(w,engine,teacher,ids,k,x,Jx,cell/'round0',deadline)
        teacher_path=[];path_status='NOT_REQUESTED';xs=x.copy();js=Jx.copy()
        if args.path_actions==3:
            path_status='LOCAL_3ACTION_BUDGET_COMPLETE'
            for rnd in range(3):
                if rnd:
                    if time.perf_counter()>=deadline:path_status='PARTIAL_OFFLINE_BUDGET';break
                    summary_next,child=assess_checkpoint(w,engine,teacher,ids,k,xs,js,cell/f'round{rnd}',deadline)
                    if summary_next['status']!='COMPLETE':path_status='PARTIAL_OFFLINE_BUDGET';break
                elif summary['status']!='COMPLETE':path_status='PARTIAL_OFFLINE_BUDGET';break
                if child is None:path_status='NUMERICAL_LOCAL_1SWAP_STOP_NO_DETERMINISTIC_BOUND';break
                ids=child['ids'];xs=child['step'];js=child['Js'];teacher_path.append(dict(round=rnd,selected_ids=ids,gain=child['gain'],tau=child['tau']))
                np.savez_compressed(cell/f'teacher_round{rnd}.npz',step=xs,Js=js,selected_ids=ids)
        teacher_risk=evaluator.risk(xs,js) if args.path_actions==3 else None
        actual_risk=evaluator.risk(actual,actualJ)
        record=dict(k=k,initial_coverage=summary,actual_online_final_risk=actual_risk,
            teacher_path_requested_actions=args.path_actions,teacher_path_status=path_status,teacher_path=teacher_path,
            teacher_final_risk=teacher_risk,actual_minus_local_teacher_risk=(actual_risk['full_gap']-teacher_risk['full_gap']) if teacher_risk else None,
            comparison_scope='Trajectory gap includes candidate coverage/ranking/search; cannot be attributed solely to score',
            local_not_global_oracle=True)
        jswrite(cell/'result.json',clean(record));cells.append(record)
        if time.perf_counter()>=deadline:break
    terminal='COMPLETED' if len(cells)==3 and all(c['initial_coverage']['status']=='COMPLETE' for c in cells) and all(c['teacher_path_status']!='PARTIAL_OFFLINE_BUDGET' for c in cells) else 'PARTIAL_OFFLINE_BUDGET'
    result=dict(status=terminal,cells=cells,requested_k=[4,8,16],completed_k=[c['k'] for c in cells],wall_total_s=time.perf_counter()-started,
        teacher_direction_rhs=teacher.counts,teacher_wall_s=teacher.wall_s,dense_J_acquisition_rhs=acquisition_rhs,
        physical_counters_before=physical_before,physical_counters_after=model.counters.as_dict(),
        context_cost=ctx.records(),runtime=runtime.receipt(),receipt=receipt)
    jswrite(out/'result.json',clean(result));jswrite(out/'status.json',dict(status=terminal,completed_k=result['completed_k']))
    return result


def parser():
    p=argparse.ArgumentParser();p.add_argument('--baseline-root',required=True);p.add_argument('--baseline-lock')
    p.add_argument('--completed-state',required=True);p.add_argument('--config',required=True);p.add_argument('--coverage-lock',required=True);p.add_argument('--inputs-root');p.add_argument('--input-manifest')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True)
    p.add_argument('--path-actions',type=int,choices=[0,3],default=0);p.add_argument('--offline-wall-limit-s',type=float,default=3600)
    return p


if __name__=='__main__':
    args=parser().parse_args();out=Path(args.out).resolve()
    if out.exists():raise SystemExit('new coverage output required')
    try:run(args)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True)
        for name in ['failure.json','status.json']:
            (out/name).write_text(json.dumps(dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
