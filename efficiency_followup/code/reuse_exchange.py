"""Isolated P6 frozen current-space starter; unchanged original top2/cap3."""
import copy
import time
from pathlib import Path
import numpy as np
from online_tolerance import directional_check

VERSION = 'a17.efficiency.previous_space_exchange.v1'


class DirectionAction:
    """Pure real FP64 full physical J action, counted independently of evaluator."""
    def __init__(self, action, P, synchronize=None, counter_supplier=None):
        self._action, self.P = action, int(P)
        self._sync = synchronize or (lambda: None)
        self._counters = counter_supplier
        self.calls = 0; self.rhs = 0; self.wall_s = 0.

    def counters(self):
        return copy.deepcopy(self._counters()) if self._counters else None

    def snapshot(self):
        return dict(calls=self.calls, tangent_rhs=self.rhs, wall_s=self.wall_s)

    def j(self, directions):
        directions = np.asarray(directions)
        if directions.dtype != np.float64 or np.iscomplexobj(directions) or not np.isfinite(directions).all():
            raise ValueError('pure action requires finite FP64 real directions')
        self._sync(); started = time.perf_counter()
        self.calls += 1; self.rhs += self.P * (1 if directions.ndim == 1 else directions.shape[1])
        try:
            out = np.asarray(self._action(directions))
            if out.dtype != np.float64 or np.iscomplexobj(out) or not np.isfinite(out).all():
                raise ValueError('physical action must return finite real FP64 output')
            return out
        finally:
            self._sync(); self.wall_s += time.perf_counter() - started


def online_exchange(ctx, workspace, engine, anchor, direction, k, out,
                    identity=None, max_actions=3, initial_ids=None):
    """No evaluator, reference step, floor, teacher, fullJ or truth arguments."""
    from run_exchange import Neighborhood, balanced_ids, moves, gains_on_anchor, basis, clean, jswrite, append_rows
    from maxwell_state import array_hash
    if max_actions != 3 or k not in (4, 8, 16):
        raise ValueError('locked max3 actions and k4/8/16')
    if ctx is not workspace.ctx:
        raise ValueError('online uses the declared public coefficient context')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    identity = identity or {}
    start = time.perf_counter(); before = direction.snapshot()
    physical_before = direction.counters()
    if initial_ids is None:
        ids = [i for i, family in enumerate(workspace.pool.families) if family == 'receiver'][:k]
    else:
        ids = list(initial_ids)
        if len(ids)!=k or len(set(ids))!=k or any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(workspace.pool.families) for i in ids):
            raise ValueError('initial IDs must identify exactly k distinct current pool atoms')
    U = basis(workspace.pool.vectors, ids)
    if len(ids) != k or U.shape[1] != k:
        raise ValueError('ACTUAL_RECEIVER_RANK_MISMATCH')
    receiver_start = time.perf_counter(); receiver = engine.evaluate(U[None])
    receiver_wall = time.perf_counter() - receiver_start
    if receiver['status'] != ['OK']:
        raise ValueError('RECEIVER_ENDPOINT_' + receiver['status'][0])
    x = receiver['steps'][0].copy(); initial = x.copy()
    t = time.perf_counter(); Jx = direction.j(x); initial_Jx = Jx.copy()
    init_wall = time.perf_counter() - t
    jswrite(out/'receiver_cost.json', clean(dict(receiver_wall_s=receiver_wall,
             statistics=receiver['statistics'], receiver_has_no_extra_online_full_rhs=True,
             guarded_Jx_initialization_wall_s=init_wall, guarded_Jx_initialization_rhs=ctx.P)))
    np.savez_compressed(out/'receiver_endpoint.npz', step=x, Js=Jx, selected_ids=ids, U_coeff=U)
    trace=[]; accepted=[]; statistics=[]; stop='MOVE_BUDGET_EXHAUSTED'
    stage=dict(receiver_wall_s=receiver_wall, Jx_initialization_wall_s=init_wall,
               endpoint_wall_s=0., anchor_wall_s=0., Jd_wall_s=0., acceptance_wall_s=0.)
    for roundno in range(3):
        incoming = balanced_ids(workspace.pool, ids, 12)
        proposals = moves(ids, incoming, 'swap', k)
        neigh = Neighborhood(workspace, engine, out, identity)
        t=time.perf_counter(); valid, steps, info, build = neigh.endpoints(proposals)
        stage['endpoint_wall_s'] += time.perf_counter()-t; statistics.extend(neigh.timings)
        D = steps-x[:, None]
        t=time.perf_counter(); qhat = gains_on_anchor(anchor, x, D, ctx) if len(valid) else np.empty(0)
        stage['anchor_wall_s'] += time.perf_counter()-t
        finalists = np.argsort(-qhat, kind='stable')[:2]
        t=time.perf_counter(); z = direction.j(D[:,finalists]) if len(finalists) else np.empty((len(ctx.r),0))
        stage['Jd_wall_s'] += time.perf_counter()-t
        u=ctx.r+Jx
        t=time.perf_counter(); check=directional_check(ctx.r,u,z,x,D[:,finalists],ctx.ell,ctx.lam)
        stage['acceptance_wall_s'] += time.perf_counter()-t
        eligible=[j for j in range(len(finalists)) if check['gain'][j] > check['tau'][j]]
        chosen=max(eligible,key=lambda j:check['gain'][j]) if eligible else None
        rows=[]; position={int(i):j for j,i in enumerate(valid)}
        for i, move in enumerate(proposals):
            row=dict(move_index=i, round=roundno, **move, **info[i], **identity)
            if i in position:
                j=position[i]; row.update(Qhat=float(qhat[j]),step_hash=array_hash(steps[:,j]))
            else: row.update(Qhat=None,step_hash=None)
            row.update(base_step_hash=array_hash(x),base_U_hash=array_hash(basis(workspace.pool.vectors,ids)),accepted=False)
            rows.append(row)
        finalists_json=[]
        for j, pos in enumerate(finalists):
            index=int(valid[pos]); rec={key: (float(value[j]) if isinstance(value,np.ndarray) else value)
                                      for key,value in check.items()}
            rec.update(move_index=index,Qhat=float(qhat[pos]),accepted_candidate=j in eligible,
                       selected=j==chosen,eps_det=None,certificate_type='HIGH_FIDELITY_CHECK')
            finalists_json.append(rec);rows[index].update(Q_true=rec['gain'],tau=rec['tau'],accepted=j==chosen)
        np.savez_compressed(out/f'finalists_round_{roundno}.npz', x=x,d=D[:,finalists],u=u,z=z,
              r=ctx.r,ell=ctx.ell,lambda_total=ctx.lam,finalist_positions=finalists,
              move_indices=valid[finalists],Qhat=qhat[finalists],**{key:value for key,value in check.items() if isinstance(value,np.ndarray)})
        append_rows(out/'moves.jsonl',rows)
        round_record=dict(round=roundno,incoming=incoming,selected_ids_before=list(ids),
               endpoint_count=len(proposals),feasible=len(valid),endpoint_info=info,
               finalists=finalists_json,accepted=chosen is not None)
        if chosen is None:
            stop='ABSTAIN_UNRESOLVED';round_record['stop']=stop;trace.append(round_record);break
        pos=int(finalists[chosen]); move=proposals[int(valid[pos])]
        ids=list(move['ids']);x=steps[:,pos].copy();Jx=Jx+z[:,chosen]
        if len(ids)!=k or basis(workspace.pool.vectors,ids).shape[1]!=k:
            raise RuntimeError('fixed-swap rank invariant violated')
        action=dict(round=roundno,**move,Qhat=float(qhat[pos]),Q_true=float(check['gain'][chosen]),tau=float(check['tau'][chosen]),actual_rank=k)
        accepted.append(action);round_record['selected_move']=action;trace.append(round_record)
        np.savez_compressed(out/f'round_{roundno}.npz',step=x,Js=Jx,selected_ids=ids,U_coeff=basis(workspace.pool.vectors,ids))
        jswrite(out/'checkpoint.json',clean(dict(completed_round=roundno,selected_ids=ids,accepted=accepted)))
    direction._sync();after=direction.snapshot();physical_after=direction.counters()
    result=dict(schema='a17.efficiency.previous_space_exchange.v1',requested_k=k,actual_rank=k,
          selected_ids=ids,accepted=accepted,rounds=trace,stop_reason=stop,
          certificate_type='HIGH_FIDELITY_CHECK',eps_det=None,deterministic_certificate=False,
          no_op_included=True,full_dictionary_stopping_certificate=False,
          wall_online_total_s=time.perf_counter()-start,stage_wall_s=stage,
          receiver_cost_separate=True,online_exchange_wall_excluding_receiver_s=time.perf_counter()-start-receiver_wall,
          direction_cost={key:after[key]-before[key] for key in after},
          physical_counters_before=physical_before,physical_counters_after=physical_after,
          adjoint_rhs=0,endpoint_statistics=statistics,**identity)
    result['reuse_policy'] = dict(version=VERSION,initialization='receiver' if initial_ids is None else 'previous_physical_current_atoms',
          initial_ids=None if initial_ids is None else list(initial_ids),max_finalists_per_round=2,max_accepted_exchanges=3,
          full_Jx_directions=1,full_Jd_directions=sum(len(r['finalists']) for r in trace),
          stale_Jx_Jd_or_material_step_reuse=False)
    np.savez_compressed(out/'endpoint.npz',step=x,Js=Jx,selected_ids=ids,U_coeff=basis(workspace.pool.vectors,ids),
                        initial_step=initial,initial_Js=initial_Jx)
    jswrite(out/'online_result.json',clean(result))
    return result,dict(step=x,Js=Jx,initial_step=initial,initial_Js=initial_Jx)
