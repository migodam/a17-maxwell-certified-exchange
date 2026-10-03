"""Locked bounded nonlinear transfer; existing full outer acceptance rules.

No reference optimal step, teacher, full Jacobian or new optimizer is acquired.
Receiver-only acquisition is independent of exchange anchor/pool acquisition.
Root must hash-lock this source and authorize this phase after primary results.
"""
from pathlib import Path
import argparse
import copy
import gc
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
import closure_runtime as runtime

VERSION = 'a17.closure.nonlinear_transfer.v1'
OBJECTS = [2002, 2005, 2009, 2016]
POLICIES = ['receiver_only', 'verified_exchange']
OUTER_RULES = dict(initialization='existing_scene_init', prior=1e-5, lm0=.01,
                   lm_decay=.3, lm_decay_period=3, max_updates=18,
                   real_lower=-.5, imag_lower=0., real_trial_tolerance=1e-8,
                   imag_trial_tolerance=1e-8, armijo=1e-4, line_trials=24,
                   backtracking=.5, gradient_fallback_denominator_floor=1e-8,
                   small_step=1e-6, small_step_first_iteration=9)


def validate_contract(config, object_id, policy):
    """Pure pre-result contract check; deliberately rejects replay prelock."""
    required = dict(nonlinear_objects=OBJECTS, nonlinear_k=8,
                    nonlinear_noise_percent=0, nonlinear_outer_cap=18,
                    nonlinear_initialization='existing_scene_init',
                    nonlinear_policies=POLICIES, nonlinear_version=VERSION,
                    nonlinear_outer_rules=OUTER_RULES,
                    online_incoming_count=12, max_finalists_per_round=2,
                    max_actions=3, anchor_rank=64, pool_count=32,
                    residual_up_count=16, random_pool_count=10,
                    pool_and_anchor_seed=20260930, cuda_candidate_batch=16,
                    online_tolerance_schema='a17.online.numerical_significance.v1',
                    online_tolerance_safety=32, prior_paid_occupation_s=4889.515,
                    cumulative_limit_s=43200, max_job_s=7200,
                    memory_warning=.8, memory_stop=.9)
    if 'nonlinear' not in config.get('permitted_modes', []):
        raise ValueError('nonlinear phase not yet authorized by final lock')
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError('nonlinear frozen contract differs: ' + key)
    if object_id not in OBJECTS or policy not in POLICIES:
        raise ValueError('object/policy outside fixed nonlinear matrix')
    return required


def validate_sources(args):
    config = json.loads(Path(args.config).read_text())
    validate_contract(config, args.object, args.policy)
    receipt = runtime.check_config_sources(config, args.config)
    if receipt['source_hashes'].get('code/run_nonlinear.py') != runtime.sha(__file__):
        raise ValueError('nonlinear driver must be locked in deployment_sources')
    manifest = Path(args.input_manifest).resolve()
    holdout = runtime.ROOT/'HOLDOUT_OBJECT_MANIFEST.json'
    for key, path in [('input_manifest_sha256', manifest), ('holdout_manifest_sha256', holdout)]:
        if config.get(key) != runtime.sha(path):
            raise ValueError('nonlinear input/holdout hash binding differs: '+key)
    declared = json.loads(holdout.read_text())
    if declared.get('nonlinear_objects') != OBJECTS:
        raise ValueError('pre-result holdout nonlinear subset differs')
    gate = Path(args.primary_gate).resolve()
    if config.get('nonlinear_primary_gate_sha256') != runtime.sha(gate):
        raise ValueError('after-primary owner authorization must be hash-bound')
    decision = json.loads(gate.read_text())
    if decision.get('owner_decision') != 'AUTHORIZE_NONLINEAR_AFTER_PRIMARY' or decision.get('primary_complete') is not True:
        raise ValueError('root has not authorized nonlinear after primary')
    return config, dict(config=receipt, primary_gate=dict(path=str(gate), sha256=runtime.sha(gate)))


def run(args):
    runtime.initialize(args.baseline_root, args.baseline_lock)
    config, lock_receipt = validate_sources(args)
    import numpy as np
    from scipy import linalg as la
    from a10_common import DenseDDA, Tangent, receiver_operator
    from scenes import acquisition
    from operators import make_problem, orth
    from material_gauge_persistence import restore_tangent, fingerprint, coefficient_payload
    from feasible_step import project_step
    from closure_state import manifest_map, check_input
    from maxwell_state import public_workspace, array_hash
    from removal_core_batch import CoreEndpointBatch
    from verified_exchange import online_exchange, DirectionAction
    from run_exchange import clean, jswrite

    out = Path(args.out).resolve()
    if out.exists() or out.is_relative_to(runtime.BASELINE):
        raise ValueError('fresh output outside read-only baseline required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); model = None; records = []; active_iteration = None

    def sync():
        if args.device == 'cuda':
            import torch
            torch.cuda.synchronize()

    def peak():
        if args.device != 'cuda':
            return None
        import torch
        return dict(allocated=int(torch.cuda.max_memory_allocated()), reserved=int(torch.cuda.max_memory_reserved()))

    def stamp(phase, **values):
        event = dict(phase=phase, iteration=active_iteration,
                     elapsed_wall_s=time.perf_counter()-started,
                     utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
        jswrite(out/'status.json', clean(dict(status='RUNNING', **event)))
        with (out/'phases.jsonl').open('a') as handle:
            handle.write(json.dumps(clean(event))+'\n')
        print(json.dumps(clean(event)), flush=True)

    def phase_begin(name):
        sync(); stamp(name+'_start')
        return time.perf_counter(), copy.deepcopy(model.counters.as_dict()) if model else None

    def phase_end(name, before):
        sync()
        info = dict(wall_s=time.perf_counter()-before[0],
                    physical_counters_before=before[1],
                    physical_counters_after=copy.deepcopy(model.counters.as_dict()) if model else None)
        stamp(name+'_end', **info)
        return info

    def save_arrays(path, **arrays):
        temp = path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(temp, **arrays); temp.replace(path)

    try:
        before = phase_begin('input_loading')
        inputs = Path(args.inputs_root).resolve()
        known = manifest_map(json.loads(Path(args.input_manifest).read_text()))
        paths = [inputs/'scenes.json', inputs/f'object_{args.object}/common_data.npz',
                 inputs/'original_resolved_config.json']
        input_receipts = [check_input(p, inputs, args.input_manifest, known) for p in paths]
        scenes = json.loads(paths[0].read_text())['scenes']
        scene = next(s for s in scenes if s['object_id'] == args.object)
        with np.load(paths[1], allow_pickle=False) as handle:
            common = {key: handle[key].copy() for key in handle.files}
        original = json.loads(paths[2].read_text())
        if original['prior'] != OUTER_RULES['prior'] or original['lm0'] != OUTER_RULES['lm0'] or original['reference_max_updates'] != 18:
            raise ValueError('original outer prior/LM/cap differs')
        points = common['points']; volume = float(scene['edge']/scene['n'])**3
        tangent, gauge = restore_tangent(points, volume, scene['representation'], common,
                                         tangent_class=Tangent, allow_legacy=False)
        init = np.full(len(points), complex(*scene['init']), dtype=np.complex128)
        if not np.array_equal(init, common['init']):
            raise ValueError('saved initial material differs from declared scene init')
        data = common['data0']; scale = float(common['scale'])
        if not np.isclose(scale, la.norm(data), rtol=1e-12, atol=0):
            raise ValueError('saved data scaling differs')
        truth = common.get('truth'); held_truth = common.get('held_truth')
        input_cost = phase_end('input_loading', before)
        jswrite(out/'run_config.json', clean(dict(version=VERSION, cli=vars(args),
                  lock_receipt=lock_receipt, input_receipts=input_receipts,
                  runtime_start=runtime.receipt(), scene=scene, gauge=gauge,
                  outer_rules=OUTER_RULES, noise_percent=0,
                  cost_contract='Each policy pays independent model and policy-specific setup; proposed anchor/pool never charged as common.',
                  reference_optimal_step_access=False)))
        before = phase_begin('physical_model_setup')
        model = DenseDDA(points, volume, scene['wavenumber'], *acquisition(scene), device=args.device)
        setup_cost = phase_end('physical_model_setup', before)
        chi = init.copy(); state = None; status = 'completed_max_updates'
        for it in range(18):
            active_iteration = it; iteration_started = time.perf_counter(); phase_costs = {}
            before = phase_begin('full_linearization')
            state = model.state(chi) if state is None else state
            reg = tangent.project(chi-init)
            lam = original['prior']+original['lm0']*(.3**(it//3))
            ell = original['prior']*reg
            ctx, ev = make_problem(state, tangent, data, scale, lam, ell)
            ctx.material_gauge = gauge
            objective = float(.5*(ctx.r@ctx.r)+.5*original['prior']*(reg@reg))
            phase_costs['full_linearization'] = phase_end('full_linearization', before)
            save_arrays(out/f'anchor_{it:02d}.npz', chi=chi, iteration=it, ell=ell,
                        lambda_total=lam, objective=objective,
                        material_tangent_fingerprint=np.asarray(fingerprint(tangent)))
            before = phase_begin('full_gradient')
            with ctx.phase('nonlinear_full_gradient'):
                h = ev.jt(ctx.r)+ell
            phase_costs['full_gradient'] = phase_end('full_gradient', before)
            iteration_out = out/f'iteration_{it:02d}'; iteration_out.mkdir()
            if args.policy == 'receiver_only':
                before = phase_begin('receiver_only_acquisition_and_endpoint')
                with ctx.phase('nonlinear_receiver_only'):
                    U = orth(ctx.receiver_modes(8), rank=8)
                    if U.shape[1] != 8:
                        raise ValueError('receiver-only actual rank differs')
                    L = U.conj().T@ctx.apply_L(U); S = ctx.apply_S(U); PB = ctx.project_B(U)
                    engine_device = 'cuda' if args.device == 'cuda' and ctx.q >= 256 else 'cpu'
                    engine = CoreEndpointBatch(L, S, PB, ctx.r, ell, lam,
                                               device=engine_device, batch_size=16)
                    endpoint = engine.evaluate(np.eye(8, dtype=np.complex128)[None])
                    if endpoint['status'] != ['OK']:
                        raise ValueError('receiver-only endpoint failed: '+str(endpoint['status']))
                    proposal = endpoint['steps'][0].copy()
                policy_record = dict(policy=args.policy, actual_rank=8,
                     engine_device=engine_device, engine_setup_wall_s=engine.setup_wall_s,
                     endpoint_statistics=endpoint['statistics'],
                     anchor_pool_acquisition=False, standalone_policy_setup=True)
                phase_costs['receiver_only_acquisition_and_endpoint'] = phase_end('receiver_only_acquisition_and_endpoint', before)
                save_arrays(iteration_out/'receiver_endpoint.npz', step=proposal, U=U)
            else:
                before = phase_begin('exchange_anchor_pool_workspace_acquisition')
                work = public_workspace(dict(ctx=ctx))
                w = work['workspace']
                engine_device = 'cuda' if args.device == 'cuda' and ctx.q >= 256 else 'cpu'
                engine = CoreEndpointBatch(work['L'], work['S'], work['PB'], ctx.r, ell, lam,
                                          device=engine_device, batch_size=16)
                phase_costs['exchange_anchor_pool_workspace_acquisition'] = phase_end('exchange_anchor_pool_workspace_acquisition', before)
                save_arrays(iteration_out/'public_workspace.npz', Q=w.Q, pool=w.pool.vectors,
                            anchor=w.anchor.U, L=work['L'], S=work['S'], PB=work['PB'],
                            r=ctx.r, ell=ell, lambda_total=lam,
                            labels=np.asarray(w.pool.labels), families=np.asarray(w.pool.families))
                before = phase_begin('verified_online_exchange')
                action = DirectionAction(ev.j, ctx.P, synchronize=sync, counter_supplier=model.counters.as_dict)
                policy_record, endpoint = online_exchange(w.ctx, w, engine, w.anchor, action, 8,
                    iteration_out/'exchange', dict(object_id=args.object, iteration=it,
                    policy=args.policy, state_hash=array_hash(chi), pool_hash=work['pool_hash']))
                proposal = endpoint['step'].copy()
                policy_record.update(anchor_pool_acquisition=True,
                   workspace_wall_s=work['workspace_wall_s'], pool_wall_s=work['pool_wall_s'],
                   anchor_metadata=work['anchor_meta'], engine_device=engine_device,
                   engine_setup_wall_s=engine.setup_wall_s,
                   acquisition_is_policy_specific=True, not_charged_as_common=True)
                phase_costs['verified_online_exchange'] = phase_end('verified_online_exchange', before)
            save_arrays(out/f'proposal_{it:02d}.npz', gradient=h, **coefficient_payload(tangent, proposal))
            before = phase_begin('passive_projection_and_full_gradient_fallback')
            step, projection = project_step(tangent, chi, proposal); fallback = False
            if h@step >= 0:
                step, projection = project_step(tangent, chi, -h/max(lam, la.norm(h), 1e-8)); fallback = True
            slope = float(h@step); dc = tangent.expand(step)
            phase_costs['passive_projection_and_full_gradient_fallback'] = phase_end('passive_projection_and_full_gradient_fallback', before)
            before = phase_begin('full_nonlinear_armijo')
            alpha = 1.; accepted = False; trials = []; infeasible = 0
            for trial_index in range(24):
                cand = chi+alpha*dc
                if cand.imag.min() < -1e-8 or cand.real.min() < -.50000001:
                    trials.append(dict(index=trial_index, alpha=alpha, status='INFEASIBLE', objective=None))
                    alpha *= .5; infeasible += 1; continue
                trial = model.state(cand)
                rr = (trial.field-data)/scale; rg = tangent.project(cand-init)
                val = float(.5*la.norm(rr)**2+.5*original['prior']*(rg@rg))
                ok = val <= objective+1e-4*alpha*min(slope, 0.)
                trials.append(dict(index=trial_index, alpha=alpha, objective=val,
                                   status='ACCEPT' if ok else 'REJECT'))
                if ok:
                    chi, state, accepted = cand, trial, True; break
                del trial; alpha *= .5
            phase_costs['full_nonlinear_armijo'] = phase_end('full_nonlinear_armijo', before)
            record = dict(iteration=it, policy=args.policy, objective_before=objective,
                lambda_total=lam, actual_rank=8, policy_result=policy_record,
                projection=projection, gradient_fallback=fallback, slope=slope,
                alpha=alpha, accepted=accepted, trials=trials, infeasible_trials=infeasible,
                executed_step_norm=float(la.norm(alpha*step)),
                unconstrained_exchange_Q_scope='Within this frozen quadratic only; projection/fallback/alpha changes direction and nonlinear objective acceptance is independently full-physics Armijo.',
                physical_counters=copy.deepcopy(model.counters.as_dict()),
                context_cost=ctx.records(), phase_costs=phase_costs,
                iteration_wall_s=time.perf_counter()-iteration_started, gpu_peak=peak())
            records.append(record); jswrite(out/'trajectory.json', clean(records))
            save_arrays(out/f'chi_checkpoint_{it:02d}.npz', chi=chi, iteration=it,
                        accepted=accepted, projected_step=step, alpha=alpha)
            if accepted:
                save_arrays(out/'last_safe_state.npz', chi=chi, iteration=it+1)
            stamp('outer_checkpoint_saved', accepted=accepted)
            # Release callback cycles before the next material-state LU.
            del ctx, ev, engine
            if args.policy == 'verified_exchange':
                del work, w, action, endpoint
            gc.collect()
            if not accepted:
                status = 'stopped_no_armijo_accept'; break
            if la.norm(alpha*step) < 1e-6 and it >= 9:
                status = 'stopped_small_step'; break
        before = phase_begin('final_offline_metrics')
        metrics = dict(material_complex_relative=None, material_real_relative=None,
                       material_imag_relative=None, measured_data_relative=float(la.norm(state.field-data)/scale),
                       heldout_data_relative=None, heldout_data_status='NOT_PROVIDED')
        if truth is not None:
            for label, estimated, actual in [('complex', chi, truth), ('real', chi.real, truth.real), ('imag', chi.imag, truth.imag)]:
                denom = float(la.norm(actual))
                metrics['material_'+label+'_relative'] = float(la.norm(estimated-actual)/denom) if denom > 0 else None
        held = None
        if held_truth is not None:
            _, _, hr, ho = acquisition(scene, heldout=True)
            held = state.current@(receiver_operator(points, scene['wavenumber'], hr, ho)*np.sqrt(volume)).T
            denom = float(la.norm(held_truth))
            metrics.update(heldout_data_relative=float(la.norm(held-held_truth)/denom) if denom > 0 else None,
                           heldout_data_status='EXISTING_PROVIDED_EVALUATION_ONLY')
        final_cost = phase_end('final_offline_metrics', before)
        arrays = dict(chi=chi, field=state.field)
        if held is not None:
            arrays['held'] = held
        save_arrays(out/'final.npz', **arrays)
        result = dict(status=status, version=VERSION, object_id=args.object, policy=args.policy,
          noise_percent=0, nonlinear_k=8, records=len(records), metrics=metrics,
          input_cost=input_cost, physical_setup_cost=setup_cost, final_metrics_cost=final_cost,
          full_physical_counters=model.counters.as_dict(), gpu_peak=peak(),
          wall_total_s=time.perf_counter()-started, runtime_end=runtime.receipt(),
          deterministic_certificate=False, no_reference_optimal_step_acquired=True,
          full_gradient_outer_only=True, scientific_acceptance='PARENT_REVIEW_REQUIRED')
        jswrite(out/'result.json', clean(result)); jswrite(out/'status.json', clean(dict(status=status, completed_iterations=len(records))))
        return result
    except Exception as exc:
        jswrite(out/'failure.json', clean(dict(status='FAILED', error=repr(exc),
            traceback=traceback.format_exc(), iteration=active_iteration,
            completed_iterations=len(records), wall_total_s=time.perf_counter()-started,
            full_physical_counters=model.counters.as_dict() if model else None,
            runtime_end=runtime.receipt())))
        jswrite(out/'status.json', dict(status='FAILED', iteration=active_iteration))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True); p.add_argument('--baseline-lock')
    p.add_argument('--config', required=True); p.add_argument('--input-manifest', required=True)
    p.add_argument('--inputs-root', required=True); p.add_argument('--primary-gate', required=True)
    p.add_argument('--object', type=int, choices=OBJECTS, required=True)
    p.add_argument('--policy', choices=POLICIES, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out', required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
