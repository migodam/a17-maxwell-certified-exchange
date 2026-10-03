"""Bounded real-Maxwell implementation gate; no scientific holdout claim.

Explicit baseline source binding and fresh output directory are mandatory.
CPU is the default. CUDA is opt-in and compares the same tiny physical inputs
and both endpoint normal branches to CPU; this script never opens a remote job.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
CLOSURE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLOSURE / 'code'))
import closure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    baseline = Path(args.baseline_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out == baseline or out.is_relative_to(baseline):
        parser.error('output cannot be inside the immutable baseline')
    if out.exists():
        parser.error('output must be a fresh directory')
    closure_runtime.initialize(baseline)
    import numpy as np
    from scipy import linalg as la
    from a10_common import DenseDDA, Tangent, grid, balanced
    from operators import make_problem, stack_real_illuminations
    from endpoint_batch import EndpointBatch
    from removal_core_batch import CoreEndpointBatch
    from online_tolerance import directional_check
    from verified_exchange import DirectionAction
    import exchange_core as ec

    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable; no CPU substitution')
        torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    tests, admissions, branches = {}, [], []

    def rel(x, y):
        return float(la.norm(x-y)/max(la.norm(x), la.norm(y), 1e-30))

    def ah(x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

    result = dict(schema='a17.closure.small_maxwell_smoke.v1', status='FAIL',
                  seed=20261002, device=args.device, grid_n=3,
                  tests=tests, branches=branches, admissions=admissions,
                  threshold=1e-9, scientific_claim=False,
                  scope='Tiny real physical implementation/checkpoint gate only; no holdout outcomes, nonlinear imaging or deterministic output certificate.')
    try:
        rng = np.random.default_rng(result['seed'])
        pts, volume = grid(3, 1.5)
        dirs, pols, rx, obs = balanced(12)
        tangent = Tangent(pts, volume, 'voxel')
        chi = .4+.05j+.1*rng.random(len(pts))
        ell = .01*rng.normal(size=tangent.d)
        cpu = DenseDDA(pts, volume, 2., dirs, pols, rx, obs, device='cpu')
        cpu_state = cpu.state(chi)
        data = cpu_state.field+.01*(rng.normal(size=cpu_state.field.shape)+1j*rng.normal(size=cpu_state.field.shape))
        cpu_ctx, cpu_ev = make_problem(cpu_state, tangent, data, .5, .2, ell)
        model, state, ctx, ev = cpu, cpu_state, cpu_ctx, cpu_ev
        if args.device == 'cuda':
            model = DenseDDA(pts, volume, 2., dirs, pols, rx, obs, device='cuda')
            state = model.state(chi)
            ctx, ev = make_problem(state, tangent, data, .5, .2, ell)
            tests['CPU_CUDA_state_fields'] = rel(cpu_state.field, state.field)
        J = ev.dense_j()
        material = rng.normal(size=(tangent.d, 3))
        w = rng.normal(size=len(ctx.r))
        tests['real_adjoint'] = rel(np.asarray(ev.j(material[:, 0])@w), np.asarray(material[:, 0]@ev.jt(w)))
        tests['full_material_jvp'] = rel(J@material, ev.j(material))
        eps = 1e-5
        dc = tangent.expand(material[:, 0])
        fd = (model.state(chi+eps*dc).field-model.state(chi-eps*dc).field)/(2*eps*.5)
        tests['polarizability_derivative_finite_difference'] = rel(stack_real_illuminations(fd), J@material[:, 0])
        if args.device == 'cuda':
            tests['CPU_CUDA_full_jvp'] = rel(cpu_ev.j(material), ev.j(material))
            tests['CPU_CUDA_real_adjoint'] = rel(cpu_ev.jt(w), ev.jt(w))
        atoms = ec.orth(rng.normal(size=(model.n, 10))+1j*rng.normal(size=(model.n, 10)))
        PB = ctx.project_B(np.eye(model.n, dtype=complex))
        B = np.concatenate([PB, 1j*PB], axis=2)
        sw = ec.atom_swap(state.L, model.GS/.5, B, atoms, (0, 1, 2, 3), (1,), (5,))
        tests['Schur_swap_vs_direct_endpoints'] = rel(sw['D'], sw['D_schur'])
        base = ctx.make_model(atoms[:, :4])
        child = ctx.make_model(ec.orth(atoms[:, (0, 2, 3, 5)]))
        Jo, Jn = base.j(np.eye(tangent.d)), child.j(np.eye(tangent.d))
        tests['existing_vs_independent_base'] = rel(Jo, sw['old']['J'])
        tests['existing_vs_independent_child'] = rel(Jn, sw['new']['J'])
        x = base.step(); delta = child.step()-x
        D = Jn-Jo; e = ctx.r+Jo@x
        h = Jo.T@(D@x)+D.T@e+D.T@(D@x)
        tests['actual_step_change'] = rel(delta, -child.solve_H(h))
        phase = np.exp(1j*rng.normal(size=4))
        permuted = ctx.make_model(atoms[:, [3, 1, 0, 2]]*phase)
        tests['phase_permutation_step'] = rel(x, permuted.step())
        action = DirectionAction(ev.j, model.P, counter_supplier=lambda: model.counters.as_dict())
        Jx = action.j(np.asarray(x, dtype=np.float64))
        z = action.j(np.asarray(delta, dtype=np.float64))
        q = ec.gain_directional(ctx.r+Jx, z, x, delta, ctx.lam, ctx.ell)
        direct = ev.phi(x)-ev.phi(x+delta)
        tests['directed_gain_vs_objective_difference'] = rel(np.asarray(q), np.asarray(direct))
        check = directional_check(ctx.r, ctx.r+Jx, z, x, delta, ctx.ell, ctx.lam)
        tests['online_gain_vs_objective_difference'] = rel(check['gain'], np.asarray([direct]))
        tests['online_gain_vs_legacy_directional'] = rel(check['gain'], np.asarray([q]))
        zero = directional_check(ctx.r, ctx.r+Jx, np.zeros_like(z), x, np.zeros_like(delta), ctx.ell, ctx.lam)
        admissions.append(dict(check='online_tolerance_contract', passed=bool(np.isfinite(check['tau']).all() and (check['tau']>0).all() and not check['deterministic_certificate'] and zero['gain'][0] == 0 and zero['tau'][0] > 0)))
        result['online_check'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in check.items()}
        result['direction_action_counts'] = action.snapshot()
        result['objective_difference'] = float(direct)
        result['gain_error_absolute'] = abs(float(check['gain'][0])-direct)
        admissions.append(dict(check='physical_online_gain_within_numerical_tau',
                               passed=bool(result['gain_error_absolute'] <= check['tau'][0]),
                               gain_error_absolute=result['gain_error_absolute'], tau=float(check['tau'][0]),
                               deterministic_output_error_bound=False))
        # Public current coefficient cache from the same tiny physical state.
        Q = ec.orth(rng.normal(size=(model.n, 24))+1j*rng.normal(size=(model.n, 24)))
        L = Q.conj().T@ctx.apply_L(Q); S = ctx.apply_S(Q); PBC = ctx.project_B(Q)
        bases = np.stack([ec.orth(rng.normal(size=(24, 4))+1j*rng.normal(size=(24, 4))) for _ in range(17)])
        reference_outputs = {}
        for branch in ('direct', 'woodbury'):
            engine = EndpointBatch(L, S, PBC, ctx.r, ell, .2, device='cpu', branch=branch)
            endpoint = engine.evaluate(bases)
            core_engine = CoreEndpointBatch(L, S, PBC, ctx.r, ell, .2, device='cpu', branch=branch)
            V = bases[0, :, :3]; incoming = bases[:, :, -1:]
            core = core_engine.prepare_removal_core(V)
            core_out = core_engine.evaluate_incoming(core, incoming)
            direct_bases = np.stack([np.concatenate([V, c], axis=1) for c in core_out['incoming_basis']])
            direct_out = engine.evaluate(direct_bases)
            for key in ('steps', 'j_steps', 'info_trace'):
                tests[branch+'_core_vs_direct_'+key] = rel(core_out[key], direct_out[key])
            admissions.append(dict(check=branch+'_CPU_admission', passed=all(s == 'OK' for s in endpoint['status']+core_out['status']+direct_out['status'])))
            row = dict(branch=branch, CPU_endpoint_statistics=endpoint['statistics'], CPU_core_prepare_statistics=core.statistics, CPU_core_statistics=core_out['statistics'])
            reference_outputs[branch] = endpoint
            if args.device == 'cuda':
                gpu_engine = EndpointBatch(L, S, PBC, ctx.r, ell, .2, device='cuda', branch=branch)
                gpu_endpoint = gpu_engine.evaluate(bases)
                gpu_core_engine = CoreEndpointBatch(L, S, PBC, ctx.r, ell, .2, device='cuda', branch=branch)
                gpu_core = gpu_core_engine.prepare_removal_core(V)
                gpu_out = gpu_core_engine.evaluate_incoming(gpu_core, incoming)
                for key in ('steps', 'j_steps', 'info_trace'):
                    tests[branch+'_CPU_CUDA_endpoint_'+key] = rel(endpoint[key], gpu_endpoint[key])
                    tests[branch+'_CPU_CUDA_core_'+key] = rel(core_out[key], gpu_out[key])
                admissions.append(dict(check=branch+'_CUDA_admission', passed=all(s == 'OK' for s in gpu_endpoint['status']+gpu_out['status'])))
                row.update(CUDA_endpoint_statistics=gpu_endpoint['statistics'], CUDA_core_prepare_statistics=gpu_core.statistics, CUDA_core_statistics=gpu_out['statistics'])
            branches.append(row)
        for key in ('steps', 'j_steps', 'info_trace'):
            tests['CPU_direct_vs_woodbury_'+key] = rel(reference_outputs['direct'][key], reference_outputs['woodbury'][key])
        fixture = dict(points=pts, chi=chi, data=data, ell=ell, material=material, atoms=atoms, Q=Q, bases=bases)
        result['input_array_sha256'] = {k: ah(v) for k, v in fixture.items()}
        np.savez_compressed(out/'smoke_inputs.npz', **fixture)
        result.update(model_size=model.n, material_dimension=tangent.d, P=model.P, batch_candidates=17,
                      source_residual=state.source_residual(), physical_CPU_counters=cpu.counters.as_dict(),
                      physical_test_device_counters=model.counters.as_dict())
        if args.device == 'cuda':
            result['peak_memory'] = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        result['status'] = 'PASS' if all(np.isfinite(v) and v < result['threshold'] for v in tests.values()) and all(x['passed'] for x in admissions) else 'FAIL'
    except Exception as exc:
        import traceback
        result.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        result['wall_s'] = time.perf_counter()-start
        result['runtime'] = closure_runtime.receipt()
        result['checkpoint'] = dict(smoke_sha256=closure_runtime.sha(__file__),
                                    verified_exchange_sha256=closure_runtime.sha(CLOSURE/'code/verified_exchange.py'),
                                    online_tolerance_sha256=closure_runtime.sha(CLOSURE/'code/online_tolerance.py'))
        (out/'small_maxwell_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(status=result['status'], device=args.device, tests=tests, error=result.get('error'), wall_s=result['wall_s'])), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
