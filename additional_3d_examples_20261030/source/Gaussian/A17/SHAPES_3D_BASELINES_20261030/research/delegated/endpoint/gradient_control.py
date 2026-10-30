"""POSTHOC full-gradient control; define code now, run only after main terminal.

Fixed digit3 then torus, at most 18 attempts each. Uses the current main-run
physical kernels, inputs, constraints, projection, prior/LM and Armijo objective.
No ROM, anchor, full Jacobian or truth-based selection. Distinct histories make
this a diagnostic of the ordinary safeguard itself, not a unique causal
attribution of the 36 main trajectories. No changes to frozen main code/caps.

Invocation, only when explicitly authorized after the main queue stops:
  Gaussian/.venv_nn/bin/python <this-file>
Output: <new experiment root>/gradient_control/{digit3,torus}/.
"""
from pathlib import Path
import gc
import json
import sys
import time
import traceback

EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT_ROOT))
# Import the NEW root runner explicitly; instantiate its Budget only, never the
# historical old.Budget. Importing it defines kernels without constructing states.
import run_comparison as comparison
import numpy as np
from scipy import linalg as la

CONTROL_ROOT = EXPERIMENT_ROOT / 'gradient_control'
CASES = ('digit3', 'torus')
POLICY = 'full_gradient'
MAX_ATTEMPTS = 18
INITIAL_WALL_RESERVE_S = 600.0
SECOND_CASE_WALL_RESERVE_S = 300.0
PHASE = 'posthoc_full_gradient'
SCOPE = {
    'posthoc': True,
    'policy': POLICY,
    'fixed_cases': list(CASES),
    'max_attempts_per_case': MAX_ATTEMPTS,
    'direction': '-h/max(lambda,norm(h),1e-8); h=ev.jt(ctx.r)+ell',
    'projection_calls_per_proposal': 1,
    'selection_uses_truth': False,
    'ROM_anchor_fullJ_used': False,
    'history_matched_to_main': False,
    'causal_unique_attribution': False,
    'interpretation': 'ordinary projected-gradient safeguard diagnostic only',
    'budget_scope': 'same COST_STATE and COST_LEDGER; cumulative caps unchanged',
    'trajectory_deadline_scope': 'same outer-admission check as frozen main runner',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def remaining_wall(budget=None):
    if budget is None:
        state = read_json(EXPERIMENT_ROOT / 'COST_STATE.json')
        elapsed = 0.0
    else:
        state = budget.data
        elapsed = time.perf_counter() - budget.start
    return float(comparison.CFG['wall_cap_s'] - state['wall_s'] - elapsed)


def write_not_run(case, status, reason, reserve_s=None):
    out = CONTROL_ROOT / case
    if out.exists():
        raise RuntimeError('REFUSE_EXISTING_CONTROL_RESULT_' + str(out))
    comparison.write(out / 'final.json', {
        **SCOPE, 'case': case, 'status': status, 'reason': reason,
        'attempted_updates': 0, 'updates': 0, 'metrics': None,
        'physical_work_started': False, 'remaining_wall_s': reserve_s,
    })
    comparison.write(out / 'trajectory.json', [])


def preflight():
    """Read-only gate. No Budget/session or physical model exists here."""
    terminal = read_json(EXPERIMENT_ROOT / 'RUN_TERMINAL.json')
    if terminal.get('status') != 'ALL_PLANNED_TERMINAL':
        raise RuntimeError('MAIN_NOT_ALL_PLANNED_TERMINAL')
    expected = {(case, policy) for case in comparison.CASES
                for policy in comparison.POLICIES}
    outcomes = terminal.get('outcomes', [])
    observed = {(row['case'], row['policy']) for row in outcomes}
    if len(expected) != 36 or len(outcomes) != 36 or observed != expected:
        raise RuntimeError('MAIN_36_TERMINAL_OUTCOMES_REQUIRED')
    for row in outcomes:
        if row['status'] == 'RUNNING' or row['status'].startswith('BUDGET_STOP'):
            raise RuntimeError('MAIN_NONTERMINAL_OR_BUDGET_STOP_NO_RETRY')
        final = read_json(EXPERIMENT_ROOT / 'results' / row['case'] /
                          row['policy'] / 'final.json')
        if final['status'] != row['status']:
            raise RuntimeError('MAIN_TERMINAL_FINAL_STATUS_MISMATCH')
    state = read_json(EXPERIMENT_ROOT / 'COST_STATE.json')
    if any(session.get('status') == 'RUNNING'
           for session in state.get('sessions', [])):
        raise RuntimeError('ACTIVE_SESSION_NO_CONTROL_ALLOWED')
    if read_json(EXPERIMENT_ROOT / 'CONFIG.json') != comparison.CFG:
        raise RuntimeError('FROZEN_CONFIG_DIFFERENT')
    if CONTROL_ROOT.exists():
        raise RuntimeError('REFUSE_EXISTING_CONTROL_ROOT_NO_OVERWRITE_OR_RETRY')
    for case in CASES:
        if not (EXPERIMENT_ROOT / 'inputs' / (case + '.npz')).is_file():
            raise RuntimeError('FIXED_INPUT_MISSING_' + case)
    return remaining_wall()


def run_control_case(case, budget):
    """Explicit root-owned steps; no monkeypatch and no shared runner mutation."""
    out = CONTROL_ROOT / case
    if out.exists():
        raise RuntimeError('REFUSE_EXISTING_CONTROL_RESULT_' + str(out))
    out.mkdir(parents=True)
    started = time.perf_counter()
    records = []
    state = model = None
    status = 'RUNNING'
    setup = None
    last_accepted_attempt = None
    # Truth and held_truth below are passed only to offline metrics/output.
    with np.load(EXPERIMENT_ROOT / 'inputs' / (case + '.npz')) as saved:
        inp = {key: saved[key].copy() for key in saved.files}
    chi = inp['init'].copy()
    initial = chi.copy()
    truth = inp['truth']
    scale = float(inp['scale'])
    current = None
    try:
        budget.stage = f'{case}/{POLICY}/setup'
        t = time.perf_counter()
        model, tangent = comparison.new_model(budget)
        setup = time.perf_counter() - t
        for iteration in range(MAX_ATTEMPTS):
            budget.check()
            if time.perf_counter() - started >= comparison.CFG['trajectory_cap_s']:
                raise comparison.BudgetStop('TRAJECTORY_WALL_CAP')
            budget.stage = f'{case}/{POLICY}/update_{iteration}'
            begin = time.perf_counter()
            current = dict(iteration=iteration, accepted=False,
                           accepted_update_count=sum(row['accepted'] for row in records),
                           proposal_computed=False, completed=False,
                           attempt_status='STARTED', trials=[], phases={})
            records.append(current)
            comparison.write(out / 'trajectory.json', records)
            t = time.perf_counter()
            if state is None:
                state = model.state(chi)
            reg = tangent.project(chi - initial)
            lam = comparison.CFG['prior'] + comparison.CFG['lm0'] * (
                comparison.CFG['lm_decay'] ** (iteration // 3))
            ell = comparison.CFG['prior'] * reg
            ctx, evaluator = comparison.op.make_problem(
                state, tangent, inp['data'], scale, lam, ell)
            objective = float(.5 * (ctx.r @ ctx.r) +
                              .5 * comparison.CFG['prior'] * (reg @ reg))
            before = comparison.old.metrics(
                chi, truth, state, inp['data'], scale, inp['held_truth'])
            current.update(before_metrics=before, objective_before=objective,
                           lambda_normal=lam)
            current['phases']['linearization_s'] = time.perf_counter() - t
            if before['measured_relative_residual'] <= comparison.CFG['residual_stop']:
                status = 'DISCREPANCY_REACHED'
                current.update(attempt_status=status, completed=True,
                               objective_after=objective, metrics=before)
                break
            t = time.perf_counter()
            h = evaluator.jt(ctx.r) + ell
            proposal = -h / max(lam, la.norm(h), 1e-8)
            current['phases']['policy_s'] = time.perf_counter() - t
            current['proposal_computed'] = True
            t = time.perf_counter()
            # Exactly one projection, as in the public safeguard branch.
            step, projection = comparison.M['feasible_step'].project_step(
                tangent, chi, proposal)
            slope = float(h @ step)
            direction = tangent.expand(step)
            current.update(direction_source='projected_gradient_safeguard',
                           policy=POLICY, gradient_fallback=False,
                           gradient_only_control=True, slope=slope,
                           projection=projection, step_norm=float(la.norm(step)),
                           alpha=1.0, executed_step_norm=0.0)
            current['phases']['projection_s'] = time.perf_counter() - t
            alpha = 1.0
            t = time.perf_counter()
            for trial_number in range(comparison.CFG['max_trials']):
                if slope >= 0 or la.norm(step) <= 1e-15:
                    break
                candidate = chi + alpha * direction
                if (candidate.imag.min() < -1e-9 or
                        candidate.real.min() < -.500000001):
                    raise RuntimeError('FIXED_RAY_FEASIBILITY')
                trial_record = dict(alpha=alpha, objective=None, accepted=False,
                                    trial_status='REQUESTED')
                current['trials'].append(trial_record)
                trial = model.state(candidate)
                trial_reg = tangent.project(candidate - initial)
                data_residual = (trial.field - inp['data']) / scale
                value = float(.5 * la.norm(data_residual) ** 2 +
                              .5 * comparison.CFG['prior'] * (trial_reg @ trial_reg))
                accepted = bool(value <= objective +
                                comparison.CFG['armijo'] * alpha * slope)
                trial_record.update(objective=value, accepted=accepted,
                                    source_residual=trial.source_residual(),
                                    trial_status='COMPLETED')
                if accepted:
                    # The sole material-state assignment; final output is last safe.
                    chi, state = candidate, trial
                    current['accepted'] = True
                    last_accepted_attempt = iteration
                    break
                del trial
                alpha *= .5
            current['phases']['line_search_s'] = time.perf_counter() - t
            after = comparison.old.metrics(
                chi, truth, state, inp['data'], scale, inp['held_truth'])
            current.update(
                alpha=alpha,
                executed_step_norm=float(la.norm(alpha * step)) if current['accepted'] else 0.0,
                accepted_update_count=sum(row['accepted'] for row in records),
                objective_after=current['trials'][-1]['objective'] if current['accepted'] else objective,
                metrics=after, completed=True, attempt_status='COMPLETED',
                wall_s=time.perf_counter() - begin)
            comparison.write(out / 'trajectory.json', records)
            np.savez_compressed(out / f'chi_{iteration+1:02d}.npz', chi=chi)
            comparison.write(out / 'progress.json', dict(
                case=case, policy=POLICY, attempts=len(records),
                updates=sum(row['accepted'] for row in records),
                metrics=after, elapsed_s=time.perf_counter()-started,
                status='RUNNING', **{'posthoc': True}))
            print(json.dumps(dict(event='GRADIENT_CONTROL_UPDATE', case=case,
                                  iteration=iteration+1, accepted=current['accepted'],
                                  nrmse=after['complex_nrmse'], alpha=alpha)), flush=True)
            del ctx, evaluator, proposal
            gc.collect()
            budget.flush()
            if not current['accepted']:
                status = 'NO_ARMIJO_ACCEPT'
                break
            if current['executed_step_norm'] < 1e-6 and iteration >= 9:
                status = 'SMALL_STEP'
                break
        else:
            status = 'MAX_UPDATES'
    except comparison.BudgetStop as exc:
        status = 'BUDGET_STOP_' + str(exc)
        if current is not None and not current['completed']:
            current['attempt_status'] = status
            if current['trials'] and current['trials'][-1]['trial_status'] == 'REQUESTED':
                current['trials'][-1]['trial_status'] = status
    except Exception as exc:
        status = 'ERROR'
        if current is not None and not current['completed']:
            current['attempt_status'] = status
            if current['trials'] and current['trials'][-1]['trial_status'] == 'REQUESTED':
                current['trials'][-1]['trial_status'] = status
        comparison.write(out / 'error.json', dict(
            error=str(exc), traceback=traceback.format_exc()))
    finally:
        final_metrics = None
        metrics_error = None
        if state is not None:
            try:
                final_metrics = comparison.old.metrics(
                    chi, truth, state, inp['data'], scale, inp['held_truth'], held=True)
            except Exception as exc:
                metrics_error = str(exc)
                status = 'ERROR_FINAL_METRICS_AFTER_' + status
        np.savez_compressed(out / 'final.npz', chi=chi, truth=truth,
                            points=inp['points'])
        comparison.write(out / 'trajectory.json', records)
        final = dict(
            **SCOPE, case=case, status=status, attempted_updates=len(records),
            completed_attempts=sum(row['completed'] for row in records),
            updates=sum(row['accepted'] for row in records),
            last_accepted_attempt=last_accepted_attempt,
            final_state_policy='last actually accepted chi, or common initial chi',
            metrics=final_metrics, metrics_error=metrics_error,
            wall_s=time.perf_counter()-started, model_setup_s=setup,
            counters=None if model is None else model.counters.as_dict(),
            same_grid=True, native_SOM_or_TSOM=False,
            physical_work_started=model is not None)
        comparison.write(out / 'final.json', final)
        print(json.dumps(dict(event='GRADIENT_CONTROL_FINAL', case=case,
                              policy=POLICY, status=status, updates=final['updates'])), flush=True)
        del model, state
        gc.collect()
    return status


def main():
    reserve = preflight()
    if reserve < INITIAL_WALL_RESERVE_S:
        for case in CASES:
            write_not_run(case, 'NOT_RUN_RESOURCE_RESERVE',
                          'entry remaining wall below 600 seconds', reserve)
        comparison.write(CONTROL_ROOT / 'CONTROL_TERMINAL.json', dict(
            **SCOPE, status='NOT_RUN_RESOURCE_RESERVE',
            remaining_wall_s=reserve, new_session_started=False))
        return
    # Mandatory primary gate already passed. This starts one NEW session in the
    # existing cumulative budget, with no reset and no change to any cap.
    budget = comparison.Budget(PHASE)
    outcomes = []
    status = 'ERROR'
    try:
        comparison.write(CONTROL_ROOT / 'SESSION_STARTED.json', dict(
            **SCOPE, entry_remaining_wall_s=remaining_wall(budget),
            starting_totals={key: budget.data[key] for key in ('states','rhs','wall_s')}))
        for index, case in enumerate(CASES):
            reserve = remaining_wall(budget)
            # The 600s gate was checked at entry; only the second case has
            # a new 300s reserve gate after the first case consumes resources.
            required = SECOND_CASE_WALL_RESERVE_S
            if index > 0 and reserve < required:
                write_not_run(case, 'NOT_RUN_RESOURCE_RESERVE',
                              f'remaining wall below {required:g} seconds before case', reserve)
                outcomes.append(dict(case=case, status='NOT_RUN_RESOURCE_RESERVE'))
                continue
            result = run_control_case(case, budget)
            outcomes.append(dict(case=case, status=result))
            comparison.write(CONTROL_ROOT / 'OUTCOMES.json', outcomes)
            if 'BUDGET_STOP' in result:
                status = 'BUDGET_STOP'
                for later in CASES[index+1:]:
                    write_not_run(later, 'NOT_RUN_PREVIOUS_BUDGET_STOP',
                                  'prior control case budget-stopped; no retry', remaining_wall(budget))
                    outcomes.append(dict(case=later, status='NOT_RUN_PREVIOUS_BUDGET_STOP'))
                break
        else:
            status = 'ALL_CONTROL_TERMINAL'
    finally:
        budget.close(status)
        comparison.write(CONTROL_ROOT / 'OUTCOMES.json', outcomes)
        comparison.write(CONTROL_ROOT / 'CONTROL_TERMINAL.json', dict(
            **SCOPE, status=status, outcomes=outcomes,
            cumulative_budget_snapshot=budget.data,
            new_physical_work_stopped=True))


if __name__ == '__main__':
    with comparison.threadpool_limits(limits=4):
        main()
