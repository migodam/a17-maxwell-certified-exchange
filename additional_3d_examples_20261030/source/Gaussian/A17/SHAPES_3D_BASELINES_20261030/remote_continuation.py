"""XINAN i9 continuation: same numerical code, only never-started tasks."""
import os
import json
import time
from pathlib import Path
import run_comparison as r
from threadpoolctl import threadpool_limits, threadpool_info

ROOT=Path(__file__).resolve().parent

def main():
    plan=json.loads((ROOT/'REMOTE_CONTINUATION_PLAN.json').read_text())
    prior=plan['completed_local_outcomes']
    budget=r.Budget('XINAN_i9_continuation');outcomes=list(prior);status='ERROR'
    with threadpool_limits(limits=8):
        r.write(ROOT/'REMOTE_EXECUTION.json',dict(host='XINAN',processor='i9-14900HX',backend='cpu',threads=8,pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pools=threadpool_info(),previous_local_outcomes=len(prior),scheduled=len(plan['pending']),numerical_source_unchanged=True,physics_rules_unchanged=True,extra_parity_or_benchmark_run=False))
        try:
            for task in plan['pending']:
                budget.check();s=r.run_case(task['case'],task['policy'],budget)
                outcomes.append(dict(case=task['case'],policy=task['policy'],status=s,execution_host='XINAN_i9'))
                r.write(ROOT/'REMOTE_OUTCOMES.json',outcomes)
                if s.startswith('BUDGET_STOP'):raise r.BudgetStop(s)
            status='ALL_PLANNED_TERMINAL'
        except r.BudgetStop as exc:
            status='BUDGET_STOP';r.write(ROOT/'REMOTE_BUDGET_STOP.json',dict(reason=str(exc),outcomes=outcomes))
        finally:
            budget.close(status)
            r.write(ROOT/'RUN_TERMINAL.json',dict(status=status,outcomes=outcomes,new_physical_work_stopped=True,continuation='user-requested XINAN i9; first two complete objects retained from Mac',local_handoff_original_status='ERROR_FROM_DOCUMENTED_REFUSE_OVERWRITE_BOUNDARY_GUARD'))
            r.write(ROOT/'REMOTE_TERMINAL.json',dict(status=status,outcomes=outcomes,additional_jobs=len(outcomes)-len(prior)))

if __name__=='__main__':main()
