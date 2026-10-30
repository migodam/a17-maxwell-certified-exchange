"""Remaining actual experiments: original CUDA physics, original CPU endpoints.

The explicit model factory below changes only the existing DenseDDA device.
No new comparison test, training, selector, precision or solve rule is added.
"""
import json,os,time
from pathlib import Path
import run_comparison as r
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent

def cuda_model(budget,n=None):
    budget.check();p,v=r.old.grid(n or r.CFG['n'],r.CFG['edge'])
    model=r.old.DenseDDA(p,v,r.CFG['wavenumber'],*r.old.acquisition(),device='cuda')
    model.solver=r.old.CountedSolver(model.solver,budget)
    return model,r.old.Tangent(p,v,'voxel')

def main():
    prior=json.loads((ROOT/'REMOTE_OUTCOMES.json').read_text())
    done={(v['case'],v['policy']) for v in prior};pending=[]
    for i,name in enumerate(r.CASES):
        for policy in r.POLICIES[i:]+r.POLICIES[:i]:
            if (name,policy) not in done:pending.append((name,policy))
    # Named, recorded runtime dependency injection, not a hidden numerical patch.
    r.new_model=cuda_model
    b=r.Budget('XINAN_4060_continuation');status='ERROR';outcomes=list(prior)
    with threadpool_limits(limits=4):
        r.write(ROOT/'GPU_EXECUTION.json',dict(pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),physics_backend='original_DenseDDA_cuda_complex128',endpoint_backend='same_RefinedCoreEndpointBatch_CPU_FP64',cpu_threads=4,scheduled=len(pending),factory_binding='run_comparison.new_model = remote_gpu_continuation.cuda_model',extra_test_or_benchmark=False))
        try:
            for name,policy in pending:
                b.check();torch.cuda.reset_peak_memory_stats()
                s=r.run_case(name,policy,b)
                outcomes.append(dict(case=name,policy=policy,status=s,execution_host='XINAN_4060',gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                r.write(ROOT/'GPU_OUTCOMES.json',outcomes)
                if s.startswith('BUDGET_STOP'):raise r.BudgetStop(s)
                torch.cuda.empty_cache()
            status='ALL_PLANNED_TERMINAL'
        except r.BudgetStop as exc:
            status='BUDGET_STOP';r.write(ROOT/'GPU_BUDGET_STOP.json',dict(reason=str(exc),outcomes=outcomes))
        finally:
            b.close(status)
            r.write(ROOT/'RUN_TERMINAL.json',dict(status=status,outcomes=outcomes,new_physical_work_stopped=True,continuation='documented user-requested hardware execution; complete jobs never repeated'))
            r.write(ROOT/'GPU_TERMINAL.json',dict(status=status,outcomes=outcomes))

if __name__=='__main__':main()
