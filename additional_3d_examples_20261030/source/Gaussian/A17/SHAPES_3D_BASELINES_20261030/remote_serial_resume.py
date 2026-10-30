"""Single-GPU formal continuation after the native process crash.

RESUME_PLAN_NATIVE.json layout:
  completed_outcomes: existing terminal outcome records (25 at this handoff)
  pending: [{"case": "cross3d", "policy": "exchange"}, ...] in fixed order
  partials: {"cross3d/exchange": {"next_iteration": 9, "prior_wall_s": ...},
             "cross3d/full_gn": {"next_iteration": 3, "prior_wall_s": ...}}

The parent records the crash cost and archives partial/prefix artifacts before
entry. No completed trajectory is repeated. No tests, benchmark, added solver,
concurrency or CPU fallback is introduced. The user approved cumulative wall
cap is 10800 seconds; existing state/RHS caps and cumulative costs are retained.
"""
import ctypes
import json
import os
from pathlib import Path
import time

import run_comparison as r
from remote_gpu_continuation import cuda_model
from remote_fast_gpu_parallel import cuda_endpoint_factory
from resumable_case import continue_case
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent


def main():
    if os.name=='nt':
        kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
        kernel32.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_uint32)
        handle=kernel32.GetCurrentProcess()
        kernel32.SetProcessAffinityMask(handle,0xffff)
        kernel32.SetPriorityClass(handle,0x00008000)
    torch.set_num_threads(4)
    r.CFG['wall_cap_s']=10800.
    r.new_model=cuda_model
    r.RefinedCoreEndpointBatch=cuda_endpoint_factory
    plan=json.loads((ROOT/'RESUME_PLAN_NATIVE.json').read_text())
    outcomes=list(plan['completed_outcomes'])
    completed={(row['case'],row['policy']) for row in outcomes}
    pending=plan['pending']
    partials=plan['partials']
    b=r.Budget('XINAN_4060_serial_native_resume')
    status='ERROR';not_run=[];next_pending=0
    with threadpool_limits(limits=4):
        r.write(ROOT/'SERIAL_EXECUTION.json',dict(
            pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            execution_mode='single GPU job; serial scheduler',
            physics_factory='run_comparison.new_model = remote_gpu_continuation.cuda_model',
            endpoint_factory='run_comparison.RefinedCoreEndpointBatch = remote_fast_gpu_parallel.cuda_endpoint_factory',
            physics_backend='original_DenseDDA_cuda_complex128',
            endpoint_backend='GpuRefinedCoreEndpointBatch_CUDA_complex128_float64',
            cpu_threads=4,torch_num_threads=4,
            process_affinity_mask='0xffff' if os.name=='nt' else None,
            process_priority='AboveNormal' if os.name=='nt' else None,
            priority_scope='current process only',
            caps={key:r.CFG[key] for key in ('wall_cap_s','state_cap','rhs_cap')},
            approved_wall_cap_seconds=10800,
            prior_terminal_count=len(outcomes),scheduled=len(pending),
            partial_resume_metadata=partials,
            counters_scope='new model counters cover continuation only; old global ledger retained',
            extra_test_or_benchmark=False,extra_checks=False,
            automatic_retry=False,additional_cpu_fallback=False))
        try:
            for index,job in enumerate(pending):
                next_pending=index
                name,policy=job['case'],job['policy']
                if (name,policy) in completed:
                    next_pending=index+1
                    continue
                final_path=ROOT/'results'/name/policy/'final.json'
                if final_path.exists():
                    # A terminal final already exists: retain it, never rerun it.
                    final=json.loads(final_path.read_text())
                    outcomes.append(dict(case=name,policy=policy,status=final['status'],
                                         execution_host='existing_terminal_preserved'))
                    completed.add((name,policy))
                    r.write(ROOT/'SERIAL_OUTCOMES.json',outcomes)
                    next_pending=index+1
                    continue
                b.check()
                key=name+'/'+policy
                if key in partials:
                    result=continue_case(name,policy,b,partials[key])
                else:
                    result=r.run_case(name,policy,b)
                outcomes.append(dict(case=name,policy=policy,status=result,
                                     execution_host='XINAN_4060_serial_resume',
                                     resumed_prefix=key in partials))
                completed.add((name,policy))
                next_pending=index+1
                r.write(ROOT/'SERIAL_OUTCOMES.json',outcomes)
                if result.startswith('BUDGET_STOP'):
                    raise r.BudgetStop(result)
            status='ALL_PLANNED_TERMINAL'
        except r.BudgetStop as exc:
            status='BUDGET_STOP'
            not_run=[dict(case=job['case'],policy=job['policy'],status='NOT_RUN_BUDGET_STOP')
                     for job in pending[next_pending:]
                     if (job['case'],job['policy']) not in completed]
            r.write(ROOT/'SERIAL_BUDGET_STOP.json',dict(
                reason=str(exc),outcomes=outcomes,not_run=not_run))
        finally:
            if status=='ERROR':
                not_run=[dict(case=job['case'],policy=job['policy'],status='NOT_RUN_SERIAL_ERROR')
                         for job in pending[next_pending:]
                         if (job['case'],job['policy']) not in completed]
            b.close(status)
            r.write(ROOT/'SERIAL_OUTCOMES.json',outcomes)
            r.write(ROOT/'SERIAL_NOT_RUN.json',not_run)
            r.write(ROOT/'RUN_TERMINAL.json',dict(
                status=status,outcomes=outcomes,not_run=not_run,
                new_physical_work_stopped=True,
                continuation='single-GPU native-crash recovery; committed prefixes and prior terminal jobs retained',
                approved_wall_cap_seconds=10800))
            r.write(ROOT/'SERIAL_TERMINAL.json',dict(
                status=status,outcomes=outcomes,not_run=not_run,
                master_phase_wall_s=b.data['sessions'][-1]['wall_s'],
                cumulative_budget_snapshot=b.data))


if __name__=='__main__':
    main()
