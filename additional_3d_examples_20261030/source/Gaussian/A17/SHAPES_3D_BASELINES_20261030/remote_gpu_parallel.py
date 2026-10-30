"""Pending jobs with two threads: original CUDA physics and CPU endpoints.

This entry is deployed only after the serial GPU job stops at its recorded
boundary. GPU_OUTCOMES supplies prior terminal jobs; none are retried.
No additional comparison, test, benchmark or CPU fallback is introduced.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import run_comparison as r
from remote_gpu_continuation import cuda_model
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


class JobBudgetProxy:
    """One job's stage, with serialized access to the shared original Budget."""
    def __init__(self, master, lock, case, policy):
        self.master = master
        self.lock = lock
        self.stage = f'{case}/{policy}/setup'

    def check(self):
        with self.lock:
            return self.master.check()

    def charge(self, kind, count=1):
        with self.lock:
            self.master.stage = self.stage
            return self.master.charge(kind, count)

    def flush(self):
        with self.lock:
            return self.master.flush()


def main():
    prior = json.loads((ROOT / 'GPU_OUTCOMES.json').read_text())
    completed = {(row['case'], row['policy']) for row in prior}
    pending = [
        (case, policy)
        for index, case in enumerate(r.CASES)
        for policy in r.POLICIES[index:] + r.POLICIES[:index]
        if (case, policy) not in completed
    ]
    # Explicit, recorded factory binding; all numerical functions stay original.
    r.new_model = cuda_model
    master = r.Budget('XINAN_4060_parallel')
    lock = threading.RLock()
    outcomes = []
    next_job = 0
    budget_stopped = False
    status = 'ERROR'
    not_run = []
    with threadpool_limits(limits=4):
        torch.cuda.reset_peak_memory_stats()
        r.write(ROOT / 'PARALLEL_EXECUTION.json', dict(
            pid=os.getpid(), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            parallel_jobs=2, process_count=1,
            executor='ThreadPoolExecutor(max_workers=2)',
            physics_backend='original_DenseDDA_cuda_complex128',
            endpoint_backend='same_RefinedCoreEndpointBatch_CPU_FP64',
            cpu_threads=4, scheduled=len(pending), prior_terminal_count=len(prior),
            factory_binding='run_comparison.new_model = remote_gpu_continuation.cuda_model',
            budget_phase='XINAN_4060_parallel',
            shared_budget=True, budget_lock='RLock around check/charge/flush',
            caps={key: r.CFG[key] for key in ('wall_cap_s','state_cap','rhs_cap')},
            wall_scope='master phase elapsed wall; overlapping child walls are not additive',
            physical_count_scope='all state and full RHS charges from both jobs accumulate',
            thread_context='original helper contextvars; separate models and job output paths',
            gpu_peak_scope='one reset at phase start, one whole-phase reading at terminal; not per job',
            extra_test_or_benchmark=False, extra_checks=False,
            automatic_retry=False, additional_cpu_fallback=False))
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                while next_job < len(pending) and len(futures) < 2:
                    case, policy = pending[next_job]
                    next_job += 1
                    proxy = JobBudgetProxy(master, lock, case, policy)
                    future = executor.submit(r.run_case, case, policy, proxy)
                    futures[future] = (case, policy)
                while futures:
                    finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                    # Process all terminal futures before considering more jobs.
                    for future in finished:
                        case, policy = futures.pop(future)
                        result = future.result()
                        outcomes.append(dict(
                            case=case, policy=policy, status=result,
                            execution_host='XINAN_4060', parallel_jobs=2))
                        r.write(ROOT / 'PARALLEL_OUTCOMES.json', outcomes)
                        if result.startswith('BUDGET_STOP'):
                            budget_stopped = True
                    if budget_stopped:
                        not_run = [dict(case=case, policy=policy,
                                        status='NOT_RUN_BUDGET_STOP')
                                   for case, policy in pending[next_job:]]
                        r.write(ROOT / 'PARALLEL_NOT_RUN.json', not_run)
                        # Existing futures finish through the original run_case;
                        # no cancellation, new submission, retry or CPU fallback.
                        continue
                    while next_job < len(pending) and len(futures) < 2:
                        case, policy = pending[next_job]
                        next_job += 1
                        proxy = JobBudgetProxy(master, lock, case, policy)
                        future = executor.submit(r.run_case, case, policy, proxy)
                        futures[future] = (case, policy)
            status = 'BUDGET_STOP' if budget_stopped else 'ALL_PLANNED_TERMINAL'
        finally:
            # The executor has waited for every submitted job before this block.
            if status == 'ERROR':
                not_run = [dict(case=case, policy=policy, status='NOT_RUN_PHASE_ERROR')
                           for case, policy in pending[next_job:]]
            phase_peak = dict(
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                scope='entire parallel phase, both jobs together; PyTorch allocator statistics')
            master.close(status)
            merged = list(prior) + outcomes
            r.write(ROOT / 'PARALLEL_OUTCOMES.json', outcomes)
            r.write(ROOT / 'PARALLEL_NOT_RUN.json', not_run)
            r.write(ROOT / 'PARALLEL_TERMINAL.json', dict(
                status=status, outcomes=outcomes, not_run=not_run,
                parallel_jobs=2, gpu_phase_peak=phase_peak,
                master_phase_wall_s=master.data['sessions'][-1]['wall_s'],
                child_wall_times_additive=False,
                new_physical_work_stopped=True))
            r.write(ROOT / 'RUN_TERMINAL.json', dict(
                status=status, outcomes=merged, not_run=not_run,
                parallel_jobs=2, gpu_phase_peak=phase_peak,
                new_physical_work_stopped=True,
                continuation='prior terminal jobs retained; remaining original jobs use two-thread CUDA physics / CPU endpoint overlap'))


if __name__ == '__main__':
    main()
