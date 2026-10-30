"""Named runtime entry: original CUDA physics and CUDA endpoint batches.

Uses the existing parallel scheduler and its GPU_OUTCOMES prior. No additional
numerical checks, comparison run, benchmark or CPU fallback is performed.
"""
import ctypes
import os
from pathlib import Path
import sys
import time

import remote_gpu_parallel as parallel
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research/delegated/endpoint'))
from gpu_refined_endpoint_batch import GpuRefinedCoreEndpointBatch


def cuda_endpoint_factory(*args,**kwargs):
    # Existing callers still pass device='cpu'; this named factory explicitly
    # overrides that argument to select the original CUDA batch implementation.
    kwargs['device']='cuda'
    return GpuRefinedCoreEndpointBatch(*args,**kwargs)


def main():
    affinity_ok=priority_ok=None
    if os.name=='nt':
        kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
        kernel32.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_uint32)
        handle=kernel32.GetCurrentProcess()
        affinity_ok=bool(kernel32.SetProcessAffinityMask(handle,0xffff))
        priority_ok=bool(kernel32.SetPriorityClass(handle,0x00008000))  # ABOVE_NORMAL_PRIORITY_CLASS
    torch.set_num_threads(4)
    parallel.r.RefinedCoreEndpointBatch=cuda_endpoint_factory
    parallel.r.write(ROOT/'FAST_ENDPOINT_EXECUTION.json',dict(
        pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        physics_backend='original_DenseDDA_cuda_complex128',
        endpoint_backend='GpuRefinedCoreEndpointBatch_CUDA_complex128_float64',
        endpoint_device='cuda',physics_device='cuda',parallel_jobs=2,
        worker_threads=2,cpu_numerical_threads=4,torch_num_threads=4,
        process_affinity_mask='0xffff' if os.name=='nt' else None,
        process_priority='AboveNormal' if os.name=='nt' else None,
        affinity_call_succeeded=affinity_ok,priority_call_succeeded=priority_ok,
        priority_scope='current process only; no system power/security changes',
        runtime_endpoint_factory='run_comparison.RefinedCoreEndpointBatch = remote_fast_gpu_parallel.cuda_endpoint_factory',
        endpoint_device_argument_override='existing caller cpu -> cuda, explicitly overridden by named factory',
        runtime_physics_factory='run_comparison.new_model = remote_gpu_continuation.cuda_model (existing parallel entry)',
        scheduler='remote_gpu_parallel.main, ThreadPoolExecutor(max_workers=2)',
        shared_budget='existing parallel RLock-protected original Budget',
        extra_test_or_benchmark=False,extra_checks=False,
        cpu_fallback=False,numerical_rule='original CUDA branch plus maximum 3 same-SPD FP64 refinements; unchanged TOL/lambda/ell/Gamma/rank/packing'))
    parallel.main()


if __name__=='__main__':
    main()
