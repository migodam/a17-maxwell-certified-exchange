"""Previously declared digit3/torus gradient controls; serial CUDA execution."""
import ctypes
import json
import sys
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
import run_comparison as r
from remote_gpu_continuation import cuda_model

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research/delegated/endpoint'))
import gradient_control as control

def main():
    r.CFG['wall_cap_s']=10800.0
    cost=json.loads((ROOT/'COST_STATE.json').read_text())
    remaining=r.CFG['wall_cap_s']-cost['wall_s']
    if remaining<600:
        for case in control.CASES:
            control.write_not_run(case,'NOT_RUN_RESOURCE_RESERVE','entry remaining wall below declared 600-second reserve',remaining)
        r.write(control.CONTROL_ROOT/'CONTROL_TERMINAL.json',dict(status='NOT_RUN_RESOURCE_RESERVE',remaining_wall_s=remaining))
        return
    kernel=ctypes.WinDLL('kernel32')
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
    kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),0xffff)
    torch.set_num_threads(4)
    r.new_model=cuda_model
    b=r.Budget('XINAN_GPU_serial_ordinary_gradient_control')
    outcomes=[];status='ERROR'
    try:
        with threadpool_limits(limits=4):
            for index,case in enumerate(control.CASES):
                remaining=control.remaining_wall(b)
                if index and remaining<300:
                    control.write_not_run(case,'NOT_RUN_RESOURCE_RESERVE','remaining below declared 300-second second-case reserve',remaining)
                    outcomes.append(dict(case=case,status='NOT_RUN_RESOURCE_RESERVE'))
                    continue
                result=control.run_control_case(case,b)
                outcomes.append(dict(case=case,status=result))
                r.write(control.CONTROL_ROOT/'OUTCOMES.json',outcomes)
                if result.startswith('BUDGET_STOP'):
                    status='BUDGET_STOP';break
            else:status='ALL_CONTROL_TERMINAL'
    finally:
        b.close(status)
        r.write(control.CONTROL_ROOT/'CONTROL_TERMINAL.json',dict(
            status=status,outcomes=outcomes,posthoc=True,
            physics_backend='original_DenseDDA_CUDA_complex128',
            cpu_threads=4,P_core_mask='0xffff',parallel_jobs=1,
            cost_caps_unchanged=True,new_physical_work_stopped=True))

if __name__=='__main__':main()
