"""User-approved total-wall extension; all experiment rules stay frozen."""
import remote_fast_gpu_parallel as fast

if __name__=='__main__':
    fast.parallel.r.CFG['wall_cap_s']=10800.0
    fast.parallel.r.write(fast.ROOT/'CONFIG_EXTENDED.json',fast.parallel.r.CFG)
    fast.main()
