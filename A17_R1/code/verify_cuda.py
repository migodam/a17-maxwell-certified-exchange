"""Real small DDA CPU/CUDA branch gate, before full frozen pilot."""
import pinned_runtime as runtime
import numpy as np,json,time
from pathlib import Path
from scipy import linalg as la
from a10_common import DenseDDA,Tangent,grid,balanced
from operators import make_problem
from endpoint_batch import EndpointBatch
from removal_core_batch import CoreEndpointBatch

def rel(x,y):return float(la.norm(x-y)/max(la.norm(x),la.norm(y),1e-30))
def main():
    import torch
    start=time.perf_counter();rng=np.random.default_rng(20261002)
    pts,v=grid(4,1.5);dirs,pols,rx,obs=balanced(12);tan=Tangent(pts,v,'voxel')
    chi=.4+.05j+.05*rng.random(len(pts))
    cpu=DenseDDA(pts,v,2.,dirs,pols,rx,obs,device='cpu');cs=cpu.state(chi)
    gpu=DenseDDA(pts,v,2.,dirs,pols,rx,obs,device='cuda');gs=gpu.state(chi)
    ell=.01*rng.normal(size=tan.d);data=cs.field+.01*(rng.normal(size=cs.field.shape)+1j*rng.normal(size=cs.field.shape))
    c,ce=make_problem(cs,tan,data,.5,.2,ell);g,ge=make_problem(gs,tan,data,.5,.2,ell)
    material=rng.normal(size=(tan.d,3));tests=dict(state_fields=rel(cs.field,gs.field),full_jvp=rel(ce.j(material),ge.j(material)))
    from operators import orth
    Q=orth(rng.normal(size=(cpu.n,24))+1j*rng.normal(size=(cpu.n,24)))
    L=Q.conj().T@c.apply_L(Q);S=c.apply_S(Q);B=c.project_B(Q)
    bases=np.stack([orth(rng.normal(size=(24,4))+1j*rng.normal(size=(24,4))) for _ in range(17)])
    branches=[]
    for branch in ('direct','woodbury'):
        eb=EndpointBatch(L,S,B,c.r,ell,.2,device='cpu',branch=branch)
        eg=EndpointBatch(L,S,B,c.r,ell,.2,device='cuda',branch=branch)
        a=eb.evaluate(bases);b=eg.evaluate(bases)
        tests[branch+'_info_trace']=rel(a['info_trace'],b['info_trace'])
        tests[branch+'_steps']=rel(a['steps'],b['steps']);tests[branch+'_Jsteps']=rel(a['j_steps'],b['j_steps'])
        branches.append(dict(branch=branch,CPU_status=a['status'],CUDA_status=b['status'],CUDA_statistics=b['statistics']))
        if any(x!='OK' for x in a['status']+b['status']):raise RuntimeError('endpoint stability/normal gate failed')
        V=bases[0,:,:3];C=bases[:,:,-1:]
        ca=CoreEndpointBatch(L,S,B,c.r,ell,.2,device='cpu',branch=branch)
        cb=CoreEndpointBatch(L,S,B,c.r,ell,.2,device='cuda',branch=branch)
        ac=ca.evaluate_incoming(ca.prepare_removal_core(V),C)
        bc=cb.evaluate_incoming(cb.prepare_removal_core(V),C)
        tests[branch+'_shared_core_steps']=rel(ac['steps'],bc['steps'])
        tests[branch+'_shared_core_Jsteps']=rel(ac['j_steps'],bc['j_steps'])
        tests[branch+'_shared_core_info_trace']=rel(ac['info_trace'],bc['info_trace'])
        if any(x!='OK' for x in ac['status']+bc['status']):raise RuntimeError('shared-core CPU/CUDA admission failed')
        branches[-1]['CUDA_shared_core_statistics']=bc['statistics']
    fullJ=ge.dense_j();z=rng.normal(size=len(c.r))
    tests['real_adjoint_cuda']=rel(np.array(ge.j(material[:,0])@z),np.array(material[:,0]@ge.jt(z)))
    result=dict(status='PASS' if max(tests.values())<1e-9 else 'FAIL',threshold=1e-9,tests=tests,model_size=cpu.n,material_d=tan.d,P=cpu.P,batch_candidates=17,branches=branches,wall_s=time.perf_counter()-start,runtime=runtime.receipt(),peak_memory=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),scope='Small physical DDA and two material-normal branches, not all frozen A17 states or gains.')
    (runtime.ROOT/'tests/cuda_Maxwell_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','tests','wall_s')}),flush=True)
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
