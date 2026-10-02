"""New independent small 3-D tests; NOT old A17 attachment reproduction."""
import pinned_runtime as runtime
import numpy as np, json, time
from pathlib import Path
from scipy import linalg as la
from a10_common import DenseDDA,Tangent,grid,balanced
from operators import make_problem
import exchange_core as ec

def rel(a,b):return float(la.norm(a-b)/max(la.norm(a),la.norm(b),1e-30))
def main():
    start=time.perf_counter();rng=np.random.default_rng(20261002)
    pts,v=grid(3,1.5);d,p,rx,obs=balanced(12)
    model=DenseDDA(pts,v,2.,d,p,rx,obs,device='cpu')
    chi=.4+.05j+.1*rng.random(len(pts));state=model.state(chi)
    tangent=Tangent(pts,v,'voxel');ell=rng.normal(size=tangent.d)*.01
    ctx,ev=make_problem(state,tangent,state.field+.01*(rng.normal(size=state.field.shape)+1j*rng.normal(size=state.field.shape)),.5,.2,ell)
    Bc=ctx.project_B(np.eye(model.n,dtype=complex));B=np.concatenate([Bc,1j*Bc],axis=2)
    S=model.GS/.5;L=state.L
    rngz=rng.normal(size=tangent.d);w=rng.normal(size=2*model.P*model.m)
    J=ev.dense_j();adj=rel(np.array(ev.j(rngz)@w),np.array(rngz@ev.jt(w)))
    eps=1e-5;dc=tangent.expand(rngz)
    fd=(model.state(chi+eps*dc).field-model.state(chi-eps*dc).field)/(2*eps*.5)
    from operators import stack_real_illuminations
    tests=dict(real_adjoint=adj,full_material_jvp=rel(J@rngz,ev.j(rngz)),polarizability_derivative_finite_difference=rel(stack_real_illuminations(fd),J@rngz))
    atoms=ec.orth(rng.normal(size=(model.n,10))+1j*rng.normal(size=(model.n,10)))
    sw=ec.atom_swap(L,S,B,atoms,(0,1,2,3),(1,),(5,))
    tests['Schur_swap_vs_endpoints']=rel(sw['D'],sw['D_schur'])
    base=ctx.make_model(atoms[:,:4]);child=ctx.make_model(ec.orth(atoms[:,(0,2,3,5)]))
    Jo,Jn=base.j(np.eye(tangent.d)),child.j(np.eye(tangent.d))
    tests['existing_vs_independent_endpoint']=rel(Jo,sw['old']['J'])
    tests['existing_vs_independent_child']=rel(Jn,sw['new']['J'])
    x=base.step();delta=child.step()-x;D=Jn-Jo;e=ctx.r+Jo@x
    h=Jo.T@(D@x)+D.T@e+D.T@(D@x)
    tests['actual_step_change']=rel(delta,-child.solve_H(h))
    q=ec.gain_directional(ctx.r+J@x,J@delta,x,delta,ctx.lam,ctx.ell)
    direct=ev.phi(x)-ev.phi(x+delta);tests['directed_gain']=rel(np.array(q),np.array(direct))
    phase=np.exp(1j*rng.normal(size=4));permuted=ctx.make_model(atoms[:,[3,1,0,2]]*phase)
    tests['phase_permutation_step']=rel(x,permuted.step())
    # Nontrivial real Λ verified by the independent core; production retains λI.
    output=dict(status='PASS' if max(tests.values())<1e-9 else 'FAIL',new_verification=True,source_receipt=runtime.receipt(),tests=tests,threshold=1e-9,model_size=model.n,P=model.P,material_dimension=tangent.d,source_residual=state.source_residual(),wall_s=time.perf_counter()-start)
    out=runtime.ROOT/'tests/small_Maxwell_verification.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:output[k] for k in ('status','tests','wall_s')}))
    if output['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
