"""Production controller smoke on independently generated small 3D Maxwell.
Only implementation acceptance, not evidence of A17 selection advantage.
"""
import pinned_runtime as runtime
import numpy as np,json,tempfile
from pathlib import Path
from a10_common import DenseDDA,Tangent,grid,balanced
from operators import make_problem
from types import SimpleNamespace
from selectors_current import CandidatePool
from selector_coordinates import prepare_selector_workspace
from maxwell_state import ProbeContext,TeacherContext
from run_exchange import Neighborhood,policy_run,first_checkpoint_diagnostics,basis,CoreEndpointBatch

def main():
    rng=np.random.default_rng(20261002);pts,v=grid(3,1.5);dirs,pols,rx,obs=balanced(10)
    model=DenseDDA(pts,v,2.,dirs,pols,rx,obs,device='cpu');state=model.state(np.full(len(pts),.4+.05j));tan=Tangent(pts,v,'voxel')
    data=state.field+.01*(rng.normal(size=state.field.shape)+1j*rng.normal(size=state.field.shape));ell=.01*rng.normal(size=tan.d)
    ctx,ev=make_problem(state,tan,data,1.,.02,ell)
    from operators import orth
    Q=orth(rng.normal(size=(model.n,20))+1j*rng.normal(size=(model.n,20)))
    J=ev.dense_j();ref=-np.linalg.solve(J.T@J+.02*np.eye(tan.d),J.T@ctx.r+ell)
    pool=CandidatePool(Q,['atom'+str(i) for i in range(20)],['receiver']*4+['random']*16)
    anchor=ctx.make_model(Q[:,:12]);w=prepare_selector_workspace(ctx,pool,anchor,chunk_size=16);c=w.ctx
    eye=np.eye(c.n_current);engine=CoreEndpointBatch(c.apply_L(eye),c.apply_S(eye),c.project_B(eye),c.r,c.ell,c.lam,device='cpu')
    teacher=TeacherContext(ev.j,ctx.P,J);probe=ProbeContext(ev.j,ctx.P)
    class E:
        floor=1e-18
        def risk(self,s,Js):
            d=s-ref;v=J@d;return dict(full_gap=float(.5*(v@v+.02*(d@d))))
    evaluator=E();seed=[0,1,2,3];a=engine.evaluate(basis(w.pool.vectors,seed)[None]);x=a['steps'][0];jx=probe.j(x)
    checks={};identity=dict(object_id='small_implementation_only',k=4)
    with tempfile.TemporaryDirectory() as tmp:
        from info_logger import InfoLogger
        out=Path(tmp);information=InfoLogger(c,fullJ=J,precision=1e-5,output_dir=out/'rhs')
        information.base_summary(seed,basis(w.pool.vectors,seed),x,evaluator=True)
        raw=np.sum(np.abs(c.apply_S(w.pool.vectors))**2,axis=0)
        diag,nb=first_checkpoint_diagnostics(seed,4,w,engine,w.anchor,evaluator,teacher,x,jx,out,identity,information,raw)
        for name,menu in [('surrogate_exchange','swap'),('exact_score_teacher','swap'),('directed_verified','swap'),('adaptive_leq_k','adaptive')]:
            result=policy_run(name,seed,4,w,engine,w.anchor,evaluator,teacher,probe,x,jx,out,identity,menu,nb if name=='exact_score_teacher' else None)
            checks[name]=dict(final_rank=result['actual_rank'],accepted=len(result['accepted']),stop=result['stop_reason'],true_gain_sum=sum(a['Q_true'] for a in result['accepted']),normal_status='all endpoints require normal residual admission')
            assert result['actual_rank']<=4
            if name in ('exact_score_teacher','directed_verified','adaptive_leq_k'):assert all(a['Q_true']>a['tau'] for a in result['accepted'])
        result=dict(status='PASS',teacher_neighborhood=diag['declared_neighborhood_count'],policies=checks,scope='production policy controller smoke on small synthetic DDA, no advantage/generalization claim',runtime=runtime.receipt())
    (runtime.ROOT/'tests/driver_small_Maxwell_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['policies']))
if __name__=='__main__':main()
