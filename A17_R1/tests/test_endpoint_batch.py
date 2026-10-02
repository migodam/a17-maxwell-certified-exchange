"""Synthetic CPU algebra against unmodified local ReducedModel/NormalInverse."""
import hashlib,json,pathlib,sys
import numpy as np
BASE=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(BASE/'Gaussian/A16/CURRENT_SELECTION_R1/code'))
from operators import SelectorContext,ReducedModel
from pairtsom_adapter import PhaseLedger
from reduced_normal import ReducedNormalInverse
from endpoint_batch import EndpointBatch,TOL

SOURCE=[BASE/'Gaussian/A17/A17_CODEX_PROMPT.md',BASE/'Gaussian/A17/A17_COMPUTATIONAL_OPTIMIZATION.md',BASE/'Gaussian/A16/CURRENT_SELECTION_R1/code/operators.py',BASE/'Gaussian/A16/CURRENT_SELECTION_R1/code/fast_contractions.py',BASE/'Gaussian/A10/code/reduced_normal.py',BASE/'Gaussian/A16/CURRENT_SELECTION_R1/research/efficiency_v16/a16_gpu_candidates_v16/torch_candidates.py']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE}
rng=np.random.default_rng(1702)
def complex_array(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
def context(L,S,PB,r,ell,lam):
    c=SelectorContext();c.n_current=L.shape[0];c.P=PB.shape[0];c.q=PB.shape[2];c.m=S.shape[0];c.dim_material=2*c.q;c.lam=lam;c.ell=ell;c.r=r;c.ledger=PhaseLedger()
    c._L=lambda U:L@U;c._S=lambda U:S@U;c._B=lambda U:np.stack([U.conj().T@p for p in PB])
    return c
cases=[]
for q,k,P,m,branch in [(7,3,3,5,'direct'),(90,3,3,5,'woodbury'),(300,3,3,5,'auto'),(7,6,2,3,'direct'),(7,0,3,5,'auto')]:
    qc=9;L=4*np.eye(qc)+.12*complex_array((qc,qc));S=complex_array((m,qc));PB=complex_array((P,qc,q));r=rng.normal(size=P*2*m);ell=rng.normal(size=2*q);lam=.13
    U=np.stack([np.linalg.qr(complex_array((qc,k)))[0] for _ in range(17)])
    ctx=context(L,S,PB,r,ell,lam)
    reference=[ReducedModel(ctx,u) for u in U];expected=np.stack([v.step() for v in reference]);expectedj=np.stack([v.j(s) for v,s in zip(reference,expected)])
    for device in ['cpu','torch_cpu']:
        engine=EndpointBatch(L,S,PB,r,ell,lam,device=device,branch=branch);out=engine.evaluate(U)
        err=np.linalg.norm(out['steps']-expected)/max(np.linalg.norm(expected),1e-300);jerr=np.linalg.norm(out['j_steps']-expectedj)/max(np.linalg.norm(expectedj),1e-300)
        assert out['status']==['OK']*17,(device,q,out['status'])
        assert max(err,jerr)<TOL,(err,jerr)
        assert out['statistics']['public_projection_batches']==2
        assert out['statistics']['normal_matrix_batches']==(0 if not k else 2)
        # Exact H solve vs existing normal direct/woodbury explicitly.
        if k:
            from types import SimpleNamespace
            v=reference[0];n=ReducedNormalInverse(v,SimpleNamespace(d=2*q,Q=None,volume=1.),1.,lam,branch='woodbury')
            assert np.linalg.norm(n.solve(v.jt(r)+ell)+expected[0])/np.linalg.norm(expected[0])<TOL
        cases.append({'q':q,'k':k,'P':P,'m':m,'backend':device,'branch':out['statistics']['branch'],'step_relative':err,'jstep_relative':jerr,'max_normal_residual':float(np.max(out['normal_relative_residual'])),'statistics':out['statistics']})
    # Same current space: phase and column permutation cannot change endpoint.
    if k:
        transformed=U[:,:,::-1]*np.exp(1j*rng.uniform(-3,3,size=(17,1,k)))
        result=EndpointBatch(L,S,PB,r,ell,lam).evaluate(transformed)
        assert np.linalg.norm(result['steps']-expected)/np.linalg.norm(expected)<TOL

# Full coupling and isolating singular, nonfinite and nonorthonormal units.
qc,q,k,P,m=4,3,2,2,3;L=np.diag([1.,1e-12,2.,3.]).astype(complex);L[0,2]=.31j;S=complex_array((m,qc));PB=complex_array((P,qc,q));r=rng.normal(size=P*2*m);ell=rng.normal(size=2*q)
U=np.stack([np.eye(qc,dtype=complex)[:,[0,2]],np.eye(qc,dtype=complex)[:,[0,1]],np.eye(qc,dtype=complex)[:,[0,2]],np.eye(qc,dtype=complex)[:,[0,2]]]);U[2,0,0]=np.nan;U[3]*=2
failure_checks=[]
for dev in ['cpu','torch_cpu']:
    out=EndpointBatch(L,S,PB,r,ell,.2,device=dev).evaluate(U)
    assert out['status']==['OK','CURRENT_NEAR_SINGULAR','NONFINITE_BASIS','NONORTHONORMAL_BASIS']
    assert np.isnan(out['steps'][1:]).all()
    assert out['stability'][0]['relative']>TOL and out['stability'][1]['relative']<=TOL
    ref=ReducedModel(context(L,S,PB,r,ell,.2),U[0]);assert np.linalg.norm(out['steps'][0]-ref.step())<TOL
    failure_checks.append({'device':dev,'status':out['status'],'stability':out['stability']})
assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE}
result={'status':'PASS_CPU_SYNTHETIC_IMPLEMENTATION_ONLY','cases':cases,'isolated_failure_checks':failure_checks,'tolerance':TOL,'source_hashes':hashes,'source_inputs_unchanged':True,'CUDA_run':False,'Maxwell_state_or_scientific_pilot_run':False,'limitations':['synthetic arrays only','torch CPU exercises shared CUDA-path algebra but does not validate CUDA device/FP64/runtime','not scientific endpoint admission, gain certification or speedup'],'numpy_version':np.__version__}
result['torch_version']=__import__('torch').__version__
out=pathlib.Path(__file__).parent/'CPU_TEST_RESULTS.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':len(cases),'max_step_relative':max(c['step_relative'] for c in cases),'max_jstep_relative':max(c['jstep_relative'] for c in cases),'max_normal_residual':max(c['max_normal_residual'] for c in cases),'CUDA_run':False},indent=2))
