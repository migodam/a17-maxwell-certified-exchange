"""Object2007 only: offline paid fullJ development consistency, never deployment callback.
No physical state, Maxwell action, GPU action, reference optimum/floor/truth is loaded.
"""
from pathlib import Path
import sys,hashlib,json,csv,time,platform,importlib.util,os
import numpy as np
HERE=Path(__file__).resolve().parents[1];R1=HERE.parent/'EXCHANGE_R1';OUT=HERE/'analysis'
os.environ['A17_MATCHED_R1_ROOT']=str(R1)
sys.path.insert(0,str(R1/'tools/matched'));sys.path.insert(0,str(R1/'code'))
import pinned_runtime as runtime
import portable_cached_onepass_v2 as cached
from run_exchange import Neighborhood,balanced_ids,evaluate_neighborhood,basis
spec=importlib.util.spec_from_file_location('closure_online_tolerance',HERE/'code/online_tolerance.py');ot=importlib.util.module_from_spec(spec);spec.loader.exec_module(ot)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[];rows=[];started=time.perf_counter();repetitions=[]
def objective(r,J,x,ell,lam):
 u=r+J@x;return float(.5*(u@u)+ell@x+.5*lam*(x@x))
def complex_outputs(a,P):
 n=a.shape[0]//P;m=n//2
 if a.ndim==1:
  v=a.reshape(P,n);return (v[:,:m]+1j*v[:,m:]).reshape(-1)
 v=a.reshape(P,n,-1);return (v[:,:m]+1j*v[:,m:]).reshape(P*m,-1)
for rep in range(2):
 t=time.perf_counter()
 for phase,folder in [('early','object_2007_early_portable'),('middle','object_2007_middle'),('late','object_2007_late')]:
  d=R1/'results'/folder;paths=[d/'public_workspace.npz',d/'offline_information_reference.npz']
  if rep==0:
   inputs.extend(dict(path=str(p),sha256=sha(p),scope='OBJECT2007_DEVELOPMENT_ONLY_OFFLINE_PAID_FULLJ' if 'reference' in p.name else 'OBJECT2007_SAVED_PUBLIC_WORKSPACE') for p in paths)
  with np.load(paths[0],allow_pickle=False) as a:bank={k:a[k].copy() for k in a.files}
  with np.load(paths[1],allow_pickle=False) as a:J=a['fullJ'].copy()
  control=cached.CachedControls(bank,device='cpu',direction=None,source=str(paths[0]));ctx=control.ctx;lam=float(ctx.lam)
  for k in (4,8,16):
   ids,U=control.receiver(k);ev=control.engine.evaluate(U[None])
   if ev['status']!=['OK']:raise RuntimeError('receiver inadmissible')
   x=ev['steps'][0];Jx=J@x;incoming=balanced_ids(control.pool,ids,12)
   neigh=Neighborhood(control.w,control.engine,OUT,{'object_id':2007,'phase':phase,'k':k})
   nb=evaluate_neighborhood(neigh,ids,incoming,'swap',k,control.anchor,x=x,Jx=Jx)
   top=np.argsort(-nb['qhat'],kind='stable')[:2];D=nb['D'][:,top];Z=J@D;u=ctx.r+Jx
   a=ot.directional_check(ctx.r,u,Z,x,D,ctx.ell,lam)
   uc=complex_outputs(u,ctx.P);zc=complex_outputs(Z,ctx.P);rc=complex_outputs(ctx.r,ctx.P)
   b=ot.directional_check(rc.real,uc,zc,x,D,ctx.ell,lam)
   phi0=objective(ctx.r,J,x,ctx.ell,lam)
   for j,pos in enumerate(top):
    qi=phi0-objective(ctx.r,J,nb['steps'][:,pos],ctx.ell,lam);err=abs(float(a['gain'][j])-qi);cerr=abs(float(b['gain'][j])-qi);ri=int(nb['valid'][pos]);move=nb['move_list'][ri]
    rows.append(dict(repetition=rep,object_id=2007,phase=phase,k=k,finalist_order=j,move_index=ri,drop=json.dumps(move['drop']),add=json.dumps(move['add']),Qhat=float(nb['qhat'][pos]),Q_directional=float(a['gain'][j]),Q_complex_output=float(b['gain'][j]),Q_objective_difference=qi,tau=float(a['tau'][j]),tau_complex_output=float(b['tau'][j]),observed_error=err,observed_error_over_tau=err/float(a['tau'][j]),complex_observed_error=cerr,complex_observed_error_over_tau=cerr/float(b['tau'][j]),dual_form_error=abs(float(a['gain'][j])-float(b['gain'][j])),passes=err<=float(a['tau'][j]) and cerr<=float(b['tau'][j]),normal_residual_receiver=float(ev['normal_relative_residual'][0]),normal_residual_child=nb['rows'][ri]['child_normal_relative_residual'],scope='OFFLINE_DEVELOPMENT_FULLJ_CONSISTENCY_NOT_DEPLOYMENT_CALLBACK'))
   if rep==0:np.savez_compressed(OUT/f'online_tolerance_dev_{phase}_k{k}.npz',r=ctx.r,u=u,z=Z,x=x,d=D,ell=ctx.ell,lam=lam,top_move_indices=nb['valid'][top])
 repetitions.append(dict(repetition=rep,wall_s=time.perf_counter()-t))
with (OUT/'online_tolerance_dev_rows.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
by=lambda rep:[r for r in rows if r['repetition']==rep]
a,b=by(0),by(1);repeat_same=all(all(x[k]==y[k] for k in x if k!='repetition') for x,y in zip(a,b))
loaded=[]
for name,module in sorted(sys.modules.items()):
 p=getattr(module,'__file__',None)
 if p and (str(R1) in p or str(HERE) in p) and Path(p).is_file():loaded.append(dict(module=name,path=str(Path(p).resolve()),sha256=sha(p)))
receipt=dict(status='PASS_DEVELOPMENT_ONLY' if all(r['passes'] for r in rows) and repeat_same else 'FAIL_NO_THRESHOLD_RELAXATION',scope='OFFLINE_OBJECT2007_SAVED_FULLJ_CONSISTENCY; NOT deployment callback, physical replay, holdout test or Maxwell error certificate',rows=len(rows),repetitions=repetitions,repeat_identical=repeat_same,max_observed_error=max(r['observed_error'] for r in rows),max_observed_error_over_tau=max(r['observed_error_over_tau'] for r in rows),max_complex_observed_error=max(r['complex_observed_error'] for r in rows),max_complex_observed_error_over_tau=max(r['complex_observed_error_over_tau'] for r in rows),max_dual_form_error=max(r['dual_form_error'] for r in rows),inputs=inputs,online_tolerance_sha256=sha(HERE/'code/online_tolerance.py'),script_sha256=sha(__file__),loaded_sources=loaded,python=platform.python_version(),numpy=np.__version__,device='cpu',precision='float64/complex128',new_Maxwell_actions=0,new_GPU_actions=0,holdout_labels_read=0,full_reference_optimum_loaded=False,floor_loaded=False,truth_loaded=False,physical_callback_used=False,wall_s=time.perf_counter()-started,threads={k:os.getenv(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
(OUT/'online_tolerance_dev_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('loaded_sources','inputs')},indent=2))
if not all(r['passes'] for r in rows) or not repeat_same:sys.exit(1)
