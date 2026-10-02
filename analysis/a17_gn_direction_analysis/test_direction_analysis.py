"""Independent objective / known band / degeneracy tests; no physical solver."""
import json,time
from pathlib import Path
import numpy as np
from scipy import linalg as la
from direction_analysis import fixed_reference,compare,receiver_weak_overlap
HERE=Path(__file__).resolve().parent;rng=np.random.default_rng(20261002);errors={};cases=[];start=time.perf_counter();P=1e-5;lam=.01001
for seed in range(16):
 e=np.array([0.,0.,P*.2,P,5*P,10*P,1.,1.]);W=la.qr(rng.normal(size=(8,8)))[0];J=np.diag(np.sqrt(e))@W.T
 sref=rng.normal(size=8);before=rng.normal(size=8);after=rng.normal(size=8);r=rng.normal(size=8);g=rng.normal(size=8)*1e-8;ell=g-J.T@(r+J@sref)-lam*sref
 ref=fixed_reference(J,lam);out=compare(ref,r,ell,sref,before,after)
 # Direct objective difference, not spectral/halfH implementation.
 def phi(x):return .5*la.norm(r+J@x)**2+ell@x+.5*lam*(x@x)
 actual=phi(before)-phi(after);err=abs(out['sum_band_gain']-actual)/max(1.,abs(actual));errors['objective_delta']=max(errors.get('objective_delta',0),err)
 # Exact analytic directional decomposition in prescribed W, using diagonal known curvature.
 analytic=.5*np.sum((e+lam)*((W.T@(before-sref))**2-(W.T@(after-sref))**2))+g@(before-after)
 errors['known_curvature']=max(errors.get('known_curvature',0),abs(analytic-out['sum_band_gain'])/max(1.,abs(analytic)))
 require_cross=abs(out['reference_gradient_cross_gain']-g@(before-after));errors['reference_gradient_cross']=max(errors.get('reference_gradient_cross',0),require_cross)
 # Rotate repeated eigenblocks. Risk band/block totals must survive arbitrary gauges.
 ref2=dict(ref);V=ref['V'].copy()
 for b in ref['blocks']:
  if b['dimension']>1:
   ix=b['indices'];R=la.qr(rng.normal(size=(len(ix),len(ix))))[0];V[:,ix]=V[:,ix]@R
 ref2['V']=V;rot=compare(ref2,r,ell,sref,before,after)
 errors['degenerate_band_rotation']=max(errors.get('degenerate_band_rotation',0),max(abs(a['actual_RGN_gain']-b['actual_RGN_gain']) for a,b in zip(out['bands'],rot['bands'])))
 cases.append(dict(seed=20261002+seed,reference_residual_norm=out['reference_gradient_norm'],cross_gain=out['reference_gradient_cross_gain'],bands=out['bands']))
# Known false-weak coverage: ROM null direction belongs to full strong information.
J=np.diag(np.sqrt([0.,P*.5,P*5,1.]));Jrom=np.diag(np.sqrt([0.,P*.5,P*5,0.]));ref=fixed_reference(J,lam);overlap=receiver_weak_overlap(ref,Jrom);strong=next(x for x in overlap['rows'] if x['full_reference_band']=='strong');errors['known_false_weak_overlap']=abs(strong['projector_overlap_trace']-1.)
# Numerically adjacent directions crossing P boundary are grouped, not arbitrarily split.
ref=fixed_reference(np.diag(np.sqrt([P*(1-2e-11),P*(1+2e-11),1.])),lam);mixed=any(b['band'].startswith('BOUNDARY_MIXED') and b['dimension']==2 for b in ref['blocks']);errors['boundary_block_grouping']=0. if mixed else 1.
out=dict(status='PASS' if max(errors.values())<=1e-9 else 'FAIL',threshold=1e-9,max_errors=errors,cases=cases,known_overlap=overlap,boundary_blocks=ref['blocks'],cpu_wall_s=time.perf_counter()-start,new_full_actions=0);(HERE/'TEST_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['status','max_errors','cpu_wall_s']}))
