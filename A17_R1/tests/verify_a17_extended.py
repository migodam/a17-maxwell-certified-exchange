"""New bounded tests, not recovered original attachments or Maxwell evidence."""
import json, traceback, itertools, sys
from pathlib import Path
import numpy as np
from scipy import linalg as la
import exchange_core as ec
from verify_a17 import trial, problem, rel, cnormal, OUT, THRESHOLD

def must_reject(fn):
    try: fn()
    except (ValueError,la.LinAlgError): return True
    raise AssertionError('expected rejection')

def extended(seed,bank_rank):
    row=trial(seed,p=53,bank_rank=bank_rank,P=2,m=3)
    assert row['support_rank']<53, 'paired-support containment must be nontrivial'
    rng,L,S,B,V,A,C,F,r,ell,Lambda=problem(seed,p=53,bank_rank=bank_rank,P=2,m=3)
    sw=ec.swap_from_core(ec.removal_core(L,S,B,V,A),C)
    Jo,Jn,D=sw['old']['J'],sw['new']['J'],sw['D']
    z=ec.exact_step_change(Jo,Jn,r,Lambda,ell)
    so,sn=z['old']['s'],z['new']['s']; Ho,Hn=z['old']['H'],z['new']['H']
    # Clipped and inexact endpoint steps: identity includes normal-equation residuals.
    clipped=np.clip(sn,-.01,.01); do=clipped-so
    T=ec.paired_support(Jo,D,Ho,exact_unconstrained=True)['T']
    clip_outside=la.norm(do-T@(T.T@do))
    sno=sn+rng.normal(size=53)*.01; soo=so+rng.normal(size=53)*.01
    rho_n=Hn@sno+Jn.T@r+ell; rho_o=Ho@soo+Jo.T@r+ell
    rhs=-Jo.T@D@sno-D.T@(r+Jn@sno)+rho_n-rho_o
    row['errors']['inexact_residual_correction']=rel(Ho@(sno-soo),rhs)
    row['clipped_step_outside_exact_support']=float(clip_outside)
    row['closure_responsibility']='API rejects absent exact_unconstrained attestation; clipping/inexact users must not attest'
    must_reject(lambda: ec.paired_support(Jo,D,Ho))
    assert clip_outside>1e-5
    # Finite covariance expectation of swap gain with shared ell/Lambda is independent direct differences.
    r0=rng.normal(size=len(r)); E=rng.normal(size=(len(r),4))*.1
    values=[]
    # Every endpoint step is affine in residual; full objective difference quadratic in residual.
    def gain(rr):
        xo=ec.gn_step(Jo,rr,Lambda,ell)['s']; xn=ec.gn_step(Jn,rr,Lambda,ell)['s']
        return ec.objective(F,rr,Lambda,ell,xo)-ec.objective(F,rr,Lambda,ell,xn)
    for j in range(4): values.extend([gain(r0+2*E[:,j]),gain(r0-2*E[:,j])])
    # Polarization reconstructs Hessian contraction without sharing the expected code.
    closed= gain(r0)+sum((gain(r0+E[:,j])+gain(r0-E[:,j])-2*gain(r0))*.5 for j in range(4))
    row['errors']['mean_swap_gain']=rel(np.array(np.mean(values)),np.array(closed))
    assert max(row['errors'].values())<=THRESHOLD
    return row

def counterexamples():
    out={}
    # Stable endpoints with singular shared core: no jitter, direct fallback retains exact endpoint.
    L=np.array([[0,1,1],[1,0,0],[1,0,1]],complex); S=np.array([[1,2,3]],complex)
    B=np.array([[[1,2],[2,1],[3,-1]]],complex); atoms=np.eye(3,dtype=complex)
    core=ec.removal_core(L,S,B,atoms[:,:1],atoms[:,1:2]); sw=ec.swap_from_core(core,atoms[:,2:3])
    assert not sw['fast_available'] and sw['old']['stability']['stable'] and sw['new']['stability']['stable']
    assert np.all(np.isfinite(sw['D']))
    must_reject(lambda: ec.bank_increment(core,atoms[:,2:3]))
    out['singular_core_stable_endpoints']=dict(reason=sw['reason'],old=sw['old']['stability'],new=sw['new']['stability'],direct_D=sw['D'].tolist())
    # Rank-deficient atom set remains explicit invalid fixed-budget move; orth never reassigns identities.
    atoms=np.column_stack([np.eye(3),np.array([1,0,0])]).astype(complex)
    rankcase=ec.atom_swap(np.eye(3),S,B,atoms,[0,1],[1],[3])
    assert not rankcase['feasible_fixed_budget'] and rankcase['new']['actual_rank']==1
    out['rank_deficient_endpoint']=dict(old_ids=rankcase['old_ids'],new_ids=rankcase['new_ids'],new_actual_rank=rankcase['new']['actual_rank'],reason=rankcase['rank_reason'])
    # Identical local Gamma/Y/N but opposite full gain; spec's exact scalar witness.
    gains=[]
    for t in [.5,-1.5]:
        J=np.array([[.5+t]]); x=np.array([0.]); d=np.array([-.4]); r=np.array([1.]); ell=np.array([0.])
        gains.append(ec.gain_directional(r,J@d,x,d,np.array([[1.]]),ell))
    assert gains[0]>0 and gains[1]<0
    out['identical_local_opposite_full_gain']=dict(Gamma=1,Y=.5,N=1,d=-.4,full_J=[1,-1],gain=gains)
    # Lower bounds cannot prove stop, and shortlist lacks full-neighborhood coverage.
    unresolved=ec.decide([ec.interval(0,1)],neighborhood_total=1)
    omitted=ec.decide([ec.interval(-2,.1)],neighborhood_total=2)
    stopped=ec.decide([ec.interval(-2,.1)],neighborhood_total=2,omitted_upper_bound=-.1)
    assert unresolved['status']=='ABSTAIN_UNRESOLVED' and omitted['status']=='ABSTAIN_UNRESOLVED'
    assert stopped['status']=='CERTIFIED_LOCAL_STOP'
    # Cost-saving move cannot be risk-safe accepted despite cost-aware LB passing.
    costsafe=ec.decide([ec.interval(-.5,0)],neighborhood_total=1,gamma=1,state_costs=[-2],transition_costs=[0])
    assert costsafe['status']!='ACCEPT'
    out['stopping']=dict(negative_LB_positive_UB=unresolved,omitted=omitted,bounded_omitted=stopped,cost_risk_safe=costsafe)
    must_reject(lambda: ec.OutputErrorBounds(.1,.1,'iterative residual only'))
    must_reject(lambda: ec.OutputErrorBounds(-1,1,'invalid',True))
    must_reject(lambda: ec.gn_step(np.eye(2),np.ones(2),np.array([[1,2],[0,1]]),np.zeros(2)))
    must_reject(lambda: ec.gn_step(np.eye(2),np.ones(2),-np.eye(2),np.zeros(2)))
    must_reject(lambda: ec.gn_step(np.eye(2),np.ones(2),np.eye(2),np.ones(2,dtype=complex)))
    out['rejection_tests']='PASS: uncertified residual-only output bounds; negative errors; asymmetric/non-SPD Lambda; complex material; unasserted closure'
    # Find an explicit finite-dictionary 1-exchange trap using real Galerkin L=I, B=ones.
    # Scalar full J=sum(a), subset J=sum(a[selected]); no invented set-function objective.
    rng=np.random.default_rng(20261003); found=None
    for attempt in range(10000):
        a=rng.normal(size=6)*2; full=a.sum()
        def phi(ids):
            j=a[list(ids)].sum(); s=-j/(j*j+1)
            return .5*(1+full*s)**2+.5*s*s
        vals={ids:phi(ids) for ids in itertools.combinations(range(6),2)}
        for ids,value in vals.items():
            neighbors=[v for child,v in vals.items() if len(set(ids)&set(child))==1]
            best=min(vals,key=vals.get)
            if value<=min(neighbors)+1e-13 and value-vals[best]>1e-3:
                found=dict(attempt=attempt,atoms=a.tolist(),selected=ids,risk=value,best=best,best_risk=vals[best],two_exchange_gain=value-vals[best],max_one_exchange_gain=max(value-v for v in neighbors));break
        if found:break
    assert found is not None
    out['one_exchange_trap']=found
    # Exact nonmonotone augmentation example: full J=1 but singleton contribution is 2.
    # old empty step 0 -> singleton -2/5 has improved loss; adding cancelling atom -1 -> full -1/2 improves further.
    # Reverse pick: old singleton J=1 has optimal full step, adding +2 drives model J=3 away.
    # Full sum1 achieved by contributions [1,2,-2].
    full=np.array([[1.]]); r=np.array([1.]); ell=np.array([0.]); lam=np.array([[1.]])
    old=ec.gn_step(np.array([[1.]]),r,lam,ell)['s']; new=ec.gn_step(np.array([[3.]]),r,lam,ell)['s']
    G=ec.objective(full,r,lam,ell,old)-ec.objective(full,r,lam,ell,new)
    assert G<0
    out['nonmonotone_augmentation']=dict(current_atom_contributions=[1,2,-2],old_selected=[0],new_selected=[0,1],full_J=1,old_step=old.tolist(),new_step=new.tolist(),gain=G)
    return out

def private_interface_checks():
    root=OUT.parents[2]; code=root/'Gaussian/A16/CURRENT_SELECTION_R1/code'
    sys.path.insert(0,str(code))
    import operators, fast_contractions as fc
    rng=np.random.default_rng(20261003); P,n,k,q,m=3,9,4,5,6
    X=cnormal(rng,(n,k)); V=ec.orth(cnormal(rng,(n,2)))
    a=ec.orth(X,V); b=operators.orth(X,V)
    orth_error=rel(a@a.conj().T,b@b.conj().T)
    tiny=ec.orth(X*1e-25); oldtiny=operators.orth(X*1e-25)
    tiny_error=rel(tiny@tiny.conj().T,oldtiny@oldtiny.conj().T)
    F=cnormal(rng,(P,k,q)); CV=cnormal(rng,(m,k)); Z=rng.normal(size=2*q)
    # A16 complex material basis -> our shared real map through [B, iB].
    B_real=np.concatenate([F,1j*F],axis=2)
    expected=ec.realify(CV@B_real)
    e_j=rel(expected@Z,fc.reduced_j(F,CV,Z)); W=rng.normal(size=2*P*m)
    e_jt=rel(expected.T@W,fc.reduced_jt(F,CV,W))
    sigma=operators._stability(np.diag([1.,1e-9]))
    must_reject(lambda: operators._stability(np.diag([1.,1e-11])))
    assert not ec.stability(np.diag([1.,1e-11]))['stable']
    assert max(orth_error,tiny_error,e_j,e_jt)<=THRESHOLD
    return dict(status='PASS',private_operators_import=True,A16_material_embedding='B_real=[B_complex, i*B_complex]',
                errors=dict(orth_projector=orth_error,tiny_scale_projector=tiny_error,j_pack=e_j,jt_real_pullback=e_jt),
                rtol_match=operators.TOL==ec.RTOL,stable_A16_example=sigma,
                limitation='dense local interfaces only; no real Maxwell frozen state built')

def run():
    rows=[];failures=[];witness=None;interface=None
    for j in range(16):
        seed=20261003+j
        try: rows.append(extended(seed,1+j%2))
        except Exception as exc:failures.append(dict(seed=seed,error=str(exc),traceback=traceback.format_exc()))
    for name,fn in [('counterexamples',counterexamples),('private_interfaces',private_interface_checks)]:
        try:
            value=fn()
            if name=='counterexamples':witness=value
            else:interface=value
        except Exception as exc:failures.append(dict(group=name,error=str(exc),traceback=traceback.format_exc()))
    result=dict(status='PASS' if not failures else 'FAIL',requested=16,passed=len(rows),seed_start=20261003,
                threshold=THRESHOLD,new_independent_verification=True,rows=rows,counterexamples=witness,
                private_interface_checks=interface,failures=failures,
                max_errors={k:max(row['errors'][k] for row in rows) for k in rows[0]['errors']} if rows else {})
    (OUT/'verify_a17_extended_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['status','requested','passed','max_errors','failures']}))
    return result
if __name__=='__main__':
    out=run();sys.exit(0 if out['status']=='PASS' else 1)
