"""Bounded independent CPU tests of new side-diagnostic layer."""
import json,time,traceback,hashlib,sys
from pathlib import Path
import numpy as np
from scipy import linalg as la
import information_diagnostics as d
OUT=Path(__file__).resolve().parent; TOL=1e-9

def cn(rng,shape): return rng.normal(size=shape)+1j*rng.normal(size=shape)
def error(a,b): return float(la.norm(np.asarray(a)-np.asarray(b))/max(1.,la.norm(np.asarray(a)),la.norm(np.asarray(b))))
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('required rejection missing')
def trial(seed):
    rng=np.random.default_rng(seed); q=17; rows=8; lam=.7
    Z=cn(rng,(rows,q))*.3; C=Z+cn(rng,(rows,q))*.1; step=rng.normal(size=2*q)
    J=d.realify_complex_linear(Z); Jc=d.realify_complex_linear(C)
    actual=la.svdvals(J); summary=d.information_summary(Z,lam,precision=lam)
    e=la.eigvalsh(J.T@J); keep=e>lam; strong=la.eigh(J.T@J)[1][:,keep]
    Pweak=np.eye(2*q)-strong@strong.T
    weak=d.fixed_weak_space(Z,lam); out=d.candidate_information(Z,C,lam,precision=lam,weak=weak,gnstep=step,spectrum=True)
    Delta=Jc.T@Jc-J.T@J; pw=Pweak@step
    expected=list(np.sqrt(np.maximum(e,0))[::-1])[:min(J.shape)]
    errors=dict(spectrum=error(actual,summary['real_singular_values']),trace=error(np.trace(J.T@J),summary['info_trace']),
                d_eff=error(np.sum(np.maximum(e,0)/(np.maximum(e,0)+lam)),summary['info_effective_dim']),
                trace_change=error(np.trace(Delta),out['delta_info_trace']),
                weak_trace=error(np.trace(Pweak@Delta),out['delta_info_weak']),
                strong_trace=error(np.trace((np.eye(2*q)-Pweak)@Delta),out['delta_info_strong']),
                weak_task=error(pw@Delta@pw,out['weak_task_score']))
    assert summary['info_effective_rank_lam']==np.count_nonzero(e>lam)
    assert summary['info_effective_rank_10lam']==np.count_nonzero(e>10*lam)
    wc=d.project_weak(step,weak); wreal=np.concatenate([wc.real,wc.imag]);errors['weak_projection']=error(pw,wreal)
    gram=d.info_from_gram(Z@Z.conj().T,float(np.vdot(Z,Z).real),lam,qcomplex=q,precision=lam)
    errors['gram_eff']=error(gram['info_effective_dim'],summary['info_effective_dim'])
    # material permutation must transform the packed material step consistently.
    perm=rng.permutation(q); zp=Z[:,perm]; cp=C[:,perm]; sp=np.concatenate([step[:q][perm],step[q:][perm]])
    wo=d.fixed_weak_space(zp,lam); po=d.candidate_information(zp,cp,lam,precision=lam,weak=wo,gnstep=sp,spectrum=True)
    for key in ['delta_info_trace','delta_info_weak','weak_task_score','delta_effective_dim']:
        errors['material_permutation_'+key]=error(out[key],po[key])
    # complex row phases/data-row permutation don't change information or tasks.
    phases=np.exp(1j*rng.uniform(-np.pi,np.pi,rows))[:,None]; rp=rng.permutation(rows)
    zz=(Z*phases)[rp]; cc=(C*phases)[rp]
    wo=d.fixed_weak_space(zz,lam); po=d.candidate_information(zz,cc,lam,precision=lam,weak=wo,gnstep=step,spectrum=True)
    for key in ['delta_info_trace','delta_info_weak','weak_task_score','delta_effective_dim']:
        errors['data_phase_permutation_'+key]=error(out[key],po[key])
    # Arbitrary real Gaussian-manifold tangent needs actual J, not spectrum duplication.
    Tangent=cn(rng,(q,7)); Jg=d.realify_tangent(Z,Tangent); general=d.information_from_tangent(Z,Tangent,lam,precision=lam)
    eg=la.svdvals(Jg)**2
    errors['general_Gaussian_trace']=error(np.trace(Jg.T@Jg),general['info_trace'])
    errors['general_Gaussian_spectrum']=error(la.svdvals(Jg),general['real_singular_values'])
    errors['general_Gaussian_eff']=error(np.sum(eg/(eg+lam)),general['info_effective_dim'])
    # Explicit realizable current-basis rephase/permutation, not a change of material physics.
    n,k,P,m=10,4,2,3; L=4*np.eye(n)+cn(rng,(n,n))*.1; S=cn(rng,(m,n)); B=cn(rng,(P,n,q)); U=la.qr(cn(rng,(n,k)),mode='economic')[0]
    def current_Z(u):
        A=u.conj().T@L@u; return np.vstack([S@u@la.solve(A,u.conj().T@b) for b in B])
    z1=current_Z(U); up=(U*np.exp(1j*rng.uniform(-np.pi,np.pi,k)))[...,rng.permutation(k)]; z2=current_Z(up)
    errors['current_phase_permutation_Z']=error(z1,z2)
    errors['current_phase_permutation_trace']=error(d.information_summary(z1,lam)['info_trace'],d.information_summary(z2,lam)['info_trace'])
    reject(lambda:d.candidate_information(Z,C,lam,weak=d.fixed_weak_space(C,lam),gnstep=step))
    assert max(errors.values())<=TOL
    return dict(seed=seed,errors=errors,strong_dimension_real=2*weak.strong_dimension_complex,weak_dimension_real=weak.weak_dimension_real,material_dimension_real=2*q)

def witnesses():
    Z=np.array([[3,0,0]],complex); C=np.array([[3,0,0],[0,1,1]],complex); step=np.array([0,1,1,0,0,0.])
    weak=d.fixed_weak_space(Z,1.); out=d.candidate_information(Z,C,1.,precision=1.,weak=weak,gnstep=step)
    assert weak.weak_dimension_real==4 and out['weak_task_score']==4 and out['delta_info_weak']==4
    diagonal_original=2.; diagonal_rotated=4.; invariant=out['weak_task_score']
    # Entire base nullspace has projector-invariant result; diagonal weighting depends on basis.
    assert diagonal_original!=diagonal_rotated
    zero=d.information_summary(np.zeros((2,5),complex),.5); assert zero['info_effective_dim']==0 and zero['info_effective_rank_lam']==0
    wz=d.fixed_weak_space(np.zeros((2,5),complex),.5); assert wz.weak_dimension_real==10
    boundary=d.info_from_gram(np.diag([1.,10.,11.]),22.,1.,precision=1.)
    assert boundary['info_effective_rank_lam']==4 and boundary['info_effective_rank_10lam']==2
    negative=d.candidate_information(np.array([[2,0]],complex),np.array([[1,0]],complex),1.)
    assert negative['delta_info_trace']<0
    reject(lambda:d.info_from_gram(np.array([[1,2],[0,1]],complex),2,1))
    reject(lambda:d.info_from_gram(np.diag([1,-1]),0,1))
    reject(lambda:d.info_from_gram(np.eye(2),10,1))
    reject(lambda:d.information_summary(Z,0))
    reject(lambda:d.project_weak(step.astype(complex),weak))
    ids=tuple('incoming_'+str(i) for i in range(20)); diag=d.fixed_diagnostic_ids(ids); cov=d.coverage(ids,diag,['base','accepted_final'])
    assert len(diag)==12 and cov['incoming_spectrum_weak_evaluated']==12 and len(cov['incoming_omitted_ids'])==8
    traceonly=d.candidate_information(Z,C,1.); assert traceonly['delta_effective_dim'] is None and traceonly['delta_info_weak'] is None
    return dict(nullspace_basis_dependence=dict(original_diagonal_sum=diagonal_original,rotated_diagonal_sum=diagonal_rotated,invariant_quadratic=invariant,full_nullspace_real_dimension=4),
                threshold_boundary=boundary,nonmonotone_trace=negative['delta_info_trace'],coverage_example=cov,
                trace_only_nulls=traceonly,zero_information=zero,invalid_input_rejections='PASS')

def integration_trial(seed):
    rng=np.random.default_rng(seed); q=7; rows=4; precision=1e-5
    Z=cn(rng,(rows,q))*.1; C=Z+cn(rng,(rows,q))*.03; J=d.realify_complex_linear(Z); Jc=d.realify_complex_linear(C)
    a=d.information_summary(Z,precision+.01,precision=precision)
    b=d.information_summary(Z,precision+.00027,precision=precision)
    for key in ['info_trace','info_effective_dim','info_logdet_volume','info_effective_rank_prior1','info_effective_rank_prior10']:
        assert a[key]==b[key], 'prior information changed with damping'
    assert a['solve_effective_dim']<b['solve_effective_dim']
    weak=d.fixed_weak_space(Z,precision=precision); r=cn(rng,(rows,)); rr=np.concatenate([r.real,r.imag])
    db=d.delta_b(Z,C,r); rhs=Jc.T@rr-J.T@rr
    errors=dict(delta_b=error(db,rhs))
    ev=la.eigvalsh(J.T@J); ev=np.maximum(ev,0)
    errors['prior_eff']=error(a['info_effective_dim'],np.sum(ev/(ev+precision)))
    sign,ld=np.linalg.slogdet(np.eye(2*q)+(J.T@J)/precision)
    assert sign==1
    errors['prior_logdet']=error(a['info_logdet_volume'],.5*ld)
    H=J.T@J+(precision+.01)*np.eye(2*q); Hc=Jc.T@Jc+(precision+.01)*np.eye(2*q)
    so=-la.solve(H,J.T@rr); sn=-la.solve(Hc,Jc.T@rr)
    form=-la.solve(Hc,db+(Jc.T@Jc-J.T@J)@so)
    errors['step_rhs_information']=error(sn-so,form)
    F=J+ rng.normal(size=J.shape)*.02
    fid=d.information_fidelity_real(J,F,precision)
    HF=F.T@F+precision*np.eye(2*q); vals,V=la.eigh(HF); invsqrt=(V/np.sqrt(vals))@V.T
    E=invsqrt@(J.T@J-F.T@F)@invsqrt
    errors['fidelity_spectral']=error(fid['info_fidelity_spectral'],la.norm(E,2))
    errors['fidelity_frobenius']=error(fid['info_fidelity_frobenius'],la.norm(E,'fro'))
    P,meta=d.common_material_probes(60,'object_'+str(seed),precision=precision)
    P2,meta2=d.common_material_probes(60,'object_'+str(seed),precision=precision)
    assert np.array_equal(P,P2) and meta['probe_hash']==meta2['probe_hash']
    errors['probe_P_orth']=error(precision*P.T@P,np.eye(16))
    full=rng.normal(size=(30,60))*.01; out=d.projected_information(full@P,metadata=meta,full_tangent_rhs=16*3)
    assert out['scope']=='PROJECTED_ONLY' and out['full_info_effective_dim'] is None and out['full_tangent_rhs']==48
    errors['projected_logdet']=error(out['projected_logdet_volume'],.5*np.linalg.slogdet(np.eye(16)+(full@P).T@(full@P))[1])
    # Scaling material units with prior transformed consistently preserves normalized quantities.
    scale=3.; res=d.information_summary(Z*scale,(precision+.01)*scale**2,precision=precision*scale**2)
    errors['material_scale_prior_eff']=error(res['info_effective_dim'],a['info_effective_dim'])
    errors['material_scale_logdet']=error(res['info_logdet_volume'],a['info_logdet_volume'])
    assert max(errors.values())<TOL
    return dict(seed=seed,errors=errors,prior_precision=precision,solve_lams=[precision+.01,precision+.00027],
                prior_d_eff=a['info_effective_dim'],solve_eff=[a['solve_effective_dim'],b['solve_effective_dim']],
                weak_dimension_real=weak.weak_dimension_real,probe_metadata=meta,projected_scope=out['scope'])

def run():
    started=time.perf_counter();rows=[];failures=[];w=None;integration=[]
    for j in range(80):
        try: rows.append(trial(20261002+j))
        except Exception as e:failures.append(dict(seed=20261002+j,error=str(e),traceback=traceback.format_exc()))
    try:w=witnesses()
    except Exception as e:failures.append(dict(group='witnesses',error=str(e),traceback=traceback.format_exc()))
    for j in range(16):
        try: integration.append(integration_trial(20261002+j))
        except Exception as e: failures.append(dict(integration_seed=20261002+j,error=str(e),traceback=traceback.format_exc()))
    result=dict(status='PASS' if not failures else 'FAIL',requested=80,passed=len(rows),threshold=TOL,
                seed_start=20261002,wall_s=time.perf_counter()-started,device='CPU',rows=rows,integration_rows=integration,integration_requested=16,integration_passed=len(integration),witnesses=w,failures=failures,
                max_errors={key:max(row['errors'][key] for row in rows) for key in rows[0]['errors']} if rows else {})
    (OUT/'TEST_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','requested','passed','wall_s','max_errors','failures']}))
    return result
if __name__=='__main__':
    r=run();sys.exit(0 if r['status']=='PASS' else 1)
