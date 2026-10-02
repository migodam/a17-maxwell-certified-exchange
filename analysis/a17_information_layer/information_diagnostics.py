"""CPU side diagnostics for complex-linear compressed A16 material maps.

Z=stack_p(Rc Fp) has (complex data rows, qcomplex). Shared real material is
[Re(material); Im(material)]. This complex-linear contract is necessary for
singular values to repeat twice. Arbitrary real Gaussian tangents use
information_from_real or information_from_tangent instead.
No selector, physics, nuisance, prior, Flow or acceptance logic is changed.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import linalg as la


def _lam(lam):
    lam=float(lam)
    if not np.isfinite(lam) or lam<=0: raise ValueError('finite scalar lam > 0 required')
    return lam


def _Z(Z):
    Z=np.asarray(Z,dtype=np.complex128)
    if Z.ndim!=2 or not np.all(np.isfinite(Z)): raise ValueError('finite 2D compressed Z required')
    return Z


def realify_complex_linear(Z):
    """Real J for q complex materials: [[ReZ,-ImZ],[ImZ,ReZ]]."""
    Z=_Z(Z)
    return np.block([[Z.real,-Z.imag],[Z.imag,Z.real]])


def realify_tangent(Z,T):
    """General REAL p-coordinate material tangent T(qcomplex,p), no duplication law."""
    Z=_Z(Z); T=_Z(T)
    if T.shape[0]!=Z.shape[1]: raise ValueError('tangent shape mismatch')
    A=Z@T
    return np.vstack([A.real,A.imag])


def _summary(eigenvalues,normtrace,lam,multiplicity,dimension=None,backend=None,precision=1e-5):
    lam=_lam(lam); precision=_lam(precision); e=np.maximum(np.asarray(eigenvalues,float),0.)
    return dict(info_trace=float(multiplicity*normtrace),
                info_effective_dim=float(multiplicity*np.sum(e/(e+precision))),
                info_effective_rank_lam=int(multiplicity*np.count_nonzero(e>precision)),
                info_effective_rank_10lam=int(multiplicity*np.count_nonzero(e>10*precision)),
                info_prior_precision=precision,solve_total_lam=lam,lm_damping=lam-precision,
                solve_effective_dim=float(multiplicity*np.sum(e/(e+lam))),
                info_logdet_volume=float(.5*multiplicity*np.sum(np.log1p(e/precision))),
                info_effective_rank_prior1=int(multiplicity*np.count_nonzero(e>precision)),
                info_effective_rank_prior10=int(multiplicity*np.count_nonzero(e>10*precision)),
                thresholds=dict(prior_precision=precision,normalized_tau1=1.,normalized_tau2=10.,tau1=precision,tau2=10*precision,weak_eigenvalue_upper=precision),
                complex_information_eigenvalues=e.tolist() if multiplicity==2 else None,
                real_singular_values=np.repeat(np.sqrt(e),multiplicity).tolist(),
                real_material_dimension=dimension,backend=backend,
                condition='whitened isometric material/data metric; scalar lam; complex-linear embedding' if multiplicity==2 else 'general real tangent; whitened/isometric metric; scalar lam',
                gn_risk_absolute=None,gn_risk_missing_reason='full reference step/H not supplied; information is not GN risk')


def info_from_gram(E,normtrace,lam=1.,*,qcomplex=None,precision=1e-5):
    """Compressed backend: E=Z Z*, normtrace=||Z||_F^2 (NOT real trace).

    Eigenvalues use the row Gram, avoiding full-q SVD / q×q normal matrix.
    qcomplex is optional but required to report full real material dimension.
    Tiny negative Gram eigenvalues are clipped only as roundoff; substantial
    negativity or inconsistency with supplied normtrace is rejected.
    """
    E=_Z(E)
    if E.shape[0]!=E.shape[1]: raise ValueError('square Gram required')
    scale=max(1.,la.norm(E)); herm_error=la.norm(E-E.conj().T)
    if herm_error>1e-10*scale: raise ValueError('non-Hermitian Gram')
    e=la.eigvalsh((E+E.conj().T)*.5)[::-1]
    if e.size and e[-1]< -1e-10*scale: raise ValueError('Gram is not PSD')
    normtrace=float(normtrace)
    if not np.isfinite(normtrace) or normtrace<0: raise ValueError('nonnegative ||Z||F^2 required')
    if abs(E.trace().real-normtrace)>1e-9*max(1.,normtrace): raise ValueError('Gram trace/normtrace mismatch')
    if qcomplex is not None:
        if qcomplex<0 or int(qcomplex)!=qcomplex: raise ValueError('qcomplex must be nonnegative integer')
        e=e[:qcomplex]
    return _summary(e,normtrace,lam,2,None if qcomplex is None else 2*qcomplex,'compressed_row_gram',precision)


def information_summary(Z,lam,*,precision=1e-5,spectrum=True,space_id=None):
    Z=_Z(Z); lam=_lam(lam); precision=_lam(precision); trace=float(np.vdot(Z,Z).real)
    if spectrum:
        out=info_from_gram(Z@Z.conj().T,trace,lam,qcomplex=Z.shape[1],precision=precision)
    else:
        out=dict(info_trace=2*trace,info_effective_dim=None,info_effective_rank_lam=None,
                 info_effective_rank_10lam=None,real_singular_values=None,
                 info_prior_precision=precision,solve_total_lam=lam,lm_damping=lam-precision,
                 info_logdet_volume=None,solve_effective_dim=None,
                 info_effective_rank_prior1=None,info_effective_rank_prior10=None,
                 thresholds=dict(prior_precision=precision,normalized_tau1=1.,normalized_tau2=10.,tau1=precision,tau2=10*precision,weak_eigenvalue_upper=precision),
                 real_material_dimension=2*Z.shape[1],backend='trace_only',
                 missing_reason='spectrum intentionally not evaluated; diagnostic coverage limit',
                 gn_risk_absolute=None,gn_risk_missing_reason='full reference step/H not supplied')
    out.update(space_id=space_id,spectrum_evaluated=bool(spectrum),complex_rows=Z.shape[0],qcomplex=Z.shape[1])
    return out


def information_from_real(J,lam,*,precision=1e-5):
    J=np.asarray(J)
    if np.iscomplexobj(J) or J.ndim!=2 or not np.all(np.isfinite(J)): raise ValueError('finite real J required')
    J=np.asarray(J,float); e=la.eigvalsh(J@J.T)[::-1]
    e=e[:J.shape[1]]
    return _summary(e,float(np.vdot(J,J)),lam,1,J.shape[1],'general_real_row_gram',precision)


def information_from_tangent(Z,T,lam,*,precision=1e-5):
    return information_from_real(realify_tangent(Z,T),lam,precision=precision)


@dataclass(frozen=True)
class FixedWeakSpace:
    Vs: np.ndarray
    precision: float
    qcomplex: int
    base_hash: str
    strong_dimension_complex: int
    weak_dimension_real: int
    definition: str='entire eigenvalue <= fixed prior precision eigenspace of BASE Z*Z, including all nullspace'


def fixed_weak_space(base_Z,precision=1e-5):
    """Store ONLY strong right basis; weak projector is identity minus Vs Vs*.

    No arbitrary bottom eigenvectors, no q×q material matrix, no explicit
    nullspace basis. The row-Gram diagonalization recovers only strong vectors.
    """
    Z=_Z(base_Z); precision=_lam(precision); E=Z@Z.conj().T
    e,U=la.eigh(E); keep=e>precision
    Vs=(Z.conj().T@U[:,keep])/np.sqrt(e[keep])[None,:]
    # Gram identity holds for these positive eigenpairs; don't reorder weak basis.
    if Vs.shape[1] and la.norm(Vs.conj().T@Vs-np.eye(Vs.shape[1]))>1e-9:
        raise ValueError('strong right basis lost orthonormality')
    Vs=np.ascontiguousarray(Vs); Vs.setflags(write=False)
    h=hashlib.sha256(np.ascontiguousarray(Z).view(np.uint8)).hexdigest()
    return FixedWeakSpace(Vs,precision,Z.shape[1],h,Vs.shape[1],2*(Z.shape[1]-Vs.shape[1]))


def project_weak(step_real,weak):
    """Return complex material pweak, equivalent to whole real weak projector."""
    p=np.asarray(step_real)
    if np.iscomplexobj(p) or p.shape!=(2*weak.qcomplex,): raise ValueError('real packed GNstep shape must be (2q,)')
    p=np.asarray(p,float)
    if not np.all(np.isfinite(p)): raise ValueError('finite real GNstep required')
    c=p[:weak.qcomplex]+1j*p[weak.qcomplex:]
    return c-weak.Vs@(weak.Vs.conj().T@c)


def candidate_information(base_Z,child_Z,lam,*,precision=1e-5,weak=None,gnstep=None,residual_complex=None,spectrum=False,candidate_id=None):
    base=_Z(base_Z); child=_Z(child_Z); lam=_lam(lam); precision=_lam(precision)
    if base.shape[1]!=child.shape[1]: raise ValueError('shared material dimension required')
    out=dict(candidate_id=candidate_id,delta_info_trace=float(2*(np.vdot(child,child).real-np.vdot(base,base).real)),
             delta_effective_dim=None,delta_logdet_volume=None,delta_info_weak=None,delta_info_strong=None,
             weak_task_score=None,weak_task_label='approximate model GNstep alignment',
             weak_task_definition='pweak.T DeltaI pweak invariant quadratic; not diagonal sum |vi*p|² delta_i, not fullGNdefect',
             spectrum_evaluated=bool(spectrum),weak_evaluated=weak is not None,
             missing_reason=None)
    if spectrum:
        a=information_summary(base,lam,precision=precision); b=information_summary(child,lam,precision=precision)
        out.update(old_effective_dim=a['info_effective_dim'],new_effective_dim=b['info_effective_dim'],
                   delta_effective_dim=b['info_effective_dim']-a['info_effective_dim'],delta_logdet_volume=b['info_logdet_volume']-a['info_logdet_volume'],
                   old_solve_effective_dim=a['solve_effective_dim'],new_solve_effective_dim=b['solve_effective_dim'])
    if weak is not None:
        h=hashlib.sha256(np.ascontiguousarray(base).view(np.uint8)).hexdigest()
        if weak.qcomplex!=base.shape[1] or weak.base_hash!=h or weak.precision!=precision:
            raise ValueError('weak space must be fixed from this exact base_Z and fixed prior precision')
        # tr(Pweak I)=||Z||²-||Z Vs||², duplicate real trace factor2.
        strong=2*(la.norm(child@weak.Vs,'fro')**2-la.norm(base@weak.Vs,'fro')**2)
        out.update(delta_info_strong=float(strong),delta_info_weak=float(out['delta_info_trace']-strong),
                   weak_dimension_real=weak.weak_dimension_real,strong_dimension_real=2*weak.strong_dimension_complex,
                   fixed_base_hash=weak.base_hash)
        if gnstep is not None:
            p=project_weak(gnstep,weak)
            out.update(weak_task_score=float(la.norm(child@p)**2-la.norm(base@p)**2),
                       approximate_GNstep_weak_norm2=float(np.vdot(p,p).real))
    out['delta_b']=delta_b(base,child,residual_complex).tolist() if residual_complex is not None else None
    out.update(info_prior_precision=precision,solve_total_lam=lam,lm_damping=lam-precision)
    missing=[]
    if residual_complex is None: missing.append('residual unavailable: delta_b missing; spectrum alone cannot determine step')
    if not spectrum: missing.append('effective information/spectrum not evaluated for this candidate')
    if weak is None: missing.append('fixed base weak projector not supplied')
    if gnstep is None: missing.append('approximate model real packed GNstep not supplied')
    out['missing_reason']='; '.join(missing) if missing else None
    return out


def fixed_diagnostic_ids(incoming_ids,limit=12):
    """Input order predeclared BEFORE gains; never cherry-pick accepted moves."""
    ids=tuple(incoming_ids)
    if len(ids)!=len(set(ids)): raise ValueError('unique incoming IDs required')
    if limit<0: raise ValueError('nonnegative limit')
    return ids[:limit]


def coverage(incoming_ids,evaluated_ids,final_endpoints=()):
    incoming=set(incoming_ids); evaluated=set(evaluated_ids)
    if not evaluated<=incoming: raise ValueError('unknown diagnostic incoming ID')
    return dict(incoming_total=len(incoming),incoming_spectrum_weak_evaluated=len(evaluated),
                incoming_diagnostic_ids=list(evaluated_ids),final_endpoint_ids=list(final_endpoints),
                incoming_omitted_ids=[i for i in incoming_ids if i not in evaluated],
                scope='side diagnostics only; omitted quantities remain null; no selector or stationarity claim')


def delta_b(base_Z,child_Z,residual_complex):
    """Normal-equation RHS change D_real.T r_real; residual order all-Re/all-Im.

    Caller converts per-illumination A16 packing into stacked COMPLEX residual.
    Real material output remains [real;imag]. Prior ell cancels between endpoints.
    """
    a,b=_Z(base_Z),_Z(child_Z)
    if a.shape!=b.shape: raise ValueError('delta_b needs identical data/material coordinates')
    r=np.asarray(residual_complex,complex)
    if r.shape!=(a.shape[0],): raise ValueError('stacked complex residual shape mismatch')
    z=(b-a).conj().T@r
    return np.concatenate([z.real,z.imag])


def information_fidelity_real(Jmodel,Jreference,precision=1e-5):
    """Small material EVALUATOR ONLY; explicitly pays dense p×p Gram/Cholesky.

    Spectral/Frobenius norms of Href^-1/2 (Imodel-Iref) Href^-1/2,
    computed via congruent Cholesky whitening (same invariant norms).
    """
    a,b=np.asarray(Jmodel),np.asarray(Jreference)
    if np.iscomplexobj(a) or np.iscomplexobj(b) or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:
        raise ValueError('shared finite real material J matrices required')
    precision=_lam(precision); Ia=a.T@a; Ib=b.T@b; H=Ib+precision*np.eye(b.shape[1])
    L=la.cholesky(H,lower=True); tmp=la.solve_triangular(L,Ia-Ib,lower=True)
    W=la.solve_triangular(L,tmp.T,lower=True).T
    eig=la.eigvalsh((W+W.T)*.5)
    return dict(info_fidelity_spectral=float(np.max(np.abs(eig),initial=0.)),info_fidelity_frobenius=float(la.norm(W,'fro')),
                scope='full small material reference, evaluator only',precision=precision,full_reference_used=True)


def common_material_probes(material_dimension,object_id,*,precision=1e-5,count=16,seed=20261002):
    """Predeclared fixed public real probe basis with V.T P V = I.

    Full actions must be paid by caller; generation is independent of candidate
    gain and truth. No spectral conclusion outside this projection is implied.
    """
    if material_dimension<1 or count<1: raise ValueError('positive material dimension/count')
    precision=_lam(precision); digest=hashlib.sha256(str(object_id).encode()).digest()
    words=np.frombuffer(digest[:16],dtype='<u4').astype(np.uint32).tolist()
    rng=np.random.default_rng(np.random.SeedSequence([seed]+words)); q=min(material_dimension,count)
    V=la.qr(rng.normal(size=(material_dimension,q)),mode='economic')[0]/np.sqrt(precision)
    h=hashlib.sha256(np.ascontiguousarray(V).view(np.uint8)).hexdigest()
    return V,dict(seed=seed,object_id=str(object_id),object_hash=digest.hex(),probe_hash=h,
                  requested_count=count,actual_count=q,precision=precision,
                  scope='PROJECTED_ONLY',full_actions_paid_by_caller=True)


def projected_information(JV,*,metadata,full_tangent_rhs,full_adjoint_rhs=0):
    """JV=J V on common P-orthonormal probes, so projected prior is identity.

    Deliberately requires charged action counters; never labels complete-space
    d_eff, weak classification or information fidelity from these 16 probes.
    """
    if metadata.get('scope')!='PROJECTED_ONLY': raise ValueError('probe scope metadata required')
    A=np.asarray(JV)
    if np.iscomplexobj(A) or A.ndim!=2 or A.shape[1]!=metadata['actual_count']: raise ValueError('real JV shape mismatch')
    if full_tangent_rhs<0 or full_adjoint_rhs<0: raise ValueError('nonnegative charged RHS counts')
    out=information_from_real(A,1.,precision=1.)
    return dict(projected_info_trace=out['info_trace'],projected_effective_dim=out['info_effective_dim'],
                projected_logdet_volume=out['info_logdet_volume'],projected_rank_tau1=out['info_effective_rank_prior1'],
                projected_rank_tau10=out['info_effective_rank_prior10'],scope='PROJECTED_ONLY',probe_metadata=metadata,
                full_tangent_rhs=int(full_tangent_rhs),full_adjoint_rhs=int(full_adjoint_rhs),
                full_info_effective_dim=None,full_weakness=None,
                missing_reason='16 fixed public probes do not span complete material space or establish full weak modes')
