"""Anchored CAIR. Finite precision normal-defect certificate is an estimated bound.
No full Jacobian; full JVP/VJP only along actual iterates/directions. Immutable lambda.
"""
from a10_common import *
from anchored_rom import AnchoredROM
from reduced_normal import ReducedNormalInverse

def cheap_screen(full,rom,tan,s,r,scale,lam,h):
    # A strict induced-norm bound if ||A||_1 ||A||_inf < 1; otherwise unavailable.
    anorm=float(np.sqrt(la.norm(full.A,1)*la.norm(full.A,np.inf)))
    if anorm>=1:return dict(available=False,reason='Neumann bound unavailable',A_upper=anorm)
    rho=1/(1-anorm);kr=rom.current_jvp(tan.expand(s));rhs=full._tangent_rhs(tan.expand(s))[0][:,:,0]
    tr=rhs-(full.L@kr.T).T;Jr=np.einsum('mn,pn->pm',full.model.GS,kr)/scale
    # ||B|| <= ||diag(da E)/sqrt(v)|| times physical tangent norm <= 1/sqrt(v)
    bn=float(np.max(np.sqrt(np.sum(abs(full.exciting.reshape(full.model.P,full.model.N,3)*full.da[None,:,None])**2,axis=(0,2))))/full.model.volume)
    cn=float(la.norm(full.model.GS,2)/scale);jbound=cn*rho*bn
    # reduced dual state for w=r+Jr; use QR public arrays through least squares independently
    LV=full.L@rom.basis;Q,R=la.qr(LV,mode='economic');rhsdual=rom.basis.conj().T@(full.model.GS.conj().T@((r+Jr)/scale).T)
    dual=Q@la.solve_triangular(R.conj().T,rhsdual,lower=True)
    ar=full.model.GS.conj().T@((r+Jr)/scale).T-full.L.conj().T@dual
    dro=Ops(rom,tan,scale).v(r+Jr)+lam*s+h
    bound=float((la.norm(dro)+bn*rho*la.norm(ar)+jbound*cn*rho*la.norm(tr))/lam)
    return dict(available=True,eta=bound,A_upper=anorm,scope='analytic induced-norm bound evaluated in float; not interval verified')

def adaptive(full,tan,r,scale,lam,h,V,method='cair',theta=.05,max_enrich=10,max_rank=180,seed=100,rom_class=AnchoredROM):
    model=full.model;fop=Ops(full,tan,scale);before=cost(model);start=time.perf_counter();hist=[];rng=np.random.default_rng(seed)
    rom=rom_class(full,V);rop=Ops(rom,tan,scale);normal=ReducedNormalInverse(rom,tan,scale,lam);s=normal.solve(-rop.v(r)-h);inner=dict(solver='reduced-state Gram',normal_residual=float(la.norm(normal.matvec(s)+rop.v(r)+h)),factor_dimension=normal.factor_dimension)
    c=certify(fop,s,r,lam,h);initial_s=s.copy();initial_c=summary_cert(c);hpbank=[];pbank=[];f=None
    for it in range(max_enrich+1):
        allowed=theta/(1+theta)*la.norm(c['s'])+1e-10
        passed=c['eta']<=allowed
        hist.append(dict(iteration=it,rank=V.shape[1],eta=c['eta'],step_norm=float(la.norm(c['s'])),threshold=float(allowed),q=c['q'],passed=bool(passed),normal_norm=float(la.norm(c['d'])),inner={k:v for k,v in inner.items() if k!='vectors'}))
        if passed or it==max_enrich or V.shape[1]>=max_rank:break
        old=c
        if method=='cair':
            # ROM inverse is a preconditioner, never a claimed full stability bound.
            if V.shape[1]+2*model.P>max_rank:
                hist[-1]['stop_reason']='insufficient capacity for complete primal/direction illumination block';break
            p=normal.solve(-c['d']);info=0
            # Flexible conjugate directions keep previous true-H information.
            for pj,hpj in zip(pbank,hpbank):p-=pj*((p@hpj)/(pj@hpj))
            Kp=fop.k(p);Jp=np.einsum('mn,pn->pm',model.GS,Kp)/scale;Hp=fop.v(Jp)+lam*p
            denom=float(p@Hp)
            if denom<=1e-30:break
            alpha=-float(c['d']@p)/denom
            snew=c['s']+alpha*p
            # Add exact states; same baseline means actual MR ROM reproduces this line.
            V=append_basis(V,np.column_stack([current_columns(c['K']),current_columns(Kp)]))
            pbank.append(p);hpbank.append(Hp)
            Js=c['Js']+alpha*Jp;Ks=c['K']+alpha*Kp;d=c['d']+alpha*Hp
            c=dict(s=snew,d=d,Js=Js,K=Ks,eta=float(la.norm(d)/lam),q=float(.5*la.norm(r+Js)**2+h@snew+.5*lam*(snew@snew)),wall_s=0.)
            rom=rom_class(full,V,previous=rom);rop=Ops(rom,tan,scale);normal=ReducedNormalInverse(rom,tan,scale,lam)
            interpolation=float(la.norm(rop.j(snew)-Js)/max(la.norm(Js),1e-30))
            inner=dict(preconditioner_info=int(info),guaranteed_line=True,interpolation_relative=interpolation,energy_drop=old['q']-c['q'])
        else:
            if method=='exact':U=current_columns(c['K']-rom.current_jvp(tan.expand(c['s'])))
            elif method=='residual':U=current_columns(rom.tangent_residual(tan.expand(c['s'])))
            elif method in ['direct','transfer']:
                k=rom.current_jvp(tan.expand(c['s']));b=full._tangent_rhs(tan.expand(c['s']))[0][:,:,0]
                U=current_columns(b if method=='direct' else (full.A@k.T).T)
            elif method=='random':U=rng.normal(size=(model.n,model.P))+1j*rng.normal(size=(model.n,model.P))
            elif method=='dual':
                w=r+c['Js'];rhs=model.GS.conj().T@(w/scale).T
                U=model.solver.solve(full._L_factor,rhs,trans=2,label='full_adjoint')
                U=np.column_stack([current_columns(c['K']),U])
            else:raise ValueError(method)
            V=append_basis(V,U,maxadd=max_rank-V.shape[1]);rom=rom_class(full,V,previous=rom);rop=Ops(rom,tan,scale)
            normal=ReducedNormalInverse(rom,tan,scale,lam);s=normal.solve(-rop.v(r)-h);inner=dict(solver='reduced-state Gram',factor_dimension=normal.factor_dimension);c=certify(fop,s,r,lam,h)
    wall=time.perf_counter()-start
    return c['s'],dict(method=method,initial_rank=hist[0]['rank'],final_rank=V.shape[1],initial_certificate=initial_c,certificate=summary_cert(c),history=hist,passed=bool(hist[-1]['passed']),status='ACCEPT' if hist[-1]['passed'] else 'FALLBACK_REQUIRED',wall_s=wall,cost=costdiff(before,cost(model)),certificate_type='estimated full normal-defect bound; exact-arithmetic theorem; empirical floating point coverage',initial_s=initial_s,basis=V,step=c['s'],active_material_dimension=len(pbank)+1 if method=='cair' else None,active_source='current full normal defect preconditioned by actual ROM; history H-conjugation' if method=='cair' else None)

def domain_bank(model,count=48,seed=102):
    """Randomized right singular subspace of GD, with residual diagnostics."""
    rng=np.random.default_rng(seed);q=min(model.n,count+12)
    O=rng.normal(size=(model.n,q))+1j*rng.normal(size=(model.n,q));Y=model.GD@O
    for _ in range(2):Y=model.GD@(model.GD.conj().T@la.qr(Y,mode='economic')[0])
    Q=la.qr(Y,mode='economic')[0];_,s,vh=la.svd(Q.conj().T@model.GD,full_matrices=False);V=vh[:count].conj().T
    AV=model.GD@V;res=la.norm(model.GD.conj().T@AV-V*(s[:count]**2))/max(la.norm(model.GD.conj().T@AV),1e-30)
    return V,dict(scope='randomized GD right singular candidate, approximate TSOM',ritz_residual=float(res),seed=seed)

def tsom(Vs,Vd,rank,gs_fraction=.5):
    a=int(round(rank*gs_fraction));return append_basis(Vs[:,:a],Vd[:,:rank-a],maxadd=rank-a)
