"""Pure-array fixed full-reference GN direction analysis. No physics actions."""
import numpy as np
from scipy import linalg as la
PRIOR=1e-5

def fixed_reference(J,lam,*,precision=PRIOR,cluster_rtol=1e-10):
    J=np.asarray(J)
    if np.iscomplexobj(J) or J.ndim!=2 or not np.all(np.isfinite(J)):raise ValueError('finite real physical J required')
    if not 0<precision<=lam:raise ValueError('require scalar total Lambda >= positive prior')
    I=J.T@J;e,V=la.eigh(I);e=np.maximum(e,0.)
    # Group numerically adjacent eigenvalues as subspaces, including repeated/null eigenvalues.
    # Clustering does not change eigenvalues, thresholds, curvature, or scores.
    groups=[]
    for i in range(len(e)):
        if not groups or abs(e[i]-e[groups[-1][-1]])>cluster_rtol*max(precision,abs(e[i]),abs(e[groups[-1][-1]])):groups.append([i])
        else:groups[-1].append(i)
    def band(x):return 'weak' if x<=precision else ('intermediate' if x<=10*precision else 'strong')
    blocks=[];bands=[]
    for g in groups:
        labels=sorted(set(band(e[i]) for i in g));label=labels[0] if len(labels)==1 else 'BOUNDARY_MIXED_'+'_'.join(labels)
        blocks.append(dict(indices=g,dimension=len(g),band=label,eigenvalue_min=float(e[g[0]]),eigenvalue_max=float(e[g[-1]]),basis_dependent_directions=len(g)>1));bands.extend([label]*len(g))
    return dict(J=J,I=I,e=e,V=V,lam=float(lam),precision=float(precision),lm=float(lam-precision),blocks=blocks,bands=bands,cluster_rtol=cluster_rtol)

def compare(ref,r,ell,sref,before,after):
    J=ref['J'];V=ref['V'];e=ref['e'];lam=ref['lam'];precision=ref['precision']
    r,ell,sref,before,after=[np.asarray(a,float) for a in [r,ell,sref,before,after]]
    if any(a.shape!=(J.shape[1],) for a in [ell,sref,before,after]) or r.shape!=(J.shape[0],):raise ValueError('physical real data/material shape mismatch')
    g=J.T@(r+J@sref)+ell+lam*sref
    h=J.T@r+ell;relative=float(la.norm(g)/max(la.norm(h),1e-300))
    eb=V.T@(before-sref);ea=V.T@(after-sref);delta=eb**2-ea**2
    likelihood=.5*e*delta;prior=.5*precision*delta;lm=.5*(lam-precision)*delta;energy=likelihood+prior+lm
    gb=V.T@g;cross=gb*(eb-ea);total=energy+cross
    rb=.5*(la.norm(J@(before-sref))**2+lam*la.norm(before-sref)**2)+g@(before-sref)
    ra=.5*(la.norm(J@(after-sref))**2+lam*la.norm(after-sref)**2)+g@(after-sref)
    directions=[dict(index=i,information_eigenvalue=float(e[i]),normalized_information=float(e[i]/precision),band=ref['bands'][i],error_before=float(eb[i]),error_after=float(ea[i]),likelihood_gain=float(likelihood[i]),prior_gain=float(prior[i]),lm_gain=float(lm[i]),halfH_gain=float(energy[i]),reference_gradient_cross_gain=float(cross[i]),actual_RGN_gain=float(total[i])) for i in range(len(e))]
    bands=[]
    for band in sorted(set(ref['bands'])):
        ix=[i for i,x in enumerate(ref['bands']) if x==band];bands.append(dict(band=band,dimension=len(ix),error_norm2_before=float(eb[ix]@eb[ix]),error_norm2_after=float(ea[ix]@ea[ix]),likelihood_gain=float(sum(likelihood[ix])),prior_gain=float(sum(prior[ix])),lm_gain=float(sum(lm[ix])),halfH_gain=float(sum(energy[ix])),reference_gradient_cross_gain=float(sum(cross[ix])),actual_RGN_gain=float(sum(total[ix])),mean_gain_per_direction=float(sum(total[ix])/len(ix))))
    blocks=[]
    for block in ref['blocks']:
        ix=block['indices'];blocks.append(dict(**block,error_norm2_before=float(eb[ix]@eb[ix]),error_norm2_after=float(ea[ix]@ea[ix]),actual_RGN_gain=float(sum(total[ix]))))
    return dict(directions=directions,bands=bands,eigenvalue_blocks=blocks,risk_before=float(rb),risk_after=float(ra),actual_RGN_delta=float(rb-ra),sum_band_gain=float(sum(x['actual_RGN_gain'] for x in bands)),sum_halfH_gain=float(sum(energy)),reference_gradient_cross_gain=float(sum(cross)),sum_consistency_abs=float(abs(sum(total)-(rb-ra))),reference_gradient_norm=float(la.norm(g)),reference_relative_residual=relative)

def receiver_weak_overlap(ref,Jreceiver):
    e,U=la.eigh(Jreceiver.T@Jreceiver);W=U[:,e<=ref['precision']];rows=[]
    for band in sorted(set(ref['bands'])):
        ix=[i for i,x in enumerate(ref['bands']) if x==band];F=ref['V'][:,ix];overlap=float(la.norm(F.T@W,'fro')**2)
        rows.append(dict(full_reference_band=band,full_band_dimension=len(ix),receiver_ROM_weak_dimension=W.shape[1],projector_overlap_trace=overlap,fraction_of_full_band=overlap/len(ix),fraction_of_ROM_weak=None if W.shape[1]==0 else overlap/W.shape[1]))
    return dict(scope='receiver-ROM eigenvalue<=prior subspace projected onto fixed full-reference bands; not full weak classification',receiver_ROM_weak_dimension=W.shape[1],rows=rows)
