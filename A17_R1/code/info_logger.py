"""Read-only diagnostic bridge into an EXISTING compressed SelectorContext.

Each model.normal.Z uses its OWN economic QR compressed data coordinates.
Such Z matrices share material columns, so information comparisons are valid,
but their rows cannot be paired with ctx.r or used to form a physical D.
Delta b MUST use child.jt(ctx.r)-base.jt(ctx.r).

No full Maxwell calls. Reference fullJ/fullJV are provided by parent evaluator;
fullJV's paid physical action counters must accompany it. Never expose them as
selector_information. Builds/projections/jt through existing model still cost
real existing-context ledger actions, and their wall times are recorded.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
from scipy import linalg as la
try:
    from . import information_diagnostics as d
except ImportError:
    import information_diagnostics as d


def _hash_array(x):
    x=np.ascontiguousarray(x)
    h=hashlib.sha256();h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.view(np.uint8))
    return h.hexdigest()


class InfoLogger:
    def __init__(self,ctx,*,precision=1e-5,fullJ=None,V=None,fullJV=None,
                 probe_metadata=None,full_tangent_rhs=0,full_adjoint_rhs=0,
                 metric_hash='UNSPECIFIED',whitening_scope='UNSPECIFIED',
                 whitening_hash='UNSPECIFIED',material_basis_hash='UNSPECIFIED',
                 output_dir=None):
        self.ctx=ctx;self.precision=d._lam(precision)
        if ctx.dim_material%2: raise ValueError('compressed complex material adapter requires even real dimension')
        if fullJ is not None and np.iscomplexobj(fullJ): raise ValueError('fullJ must be real packed physical data Jacobian')
        if V is not None and np.iscomplexobj(V): raise ValueError('public material probes must be real')
        if fullJV is not None and np.iscomplexobj(fullJV): raise ValueError('fullJV must be real packed physical probe outputs')
        self.metric_hash=metric_hash;self.whitening_scope=whitening_scope
        self.whitening_hash=whitening_hash;self.material_basis_hash=material_basis_hash
        self._fullJ=None if fullJ is None else np.asarray(fullJ,float).copy()
        self._V=None if V is None else np.asarray(V,float).copy()
        self._fullJV=None if fullJV is None else np.asarray(fullJV,float).copy()
        self._probe_metadata=probe_metadata
        self._charged_full_tangent_rhs=int(full_tangent_rhs);self._charged_full_adjoint_rhs=int(full_adjoint_rhs)
        if (self._V is None)!=(self._fullJV is None): raise ValueError('V and parent-paid fullJV must be supplied together')
        if self._V is not None:
            if probe_metadata is None or probe_metadata.get('scope')!='PROJECTED_ONLY': raise ValueError('projected probe metadata required')
            if self._V.shape[0]!=ctx.dim_material or self._V.shape[1]!=probe_metadata['actual_count']: raise ValueError('probe material shape mismatch')
            if la.norm(self.precision*self._V.T@self._V-np.eye(self._V.shape[1]))>1e-9: raise ValueError('V.T P V must equal I')
            if self._charged_full_tangent_rhs<=0: raise ValueError('parent-paid fullJV tangent RHS charges required')
        if self._fullJ is not None and (self._fullJ.ndim!=2 or self._fullJ.shape[1]!=ctx.dim_material):
            raise ValueError('fullJ must share ctx real material coordinates')
        self.output_dir=None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:self.output_dir.mkdir(parents=True,exist_ok=True)
        self._models={};self._summaries={};self._weak={};self._base_model=None;self._base_ids=None;self._base_x=None
        self._context_key=self._frozen_context_key()
        self.records=[]

    def _frozen_context_key(self):
        return (float(self.ctx.lam),_hash_array(self.ctx.r),_hash_array(self.ctx.ell),
                self.precision,self.metric_hash,self.whitening_hash,self.material_basis_hash)

    def _check_context(self):
        if self._frozen_context_key()!=self._context_key:
            raise ValueError('frozen residual/prior/LM/metric changed; create a fresh InfoLogger')

    def _key(self,ids,U):
        ids=tuple(ids)
        if len(ids)!=len(set(ids)):raise ValueError('unique original atom IDs required')
        return ids,_hash_array(np.asarray(U,complex))

    def _Z(self,model):
        if model.normal is None:return np.zeros((0,self.ctx.dim_material//2),complex)
        return np.asarray(model.normal.Z,complex)

    def _get_model(self,ids,U):
        self._check_context();key=self._key(ids,U);hit=key in self._models;before=time.perf_counter()
        if not hit:self._models[key]=self.ctx.make_model(U)
        return self._models[key],key,hit,time.perf_counter()-before

    def model(self,ids,U):
        """Reuse an existing reduced model; no evaluator arrays returned."""
        return self._get_model(ids,U)[0]

    def _base_info(self,model,key,spectrum):
        skey=key,bool(spectrum);hit=skey in self._summaries;start=time.perf_counter()
        if not hit:self._summaries[skey]=d.information_summary(self._Z(model),float(self.ctx.lam),precision=self.precision,spectrum=spectrum,space_id=str(key[0]))
        return dict(self._summaries[skey]),hit,time.perf_counter()-start

    def _rhs_record(self,rhs,key):
        rhs=np.asarray(rhs,float);h=_hash_array(rhs)
        if self.output_dir is None:return dict(delta_b=rhs.tolist(),delta_b_sha256=h,delta_b_path=None)
        path=self.output_dir/('delta_b_'+h+'.npz')
        if not path.exists():np.savez(path,delta_b=rhs)
        return dict(delta_b=None,delta_b_sha256=h,delta_b_path=str(path.resolve()),
                    delta_b_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def record(self,ids,U,x,base_model=None,*,spectrum=False,weak=False,evaluator=False,role='endpoint'):
        """Record reduced endpoint information, optionally relative to base_model.

        x is shared real material step. Task-weak alignment uses stored base x
        when base_summary established it; otherwise x supplied here is used and
        source is explicitly recorded. No full GN defect is inferred.
        """
        start=time.perf_counter();model,key,hit,buildwall=self._get_model(ids,U)
        info,infohit,infowall=self._base_info(model,key,spectrum)
        x=np.asarray(x)
        if np.iscomplexobj(x) or x.shape!=(self.ctx.dim_material,):raise ValueError('x must be real packed shared material step')
        out=dict(ids=list(ids),role=role,U_hash=key[1],model_cache_hit=hit,information_cache_hit=infohit,
                 selector_information=info,evaluator_information=None,
                 metadata=dict(prior_precision=self.precision,prior_precision_hash=_hash_array(np.array([self.precision])),lam_total=float(self.ctx.lam),
                               lm_metric='identity in existing real material gauge',
                               diagnostic_coverage_policy='caller locked family-balanced12 per outgoing bank + final/teacherbest; logger does not pick candidates',
                               lm_damping=float(self.ctx.lam)-self.precision,metric_hash=self.metric_hash,
                               whitening_scope=self.whitening_scope,whitening_hash=self.whitening_hash,
                               material_basis_hash=self.material_basis_hash,
                               current_actual_rank=model.U.shape[1],material_dimension=self.ctx.dim_material,
                               compressed_rows=self._Z(model).shape[0],
                               compressed_row_scope='model-specific economic QR isometry; not physical residual rows'),
                 wall_model_build_s=buildwall,wall_information_s=infowall,
                 wall_rhs_s=0.,wall_evaluator_s=0.,full_tangent_rhs_called=0,full_adjoint_rhs_called=0)
        if base_model is not None:
            t=time.perf_counter();rhs=np.asarray(model.jt(self.ctx.r))-np.asarray(base_model.jt(self.ctx.r))
            out.update(self._rhs_record(rhs,key),normal_rhs_difference_norm=float(la.norm(rhs)),
                       delta_b_source='child.jt(ctx.r)-base.jt(ctx.r) in common PHYSICAL real data coordinates',wall_rhs_s=time.perf_counter()-t)
            bz=self._Z(base_model);cz=self._Z(model);w=None
            if weak:
                wk=_hash_array(bz)
                if wk not in self._weak:self._weak[wk]=d.fixed_weak_space(bz,precision=self.precision)
                w=self._weak[wk]
            px=self._base_x if base_model is self._base_model and self._base_x is not None else x
            t=time.perf_counter();candidate=d.candidate_information(bz,cz,float(self.ctx.lam),precision=self.precision,
                           weak=w,gnstep=px if weak else None,spectrum=False,candidate_id=str(tuple(ids)))
            if spectrum:
                binfo,_,bwall=self._base_info(base_model,('relative_base',_hash_array(base_model.U)),True)
                candidate.update(spectrum_evaluated=True,old_effective_dim=binfo['info_effective_dim'],new_effective_dim=info['info_effective_dim'],
                                 delta_effective_dim=info['info_effective_dim']-binfo['info_effective_dim'],
                                 delta_logdet_volume=info['info_logdet_volume']-binfo['info_logdet_volume'],
                                 old_solve_effective_dim=binfo['solve_effective_dim'],new_solve_effective_dim=info['solve_effective_dim'])
                if candidate.get('missing_reason'):
                    candidate['missing_reason']=candidate['missing_reason'].replace('effective information/spectrum not evaluated for this candidate; ','').replace('effective information/spectrum not evaluated for this candidate','')
            # Its generic residual route is deliberately unused: bz/cz do NOT share physical rows.
            candidate.pop('delta_b',None)
            candidate['delta_b_missing_reason']=None
            candidate['rhs_source']='bridge physical model.jt(ctx.r); compressed rows are never paired with residual'
            if candidate.get('missing_reason'):
                candidate['missing_reason']=candidate['missing_reason'].replace('residual unavailable: delta_b missing; spectrum alone cannot determine step; ','').replace('residual unavailable: delta_b missing; spectrum alone cannot determine step','') or None
            candidate['approximate_GNstep_source']='stored base step' if px is self._base_x else 'caller supplied step'
            out['selector_information']['candidate']=candidate
            out['wall_information_s']+=time.perf_counter()-t
        if evaluator:
            t=time.perf_counter();ev=dict(permission='EVALUATOR_ONLY; never pass to proposer inputs')
            if self._fullJ is not None:
                J=model.j(np.eye(self.ctx.dim_material))
                ev['full_information_fidelity']=d.information_fidelity_real(J,self._fullJ,self.precision)
                ev['full_reference_hash']=_hash_array(self._fullJ)
                ev['model_j_material_columns']=self.ctx.dim_material
            else:
                ev.update(full_information_fidelity=None,full_reference_missing_reason='small fullJ not supplied')
            if self._V is not None:
                Jv=model.j(self._V)
                ev['model_projected']=d.projected_information(Jv,metadata=self._probe_metadata,full_tangent_rhs=0)
                ev['reference_projected']=d.projected_information(self._fullJV,metadata=self._probe_metadata,
                                          full_tangent_rhs=self._charged_full_tangent_rhs,full_adjoint_rhs=self._charged_full_adjoint_rhs)
                ev['parent_paid_reference_cost_scope']='one fixed setup cost; do not sum repeated record views'
                ev['projected_information_fidelity']=d.information_fidelity_real(Jv,self._fullJV,precision=1.)
                ev['projected_information_fidelity']['scope']='PROJECTED_ONLY common P-orthonormal probe basis'
                ev['model_j_probe_columns']=self._V.shape[1]
            else:ev.update(projected_information=None,projected_missing_reason='parent-paid public V/fullJV not supplied')
            out['evaluator_information']=ev;out['wall_evaluator_s']=time.perf_counter()-t
        out['wall_total_s']=time.perf_counter()-start
        self.records.append(out);return out

    def base_summary(self,ids,U,x,*,spectrum=True,evaluator=False):
        out=self.record(ids,U,x,spectrum=spectrum,evaluator=evaluator,role='base')
        self._base_model=self._models[self._key(ids,U)];self._base_ids=tuple(ids);self._base_x=np.asarray(x,float).copy()
        return out

    def candidate_summary(self,ids,U,x,base_model=None,*,spectrum=False,weak=False,evaluator=False):
        base=self._base_model if base_model is None else base_model
        if base is None:raise ValueError('base_summary or explicit base_model required')
        return self.record(ids,U,x,base_model=base,spectrum=spectrum,weak=weak,evaluator=evaluator,role='candidate')

    def final_summary(self,ids,U,x,*,base_model=None,evaluator=False):
        base=self._base_model if base_model is None else base_model
        return self.record(ids,U,x,base_model=base,spectrum=True,weak=base is not None,evaluator=evaluator,role='final')
