"""Frozen A16 inputs, exact saved material gauge, pinned existing kernels."""
from pathlib import Path
import json, hashlib, time
import numpy as np
from scipy import linalg as la
import pinned_runtime as runtime
from a10_common import DenseDDA,Tangent
from scenes import acquisition
from material_gauge_persistence import load_frozen_material,load_warm_step
from operators import make_problem
from selectors_current import SelectorConfig,build_independent_anchor,build_candidates
from selector_coordinates import prepare_selector_workspace

ROOT=runtime.ROOT
def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def frozen(oid,phase,device):
    started=time.perf_counter();config=json.loads((ROOT/'A17_RUN_CONFIG.json').read_text())
    scene=next(s for s in json.loads((ROOT/'inputs/scenes.json').read_text())['scenes'] if s['object_id']==oid)
    ref=ROOT/f'inputs/object_{oid}';it=config['phases'][phase]
    paths=[ref/'common_data.npz',ref/f'anchor_{it:02d}.npz',ref/f'step_{it:02d}.npz']
    known={row['path']:row['sha256'] for row in runtime.MANIFEST['inputs'] if row.get('kind')=='frozen_input'}
    for p in paths:
        if sha(p)!=known[p.relative_to(ROOT).as_posix()]:raise ValueError('frozen input hash drift')
    common=dict(np.load(paths[0],allow_pickle=False));anchor=dict(np.load(paths[1],allow_pickle=False));saved=dict(np.load(paths[2],allow_pickle=False))
    points=common['points'];v=float(scene['edge']/scene['n'])**3
    tangent,gauge,ell=load_frozen_material(points,v,scene['representation'],common,anchor,tangent_class=Tangent,allow_legacy=False)
    full_step=load_warm_step(tangent,saved,gauge,allow_legacy=False)
    dirs,pols,rx,obs=acquisition(scene)
    model=DenseDDA(points,v,scene['wavenumber'],dirs,pols,rx,obs,device=device)
    state=model.state(anchor['chi'])
    ctx,ev=make_problem(state,tangent,common['data0'],float(common['scale']),float(anchor['lambda_total']),ell)
    ctx.material_gauge=gauge
    setup=time.perf_counter()-started
    prior_config=json.loads((ROOT/'inputs/original_resolved_config.json').read_text())
    prior=float(prior_config['prior']);lm=ctx.lam-prior
    if lm<0:raise ValueError('saved lambda below original prior')
    metadata=dict(prior_precision=prior,lm_mu=lm,lm_metric='identity in saved physical orthonormal material coordinates',prior_source_hash=sha(ROOT/'inputs/original_resolved_config.json'),physical_metric_hash=gauge['basis_fingerprint'],data_whitening_hash=array_hash(np.asarray(common['scale'])),data_whitening_scope='original measured-data scale normalization, not calibrated noise covariance',nuisance_policy='none; no new nuisance projection')
    return dict(information_metadata=metadata,scene=scene,common=common,anchor=anchor,saved=saved,state=state,ctx=ctx,ev=ev,tangent=tangent,full_step=full_step,physical_setup_wall_s=setup,inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},state_hash=array_hash(anchor['chi']),phase=phase,iteration=it)

def public_workspace(bundle):
    ctx=bundle['ctx'];cfg=SelectorConfig(seed=20260930,pool_count=32,residual_up_count=16,random_pool_count=10)
    started=time.perf_counter()
    anchor,anchor_meta=build_independent_anchor(ctx,64,seed=cfg.seed)
    t=time.perf_counter();pool=build_candidates(ctx,anchor,cfg);pool_time=time.perf_counter()-t
    w=prepare_selector_workspace(ctx,pool,anchor,chunk_size=16)
    cctx=w.ctx
    # Paid extraction of cached public maps only: identity is CURRENT, not material.
    eye=np.eye(cctx.n_current,dtype=np.complex128)
    L=cctx.apply_L(eye);S=cctx.apply_S(eye);PB=cctx.project_B(eye)
    return dict(workspace=w,L=L,S=S,PB=PB,anchor_meta=anchor_meta,pool_wall_s=pool_time,workspace_wall_s=time.perf_counter()-started,pool_hash=array_hash(pool.vectors),full_J_H_truth_access=False)

class EvaluatorContext:
    """Offline evaluation: never delivered to SelectorContext or its scorer."""
    def __init__(self,bundle):
        ctx=bundle['ctx'];ev=bundle['ev'];s=bundle['full_step'];start=time.perf_counter()
        with ctx.phase('offline_reference_verification'):
            Js=ev.j(s);h=ev.jt(ctx.r)+ctx.ell
            gres=ev.jt(ctx.r+Js)+ctx.lam*s+ctx.ell
        rel=float(la.norm(gres)/max(la.norm(h),1e-300))
        if rel>1e-10:raise ValueError('saved full reference fails new true residual: '+str(rel))
        self.ctx,self.ev,self.s,self.Js,self.gres=ctx,ev,s,Js,gres
        self.relative_residual=rel
        phi=float(.5*la.norm(ctx.r+Js)**2+ctx.ell@s+.5*ctx.lam*(s@s))
        self.floor=float(64*np.finfo(float).eps*max(.5*(ctx.r@ctx.r),abs(phi),.5*ctx.lam*(s@s),Js@Js,1e-300)+.5*(gres@gres)/ctx.lam)
        self.H_reference_energy=float(Js@Js+ctx.lam*(s@s))
        self.wall_s=time.perf_counter()-start
        self.J=None;self.dense_J_wall_s=0.;self.dense_J_tangent_rhs=0
    def risk(self,s,Js=None):
        ctx=self.ctx;ds=s-self.s
        jd=self.ev.j(ds) if Js is None else Js-self.Js
        energy=float(.5*(jd@jd+ctx.lam*(ds@ds)))
        cross=float(self.gres@ds)
        return dict(full_gap=energy+cross,half_H_step_energy=energy,reference_gradient_cross=cross,numerical_floor=self.floor,H_step_error=float(np.sqrt(max(0,2*energy))),relative_H_step_error=float(np.sqrt(2*energy/self.H_reference_energy)) if self.H_reference_energy>2*self.floor else None,full_reference_H_energy=self.H_reference_energy,reference_relative_residual=self.relative_residual)
    def offline_gaussian_J(self):
        if self.ctx.dim_material>128:return None
        if self.J is None:
            t=time.perf_counter()
            with self.ctx.phase('offline_teacher_dense_Gaussian_J'):
                self.J=self.ev.dense_j()
            self.dense_J_wall_s=time.perf_counter()-t;self.dense_J_tangent_rhs=self.ctx.P*self.ctx.dim_material
        return self.J

class ProbeContext:
    """Paid directional action capability only; no full step/J/truth accessor."""
    __slots__=('action','P','counts','wall_s')
    def __init__(self,action,P):self.action=action;self.P=P;self.counts=0;self.wall_s=0.
    def j(self,Z):
        start=time.perf_counter();self.counts+=self.P*(1 if Z.ndim==1 else Z.shape[1])
        result=self.action(Z);self.wall_s+=time.perf_counter()-start;return result

class TeacherContext:
    """Explicitly offline directional labels; not accessible to online proposal."""
    __slots__=('action','J','P','counts','wall_s')
    def __init__(self,action,P,J=None):self.action=action;self.J=J;self.P=P;self.counts=0;self.wall_s=0.
    def j(self,Z):
        start=time.perf_counter()
        if self.J is None:
            self.counts+=self.P*(1 if Z.ndim==1 else Z.shape[1]);result=self.action(Z)
        else:result=self.J@Z
        self.wall_s+=time.perf_counter()-start;return result
