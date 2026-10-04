"""Pinned frozen-route instrumentation only. Importing this module runs no physics.

Write a hash-bound config first, then root separately authorizes execution.
Transparent wrappers change measurement/synchronization, not numerical algorithms.
Never report this instrumented process as an uninstrumented pinned receipt.
"""
import argparse
from contextlib import contextmanager
import copy
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

VERSION = 'a17.eff.frozen_exclusive_profile.v1'
FIXED = dict(objects=[2002, 2009], phase='middle', iteration=3, k=8,
             diagnostic_blocks=2, precision=['float64', 'complex128'],
             nonlinear_updates=0, shared_full_gradient=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


class ExclusiveProfiler:
    """Nested spans partition inclusive time, never add inclusive parent+child."""
    def __init__(self, synchronize=lambda: None, clock=time.perf_counter):
        self.sync, self.clock = synchronize, clock
        self.stack, self.rows = [], []

    @contextmanager
    def span(self, label):
        self.sync()
        node = dict(label=label, path=[x['label'] for x in self.stack]+[label],
                    start=self.clock(), children_s=0., status='OK')
        self.stack.append(node)
        try:
            yield
        except BaseException:
            node['status'] = 'ERROR'; raise
        finally:
            self.sync(); node['end']=self.clock()
            inclusive=node['end']-node['start']; exclusive=inclusive-node.pop('children_s')
            if exclusive < -1e-10:
                raise RuntimeError('overlapping child spans invalidate exclusive accounting')
            node.update(inclusive_wall_s=inclusive, exclusive_wall_s=max(0.,exclusive))
            self.stack.pop()
            if self.stack:self.stack[-1]['children_s']+=inclusive
            self.rows.append(node)

    def call(self, label, callback, *args, **kwargs):
        with self.span(label):return callback(*args, **kwargs)

    def report(self, actual_wall_s):
        if self.stack:raise RuntimeError('cannot report open spans')
        measured=sum(r['exclusive_wall_s'] for r in self.rows)
        if measured > actual_wall_s+1e-8:
            raise RuntimeError('exclusive times exceed actual elapsed wall')
        grouped={}
        for row in self.rows:
            key='/'.join(row['path']); old=grouped.setdefault(key,dict(calls=0,exclusive_wall_s=0.,inclusive_wall_s=0.))
            old['calls']+=1;old['exclusive_wall_s']+=row['exclusive_wall_s'];old['inclusive_wall_s']+=row['inclusive_wall_s']
        return dict(actual_wall_s=actual_wall_s,exclusive_partition_wall_s=measured,
                    unattributed_wall_s=max(0.,actual_wall_s-measured),groups=grouped,spans=self.rows,
                    aggregation_rule='Only exclusive fields partition work. Inclusive fields overlap ancestors; never sum them.',
                    synchronization_scope='Entry sync belongs to enclosing span/unattributed; exit sync belongs to active span.')


class Hooks:
    """Reversible transparent wrappers; explicit instrumentation provenance."""
    def __init__(self, profiler):self.profiler=profiler;self.originals=[];self.inventory=[]
    def wrap(self, owner, name, label):
        original=getattr(owner,name)
        @functools.wraps(original)
        def observed(*a,**kw):return self.profiler.call(label,original,*a,**kw)
        self.originals.append((owner,name,original));setattr(owner,name,observed)
        self.inventory.append(dict(owner=getattr(owner,'__name__',type(owner).__name__),attribute=name,label=label))
    def restore(self):
        for owner,name,original in reversed(self.originals):setattr(owner,name,original)
        self.originals=[]


def source_binding(args):
    closure=Path(args.closure_root).resolve();baseline=Path(args.baseline_root).resolve()
    return dict(version=VERSION,fixed=FIXED,profile_source_sha256=sha(__file__),
                closure_sources={p.name:sha(p) for p in sorted((closure/'code').glob('*.py'))},
                baseline_manifest_sha256=sha(baseline/'SOURCE_MANIFEST.json'),
                baseline_lock_sha256=sha(args.baseline_lock),input_manifest_sha256=sha(args.input_manifest))


def validate_binding(args):
    actual=source_binding(args);declared=json.loads(Path(args.config).read_text())
    if declared!=actual:raise ValueError('profile config/source/input binding drift')
    if args.object not in FIXED['objects'] or args.device not in ('cpu','cuda'):
        raise ValueError('fixed pilot objects and explicit CPU/CUDA only')
    if Path(args.closure_root).resolve()!=Path(__file__).resolve().parents[1]:
        raise ValueError('closure-root must be this isolated efficiency runtime with exact source copies')
    return actual


def safe_runtime(runtime, baseline, closure):
    receipt=runtime.receipt(); baseline=Path(baseline).resolve();closure=Path(closure).resolve()
    imported={}
    for name,row in receipt['actual_imports'].items():
        path=Path(row['actual_path']).resolve()
        root=baseline if path.is_relative_to(baseline) else closure
        imported[name]=dict(bundle='baseline' if root==baseline else 'closure',
                            relative_path=path.relative_to(root).as_posix(),sha256=row['sha256'])
    env=receipt['baseline']
    return dict(actual_imports=imported,
                environment={k:env.get(k) for k in ['python','numpy','scipy','torch','cuda_available','cuda_version','gpu','threads']},
                baseline_manifest_sha256=receipt['baseline_manifest_sha256'],
                baseline_lock_sha256=receipt['baseline_lock_sha256'],
                closure_sources=receipt['closure_sources'],
                callable_instrumentation_installed=True,numerical_algorithm_change=False,
                receipt_scope='Pinned file hashes plus separately declared in-process observation wrappers; not an uninstrumented run.')


def diagnostic_j(bundle, Z, profiler):
    """Separately paid equivalent ORIGINAL action, including full current download.

    No optimized observation path is used. Counter/timer labels match full tangent
    work; extra exclusive spans diagnose stages and do not replace total timing.
    """
    import numpy as np
    from scipy import linalg as la
    from pairtsom_adapter import stack_real_illuminations
    state=bundle['state'];ctx=bundle['ctx'];model=state.model;solver=model.solver;c=model.counters
    z=np.asarray(Z)
    if z.dtype!=np.float64 or np.iscomplexobj(z) or z.ndim not in (1,2) or z.shape[0]!=ctx.dim_material or not np.isfinite(z).all():
        raise ValueError('diagnostic input requires finite original real FP64 material gauge')
    B=1 if z.ndim==1 else z.shape[1]
    if B<1:raise ValueError('do not charge an empty direction block')
    tfield=time.perf_counter()
    with profiler.span('expansion'):
        dc=bundle['tangent'].expand(z)
    tjvp=time.perf_counter();c.bump('jvp_calls');ctx.ledger.add('full_tangent_rhs',ctx.P*B)
    with profiler.span('injection'):
        rhs,ncols,squeezed=state._tangent_rhs(dc)
    if ncols!=B:raise ValueError('probe shape mismatch')
    with profiler.span('RHS_pack'):
        R=rhs.transpose(1,0,2).reshape(model.n,ctx.P*B)
    with c.timed('solve_s'):
        c.bump('solve_calls');c.bump('solve_rhs',ctx.P*B)
        c.bump('solve_rhs_full_tangent',ctx.P*B);c.bump('solve_rhs_full',ctx.P*B)
        if model.device=='cuda':
            with profiler.span('RHS_upload'):device_rhs=solver._to_device(R)
            with profiler.span('stored_LU_multi_RHS_solve'):
                Xdev=solver.torch.linalg.lu_solve(state._L_factor[1],state._L_factor[2],device_rhs)
            with profiler.span('full_current_download'):X=solver._from_device(Xdev)
            del device_rhs,Xdev
        else:
            with profiler.span('stored_LU_multi_RHS_solve'):X=la.lu_solve(state._L_factor[1:],R,trans=0)
    with profiler.span('current_unpack'):
        K=X.reshape(model.n,ctx.P,B).transpose(1,0,2)
    c.add_time('jvp_s',time.perf_counter()-tjvp);c.bump('jvp_rhs',ctx.P*B)
    with profiler.span('GS_observation'):
        y=np.einsum('oc,pcb->pob',model.GS,K)
    c.add_time('field_jvp_s',time.perf_counter()-tfield)
    with profiler.span('scale_and_realpack'):
        result=stack_real_illuminations((y[:,:,0] if squeezed else y)/float(bundle['common']['scale']))
    return result,dict(material_directions=B,physical_rhs=ctx.P*B,
                       upload_bytes=R.nbytes if model.device=='cuda' else 0,
                       current_download_bytes=X.nbytes if model.device=='cuda' else 0,
                       optimized=False)


def run(args):
    binding=validate_binding(args);out=Path(args.out).resolve()
    if out.exists():raise ValueError('fresh output directory required')
    protected=[Path(args.closure_root)/'code',Path(args.baseline_root),Path(args.inputs_root)]
    inputs=Path(args.inputs_root).resolve()
    if inputs.name.lower()=='inputs' and inputs.parent!=Path(args.closure_root).resolve():
        protected.append(inputs.parent)  # Original closure root remains read-only.
    for root in protected:
        if out.is_relative_to(Path(root).resolve()):raise ValueError('profiling output must not modify pinned source/input roots')
    out.mkdir(parents=True);started=time.perf_counter();model=None;hooks=None;profile=None
    sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(args.closure_root).resolve()/'code'))
    def synchronize():
        if args.device=='cuda':
            import torch
            torch.cuda.synchronize()
    def peak():
        if args.device!='cuda':return None
        import torch
        return dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    try:
        import closure_runtime as runtime
        runtime.initialize(args.baseline_root,args.baseline_lock)
        import numpy as np
        import maxwell_state as ms
        import operators as op
        import selectors_current as sc
        import selector_coordinates as co
        import cair
        import run_exchange as re
        import verified_exchange as ve
        from closure_state import load_state
        from removal_core_batch import CoreEndpointBatch
        from endpoint_batch import EndpointBatch
        if args.device=='cuda':
            import torch
            torch.cuda.reset_peak_memory_stats()
        profile=ExclusiveProfiler(synchronize);hooks=Hooks(profile)
        # Aliases are independently wrapped, always delegating to original bodies.
        for owner,name,label in [
            (ms,'build_independent_anchor','anchor_build'),(ms,'build_candidates','pool_build'),
            (ms,'prepare_selector_workspace','prepare_workspace'),(sc,'fourier_modes','fourier_modes'),
            (sc,'orth','selector_QR'),(sc,'_model','anchor_reduced_model'),
            (sc,'anchor_snapshots','anchor_snapshots'),(sc,'normalized_pool','pool_normalization'),
            (co,'orth','workspace_QR'),(op,'domain_bank','domain_bank_geometry'),
            (cair,'domain_bank','domain_bank_geometry'),(op,'orth','operator_QR'),
            (op.ReducedModel,'__init__','reduced_model_construction'),
            (re.Neighborhood,'endpoints','candidate_endpoints'),(re,'gains_on_anchor','anchor_score'),
            (ve,'directional_check','online_acceptance_arithmetic'),
            (EndpointBatch,'__init__','engine_common_setup'),(EndpointBatch,'evaluate','endpoint_evaluate'),
            (CoreEndpointBatch,'prepare_removal_core','removal_core_prepare'),
            (CoreEndpointBatch,'evaluate_incoming','incoming_endpoint_batch')]:hooks.wrap(owner,name,label)
        # la.qr/svd functions remain original algorithms; identify internal costs
        # through their parent path rather than attributing nested time twice.
        from scipy import linalg as la
        hooks.wrap(la,'qr','scipy_QR');hooks.wrap(la,'svd','scipy_SVD')
        bundle=profile.call('pinned_physical_state_load',load_state,args.inputs_root,args.object,'middle',args.device,args.input_manifest)
        model=bundle['state'].model;ctx=bundle['ctx'];ev=bundle['ev']
        phase_counters=[]
        def counter_snapshot():
            value=model.counters.as_dict()
            return {k:copy.deepcopy(value[k]) for k in ['totals','timings_s','event_count']}
        @contextmanager
        def major_phase(label):
            before=counter_snapshot()
            try:
                with profile.span(label):yield
            finally:
                after=counter_snapshot()
                phase_counters.append(dict(phase=label,before=before,after=after,
                    counter_delta={k:after['totals'].get(k,0)-before['totals'].get(k,0) for k in set(before['totals'])|set(after['totals'])}))
        hooks.wrap(ctx,'_internal','internal_modes');hooks.wrap(ctx,'_receiver','receiver_modes')
        for name,label in [('_L','physical_L_product'),('_S','physical_S_product'),('_B','physical_B_projection')]:hooks.wrap(ctx,name,label)
        hooks.wrap(bundle['tangent'],'expand','material_expansion_original')
        hooks.wrap(bundle['state'],'_tangent_rhs','tangent_injection_original')
        hooks.wrap(model.solver,'_to_device','solver_upload_original')
        hooks.wrap(model.solver,'_from_device','solver_download_original')
        hooks.wrap(ev,'j','original_full_J_action');hooks.wrap(ev,'jt','original_full_adjoint')
        initial_counters=copy.deepcopy(model.counters.as_dict())
        with major_phase('shared_full_gradient'):
            with ctx.phase('efficiency_profile_shared_full_gradient'):h=ev.jt(ctx.r)+ctx.ell
        # Receiver-first records cold receiver SVD cost explicitly; subsequent
        # workspace uses the paid state-local receiver cache. No cold-policy ratio.
        with major_phase('receiver_endpoint_shared_state'):
            U=op.orth(ctx.receiver_modes(8),rank=8)
            if U.shape[1]!=8:raise ValueError('receiver rank differs')
            with profile.span('receiver_public_maps'):
                L=U.conj().T@ctx.apply_L(U);S=ctx.apply_S(U);PB=ctx.project_B(U)
            engine_device='cuda' if args.device=='cuda' and ctx.q>=256 else 'cpu'
            receiver_engine=CoreEndpointBatch(L,S,PB,ctx.r,ctx.ell,ctx.lam,device=engine_device,batch_size=16)
            receiver=receiver_engine.evaluate(np.eye(8,dtype=np.complex128)[None])
            if receiver['status']!=['OK']:raise ValueError('receiver endpoint failed')
        del receiver_engine
        with major_phase('original_public_workspace_acquisition'):
            work=ms.public_workspace(bundle)
        w=work['workspace']
        with major_phase('exchange_engine_common_maps_setup'):
            engine=CoreEndpointBatch(work['L'],work['S'],work['PB'],ctx.r,ctx.ell,ctx.lam,device=engine_device,batch_size=16)
        captured=[]
        def original_action(z):
            label='guarded_Jx' if not captured else 'finalist_Jd'
            with profile.span(label):result=ev.j(z)
            captured.append((z.copy(),result.copy()));return result
        action=ve.DirectionAction(original_action,ctx.P,synchronize=synchronize,counter_supplier=model.counters.as_dict)
        with major_phase('original_frozen_k8_online_exchange'):
            policy,endpoint=ve.online_exchange(w.ctx,w,engine,w.anchor,action,8,out/'exchange',
                dict(object_id=args.object,state='middle',state_hash=bundle['state_hash'],pool_hash=work['pool_hash']))
        policy_end_counters=copy.deepcopy(model.counters.as_dict());diagnostics=[]
        # Only compare directions actually generated by the completed original
        # causal run. Replayed diagnostic work is separately charged, never free.
        for index,(z,reference) in enumerate(captured[:FIXED['diagnostic_blocks']]):
            before=copy.deepcopy(model.counters.as_dict())
            with major_phase('paid_original_equivalent_J_diagnostic'):
                actual,cost=diagnostic_j(bundle,z,profile)
            error=float(np.linalg.norm(actual-reference));scale=float(np.linalg.norm(reference))
            relative=error/max(scale,1e-300)
            if not np.allclose(actual,reference,rtol=1e-12,atol=1e-13):raise ValueError('split original action differs from pinned original')
            diagnostics.append(dict(captured_action=index,absolute_error=error,relative_error=relative,cost=cost,
                                    counters_before=before,counters_after=copy.deepcopy(model.counters.as_dict()),
                                    check_scope='FP64 numerical equality check, not a deterministic output certificate'))
        np.savez_compressed(out/'profile_arrays.npz',gradient=h,receiver_step=receiver['steps'][0],exchange_step=endpoint['step'],
                            chi=bundle['anchor']['chi'],material_tangent_fingerprint=np.asarray(ctx.material_gauge['basis_fingerprint']))
        precision={key:str(getattr(value,'dtype',None)) for key,value in dict(
            chi=bundle['state'].chi,polarizability_derivative=bundle['state'].da,
            exciting=bundle['state'].exciting,current=bundle['state'].current,
            GS=model.GS,Goff=model.Goff,LU=bundle['state']._L_factor[1],
            gradient=h,receiver_step=receiver['steps'][0],exchange_step=endpoint['step']).items()}
        for key,dtype in precision.items():
            required='float64' if key in ('gradient','receiver_step','exchange_step') else 'complex128'
            if dtype not in (required,'torch.'+required):raise ValueError('FP64 invariant failed: '+key)
        try:
            import psutil
            host_memory=psutil.Process().memory_info()
            host_memory=dict(rss_bytes=host_memory.rss,peak_wset_bytes=getattr(host_memory,'peak_wset',None),
                             scope='Final process RSS; OS peak working set only where provided. External watchdog remains occupation/memory authority.')
        except ImportError:
            host_memory=dict(rss_bytes=None,peak_wset_bytes=None,scope='psutil unavailable; external watchdog record required; no dependency installed')
        synchronize();actual_wall=time.perf_counter()-started
        record=dict(status='PROFILE_COMPLETED',version=VERSION,binding=binding,
            object_id=args.object,phase='middle',iteration=bundle['iteration'],FP64=True,
            precision_evidence=precision,host_memory=host_memory,
            instrumentation=dict(callable_wrappers=True,numerical_algorithm_change=False,hooks=hooks.inventory),
            geometry_cache_policy='Existing pinned state-local receiver/internal caches only; receiver-first shares paid SVD. No new geometry cache.',
            reference_step_access=False,nonlinear_updates=0,state_hash=bundle['state_hash'],gauge=ctx.material_gauge,
            input_receipts=bundle['inputs'],runtime=safe_runtime(runtime,args.baseline_root,args.closure_root),
            initial_counters=initial_counters,policy_end_counters=policy_end_counters,
            final_counters=copy.deepcopy(model.counters.as_dict()),context_cost=ctx.records(),
            phase_physical_counter_receipts=phase_counters,
            receiver_statistics=receiver['statistics'],policy_result=policy,diagnostic_checks=diagnostics,
            profile=profile.report(actual_wall),peak_gpu_bytes=peak(),
            standalone_policy_ratio_claim=False,
            timing_scope='Total ends before final report JSON serialization; arrays and exchange files are charged within total. Backend nested timers are audit views only.')
        write_json(out/'profile_result.json',record)
        print(json.dumps(dict(status='PROFILE_COMPLETED',object_id=args.object,out=str(out))))
    except BaseException as error:
        if profile is not None:
            elapsed=time.perf_counter()-started
            write_json(out/'profile_failure.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error),
                traceback=traceback.format_exc(),binding=binding,profile=profile.report(elapsed),
                physical_counters=model.counters.as_dict() if model is not None else None,
                peak_gpu_bytes=peak(),automatic_retry=False))
        raise
    finally:
        if hooks is not None:hooks.restore()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['closure-root','baseline-root','baseline-lock','inputs-root','input-manifest','config']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--object',type=int,choices=FIXED['objects']);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--out');parser.add_argument('--write-config',action='store_true');parser.add_argument('--check-config',action='store_true')
    args=parser.parse_args()
    if args.write_config:
        if Path(args.config).exists():raise ValueError('do not overwrite a predeclared config')
        write_json(args.config,source_binding(args));print('CONFIG_WRITTEN_NO_PHYSICS');return
    if args.object is None:parser.error('--object required')
    if args.check_config:
        validate_binding(args);print('CONFIG_VALIDATED_NO_PHYSICS');return
    if not args.out:parser.error('--out required for physical profile')
    run(args)


if __name__=='__main__':main()
