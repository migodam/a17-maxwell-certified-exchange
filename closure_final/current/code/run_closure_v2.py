"""Bounded closure state driver; resource watchdog is owned by root.

No online full reference/adjoint/truth/evaluation-floor acquisition. Evaluation
starts only after all online k policies are durably saved for this state.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import traceback
import numpy as np
import closure_runtime as runtime


def synchronize(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()


def validate_locked_config(args):
    config=json.loads(Path(args.config).read_text())
    receipt=runtime.check_config_sources(config,args.config)
    if args.mode not in config.get('permitted_modes',[]):raise ValueError('mode not yet permitted by pre-result lock')
    constants=dict(phases=dict(early=0,middle=3,late=17),k=[4,8,16],online_incoming_count=12,
                   max_finalists_per_round=2,max_actions=3,anchor_rank=64,pool_count=32,
                   residual_up_count=16,random_pool_count=10,pool_and_anchor_seed=20260930,
                   master_seed=20261002,cuda_candidate_batch=16,cpu_threads=4,
                   current_stability_relative_min=1e-10,endpoint_normal_relative_residual_actual=1e-10,
                   reference_relative_residual=1e-10,online_tolerance_schema='a17.online.numerical_significance.v1',
                   online_tolerance_safety=32,max_job_s=7200,memory_warning=.8,memory_stop=.9)
    for key,value in constants.items():
        if config.get(key)!=value:raise ValueError('frozen config differs: '+key)
    mode_key=dict(replay='replay_objects',validation='validation_objects',noise='noise_objects')[args.mode]
    if args.object not in config.get(mode_key,[]):raise ValueError('object outside locked '+mode_key)
    if args.phase not in config['phases'] or any(k not in config['k'] for k in args.k) or len(set(args.k))!=len(args.k):
        raise ValueError('phase/k outside frozen config')
    if config.get('prior_paid_occupation_s')!=4889.515 or config.get('cumulative_limit_s')!=43200:
        raise ValueError('closure budget cannot reset/expand')
    manifest=Path(args.input_manifest) if args.input_manifest else runtime.ROOT/'inputs/INPUT_MANIFEST.json'
    if config.get('input_manifest_sha256')!=runtime.sha(manifest):
        raise ValueError('config must hash-bind new input manifest')
    if args.mode=='noise':
        if config.get('noise_percent')!=[1,3] or config.get('noise_realizations')!=3:
            raise ValueError('locked noise matrix differs')
        if args.phase!='middle' or args.noise_basis_points not in (100,300) or args.realization_index not in (0,1,2):
            raise ValueError('outside locked noise matrix')
    holdout=runtime.ROOT/'HOLDOUT_OBJECT_MANIFEST.json'
    if config.get('holdout_manifest_sha256')!=runtime.sha(holdout):
        raise ValueError('config must bind pre-result holdout manifest')
    declared=json.loads(holdout.read_text())
    eligible=[row['object_id'] if isinstance(row,dict) else row for row in declared['eligible_objects']]
    if sorted(config['validation_objects'])!=sorted(eligible) or config['noise_objects']!=declared['noise_objects']:
        raise ValueError('config object sets differ from pre-result manifest')
    if config['replay_objects']!=[2001,2007,2012,2014]:raise ValueError('replay objects differ from original declared fixtures')
    return config,receipt


def run(args):
    runtime.initialize(args.baseline_root,args.baseline_lock)
    locked_config,lock_receipt=validate_locked_config(args)
    from maxwell_state import frozen, public_workspace, EvaluatorContext, array_hash
    from run_exchange import clean, jswrite
    from removal_core_batch import CoreEndpointBatch
    from closure_state_v2 import load_state, persisted_workspace, load_evaluation_reference
    from verified_exchange import online_exchange, DirectionAction
    out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    jswrite(out/'status.json',dict(status='RUNNING',phase='setup'))
    config_receipt=dict(cli=vars(args),baseline_config_sha256=runtime.sha(runtime.BASELINE/'A17_RUN_CONFIG.json'),
                        closure_config_sha256=runtime.sha(args.config),lock_receipt=lock_receipt)
    jswrite(out/'run_config.json',config_receipt)
    if args.mode == 'replay':
        if args.noise_basis_points or args.inputs_root or args.input_manifest:
            raise ValueError('replay uses unchanged old inputs and zero noise')
        bundle=frozen(args.object,args.phase,args.device)
        bundle['noise']=dict(relative_norm=0.,basis_points=0,realization_index=0)
        persisted=Path(args.persisted).resolve() if args.persisted else runtime.BASELINE/'results'/f'object_{args.object}_{args.phase}'
        if not persisted.exists() and args.object==2007 and args.phase=='early':
            persisted=runtime.BASELINE/'results/object_2007_early_portable'
        # The failed original early start exists but has no public workspace.
        if not (persisted/'public_workspace.npz').exists() and args.object==2007 and args.phase=='early' and not args.persisted:
            persisted=runtime.BASELINE/'results/object_2007_early_portable'
        permitted=runtime.BASELINE/'results'/f'object_{args.object}_{args.phase}'
        if args.object==2007 and args.phase=='early':permitted=runtime.BASELINE/'results/object_2007_early_portable'
        if persisted.resolve()!=permitted.resolve():raise ValueError('replay workspace outside locked existing fixture')
        fixture_path=runtime.ROOT/'research/REPLAY_FIXTURE_MANIFEST.json'
        if locked_config.get('replay_fixture_manifest_sha256')!=runtime.sha(fixture_path):
            raise ValueError('replay fixture manifest lock mismatch')
        fixtures=json.loads(fixture_path.read_text())['fixtures']
        t=time.perf_counter();work=persisted_workspace(bundle,persisted,fixtures[f'object_{args.object}_{args.phase}'])
        work['cache_load_adapter_wall_s']=time.perf_counter()-t
    else:
        if not args.inputs_root or not args.input_manifest:
            raise ValueError('validation/noise needs explicit new inputs and manifest')
        if args.mode=='noise' and (args.phase!='middle' or args.noise_basis_points not in (100,300)):
            raise ValueError('noise is locked middle state at1/3 percent')
        if args.mode=='validation' and args.noise_basis_points:
            raise ValueError('validation is noise-free; explicit noise mode required')
        phases=locked_config['phases']
        bundle=load_state(args.inputs_root,args.object,args.phase,args.device,args.input_manifest,
                          phases=phases,noise_basis_points=args.noise_basis_points,
                          realization_index=args.realization_index)
        # Noise is already in ctx.r BEFORE residual-dependent anchor/pool generation.
        work=public_workspace(bundle)
    ctx=bundle['ctx'];w=work['workspace'];model=bundle['state'].model
    synchronize(args.device)
    engine_device='cuda' if args.device=='cuda' and ctx.q>=256 else 'cpu'
    engine=CoreEndpointBatch(work['L'],work['S'],work['PB'],ctx.r,ctx.ell,ctx.lam,
                            device=engine_device,batch_size=16)
    np.savez_compressed(out/'public_workspace.npz',Q=w.Q,pool=w.pool.vectors,anchor=w.anchor.U,
          L=work['L'],S=work['S'],PB=work['PB'],labels=np.asarray(w.pool.labels),families=np.asarray(w.pool.families),
          r=ctx.r,ell=ctx.ell,lambda_total=ctx.lam)
    if 'values' in bundle['noise']:
        np.savez_compressed(out/'measurement_noise.npz',epsilon=bundle['noise']['values'])
    setup=dict(physical_state_wall_s=bundle['physical_setup_wall_s'],endpoint_engine_setup_wall_s=engine.setup_wall_s,
               public_workspace_wall_s=work.get('workspace_wall_s'),cache_load_adapter_wall_s=work.get('cache_load_adapter_wall_s'),
               anchor_metadata=work.get('anchor_meta'),pool_wall_s=work.get('pool_wall_s'),
               projection_metadata=getattr(w,'metadata',None),engine_device=engine_device,
               independently_measured_standalone_cold_wall_s=None,
               shared_fresh_setup_wall_s=None if args.mode=='replay' else bundle['physical_setup_wall_s']+work['workspace_wall_s']+engine.setup_wall_s,
               scope='Physical and public workspace outer intervals; anchor/pool/projection are nested; replay is persisted warm workspace',
               physical_setup_counters=copy.deepcopy(model.counters.as_dict()))
    jswrite(out/'setup_cost.json',clean(setup))
    receipt=dict(runtime=runtime.receipt(),inputs=bundle['inputs'],input_manifest_sha256=bundle.get('input_manifest_sha256'),
                 state_hash=bundle['state_hash'],pool_hash=work['pool_hash'],coefficient_pool_hash=array_hash(w.pool.vectors),material_gauge=ctx.material_gauge,
                 persisted_hashes=work.get('persisted_hashes'),config=config_receipt,
                 noise={key:value for key,value in bundle['noise'].items() if key!='values'},
                 actual_dtype='complex128/float64',device=args.device,
                 online_controller_reference_access=False,online_fullJ_cache=False,
                 preparation_reference_access=args.mode=='replay',
                 preparation_scope='Old offline regression fixture loads saved step' if args.mode=='replay' else 'Deployment preparation does not open saved reference step')
    jswrite(out/'runtime_receipt.json',clean(receipt))
    action=DirectionAction(bundle['ev'].j,ctx.P,synchronize=lambda:synchronize(args.device),
                           counter_supplier=model.counters.as_dict)
    online=[];endpoints=[]
    for k in args.k:
        jswrite(out/'status.json',dict(status='RUNNING',phase='online',k=k,completed_k=[r['requested_k'] for r in online]))
        identity=dict(object_id=args.object,phase=args.phase,mode=args.mode,k=k,
                      family=bundle['scene']['family'],material_basis=bundle['scene']['representation'],
                      state_hash=bundle['state_hash'],pool_hash=work['pool_hash'],
                      noise_basis_points=args.noise_basis_points,realization_index=args.realization_index)
        record,endpoint=online_exchange(w.ctx,w,engine,w.anchor,action,k,out/f'k{k}',identity)
        record['reconstructed_standalone_fee_wall_s']=(setup['shared_fresh_setup_wall_s']+record['wall_online_total_s']) if setup['shared_fresh_setup_wall_s'] is not None else None
        record['standalone_cold_wall_s']=None
        record['cost_scope']='Same shared acquisition reconstructed standalone fee; not independently rerun cold policy'
        jswrite(out/f'k{k}'/'online_result.json',clean(record))
        online.append(record);endpoints.append(endpoint)
        print(json.dumps(dict(event='online_cell_completed',object=args.object,phase=args.phase,k=k,
                              accepted=len(record['accepted']),stop=record['stop_reason'])),flush=True)
    jswrite(out/'status.json',dict(status='RUNNING',phase='offline_evaluation',completed_k=args.k))
    # Only now acquire/evaluate full-reference quantities. No online action uses
    # Gaussian dense_J, and the full physical J action never changes callback.
    offline_start=time.perf_counter();offline_before=copy.deepcopy(model.counters.as_dict())
    if args.mode!='replay':
        evaluation_input=load_evaluation_reference(bundle)
        jswrite(out/'offline_reference_input_receipt.json',evaluation_input)
    reference_record=dict(source='original_saved_full_step',new_reference_solve=False)
    if args.mode=='noise':
        from run_reference import solve_reference
        saved=bundle['full_step'].copy()
        reference,gradient,ref_info=solve_reference(ctx,bundle['ev'],initial=saved)
        bundle['full_step']=reference
        reference_record=dict(source='existing_solve_reference_after_online',new_reference_solve=True,
                              solver=ref_info,original_step_is_offline_warm_start_only=True)
        np.savez_compressed(out/'offline_reference.npz',step=reference,gradient=gradient,original_saved_step=saved)
    evaluator=EvaluatorContext(bundle)
    evaluated=[]
    for online_record,endpoint in zip(online,endpoints):
        k=online_record['requested_k']
        rec=dict(k=k,receiver_risk=evaluator.risk(endpoint['initial_step'],endpoint['initial_Js']),
                 final_risk=evaluator.risk(endpoint['step'],endpoint['Js']),
                 reference_relative_residual=evaluator.relative_residual,
                 evaluation_floor=evaluator.floor,reference=reference_record)
        if args.mode=='noise':
            rec['original_saved_full_step_risk_under_noisy_objective']=evaluator.risk(saved)
        dc=bundle['tangent'].expand(endpoint['step']);chi=bundle['anchor']['chi']+dc
        truth=bundle['common'].get('truth')
        rec['one_step_truth']=dict(relative_material_error=float(np.linalg.norm(chi-truth)/np.linalg.norm(truth)) if truth is not None else None,
                constraints_violated=bool(chi.imag.min()<-1e-8 or chi.real.min()<-.50000001),
                scope='Offline unconstrained alpha1 diagnostic; not nonlinear accepted update')
        rec['final_nonlinear_reconstruction']=dict(status='NOT_RUN',reason='Separate owner bounded nonlinear transfer stage')
        jswrite(out/f'k{k}'/'offline_evaluation.json',clean(rec));evaluated.append(rec)
    if args.offline_dense_j:
        J=evaluator.offline_gaussian_J()
        if J is not None:np.savez_compressed(out/'offline_information_reference.npz',fullJ=J,scope=np.asarray('OFFLINE_ONLY_AFTER_ONLINE'))
    result=dict(status='COMPLETED',mode=args.mode,object_id=args.object,phase=args.phase,completed_k=args.k,
          online=online,evaluation=evaluated,setup=setup,reference=reference_record,
          offline_evaluation_wall_s=time.perf_counter()-offline_start,
          physical_offline_counters_before=offline_before,physical_offline_counters_after=model.counters.as_dict(),
          context_cost=ctx.records(),wall_total_s=time.perf_counter()-started,final_runtime=runtime.receipt())
    jswrite(out/'result.json',clean(result));jswrite(out/'status.json',dict(status='COMPLETED',completed_k=args.k,wall_total_s=result['wall_total_s']))
    return result


def parser():
    p=argparse.ArgumentParser();p.add_argument('--baseline-root',required=True);p.add_argument('--baseline-lock')
    p.add_argument('--mode',choices=['replay','validation','noise'],required=True)
    p.add_argument('--object',type=int,required=True);p.add_argument('--phase',choices=['early','middle','late'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--k',type=int,nargs='+',default=[4,8,16])
    p.add_argument('--out',required=True);p.add_argument('--inputs-root');p.add_argument('--input-manifest');p.add_argument('--persisted')
    p.add_argument('--config',required=True);p.add_argument('--noise-basis-points',type=int,choices=[0,100,300],default=0)
    p.add_argument('--realization-index',type=int,choices=[0,1,2],default=0);p.add_argument('--offline-dense-j',action='store_true')
    return p


if __name__=='__main__':
    p=parser();args=p.parse_args()
    if any(k not in (4,8,16) for k in args.k) or len(args.k)!=len(set(args.k)):
        p.error('distinct k values from4/8/16 required')
    destination=Path(args.out).expanduser().resolve()
    if destination.exists():p.error('new output directory required; no overwrite')
    try:run(args)
    except Exception as exc:
        destination.mkdir(parents=True,exist_ok=True)
        record=dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),object_id=args.object,phase=args.phase)
        for name in ['failure.json','status.json']:
            (destination/name).write_text(json.dumps(record,indent=2)+'\n')
        raise
