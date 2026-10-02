"""Explicit-root v2: shared fresh physical callback + persisted public maps.
Physical mode is for parent GPU execution only. Worker validation uses CPU test callback.
"""
from pathlib import Path
import argparse,os,sys,time,json,traceback,hashlib,copy,platform
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,data,sanitize):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(sanitize(data),indent=2,allow_nan=False)+'\n');tmp.replace(p)
def floor_from_receipts(persisted,explicit=None):
    paths=[Path(explicit)] if explicit else sorted(Path(persisted).glob('k*/receiver_no_swap.json'))
    if not paths:raise ValueError('existing common numerical_floor receipt required; no reference optimization permitted')
    floors=[];sources=[]
    for p in paths:
        r=json.loads(p.read_text());floor=r['risk']['numerical_floor']
        if not isinstance(floor,(int,float)) or not np.isfinite(floor) or floor<0:raise ValueError('invalid predeclared floor')
        floors.append(float(floor));sources.append(dict(path=str(p.absolute()),sha256=sha(p),field='risk.numerical_floor',value=floor))
    if len(set(floors))!=1:raise ValueError('common numerical floor receipts disagree; do not select by outcome')
    return floors[0],dict(value=floors[0],sources=sources,scope='Already recorded common reference_floor scalar only; no full optimal step/adjoint/fullJ acquired for ranking or recalculated.')

def initialize(args):
    r1=Path(args.r1_root).expanduser().resolve();os.environ['A17_MATCHED_R1_ROOT']=str(r1)
    if 'pinned_runtime' in sys.modules and Path(sys.modules['pinned_runtime'].ROOT).absolute()!=r1:raise RuntimeError('process already imported another R1; use fresh process')
    sys.path.insert(0,str(r1/'code'))
    import portable_cached_onepass_v2 as c
    if c.R1!=r1:raise RuntimeError('portable module root mismatch')
    p=Path(args.persisted).expanduser().absolute();load_start=time.perf_counter()
    with np.load(p/'public_workspace.npz',allow_pickle=False) as f:bank={k:f[k].copy() for k in f.files}
    floor,floor_source=floor_from_receipts(p,args.floor_receipt);publicload=time.perf_counter()-load_start
    args._setup_progress=dict(public_cache_load_wall_s=publicload,floor_source=floor_source,physical_model_setup_counters=None,physical_setup_scope='NOT_YET_CONSTRUCTED')
    physical_setup=None;bundle=None;reference=None
    if args.direction=='physical':
        if args.device!='cuda' or args.object is None or args.phase is None:raise ValueError('physical mode requires explicit --device cuda --object --phase')
        from maxwell_state import frozen,array_hash
        import torch
        def sync():torch.cuda.synchronize()
        sync();t=time.perf_counter();bundle=frozen(args.object,args.phase,'cuda');sync();physical_setup=time.perf_counter()-t
        ctx=bundle['ctx'];model=bundle['state'].model
        args._setup_progress.update(fresh_physical_state_setup_wall_s=physical_setup,physical_model_setup_counters=copy.deepcopy(model.counters.as_dict()),physical_setup_scope='FROZEN_STATE_CONSTRUCTED_BEFORE_CONTRACT_VALIDATION',state_hash=bundle['state_hash'])
        # Only frozen state and directional capability; EvaluatorContext/fullJ/step optimizer never called.
        if bank['PB'].shape[0]!=ctx.P or bank['r'].shape!=ctx.r.shape or bank['ell'].shape!=ctx.ell.shape or float(bank['lambda_total'])!=ctx.lam:raise ValueError('persisted public and frozen physical contracts differ')
        for name,a,b in [('r',bank['r'],ctx.r),('ell',bank['ell'],ctx.ell)]:
            if np.linalg.norm(a-b)>1e-9*max(np.linalg.norm(a),np.linalg.norm(b),1e-300):raise ValueError('physical '+name+' mismatch')
        recpath=p/'runtime_receipt.json'
        if not recpath.exists():raise ValueError('physical callback requires original runtime_receipt state/gauge identity')
        rec=json.loads(recpath.read_text())
        if rec['state_hash']!=bundle['state_hash']:raise ValueError('physical state hash differs from public workspace')
        if rec['material_gauge']['basis_fingerprint']!=ctx.material_gauge['basis_fingerprint']:raise ValueError('physical material coordinate fingerprint mismatch')
        direction=c.DirectionAction(P=ctx.P,action=bundle['ev'].j,kind='PHYSICAL_CALLBACK',synchronize=sync,counter_supplier=model.counters.as_dict,backend='cuda_physical_Maxwell_return_FP64_host')
        physical_setup_counters=copy.deepcopy(model.counters.as_dict())
    else:
        # Explicit test-only stand-in: no frozen(), Maxwell state, or GPU call.
        if args.device!='cpu':raise ValueError('test-cache-callback is CPU only')
        reference=p/'offline_information_reference.npz'
        with np.load(reference,allow_pickle=False) as f:
            if 'fullJ' not in f:raise ValueError('test callback fixture requires already-paid real fullJ')
            J=f['fullJ'].copy()
        if J.dtype!=np.float64 or np.iscomplexobj(J):raise ValueError('FP64 real test matrix')
        counters={'totals':{},'timings_s':{},'event_count':0,'events':[]}
        def test_action(Z):
            t=time.perf_counter();result=J@Z
            counters['totals']['test_matmul_calls']=counters['totals'].get('test_matmul_calls',0)+1
            counters['totals']['test_rhs']=counters['totals'].get('test_rhs',0)+bank['PB'].shape[0]*Z.shape[1]
            counters['timings_s']['test_matmul_s']=counters['timings_s'].get('test_matmul_s',0.)+time.perf_counter()-t
            return result
        # PHYSICAL_CALLBACK interface is exercised, but receipt explicitly records TEST not physical.
        direction=c.DirectionAction(P=bank['PB'].shape[0],action=test_action,kind='PHYSICAL_CALLBACK',counter_supplier=lambda:copy.deepcopy(counters),backend='CPU_TEST_MATRIX_CALLBACK_NOT_PHYSICAL')
        physical_setup_counters=None
    adapter_start=time.perf_counter();engine=c.CachedControls(bank,device=args.device,direction=direction,reference_floor=floor,source=str(p));direction.sync();adapterwall=time.perf_counter()-adapter_start
    setup=dict(version='portable_v2',r1_root=str(r1),persisted=str(p),direction_mode=args.direction,direction_backend=direction.backend,is_actual_physical_callback=args.direction=='physical',test_callback_is_not_physical=args.direction!='physical',fresh_physical_state_setup_wall_s=physical_setup,physical_model_setup_counters=physical_setup_counters,public_cache_load_wall_s=publicload,adapter_wall_s=adapterwall,adapter=engine.setup,floor_source=floor_source,standalone_cold_wall_s=None,cold_speedup=None,physical_objects_constructed=1 if bundle else 0,shared_physical_object_cache_count=1 if bundle else 0,offlinefullJ_constructed=False,full_reference_step_optimization=False,cold_scope='Fresh physical state setup is measured shared acquisition only. Anchor/pool/public maps are persisted; independent end-to-end cold setup is NOT measured.',CPU_threads={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    paths=[p/'public_workspace.npz']+[Path(x['path']) for x in floor_source['sources']]
    if reference:paths.append(reference)
    if (p/'runtime_receipt.json').exists():paths.append(p/'runtime_receipt.json')
    setup['input_sha256']={str(x):sha(x) for x in paths}
    if (p/'setup_cost.json').exists():setup['historical_public_acquisition_receipt']=dict(path=str(p/'setup_cost.json'),sha256=sha(p/'setup_cost.json'),record=json.loads((p/'setup_cost.json').read_text()),scope='Historical acquisition only; not newly measured independent cold cost, not added to current shared occupation.')
    if bundle:setup['fresh_frozen_input_sha256']=bundle['inputs'];setup['state_hash']=bundle['state_hash']
    return c,engine,setup,bundle

def imports_receipt(c):
    modules={}
    for name in ['portable_cached_onepass_v2','pinned_runtime','run_exchange','maxwell_state','endpoint_batch','removal_core_batch','operators','selectors_current','signed_round_reuse','a10_common','a9_engine','pairtsom_adapter']:
        m=sys.modules.get(name)
        if m is not None and getattr(m,'__file__',None):
            p=Path(m.__file__).absolute();modules[name]=dict(actual_path=str(p),sha256=sha(p))
    import scipy
    receipt=dict(python=platform.python_version(),executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,actual_imports=modules,driver=dict(actual_path=str(Path(__file__).absolute()),sha256=sha(__file__)),R1_CONFIG_sha256=sha(c.R1/'A17_RUN_CONFIG.json'))
    if 'torch' in sys.modules:
        torch=sys.modules['torch'];receipt['torch']=torch.__version__;receipt['cuda_version']=torch.version.cuda
    receipt['vendor_pinned_receipt']=c.pinned_runtime.receipt()
    return receipt

def run_trials(c,e,ks,on_row,setup):
    rows=[];failures=[];warm=[]
    def attempt(method,k,rep,pos):
        t=time.perf_counter();before=e.direction.physical_snapshot()
        try:
            r=e.run(method,k);r.update(status='OK',repetition=rep,order_position=pos,full_paid_sample_wall_s=time.perf_counter()-t)
        except Exception as exc:
            e.direction.sync();paid=getattr(e,'last_attempt',{})
            r=dict(status='FAILED',method=method,requested_k=k,repetition=rep,order_position=pos,error=repr(exc),traceback=traceback.format_exc(),wall_failed_s=time.perf_counter()-t,partial_stage_wall_s=paid.get('stage_wall_s'),partial_endpoint_statistics=paid.get('endpoint_statistics'),partial_endpoint_fee_totals=c.endpoint_fee_totals(paid.get('endpoint_statistics',[])),partial_direction_cost=e._delta(e.direction.snapshot(),paid['direction_before']) if paid.get('direction_before') else None,physical_all_counter_delta=c.counter_delta(e.direction.physical_snapshot(),before),cold_wall_s=None,cold_speedup=None)
            failures.append(r)
        r['direction_charge_scope']='NEW_PHYSICAL_DIRECTION_RHS' if setup['is_actual_physical_callback'] else 'TEST_CALLBACK_RHS_EQUIVALENT_ONLY_NOT_PHYSICAL'
        if not setup['is_actual_physical_callback']:
            for key in ['direction_selection','direction_offline_evaluation','partial_direction_cost']:
                if r.get(key) is not None:r[key]['test_callback_rhs_equivalent']=r[key].pop('new_physical_rhs',0);r[key]['new_physical_rhs']=0
        on_row(r);return r
    for k in ks:
        for pos,method in enumerate(c.METHODS):warm.append(attempt(method,k,-1,pos))
    for rep in range(5):
        order=c.METHODS[rep%3:]+c.METHODS[:rep%3]
        for k in ks:
            for pos,method in enumerate(order):rows.append(attempt(method,k,rep,pos))
    return dict(status='COMPLETE_WITH_FAILURES' if failures else 'COMPLETE',samples=rows,prewarm_charged_separately=warm,failures=failures,repetitions=5,methods=c.METHODS,setup=setup,physical_model_final_counters=e.direction.physical_snapshot(),cold_speedup=None,science_verdict=None,fee_scope='Every warm and prewarm sample paid. Selection and offline evaluation separated; physical snapshots/counters nested audit views, not extra occupation. End-to-end watchdog occupation remains parent ledger.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--r1-root',required=True);p.add_argument('--persisted',required=True);p.add_argument('--out',required=True);p.add_argument('--direction',choices=['physical','test-cache-callback'],default='physical');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--object',type=int);p.add_argument('--phase',choices=['early','middle','late']);p.add_argument('--floor-receipt');p.add_argument('--k',type=int,nargs='+',default=[4,8,16]);a=p.parse_args()
    r1=Path(a.r1_root).expanduser().resolve();target=Path(a.out).expanduser().resolve()
    permitted=target.is_relative_to(HERE) or (target.parent==r1/'results' and target.name.startswith('timing_'))
    if not permitted:raise ValueError('--out requires delegated directory or explicit R1/results/timing_*')
    target.mkdir(parents=True,exist_ok=False);tjob=time.perf_counter()
    try:
        c,e,setup,bundle=initialize(a);write_json(target/'SETUP.json',setup,c.sanitize);write_json(target/'IMPORTS_START.json',imports_receipt(c),c.sanitize)
        counter_start=e.direction.physical_snapshot()
        def on_row(row):
            scalar={k:v for k,v in row.items() if k not in ('step','U')}
            if 'step' in row:
                name=f"r{row['repetition']}_k{row['requested_k']}_p{row['order_position']}_{row['method']}.npz";dest=target/name
                np.savez_compressed(dest,step=row['step'],U=row['U']);scalar['endpoint_array_path']=str(dest);scalar['endpoint_array_sha256']=sha(dest)
            with (target/'SAMPLE_JOURNAL.jsonl').open('a') as f:f.write(json.dumps(c.sanitize(scalar),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        result=run_trials(c,e,a.k,on_row,setup)
        result['physical_all_trials_counter_delta']=c.counter_delta(e.direction.physical_snapshot(),counter_start);result['driver_wall_s']=time.perf_counter()-tjob;result['final_imports']=imports_receipt(c)
        for group in ['samples','prewarm_charged_separately']:
            for r in result[group]:r.pop('step',None);r.pop('U',None)
        write_json(target/'TIMINGS.json',result,c.sanitize)
        print(json.dumps(dict(status=result['status'],samples=len(result['samples']),failures=len(result['failures']),out=str(target),direction=a.direction,cold_speedup=None)))
    except Exception as exc:
        failure=dict(status='FAILED_SETUP_OR_DRIVER',error=repr(exc),traceback=traceback.format_exc(),driver_wall_s=time.perf_counter()-tjob,direction=a.direction,object_id=a.object,phase=a.phase,partial_setup_progress=getattr(a,'_setup_progress',None),physical_setup_failure_counter_limit='If frozen() itself raised, internal model counters unavailable; occupied job fee remains parent watchdog ledger, never zero.')
        (target/'FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n');raise
if __name__=='__main__':main()
