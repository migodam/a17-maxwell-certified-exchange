"""Freeze new A17 inputs without modifying any A16 archive."""
from pathlib import Path
import hashlib, json, shutil, datetime

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE.parents[2]
A16=PROJECT/'Gaussian/A16/CURRENT_SELECTION_R1'
SNAP=A16/'evidence_snapshots/terminal_20261001t1143_v16/workspace'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if (HERE/'SOURCE_MANIFEST.json').exists():
        raise RuntimeError('Already frozen: no overwrite')
    lock=json.loads((A16/'configs/validation_lock.json').read_text())
    rows=[];modules={}
    paths=[(A16/p,h) for p,h in lock['source_hashes'].items() if p.startswith('code/') and p.endswith('.py')]
    paths += [(PROJECT/next(p[8:] for p in row['allowed_paths'] if p.startswith('project:')),row['sha256']) for row in lock['external_modules'].values()]
    for src,expected in paths:
        if sha(src)!=expected: raise RuntimeError('Source drift: '+str(src))
        rel=src.relative_to(PROJECT);dst=HERE/'vendor'/rel
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        if src.stem in modules:raise RuntimeError('Duplicate module name')
        modules[src.stem]={'path':str(dst.relative_to(HERE)),'sha256':expected}
        rows.append(dict(kind='private_source',original_project_path=str(rel),path=str(dst.relative_to(HERE)),sha256=expected,byte_identical=True))
    scenes=json.loads((A16/'configs/scenes.json').read_text())
    selected=[s for s in scenes['scenes'] if s['object_id'] in (2001,2007,2012,2014)]
    write(HERE/'inputs/scenes.json',dict(scenes=selected,original_sha256=sha(A16/'configs/scenes.json')))
    for oid in (2001,2007,2012,2014):
        src=SNAP/f'Gaussian/A16/CURRENT_SELECTION_R1/frozen_validation/core_v10_locked/object_{oid}/reference'
        files=['result.json','common_data.npz']+[f'{name}_{it:02d}.npz' for it in (0,3,17) for name in ('anchor','step')]
        for name in files:
            p=src/name;dst=HERE/f'inputs/object_{oid}'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
            rows.append(dict(kind='frozen_input',path=str(dst.relative_to(HERE)),original_project_path=str(p.relative_to(PROJECT)),sha256=sha(p),bytes=p.stat().st_size,byte_identical=True))
    for p in sorted((PROJECT/'Gaussian/A17').glob('*.md')):
        rows.append(dict(kind='specification',original_project_path=str(p.relative_to(PROJECT)),sha256=sha(p),bytes=p.stat().st_size))
    write(HERE/'SOURCE_MANIFEST.json',dict(schema='a17.source.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),modules=modules,inputs=rows,A16_problem_commit='e17f9de3469964c8f4cd1b1c967d2e2f14b3d277',scope='Current hash-matched private kernel copies bound to NEW A17 execution; not retroactive proof for every A16 child.',missing_supplied_tests=['tests/verify_a17.py','tests/verify_a17_extended.py'],saved_reference_source='Immutable A16 terminal workspace; external first-selection authority remains relevant for objects 2007/2012; old origin seals not rewritten.'))
    config=dict(schema='a17.exchange.r1',pilot_objects=[2007,2012],extension_objects=[2001,2014],phases=dict(early=0,middle=3,late=17),k=[4,8,16],seed=20261002,extended_seed=20261003,anchor_rank=64,pool_count=32,residual_up_count=16,random_pool_count=10,online_incoming_count=12,family_balancing='Round robin in original family and atom order; remove already selected atom IDs; no full label access.',max_actions=3,max_finalists_per_round=2,cuda_candidate_batch=16,cpu_threads=4,current_stability_relative_min=1e-10,algebra_relative_tolerance=1e-9,Maxwell_relative_tolerance=1e-9,reference_relative_residual=1e-10,endpoint_normal_relative_residual=1e-8,gain_tolerance_rule='max(10*reference_floor, 256*float64_epsilon*max(base objective energy, child objective energy, residual norm squared)); tolerance is not a rigorous output-error certificate.',max_gpu_occupation_s=43200,max_job_s=7200,memory_warning=.8,memory_stop=.9,expansion_gate=dict(both_objects_positive=True,median_object_summed_risk_improvement=.10,at_least_one_object_k_ge_8_positive=True),nn_gate=dict(positive_objects_at_least=3,k_ge_8_positive_objects_at_least=2,opportunity_capture_target=.70,end_to_end_cost_reduction_target=2.,seeds=[20261002,20261003,20261004]),micro_max_endpoints=10000,two_swap_domain='First four deletable logical atoms and first six balanced incoming atoms; all 2-out/2-in combinations, recompute after receiver start only.',primary_scope='Frozen no-noise Level-1 GN fidelity; no new nonlinear trajectory, PCG, preconditioner, MURR, or Flow.',certification='No deterministic claim without validated full output-error upper bound; double-precision full directed checks are HIGH_FIDELITY_CHECK.',cost_contract='Actual jobs paid once; standalone selector pays physical state + declared anchor/pool/workspace + its own screen/probe/transition; teacher and offline evaluation separate; shared-cache timing not a cold speedup claim.')
    write(HERE/'A17_RUN_CONFIG.json',config)
    write(HERE/'A17_STATUS.json',dict(status='IMPLEMENTING',gpu_occupation_s=0,algebra='PENDING_NEW_INDEPENDENT_TESTS',pilot='NOT_RUN',extension='NOT_RUN',NN='NOT_RUN',prior='NOT_RUN'))
    print(json.dumps(dict(sources=len(modules),inputs=len(rows),root=str(HERE)),ensure_ascii=False))

if __name__=='__main__':main()
