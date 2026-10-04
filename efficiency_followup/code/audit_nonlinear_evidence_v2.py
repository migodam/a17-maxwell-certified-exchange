"""Local receipt/trajectory audit only; no solver, selector or Gate decision.

Consumes the separately validated analyzer V3 identities. Scalar diagnostics do
not stand in for constraint checks on saved material checkpoints.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


DRIVER_ROLES = {
    'a17.efficiency.nonlinear_transfer.v1': 'MAIN',
    'a17.efficiency.nonlinear_reuse_transfer.v1': 'REUSE_SECONDARY',
    'a17.efficiency.nonlinear_top1cap3_transfer.v1': 'TOP1CAP3_CONDITIONAL',
}
PHYSICS_KEYS = ('baseline_lock_sha256','baseline_manifest_sha256','input_manifest_sha256',
                'audited_geometry_provider_sha256','nonlinear_outer_rules','nonlinear_k',
                'nonlinear_noise_percent','precision','numerical_environment')


def record_input(inputs, path, status='PRESENT'):
    path = Path(path)
    inputs.append(dict(path=str(path.resolve()), status=status if path.is_file() else 'MISSING',
                       sha256=sha(path) if path.is_file() else None))


def physics_identity(lock):
    if not isinstance(lock, dict) or any(key not in lock for key in PHYSICS_KEYS):
        return None
    return hashlib.sha256(json.dumps({k:lock[k] for k in PHYSICS_KEYS},
                                    sort_keys=True,separators=(',',':')).encode()).hexdigest()


def locate_raw(raw_root, item, inputs):
    if raw_root is None:
        return None, 'NOT_MEASURED_NO_RAW_ROOT'
    root = Path(raw_root).resolve()
    if not root.is_dir():
        return None, 'NOT_MEASURED_RAW_ROOT_MISSING'
    expected = item.get('result_sha256')
    if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        return None, 'NOT_MEASURED_UNKNOWN_RESULT_HASH'
    candidates = []
    for path in sorted(root.rglob('result.json')):
        if not path.resolve().is_relative_to(root):
            continue
        record_input(inputs,path)
        if sha(path)==expected:
            candidates.append(path.parent)
    if len(candidates)==1:
        return candidates[0], 'HASH_BOUND_RAW_DIRECTORY'
    return None, 'NOT_MEASURED_RAW_HASH_MISSING_OR_AMBIGUOUS'


def load_arrays(path, inputs, issues):
    import numpy as np
    path = Path(path)
    record_input(inputs,path)
    if not path.is_file():
        issues.append(dict(path=str(path),failure='MISSING_ARRAY_FILE'))
        return None
    try:
        with np.load(path,allow_pickle=False) as handle:
            arrays={k:handle[k].copy() for k in handle.files}
        for key,value in arrays.items():
            if value.dtype.kind in 'biufc' and not np.isfinite(value).all():
                issues.append(dict(path=str(path),array=key,failure='NONFINITE_ARRAY'))
        return arrays
    except Exception as error:
        issues.append(dict(path=str(path),failure='ARRAY_READ_ERROR',error=type(error).__name__))
        return None


def check_material(arrays, path, shape, issues):
    import numpy as np
    if arrays is None:
        return False
    value=arrays.get('chi')
    if value is None:
        issues.append(dict(path=str(path),failure='MISSING_CHI'))
        return False
    valid=True
    if value.shape!=shape or value.ndim!=1 or value.dtype.kind!='c':
        issues.append(dict(path=str(path),failure='CHI_SHAPE_OR_DTYPE',shape=list(value.shape),expected=list(shape)))
        valid=False
    if not np.isfinite(value).all():
        valid=False
    if np.any(value.real<-.50000001) or np.any(value.imag<-1e-8):
        issues.append(dict(path=str(path),failure='INFEASIBLE_CHI',real_min=float(value.real.min()) if np.isfinite(value.real.min()) else None,imag_min=float(value.imag.min()) if np.isfinite(value.imag.min()) else None))
        valid=False
    return valid


def array_comparison(left,right):
    import numpy as np
    if left is None or right is None:
        return dict(status='NOT_MEASURED_MISSING_ARRAY',bitwise_equal=None)
    if left.shape!=right.shape:
        return dict(status='SHAPE_MISMATCH',bitwise_equal=False,left_shape=list(left.shape),right_shape=list(right.shape))
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return dict(status='NONFINITE',bitwise_equal=False)
    same=left.dtype==right.dtype and left.tobytes(order='C')==right.tobytes(order='C')
    difference=np.asarray(left)-np.asarray(right)
    numerator=float(np.linalg.norm(difference.ravel()))
    denominator=float(np.linalg.norm(np.asarray(right).ravel()))
    return dict(status='MEASURED_ARRAY_COMPARISON',bitwise_equal=same,
                max_abs_error=float(np.max(np.abs(difference),initial=0)),
                relative_l2_error=numerator/denominator if denominator else (0. if numerator==0 else None),
                dtype_equal=left.dtype==right.dtype)


def raw_constraints(directory,trajectory,inputs):
    issues=[]
    if directory is None:
        return dict(status='NOT_MEASURED',issues=[],accepted_checkpoint_checks=[],final_matches_last_safe=None)
    final=load_arrays(directory/'final.npz',inputs,issues)
    if final is None or 'chi' not in final:
        if final is not None:issues.append(dict(failure='MISSING_FINAL_CHI'))
        return dict(status='INCOMPLETE_RAW',issues=issues,accepted_checkpoint_checks=[],final_matches_last_safe=None)
    shape=final['chi'].shape
    check_material(final,directory/'final.npz',shape,issues)
    accepted=[]
    for record in trajectory:
        iteration=record['iteration']
        anchorpath=directory/f'anchor_{iteration:02d}.npz'
        anchor=load_arrays(anchorpath,inputs,issues);check_material(anchor,anchorpath,shape,issues)
        checkpointpath=directory/f'chi_checkpoint_{iteration:02d}.npz'
        checkpoint=load_arrays(checkpointpath,inputs,issues)
        material_ok=check_material(checkpoint,checkpointpath,shape,issues)
        if record.get('accepted'):
            accepted.append(dict(iteration=iteration,status='MEASURED_FEASIBLE' if material_ok else 'FAILED_OR_NOT_MEASURED'))
        if checkpoint is not None:
            import numpy as np
            flag=checkpoint.get('accepted')
            if flag is None or np.asarray(flag).size!=1 or bool(np.asarray(flag).reshape(-1)[0])!=bool(record.get('accepted')):
                issues.append(dict(path=str(checkpointpath),failure='CHECKPOINT_ACCEPTED_FLAG_MISMATCH'))
    # Inspect any additional synchronized checkpoints/anchors too; no silent ignored material states.
    expected={directory/f'{stem}_{r["iteration"]:02d}.npz' for r in trajectory for stem in ('anchor','chi_checkpoint')}
    for path in sorted(set(directory.glob('anchor_*.npz'))|set(directory.glob('chi_checkpoint_*.npz'))):
        if path not in expected:check_material(load_arrays(path,inputs,issues),path,shape,issues)
    safe=load_arrays(directory/'last_safe_state.npz',inputs,issues)
    check_material(safe,directory/'last_safe_state.npz',shape,issues)
    equality=array_comparison(final.get('chi'),safe.get('chi') if safe else None)
    if equality.get('bitwise_equal') is False:issues.append(dict(failure='FINAL_LAST_SAFE_MISMATCH'))
    missing=any(i['failure'].startswith('MISSING') for i in issues)
    return dict(status='INCOMPLETE_RAW' if missing else 'FAILED_ARRAY_CHECKS' if issues else 'MEASURED_ARRAY_CHECKS_ONLY',
                issues=issues,accepted_checkpoint_checks=accepted,final_matches_last_safe=equality)


def cached_array_equivalence(cached,uncached,trajectory,inputs):
    issues=[];comparisons=[]
    if cached is None or uncached is None:
        return dict(status='NOT_MEASURED',issues=[dict(failure='MISSING_HASH_BOUND_RAW_RUN')],comparisons=[])
    a=load_arrays(cached/'final.npz',inputs,issues);b=load_arrays(uncached/'final.npz',inputs,issues)
    for key in ('chi','field','held'):
        comparisons.append(dict(member='final.npz:'+key,**array_comparison(a.get(key) if a else None,b.get(key) if b else None)))
    expected=set()
    for record in trajectory:
        folder=Path(f'iteration_{record["iteration"]:02d}')/'exchange'
        for rd in record.get('policy_result',{}).get('rounds',[]):
            expected.add(folder/f'finalists_round_{rd["round"]}.npz')
        expected.add(folder/'endpoint.npz')
        accepted_moves=record.get('policy_result',{}).get('accepted',[])
        if isinstance(accepted_moves,list):
            for move in accepted_moves:
                if isinstance(move,dict) and 'round' in move:expected.add(folder/f'round_{move["round"]}.npz')
    # Compare union as well, so a member present on only one side is explicit.
    for base in (cached,uncached):
        expected.update(p.relative_to(base) for p in base.rglob('finalists_round_*.npz'))
        expected.update(p.relative_to(base) for p in base.rglob('round_*.npz'))
    for relative in sorted(expected):
        a=load_arrays(cached/relative,inputs,issues);b=load_arrays(uncached/relative,inputs,issues)
        keys=('d','z','x') if relative.name.startswith('finalists_') else ('selected_ids',)
        for key in keys:
            comparisons.append(dict(member=str(relative)+':'+key,**array_comparison(a.get(key) if a else None,b.get(key) if b else None)))
    missing=any(x['status'].startswith('NOT_MEASURED') for x in comparisons)
    return dict(status='INCOMPLETE_RAW' if missing or issues else 'MEASURED_ARRAY_COMPARISONS_ONLY',issues=issues,comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'analysis/derived_v3')
    parser.add_argument('--out', type=Path, default=ROOT/'analysis/owner_nonlinear_v2')
    parser.add_argument('--raw-root', type=Path, help='Optional hash-bound synchronized raw result tree')
    args = parser.parse_args()
    identity = read(args.data/'nonlinear_identity_audit.json')
    tolerances = read(ROOT/'configs/POLICY_PRELOCK_V1.json')['gateB']
    entries, issues, inputs = [], [], []
    record_input(inputs,args.data/'nonlinear_identity_audit.json')
    record_input(inputs,ROOT/'configs/POLICY_PRELOCK_V1.json')
    locks={}
    for path in sorted((ROOT/'configs').glob('*.json')):
        lock=read(path)
        if lock.get('schema')=='a17.efficiency.nonlinear.lock.v1':
            record_input(inputs,path);locks[sha(path)]=lock
    raw_directories={};trajectories={}
    for item in identity:
        if item['scope'] != 'eff_nonlinear':
            continue
        p = Path(item['source'])
        files=[p,p.parent/'trajectory.json',p.parent/'run_config.json']
        if any(not f.is_file() for f in files):
            for f in files:record_input(inputs,f)
            issues.append(dict(source=str(p),reasons=['SOURCE_PATH_MISSING']));continue
        if sha(p)!=item.get('result_sha256'):
            record_input(inputs,p);issues.append(dict(source=str(p),reasons=['RESULT_HASH_MISMATCH_OR_UNKNOWN']));continue
        result, trajectory, config = read(p), read(p.parent/'trajectory.json'), read(p.parent/'run_config.json')
        role=DRIVER_ROLES.get(result.get('version'),'UNKNOWN_DRIVER')
        if role=='UNKNOWN_DRIVER':issues.append(dict(source=str(p),reasons=['UNKNOWN_DRIVER_VERSION']))
        for f in [p, p.parent/'trajectory.json', p.parent/'run_config.json']:
            inputs.append(dict(path=str(f), sha256=sha(f)))
        if not item['match_identity_eligible']:
            issues.append(dict(source=str(p), reasons=item['match_identity_errors']))
        checks = []
        locked=locks.get(item.get('config_sha256'))
        binding=config.get('lock_receipt',{})
        actual_driver=result.get('runtime_end',{}).get('declared_efficiency',{}).get('nonlinear_driver_sha256')
        driver_files={'MAIN':'code/run_nonlinear_efficient.py','REUSE_SECONDARY':'code/run_nonlinear_reuse.py','TOP1CAP3_CONDITIONAL':'code/run_nonlinear_top1cap3_v1.py'}
        binding_errors=[]
        if locked is None or binding.get('config_sha256')!=item.get('config_sha256'):binding_errors.append('CONFIG_HASH_BINDING_MISSING_OR_MISMATCH')
        if locked is not None:
            if binding.get('source_hashes')!=locked.get('source_hashes'):binding_errors.append('SOURCE_HASH_BINDING_MISMATCH')
            if locked.get('nonlinear_version')!=result.get('version'):binding_errors.append('LOCK_VERSION_MISMATCH')
            if actual_driver!=locked.get('source_hashes',{}).get(driver_files.get(role)):binding_errors.append('DRIVER_HASH_MISMATCH')
        if binding_errors:issues.append(dict(source=str(p),reasons=binding_errors))
        jx, jd, exchange_moves, adjoint, block_calls = 0, 0, 0, 0, 0
        phases, online_stages = {}, {}
        feasible_labels = True
        for record in trajectory:
            pr = record['policy_result']
            for key in ('objective_before','slope','alpha','executed_step_norm'):
                value=record.get(key)
                if not isinstance(value,(int,float)) or not math.isfinite(value):checks.append(dict(iteration=record['iteration'],failure='NONFINITE_OR_MISSING_SCALAR',field=key))
            if not record.get('projection',{}).get('all_solver_success',False):checks.append(dict(iteration=record['iteration'],failure='PROJECTION_SOLVER_FAILURE'))
            if record['actual_rank'] != 8 or pr['actual_rank'] != 8:
                checks.append(dict(iteration=record['iteration'], failure='actual_rank_not_eight'))
            if record['accepted']:
                accepted = [t for t in record['trials'] if t['status'] == 'ACCEPT']
                if len(accepted) != 1:
                    checks.append(dict(iteration=record['iteration'], failure='accepted_trial_count'))
                else:
                    trial = accepted[0]
                    rhs = record['objective_before']+1e-4*trial['alpha']*min(record['slope'], 0.)
                    if trial['objective'] > rhs:
                        checks.append(dict(iteration=record['iteration'], failure='recorded_armijo_violation', difference=trial['objective']-rhs))
                    if trial['alpha'] != record['alpha']:
                        checks.append(dict(iteration=record['iteration'], failure='record_alpha_mismatch'))
            feasible_labels &= all(t['status'] in ('INFEASIBLE', 'ACCEPT', 'REJECT') for t in record['trials'])
            for label, cost in record['phase_costs'].items():
                phases[label] = phases.get(label, 0.)+cost['wall_s']
            for label, seconds in pr.get('stage_wall_s', {}).items():
                online_stages[label] = online_stages.get(label, 0.)+seconds
            direction_rhs = pr.get('direction_cost', {}).get('tangent_rhs', 0)
            # Authoritative finalist direction count is recorded explicitly in
            # controller budget/round records; total action RHS includes Jx.
            current_jd = sum(len(rd.get('finalists', [])) for rd in pr.get('rounds', []))
            budget=pr.get('reuse_policy',pr.get('efficiency_policy',{}))
            if 'full_Jd_directions' in budget:
                current_jd=budget['full_Jd_directions']
            if direction_rhs:
                # This interface acquires one Jx and only Jd thereafter. The
                # equality is audited against trajectory source identities.
                if direction_rhs % 6:
                    checks.append(dict(iteration=record['iteration'], failure='tangent_rhs_not_six_illuminations'))
                total_directions = direction_rhs//6
                if current_jd == 0 and total_directions > 1:
                    # Legacy round schemas carry finalists in the verified
                    # record as "verified", rather than "finalists".
                    current_jd = total_directions-1
                if total_directions != current_jd+1:
                    checks.append(dict(iteration=record['iteration'], failure='Jx_Jd_count_disagreement', rhs=direction_rhs, jd=current_jd))
                jx += 1
                jd += current_jd
                block_calls += pr['direction_cost']['calls']-1
            accepted_moves = pr.get('accepted', [])
            exchange_moves += len(accepted_moves) if isinstance(accepted_moves,list) else accepted_moves
            adjoint += pr.get('adjoint_rhs', 0)
        counters = result['full_physical_counters']['totals']
        tangent_rhs = counters.get('solve_rhs_full_tangent', 0)
        if tangent_rhs != 6*(jx+jd):
            checks.append(dict(failure='final_tangent_RHS_disagreement', actual=tangent_rhs, expected=6*(jx+jd)))
        if result.get('status')=='completed_max_updates' and len(trajectory)!=18:checks.append(dict(failure='COMPLETED_MAX_UPDATES_COUNT'))
        if result.get('records')!=len(trajectory):checks.append(dict(failure='RECORD_COUNT_MISMATCH'))
        if result.get('status') not in ('completed_max_updates','stopped_small_step','stopped_no_armijo_accept'):checks.append(dict(failure='UNKNOWN_TERMINAL_STATUS'))
        if result.get('status')=='stopped_no_armijo_accept' and (not trajectory or trajectory[-1].get('accepted')):checks.append(dict(failure='STOP_NO_ACCEPT_TRAJECTORY_MISMATCH'))
        if result.get('status')=='stopped_small_step' and (not trajectory or not trajectory[-1].get('accepted') or trajectory[-1].get('iteration',-1)<9 or trajectory[-1].get('executed_step_norm',float('inf'))>=1e-6):checks.append(dict(failure='SMALL_STEP_STATUS_MISMATCH'))
        rawdir,rawstatus=locate_raw(args.raw_root,item,inputs)
        rawcheck=raw_constraints(rawdir,trajectory,inputs)
        raw_directories[item['result_sha256']]=rawdir;trajectories[item['result_sha256']]=trajectory
        setup = sum(result[k]['wall_s'] for k in ('input_cost','physical_setup_cost','final_metrics_cost'))
        child = result['wall_total_s']
        extras = child-sum(phases.values())-setup
        receipt_phases = {k:v for k,v in phases.items()}
        row = dict(object=result['object_id'], policy=result['policy'], geometry_cache=result['geometry_cache'],
            repetition=result['repetition'], version=result['version'], config_sha256=item['config_sha256'],
            identity_eligible=item['match_identity_eligible'] and role!='UNKNOWN_DRIVER' and not binding_errors,
            driver_role=role,physics_identity=physics_identity(locks.get(item.get('config_sha256'))),
            raw_binding_status=rawstatus,raw_array_check=rawcheck, result_sha256=sha(p), source=str(p),
            status=result['status'], outer_records=result['records'], child_wall_s=child,
            material=result['metrics']['material_complex_relative'],
            material_real=result['metrics']['material_real_relative'], material_imag=result['metrics']['material_imag_relative'],
            heldout=result['metrics']['heldout_data_relative'], measured_data=result['metrics']['measured_data_relative'],
            accepted_outer_updates=sum(r['accepted'] for r in trajectory), exchange_moves=exchange_moves,
            Jx_directions=jx, Jd_directions=jd, Jd_block_calls=block_calls,
            avg_Jd_per_outer=jd/len(trajectory) if trajectory else None,
            tangent_rhs=tangent_rhs, full_adjoint_rhs=counters.get('solve_rhs_full_adjoint'),
            full_forward_rhs=counters.get('solve_rhs_full_forward'), full_LU=counters.get('lu_factorizations_full'),
            gradient_fallbacks=sum(r['gradient_fallback'] for r in trajectory),
            all_projection_solver_success=all(r['projection']['all_solver_success'] for r in trajectory),
            infeasible_trials=sum(r['infeasible_trials'] for r in trajectory),
            armijo_trial_count=sum(len(r['trials']) for r in trajectory),
            observable_label_checks=not checks and feasible_labels,
            no_reference_optimal_step_acquired=result.get('no_reference_optimal_step_acquired',False),
            checkpoint_constraint_check=rawcheck['status'],
            check_failures=checks, phase_seconds=receipt_phases, online_stage_seconds=online_stages,
            input_setup_final_metrics_s=setup, unassigned_child_s=extras,
            timing_scope='Child excludes pre-start imports/binding and post-end final publication; external occupation separate')
        entries.append(row)
    comparisons = []
    equivalences = []
    # References are unique MAIN receipts, never last-write-wins index entries.
    for row in entries:
        if not row['identity_eligible'] or row['physics_identity'] is None:
            continue
        peers=[x for x in entries if x['identity_eligible'] and x['driver_role']=='MAIN'
               and x['object']==row['object'] and x['repetition']==row['repetition']
               and x['physics_identity']==row['physics_identity']]
        originals=[x for x in peers if x['policy']=='original_top2_cap3' and x['geometry_cache'] is False]
        receivers=[x for x in peers if x['policy']=='receiver_only' and x['geometry_cache'] is False]
        if len(originals)!=1 or len(receivers)!=1:
            issues.append(dict(source=row['source'],reasons=['EXPLICIT_MAIN_REFERENCE_MISSING_OR_AMBIGUOUS']));continue
        original,receiver=originals[0],receivers[0]
        if any(not isinstance(x.get(metric),(int,float)) or not math.isfinite(x[metric]) for x in (row,original) for metric in ('material','heldout')):
            issues.append(dict(source=row['source'],reasons=['QUALITY_METRIC_MISSING_OR_NONFINITE']));continue
        comparison=dict(object=row['object'],policy=row['policy'],geometry_cache=row['geometry_cache'],repetition=row['repetition'],
            version=row['version'],driver_role=row['driver_role'],config_sha256=row['config_sha256'],
            source_sha256=row['result_sha256'],reference_sha256=original['result_sha256'],receiver_sha256=receiver['result_sha256'],
            reference_version=original['version'],reference_config_sha256=original['config_sha256'],physics_identity=row['physics_identity'],
            driver_comparison='SAME_MAIN_VERSION_PHYSICS_REP' if row['driver_role']=='MAIN' else 'EXPLICIT_CROSS_VERSION_SAME_PHYSICS_REP_SECONDARY_NO_INDEPENDENT_VALIDATION',
            original_wall_ratio=row['child_wall_s']/original['child_wall_s'],receiver_wall_ratio=row['child_wall_s']/receiver['child_wall_s'],
            material_ratio_to_original=row['material']/original['material'] if original['material'] else None,
            heldout_ratio_to_original=row['heldout']/original['heldout'] if original['heldout'] else None,
            observed_path_checks=row['observable_label_checks'] and row['all_projection_solver_success'],
            no_hidden_full_reference=row['no_reference_optimal_step_acquired'])
        for metric,tol_name in [('material','max_material_complex_relative_degradation'),('heldout','max_heldout_relative_degradation')]:
            limit=(1+tolerances[tol_name])*original[metric]+tolerances['absolute_metric_comparison_floor']
            comparison[metric+'_comparison_limit']=limit
            comparison[metric+'_within_predeclared_tolerance']=row[metric]<=limit
        comparisons.append(comparison)
        if row['driver_role']=='MAIN' and row['policy']=='original_top2_cap3' and row['geometry_cache'] is True:
            detail=cached_array_equivalence(raw_directories.get(row['result_sha256']),raw_directories.get(original['result_sha256']),trajectories[row['result_sha256']],inputs)
            equivalences.append(dict(object=row['object'],repetition=row['repetition'],version=row['version'],cached_source=row['source'],uncached_source=original['source'],cached_result_sha256=row['result_sha256'],uncached_result_sha256=original['result_sha256'],**detail))
    args.out.mkdir(parents=True,exist_ok=True)
    payload=dict(schema='a17.efficiency.owner_trajectory_audit.v2', numerical_jobs_launched=False,
        full_checkpoint_validation='NOT_MEASURED_NO_RAW_ROOT' if args.raw_root is None else 'PER_RUN_ARRAY_STATUS_ONLY',raw_root=str(args.raw_root.resolve()) if args.raw_root else None,cached_original_array_equivalence=equivalences, gate_assignment='ROOT_REVIEW_REQUIRED',
        quality_convention='Proposed <=1.01*original+5e-9, frozen before outcomes',
        reports=entries, comparisons=comparisons, input_manifest=inputs, identity_issues=issues,
        source_sha256=sha(__file__))
    (args.out/'AUDIT.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    for name, rows in [('runs',entries),('comparisons',comparisons),('cached_original_array_equivalence',equivalences),('input_manifest',inputs),('identity_issues',issues)]:
        (args.out/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        fields=sorted({k for row in rows for k in row})
        with (args.out/(name+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    print(json.dumps(dict(runs=len(entries), comparisons=len(comparisons), identity_issues=len(issues),
        observable_failures=sum(not r['observable_label_checks'] for r in entries), out=str(args.out))))


if __name__=='__main__':
    main()
