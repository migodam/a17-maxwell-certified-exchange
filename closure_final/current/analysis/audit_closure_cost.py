"""Offline receipt-only cost audit. No runtime imports or numerical execution."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

OLD_CARRY_S = 4889.515
CEILING_S = 43200.0
STAGES = ('receiver_wall_s', 'Jx_initialization_wall_s', 'endpoint_wall_s',
          'anchor_wall_s', 'Jd_wall_s', 'acceptance_wall_s')


def sum_s(values):
    return math.fsum(values)


def delta(before, after):
    """Only physical counter receipts are physical-action authority."""
    a, b = before.get('totals', {}), after.get('totals', {})
    return {key: b.get(key, 0)-a.get(key, 0) for key in sorted(a.keys() | b.keys())}


def ledger(events, rejected):
    starts, ends = {}, {}
    for row in events:
        event = row['event']
        if event not in ('start', 'end'):
            raise ValueError('unsupported ledger event: '+event)
        dst = starts if event == 'start' else ends
        if row['attempt'] in dst:
            raise ValueError('duplicate '+event)
        dst[row['attempt']] = row
    if starts.keys() != ends.keys():
        raise ValueError('unpaired monitored attempt')
    if len({r['attempt'] for r in rejected}) != len(rejected):
        raise ValueError('duplicate rejected preflight')
    if set(starts) & {r['attempt'] for r in rejected}:
        raise ValueError('rejected preflight also charged occupation')
    pairs = sorted([(starts[a], ends[a]) for a in starts], key=lambda pair: pair[0]['utc'])
    for left, right in zip(pairs, pairs[1:]):
        if datetime.fromisoformat(left[1]['utc']) > datetime.fromisoformat(right[0]['utc']):
            raise ValueError('recorded monitored intervals overlap')
    for start, end in pairs:
        if end['occupation_s'] < 0:
            raise ValueError('negative occupation')
        if abs(start['preflight']['accounting']['baseline']['charged_s']-OLD_CARRY_S) > 1e-6:
            raise ValueError('old carry mismatch')
    occupation = sum_s(e['occupation_s'] for _, e in pairs)
    preflight = sum_s(r['wall_s'] for r in rejected)
    total = sum_s([OLD_CARRY_S, occupation, preflight])
    return pairs, dict(old_closed_attempts=17, old_carry_s=OLD_CARRY_S,
        closure_paired_attempts=len(pairs), closure_monitored_occupation_s=occupation,
        rejected_preflight_count=len(rejected), rejected_preflight_wall_s=preflight,
        cumulative_charged_s=total, ceiling_s=CEILING_S, remaining_s=CEILING_S-total,
        mutually_exclusive_addends=['old_carry_s','closure_monitored_occupation_s','rejected_preflight_wall_s'])


class Snapshot:
    def __init__(self, root):
        self.root = Path(root)
        self.receipt = json.loads((self.root/'snapshot_receipt.json').read_text())
        self.expected = {r['path'].replace('\\', '/'): r for r in self.receipt['manifest']}
        self.used = {}

    def check(self, path):
        path = Path(path)
        rel = path.relative_to(self.root).as_posix()
        expected = self.expected.get(rel)
        if expected is None:
            raise ValueError('unmanifested input '+rel)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected['sha256'] or len(data) != expected['bytes']:
            raise ValueError('snapshot hash mismatch '+rel)
        self.used[rel] = digest
        return data

    def read(self, path):
        return json.loads(self.check(path))

    def lines(self, path):
        return [json.loads(line) for line in self.check(path).decode().splitlines() if line.strip()]


def relative_result(path):
    normalized = str(path).replace('\\', '/')
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return 'results/'+normalized.split('/results/', 1)[1]


def online_view(record):
    stages = {key: record['stage_wall_s'].get(key) for key in STAGES}
    if any(v is None for v in stages.values()):
        raise ValueError('missing measured online stage')
    residual = record['wall_online_total_s']-sum_s(stages.values())
    if residual < -1e-6:
        raise ValueError('online stages exceed containing interval')
    finalists = [f for rd in record['rounds'] for f in rd['finalists']]
    counts = Counter('selected' if f['selected'] else 'eligible_unselected' if f['accepted_candidate']
                     else 'threshold_rejected' for f in finalists)
    physical = delta(record['physical_counters_before'], record['physical_counters_after'])
    reported = record['direction_cost']['tangent_rhs']
    if physical.get('solve_rhs_full_tangent', 0) != reported:
        raise ValueError('online direction/physical RHS mismatch')
    # Jx initial receiver receipt is linked separately by caller. Batched finalist
    # Jd wall cannot be allocated among selected/unselected candidates.
    return dict(k=record['requested_k'], wall_online_total_s=record['wall_online_total_s'],
        disjoint_stage_wall_s=stages, controller_other_wall_s=residual,
        direction_nested=record['direction_cost'], physical_counter_delta=physical,
        finalist_counts={key:counts[key] for key in ('selected','eligible_unselected','threshold_rejected')},
        endpoint_count=sum(rd['endpoint_count'] for rd in record['rounds']),
        feasible_count=sum(rd['feasible'] for rd in record['rounds']),
        selected_finalist_wall_s=None, unselected_finalist_wall_s=None,
        reconstructed_standalone_fee_wall_s=record.get('reconstructed_standalone_fee_wall_s'),
        independently_measured_cold_wall_s=record.get('standalone_cold_wall_s'))


def audit(root):
    snap = Snapshot(root)
    for name in ('verified_exchange.py','run_closure_v2.py','watch_closure_job_v3.py'):
        snap.check(snap.root/'code'/name)
    pairs, totals = ledger(snap.lines(snap.root/'execution/gpu_attempts.jsonl'),
                          snap.lines(snap.root/'execution/preflight_checks.jsonl'))
    attempts, states = [], []
    for start, end in pairs:
        cmd = start['command']
        batch_root = snap.root/relative_result(cmd[cmd.index('--out')+1]) if '--out' in cmd else None
        batch_path = batch_root/'batch_receipt.json' if batch_root else None
        batch = snap.read(batch_path) if batch_path and batch_path.exists() else None
        children = []
        for task in batch['tasks'] if batch else []:
            directory = snap.root/relative_result(task['path'])
            status = snap.read(directory/'status.json')
            setup = snap.read(directory/'setup_cost.json') if (directory/'setup_cost.json').exists() else None
            result = snap.read(directory/'result.json') if (directory/'result.json').exists() else None
            policies = []
            for path in sorted(directory.glob('k*/online_result.json')):
                view = online_view(snap.read(path))
                receiver = snap.read(path.parent/'receiver_cost.json')
                initial_rhs = receiver['guarded_Jx_initialization_rhs']
                view['initial_Jx_physical_rhs'] = initial_rhs
                view['finalist_Jd_physical_rhs'] = view['physical_counter_delta'].get('solve_rhs_full_tangent',0)-initial_rhs
                view['finalist_class_physical_rhs'] = {key:count*initial_rhs for key,count in view['finalist_counts'].items()}
                if sum(view['finalist_class_physical_rhs'].values()) != view['finalist_Jd_physical_rhs']:
                    raise ValueError('finalist count/physical RHS mismatch')
                view['receiver_statistics_nested'] = receiver['statistics']
                policies.append(view)
            row = dict(path=directory.relative_to(snap.root).as_posix(),attempt=start['attempt'],
                mode=batch['mode'],object_id=batch['object'],phase=task['phase'],
                status=status['status'],child_process_wall_s=task['wall_s'],
                measured_driver_wall_s=result.get('wall_total_s') if result else None,
                setup=setup,policies=policies,
                offline_evaluation_wall_s=result.get('offline_evaluation_wall_s') if result else None,
                offline_physical_counter_delta=delta(result['physical_offline_counters_before'],result['physical_offline_counters_after']) if result else None)
            if setup:
                row['setup_outer_disjoint_wall_s']={k:setup.get(k) for k in ('physical_state_wall_s','endpoint_engine_setup_wall_s','public_workspace_wall_s','cache_load_adapter_wall_s')}
                row['workspace_nested_wall_s']=dict(anchor=setup.get('anchor_metadata',{}).get('wall_s') if setup.get('anchor_metadata') else None,
                    pool=setup.get('pool_wall_s'),projection=(setup.get('projection_metadata') or {}).get('setup_wall_s'))
            if result:
                measured = sum_s(v for v in row['setup_outer_disjoint_wall_s'].values() if v is not None)
                measured += sum_s(p['wall_online_total_s'] for p in policies)+result['offline_evaluation_wall_s']
                row['driver_other_wall_s']=result['wall_total_s']-measured
                if row['driver_other_wall_s'] < -1e-6:
                    raise ValueError('state child intervals exceed driver '+row['path'])
            else:
                row['driver_other_wall_s']=None
            children.append(row);states.append(row)
        child_sum = sum_s(c['child_process_wall_s'] for c in children) if batch else None
        remainder = end['occupation_s']-child_sum if batch else None
        if remainder is not None and remainder < -1e-6:
            raise ValueError('children exceed monitored occupation')
        attempts.append(dict(attempt=start['attempt'],unit=start.get('unit'),exit_code=end['exit_code'],
            monitored_occupation_s=end['occupation_s'],child_process_wall_sum_s=child_sum,
            occupation_outside_children_s=remainder,batch_wall_s_nested=batch.get('total_wall_s') if batch else None,
            children=[c['path'] for c in children],monitor_cpu_s_nested=end.get('monitor_cpu_s'),
            terminal_child_cpu_s=end.get('terminal_child_cpu_s'),terminal_child_cpu_missing_reason=end.get('terminal_child_cpu_missing_reason'),
            successful_preflight_wall_s_nested=start['preflight'].get('wall_s'),
            limit_at_launch_s=start.get('limit_s'),charged_before_s_snapshot_nonadditive=start.get('charged_before_s')))
    return dict(schema='a17.closure.cost.receipt_audit.v1',snapshot_zip_sha256=snap.receipt['sha256'],
        totals=totals,attempts=attempts,states=states,verified_inputs=snap.used,
        interpretation=dict(occupation='Measured watchdog wall includes successful preflight, CPU/setup, online, offline and failures; not GPU-active time.',
        additivity='Only totals.mutually_exclusive_addends sum to cumulative charged time. Children partition attempt; driver/online/setup views are nested and never additional fees.',
        workspace='anchor/pool/projection are nested inside public_workspace; not additive to its containing interval.',
        finalist='Initial Jx and batched finalist Jd physical RHS from actual counters; selected versus unselected wall separation unavailable.',
        cold='Reconstructed standalone setup+policy fees reuse shared acquisition; no independent cold timing available.',
        gaussian='Cached logical teacher labels do not imply physical RHS. This audit uses actual physical counters only, never TeacherContext.counts.',
        unavailable='No reliable terminal child CPU total, no exact continuous CUDA-active wall, no per-k offline wall, no individual accepted/rejected Jd wall.',
        overlap='UTC start/end are ordered and nonoverlapping; successful preflight begins before start UTC. Occupation uses monotonic measured duration, not UTC differences.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();report=audit(args.snapshot);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'cost_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.out/'state_costs.csv').open('w',newline='') as stream:
        keys=['path','attempt','mode','object_id','phase','status','child_process_wall_s','measured_driver_wall_s','offline_evaluation_wall_s','driver_other_wall_s']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:s.get(k) for k in keys} for s in report['states'])
    print(json.dumps(dict(totals=report['totals'],states=len(report['states']),verified_inputs=len(report['verified_inputs']))))


if __name__=='__main__':
    main()
