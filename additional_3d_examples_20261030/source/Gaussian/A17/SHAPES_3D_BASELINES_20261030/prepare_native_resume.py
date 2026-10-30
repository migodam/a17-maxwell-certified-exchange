"""One-time metadata recovery from saved files after PID23388's native crash.

No model is imported and no numerical operation or experiment is run here.
"""
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[3]
fault = json.loads((ROOT/'execution/NATIVE_CRASH_SUMMARY.json').read_text())
archive = ROOT/'execution/native_crash_23388'
archive.mkdir()
for name in ('COST_STATE.json','COST_LEDGER.jsonl','RUN_TERMINAL.json',
             'GPU_OUTCOMES.json','PARALLEL_EXECUTION.json',
             'FAST_ENDPOINT_EXECUTION.json','CONFIG_EXTENDED.json'):
    path = ROOT/name
    if path.exists():
        shutil.copy2(path, archive/name)
log = BASE/'run_extended.log'
if log.exists():
    shutil.copy2(log, archive/log.name)
partials = {}
for policy in ('exchange','full_gn'):
    out = ROOT/'results/cross3d'/policy
    shutil.copytree(out, archive/'partial_cases'/policy)
    records = json.loads((out/'trajectory.json').read_text())
    progress = json.loads((out/'progress.json').read_text())
    next_iteration = int(records[-1]['iteration']) + 1
    estimated_start = (out/'progress.json').stat().st_mtime - progress['elapsed_s']
    prior_wall_s = math.ceil(fault['fault_unix_s'] - estimated_start) + 1
    partials['cross3d/'+policy] = dict(
        next_iteration=next_iteration, prior_wall_s=prior_wall_s,
        committed_records=len(records), snapshot='chi_%02d.npz' % next_iteration,
        case_start_source='progress file mtime minus recorded elapsed_s',
        estimated_case_start_unix_s=estimated_start,
        prior_wall_scope='conservative through native-crash event, ceiling+1sec',
        fault_event_unix_s=fault['fault_unix_s'])
    abandoned = out/('exchange_%02d' % next_iteration)
    if abandoned.exists():
        shutil.move(str(abandoned),str(archive/('abandoned_'+abandoned.name)))

cost_path=ROOT/'COST_STATE.json'
cost=json.loads(cost_path.read_text())
session=cost['sessions'][-1]
if session['pid'] != 23388 or session['status'] != 'RUNNING':
    raise RuntimeError('Expected untouched crashed session; no accounting mutation made')
cost['wall_s'] += fault['charged_wall_s']
session.update(status='NATIVE_CRASH_C0000005',wall_s=fault['charged_wall_s'],
               ended=datetime.fromtimestamp(fault['fault_unix_s'],timezone.utc).isoformat(),
               accounting=fault)
cost.pop('active_wall_s',None)
cost_path.write_text(json.dumps(cost,indent=2))

outcomes=json.loads((ROOT/'GPU_OUTCOMES.json').read_text(encoding='utf-8-sig'))
by_key={(x['case'],x['policy']):x for x in outcomes}
for path in sorted((ROOT/'results').glob('*/*/final.json')):
    f=json.loads(path.read_text())
    by_key[(f['case'],f['policy'])]=dict(case=f['case'],policy=f['policy'],
        status=f['status'],execution_host='existing_remote_terminal_preserved')
pending=[dict(case='cross3d',policy=p) for p in
         ('exchange','full_gn','receiver','projected_internal','fourier')]
pending += [dict(case='offset_shell',policy=p) for p in
            ('exchange','full_gn','receiver','projected_internal','fourier','random')]
plan=dict(completed_outcomes=list(by_key.values()),pending=pending,partials=partials,
          native_crash_record=fault,completed_trajectories_never_rerun=True,
          physical_or_numeric_tests_run=False,cumulative_wall_after_fault=cost['wall_s'])
(ROOT/'RESUME_PLAN_NATIVE.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(dict(completed=len(by_key),pending=len(pending),partials=partials,
                      cumulative_wall_s=cost['wall_s'],remaining_wall_s=10800-cost['wall_s'])))
