"""Metadata-only continuation after the observed logging namespace exception."""
from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parent
archive=ROOT/'execution/namespace_error_12264'
archive.mkdir()
for name in ('COST_STATE.json','COST_LEDGER.jsonl','RUN_TERMINAL.json',
             'SERIAL_EXECUTION.json','SERIAL_OUTCOMES.json','SERIAL_TERMINAL.json',
             'SERIAL_NOT_RUN.json','RESUME_PLAN_NATIVE.json'):
    path=ROOT/name
    if path.exists():shutil.copy2(path,archive/name)
shutil.copy2(ROOT.parents[3]/'run_serial_resume.log',archive/'run_serial_resume.log')
out=ROOT/'results/offset_shell/exchange'
shutil.copytree(out,archive/'partial_case')
old_final=json.loads((out/'final.json').read_text())
records=json.loads((out/'trajectory.json').read_text())
next_iteration=records[-1]['iteration']+1
meta=dict(next_iteration=next_iteration,prior_wall_s=old_final['wall_s'],
          snapshot='chi_%02d.npz'%next_iteration,committed_records=len(records),
          reason='fresh process after logging clean(np.ndarray) namespace exception',
          prior_final_status=old_final['status'],
          prior_attempt_record='execution/namespace_error_12264/partial_case',
          scientific_rules_changed=False)
for name in ('final.json','final.npz','error.json'):
    shutil.move(str(out/name),str(archive/('pre_resume_'+name)))
abandoned=out/('exchange_%02d'%next_iteration)
if abandoned.exists():shutil.move(str(abandoned),str(archive/abandoned.name))
reservation=ROOT/'results/offset_shell/receiver'
shutil.move(str(reservation),str(archive/'handoff_receiver_reservation'))
outcomes=json.loads((ROOT/'SERIAL_OUTCOMES.json').read_text())
done=[v for v in outcomes if (v['case'],v['policy'])!=('offset_shell','exchange')]
pending=[dict(case='offset_shell',policy=p) for p in
         ('exchange','receiver','projected_internal','fourier','random')]
plan=dict(completed_outcomes=done,pending=pending,
          partials={'offset_shell/exchange':meta},
          recovery_event='logging namespace exception; process retired at next unstarted task',
          repeated_numeric_checks=False,algorithm_or_threshold_changes=False)
(ROOT/'RESUME_PLAN_NATIVE.json').write_text(json.dumps(plan,indent=2))
event=dict(pid=12264,event='LOGGING_NAMESPACE_EXCEPTION',
           error="'builtin_function_or_method' object has no attribute 'ndarray'",
           location='original_exchange_helpers.clean',cause='not established',
           committed_update_count=next_iteration,
           recovery='fresh serial GPU process; continue committed prefix; no code-rule repair',
           failed_attempt_preserved=True,prior_cost_reset=False,extra_tests_run=False)
(ROOT/'execution/NAMESPACE_ERROR_12264.json').write_text(json.dumps(event,indent=2))
print(json.dumps(dict(completed=len(done),pending=len(pending),resume=meta)))
