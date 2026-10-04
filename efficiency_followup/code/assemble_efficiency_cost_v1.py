"""Offline, whitelist-only cost assembly from a final collector V2 snapshot.

Never imports a watcher, opens live execution ledgers, or launches numerical work.
Whole external occupation includes successful preflight/monitor cost; nested CPU
or child-wall fields are never added again. Missing evidence leaves totals OPEN.
"""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='a17.efficiency.final_cost.v1'
DRIVERS={
 'run_nonlinear_efficient.py':'a17.efficiency.nonlinear_transfer.v1',
 'run_nonlinear_reuse.py':'a17.efficiency.nonlinear_reuse_transfer.v1',
 'run_nonlinear_top1cap3_v1.py':'a17.efficiency.nonlinear_top1cap3_transfer.v1',
}
POLICIES={'receiver_only','original_top2_cap3','top1_cap1','top1_cap2','top1_cap3','top2_cap1','reuse_previous','reuse_reset3'}
STOPS={'TIME_OR_CUMULATIVE_BUDGET_LIMIT','GPU_MEMORY_90PCT_STOP','MONITOR_FAILURE','WATCHDOG_EXCEPTION'}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0

def safe_hash(value):return value if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) else None

def identifier(value):return value if isinstance(value,str) and re.fullmatch('[A-Za-z0-9_-]{1,200}',value) else None

def method_metadata(row):
 # Read only whitelisted command tokens; never export argv/path/host/error strings.
 command=row.get('command',[]);command=command if isinstance(command,list) else []
 drivers=[name for token in command if isinstance(token,str) for name in DRIVERS if token.replace('\\','/').split('/')[-1]==name]
 version=DRIVERS[drivers[0]] if len(drivers)==1 else None
 policy=None;obj=None;rep=None;cache=None
 for flag in ('--policy','--object','--repetition'):
  positions=[i for i,x in enumerate(command) if x==flag]
  if len(positions)==1 and positions[0]+1<len(command):
   value=command[positions[0]+1]
   if flag=='--policy' and value in POLICIES:policy=value
   if flag=='--object' and str(value) in ('2002','2009','2005'):obj=int(value)
   if flag=='--repetition' and str(value) in ('0','1','2'):rep=int(value)
 if '--geometry-cache' in command and '--no-geometry-cache' not in command:cache=True
 if '--no-geometry-cache' in command and '--geometry-cache' not in command:cache=False
 return dict(version=version,method=policy,object=obj,repetition=rep,geometry_cache=cache,
             method_metadata_status='KNOWN_COMMAND_TOKENS_ONLY' if version else 'NOT_DECLARED')

def parse_ledger(raw,source):
 rows=[];errors=[]
 for line,text in enumerate(raw.decode('utf-8').splitlines(),1):
  if not text.strip():continue
  try:record=json.loads(text)
  except (ValueError,UnicodeError):errors.append(dict(source=source,line=line,status='MALFORMED_JSON'));continue
  if not isinstance(record,dict):errors.append(dict(source=source,line=line,status='NONOBJECT_ROW'));continue
  rows.append((line,record,digest(text.encode())))
 return rows,errors

def assemble(snapshot,out):
 snapshot=Path(snapshot).resolve();out=Path(out).resolve()
 if snapshot==ROOT or snapshot==ROOT/'execution' or snapshot.is_relative_to(ROOT/'execution'):
  raise ValueError('Only a final collector V2 snapshot is accepted')
 if out.exists():raise ValueError('Fresh output directory required')
 private=snapshot/'private_audit';issues=[];bindings=[];rawfiles={}
 summary_path=snapshot/'FINAL_EVIDENCE_SUMMARY_V2.json'
 if not summary_path.is_file():raise ValueError('Final collector V2 summary missing')
 summary_raw=summary_path.read_bytes();summary=json.loads(summary_raw)
 if summary.get('schema')!='a17.efficiency.final_private_evidence.v2' or summary.get('status')!='IMMUTABLE_PRIVATE_SNAPSHOT_VERIFIED' or summary.get('owner_final_idle') is not True:
  raise ValueError('Final immutable collector V2 evidence required')
 inventory=summary.get('files',[])
 wanted=['execution/gpu_attempts.jsonl','execution/preflight_checks.jsonl','research/BUDGET_CARRY_LOCK.json']
 for relative in wanted:
  p=private/relative;expected=[x for x in inventory if x.get('path')==relative and x.get('download') is True]
  if p.is_symlink() or not p.resolve().is_relative_to(private.resolve()):
   bindings.append(dict(source=relative,status='UNSAFE_INPUT_PATH_REJECTED',sha256=None));issues.append(dict(source=relative,status='UNSAFE_INPUT_PATH_REJECTED'));continue
  if not p.is_file():bindings.append(dict(source=relative,status='MISSING',sha256=None));issues.append(dict(source=relative,status='MISSING_NO_IMPLIED_ZERO'));continue
  raw=p.read_bytes();sha=digest(raw)
  valid=len(expected)==1 and safe_hash(expected[0].get('sha256'))==sha and expected[0].get('bytes')==len(raw)
  bindings.append(dict(source=relative,status='HASH_BOUND' if valid else 'HASH_BINDING_REJECTED',sha256=sha,bytes=len(raw)))
  if not valid:issues.append(dict(source=relative,status='HASH_BINDING_REJECTED'));continue
  rawfiles[relative]=raw
 carry=None
 if wanted[2] in rawfiles:
  try:carry=json.loads(rawfiles[wanted[2]])
  except ValueError:issues.append(dict(source=wanted[2],status='MALFORMED_CARRY'))
 prior=None;ceiling=None
 if isinstance(carry,dict):
  valid=carry.get('schema')=='a17.efficiency.budget_carry.v1'
  values=[]
  for epoch in ('baseline','closure'):
   rec=carry.get(epoch,{})
   for key in ('occupation_s','guard_s'):
    if number(rec.get(key)):values.append(rec[key])
    else:valid=False
   if not safe_hash(rec.get('sha256')) or rec.get('unpaired')!=[]:valid=False
   if rec.get('guard_s',0)>0 and not safe_hash(rec.get('guard_sha256')):valid=False
  prior=carry.get('prior_closed_occupation_s');ceiling=carry.get('total_ceiling_s')
  valid=valid and number(prior) and number(ceiling) and abs(prior-19962.060)<=1e-6 and abs(ceiling-43200.)<=1e-6 and abs(math.fsum(values)-prior)<=1e-6
  if not valid:issues.append(dict(source=wanted[2],status='CARRY_REJECTED'));prior=ceiling=None
 events=[];starts={};ends={};monitors={};attempts=[];guards=[]
 if wanted[0] in rawfiles:
  rows,errs=parse_ledger(rawfiles[wanted[0]],wanted[0]);issues.extend(errs)
  for line,r,rowsha in rows:
   event=r.get('event');attempt=identifier(r.get('attempt'));unit=identifier(r.get('unit'))
   status='RECORDED' if event in ('start','end','monitor_error') else 'UNKNOWN_EVENT_REJECTED'
   if event not in ('start','end','monitor_error'):issues.append(dict(source=wanted[0],line=line,status=status));event='UNKNOWN'
   if event in ('start','end') and attempt is None:issues.append(dict(source=wanted[0],line=line,status='INVALID_ATTEMPT_ID'));continue
   public=dict(source=wanted[0],line=line,record_sha256=rowsha,event=event,attempt=attempt,unit=unit,status=status)
   if event=='start':
    public.update(method_metadata(r));public['identity_sha256']=safe_hash(r.get('identity',{}).get('identity_sha256'));public['config_sha256']=safe_hash(r.get('identity',{}).get('config',{}).get('sha256'))
    for key in ('wall_s','cpu_s'):
     value=r.get('preflight',{}).get(key);public['nested_preflight_'+key]=value if number(value) else None
    starts.setdefault(attempt,[]).append((line,r,public))
   if event=='end':
    value=r.get('occupation_s');public['external_occupation_s']=value if number(value) else None
    value=r.get('monitor_cpu_s');public['nested_monitor_cpu_s']=value if number(value) else None
    public['exit_code']=r.get('exit_code') if isinstance(r.get('exit_code'),int) and not isinstance(r.get('exit_code'),bool) else None
    public['stop_status']='NONE' if r.get('stop_reason') is None else r['stop_reason'] if r.get('stop_reason') in STOPS else 'UNKNOWN_STOP_REASON'
    public['identity_sha256']=safe_hash(r.get('identity_sha256'));ends.setdefault(attempt,[]).append((line,r,public))
    if attempt not in starts:issues.append(dict(source=wanted[0],line=line,status='END_BEFORE_START'))
   if event=='monitor_error':
    monitors[attempt]=monitors.get(attempt,0)+1
    if r.get('child_status_unknown') is True:issues.append(dict(source=wanted[0],line=line,status='MONITOR_CHILD_STATUS_UNKNOWN_OPEN'))
   events.append(public)
  for attempt in sorted(starts.keys()|ends.keys()):
   a=starts.get(attempt,[]);b=ends.get(attempt,[]);valid=len(a)==len(b)==1
   failure='OPEN_UNPAIRED' if not a or not b else 'DUPLICATE_REJECTED' if not valid else None
   if valid:
    s,e=a[0][2],b[0][2]
    if number(e['external_occupation_s']) and e['external_occupation_s']>7200.:issues.append(dict(source=wanted[0],line=b[0][0],status='PER_JOB_OCCUPATION_LIMIT_EXCEEDED'))
    if not number(e['external_occupation_s']):failure='UNKNOWN_OCCUPATION_REJECTED';valid=False
    elif s['identity_sha256'] is None or s['identity_sha256']!=e['identity_sha256']:failure='IDENTITY_HASH_REJECTED';valid=False
    elif s['unit'] is None or s['unit']!=e['unit']:failure='UNIT_PAIR_REJECTED';valid=False
   if not valid:issues.append(dict(source=wanted[0],attempt=attempt,status=failure))
   s=a[0][2] if a else {};e=b[0][2] if b else {}
   state=failure if failure else 'CLOSED_SUCCESS' if e.get('exit_code')==0 and e.get('stop_status')=='NONE' else 'CLOSED_FAILED_OR_STOPPED' if e.get('exit_code') is not None else 'CLOSED_EXIT_STATUS_UNKNOWN'
   if state=='CLOSED_EXIT_STATUS_UNKNOWN':issues.append(dict(source=wanted[0],attempt=attempt,status=state))
   if e.get('stop_status')=='UNKNOWN_STOP_REASON':issues.append(dict(source=wanted[0],attempt=attempt,status='UNKNOWN_STOP_REASON'))
   attempts.append(dict(attempt=attempt,unit=s.get('unit',e.get('unit')),status=state,
     occupation_s=e.get('external_occupation_s') if valid else None,occupation_charged=valid,
     start_source=wanted[0],start_line=a[0][0] if a else None,end_source=wanted[0],end_line=b[0][0] if b else None,
     config_sha256=s.get('config_sha256'),identity_sha256=s.get('identity_sha256'),
     monitor_error_count=monitors.get(attempt,0),monitor_cpu_s_nested=e.get('nested_monitor_cpu_s'),
     successful_preflight_wall_s_nested=s.get('nested_preflight_wall_s'),exit_code=e.get('exit_code'),
     child_wall_s=None,child_wall_status='NOT_MEASURED_FROM_EXECUTION_LEDGERS',
     **{k:s.get(k) for k in ('version','method','object','repetition','geometry_cache','method_metadata_status')}))
 # monitor errors must reference a known attempt; CPU is nested, never charged twice.
 for attempt in monitors:
  if attempt is None or attempt not in starts:issues.append(dict(source=wanted[0],status='MONITOR_ERROR_ATTEMPT_UNKNOWN'))
 if wanted[1] in rawfiles:
  rows,errs=parse_ledger(rawfiles[wanted[1]],wanted[1]);issues.extend(errs);seen=set()
  for line,r,rowsha in rows:
   attempt=identifier(r.get('attempt'));value=r.get('wall_s');valid=r.get('event')=='preflight_rejected' and attempt is not None and number(value)
   status='CHARGED_REJECTED_PREFLIGHT' if valid else 'GUARD_REJECTED'
   if attempt in seen:valid=False;status='DUPLICATE_GUARD_REJECTED'
   if attempt in starts or attempt in ends:valid=False;status='GUARD_OVERLAPS_ATTEMPT_REJECTED'
   seen.add(attempt)
   if not valid:issues.append(dict(source=wanted[1],line=line,status=status))
   unit_match=re.fullmatch(r'a17_efficiency_([A-Za-z0-9_-]+)_[0-9a-f]{32}',attempt or '')
   guards.append(dict(unit=unit_match.group(1) if unit_match else None,unit_status='ATTEMPT_ID_UNIT' if unit_match else 'NOT_DECLARED',source=wanted[1],line=line,record_sha256=rowsha,attempt=attempt,status=status,guard_wall_s=value if valid else None,charged=valid,nested_monitor_cpu_s=r.get('monitor_cpu_s') if number(r.get('monitor_cpu_s')) else None))
 known_attempts=math.fsum(x['occupation_s'] for x in attempts if x['occupation_charged'])
 known_guards=math.fsum(x['guard_wall_s'] for x in guards if x['charged'])
 complete=not issues and len(rawfiles)==3
 charged=prior+known_attempts+known_guards if complete else None
 budget=dict(schema=SCHEMA,status='CLOSED_LEDGER_ACCOUNTING_ONLY' if complete else 'OPEN_OR_REJECTED_NO_FINAL_TOTAL',
  prior_carry_s=prior,total_ceiling_s=ceiling,new_known_paired_occupation_s=known_attempts,new_known_guard_s=known_guards,
  cumulative_charged_s=charged,remaining_s=ceiling-charged if charged is not None else None,
  ceiling_exceeded=charged>ceiling if charged is not None else None,closed_charged_attempts=sum(x['occupation_charged'] for x in attempts),
  failed_or_stopped_attempts=sum(x['status']=='CLOSED_FAILED_OR_STOPPED' for x in attempts),
  cost_scope='Occupation includes successful preflight/setup/monitor/evaluation/failures; rejected preflight guard charged separately. Nested CPU and child wall never added.',
  child_wall_status='NOT_MEASURED_FROM_EXECUTION_LEDGERS',snapshot_summary_sha256=digest(summary_raw),
  assembler_sha256=digest(Path(__file__).read_bytes()),physics_launched=False,network=False,gate_decision='NONE')
 # Detect concurrent mutation of a purported immutable snapshot before publishing.
 if summary_path.read_bytes()!=summary_raw or any((private/k).read_bytes()!=v for k,v in rawfiles.items()):raise ValueError('Snapshot changed while read')
 out.mkdir(parents=True,exist_ok=False)
 units=[]
 for unit in sorted({x['unit'] for x in attempts+guards if x.get('unit') is not None}):
  a=[x for x in attempts if x.get('unit')==unit];g=[x for x in guards if x.get('unit')==unit]
  valid=all(x['occupation_charged'] for x in a) and all(x['charged'] for x in g)
  known=math.fsum([x['occupation_s'] for x in a if x['occupation_charged']]+[x['guard_wall_s'] for x in g if x['charged']])
  units.append(dict(unit=unit,status='KNOWN_UNIT_RECORD_SUM_ONLY' if valid else 'OPEN_UNIT_RECORDS',occupation_and_rejected_guard_s=known if valid else None,known_record_charge_s=known,attempt_count=len(a),guard_count=len(g),failed_or_stopped_count=sum(x['status']=='CLOSED_FAILED_OR_STOPPED' for x in a),source='attempts.json/guard_fees.json; line links retained there'))
 tables={'unit_costs':units,'attempts':attempts,'guard_fees':guards,'whitelisted_events':events,'input_bindings':bindings,'failures_and_open':issues}
 for name,rows in tables.items():
  (out/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
  fields=list(dict.fromkeys(k for row in rows for k in row))
  with (out/(name+'.csv')).open('w',newline='') as f:
   writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
 (out/'BUDGET_RECEIPT.json').write_text(json.dumps(budget,indent=2,allow_nan=False)+'\n')
 return budget

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'analysis/final_cost_v1');a=p.parse_args()
 receipt=assemble(a.snapshot,a.out)
 print(json.dumps({k:receipt[k] for k in ('schema','status','closed_charged_attempts','failed_or_stopped_attempts','cumulative_charged_s','remaining_s')}))
if __name__=='__main__':main()
