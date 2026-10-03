"""Standard-library public CSV fee reconstruction. New reanalysis source, not monitor audit."""
from pathlib import Path
import argparse,csv,hashlib,json,math,re
ATTEMPT_COLUMNS={'attempt','partition','exit_code','occupation_s','child_process_wall_sum_s','occupation_outside_children_s','successful_preflight_nested_s','terminal_child_cpu_s','terminal_child_cpu_reason','terminal_child_cpu_reason_sha256','child_wall_partition_reason','peak_device_used_mib','charge_kind','source_json_sha256'}
PREFLIGHT_COLUMNS={'attempt','partition','wall_s','charge_kind','source_json_sha256'}
PHASES={'original_replay','primary_validation','noise','nonlinear_transfer','offline_coverage','gate_or_failed_setup'}
def check(value,message):
 if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def numeric(value):
 check(value is not None and value!='','missing additive fee must not become zero');x=float(value);check(math.isfinite(x) and x>=0,'invalid fee');return x
def load(path,allowed,required):
 with Path(path).open(newline='') as f:
  reader=csv.DictReader(f);headers=reader.fieldnames;check(headers is not None and len(headers)==len(set(headers)) and set(headers)<=allowed and required<=set(headers),'unsafe/unknown/missing CSV fields');rows=list(reader)
 for row in rows:check(None not in row and None not in row.values(),'malformed CSV row')
 return rows
def validate(attempt_csv,preflight_csv,authority_json,authority_sha256):
 check(sha(authority_json)==authority_sha256,'public authority SHA differs');a=json.loads(Path(authority_json).read_text());check(a.get('schema')=='a17.public.cost.derivation.v1' and a.get('raw_monitor_reaudit_supported') is False,'public derivative authority required');source=a['source_authoritative_json_sha256'];check(re.fullmatch('[0-9a-f]{64}',source) is not None,'private source JSON SHA binding missing')
 attempts=load(attempt_csv,ATTEMPT_COLUMNS,{'attempt','partition','exit_code','occupation_s','charge_kind','source_json_sha256'});preflights=load(preflight_csv,PREFLIGHT_COLUMNS,PREFLIGHT_COLUMNS);ids=set();rejectids=set();phase_occupation={p:[] for p in PHASES};phase_rejected={p:[] for p in PHASES}
 for rows,expected,charge,key in [(attempts,a['attempts'],'ADDITIVE_WHOLE_ATTEMPT_OCCUPATION','occupation_s'),(preflights,a['rejected_preflights'],'ADDITIVE_REJECTED_PREFLIGHT','wall_s')]:
  check(len(rows)==len(expected),'CSV/authority row count differs')
  for row,original in zip(rows,expected):
   for field in row:check(row[field]==('' if original.get(field) is None else str(original[field])),'CSV field differs from public authority: '+field)
   check(row['source_json_sha256']==source and row['charge_kind']==charge and row['partition'] in PHASES,'fee scope/source/phase differs');name=row['attempt'];check(re.fullmatch('[A-Za-z0-9_.:-]+',name) is not None,'unsafe attempt identifier');target=ids if key=='occupation_s' else rejectids;check(name not in target,'duplicate charge');target.add(name);value=numeric(row[key]);(phase_occupation if key=='occupation_s' else phase_rejected)[row['partition']].append(value)
 check(not ids & rejectids,'preflight charged again as occupation');occupation=math.fsum(numeric(r['occupation_s']) for r in attempts);rejected=math.fsum(numeric(r['wall_s']) for r in preflights);carry=numeric(a['totals']['old_carry_s']);campaign=math.fsum([carry,occupation,rejected]);t=a['totals'];check(occupation==t['closure_monitored_occupation_s'] and rejected==t['rejected_preflight_wall_s'] and campaign==t['cumulative_charged_s'],'exact additive campaign mismatch');check(len(attempts)==t['closure_paired_attempts'] and len(preflights)==t['rejected_preflight_count'],'charge count differs')
 phase_rows=[]
 for r in a['phase_totals']:
  p=r['partition'];check(p in PHASES,'unknown phase');occ=math.fsum(phase_occupation[p]);reject=math.fsum(phase_rejected[p]);check(occ==r['occupation_s'] and reject==r['rejected_preflight_s'] and math.fsum([occ,reject])==r['closure_phase_fee_s'],'phase fee mismatch');phase_rows.append({'partition':p,'occupation_s':occ,'rejected_preflight_s':reject})
 check(len(phase_rows)==len(PHASES) and {r['partition'] for r in phase_rows}==PHASES,'phase denominator differs')
 return {'schema':'a17.public.fee.csv.reconstruction.v1','status':'PASS_PUBLIC_DERIVED_CSV_FEES','source_authoritative_json_sha256':source,'public_authority_sha256':authority_sha256,'input_sha256':{'attempt_csv':sha(attempt_csv),'rejected_preflight_csv':sha(preflight_csv)},'old_carry_s':carry,'occupation_s':occupation,'rejected_preflight_s':rejected,'campaign_s':campaign,'failed_attempts_retained':sum(int(r['exit_code'])!=0 for r in attempts),'rejected_preflights_retained':len(preflights),'phase_totals':phase_rows,'nested_stages_added':False,'raw_private_monitor_reaudited':False,'scope':'Reconstruct audited public CSV fees only. Private SSH/CIM/command monitor originals are not available or audited by this program.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt-csv',required=True);p.add_argument('--preflight-csv',required=True);p.add_argument('--public-authority',required=True);p.add_argument('--public-authority-sha256',required=True);p.add_argument('--out');a=p.parse_args();result=validate(a.attempt_csv,a.preflight_csv,a.public_authority,a.public_authority_sha256)
 if a.out:
  with Path(a.out).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
