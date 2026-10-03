"""NEW_PUBLIC_REANALYSIS: public authority/CSV and aggregate checks; no private ledger/provenance audit."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math
import public_support as S
OBJECTS=(2002,2003,2004,2005,2006,2008,2009,2010,2011,2013,2016)
PHASES=('early','middle','late');KS=(4,8,16)
CAPTURES=('balanced12_capture','anchor_top1_raw_noop_capture','anchor_top1_significance_verified_noop_capture')
RISKS=('actual_online_final_full_gap','local_teacher_final_full_gap','actual_minus_local_teacher_risk')
PATH_TERMINAL={'LOCAL_3ACTION_BUDGET_COMPLETE','NUMERICAL_LOCAL_1SWAP_STOP_NO_DETERMINISTIC_BOUND'}
def aggregate_initial(cells):
    rows=[]
    for oid in OBJECTS:
        domain=[c for c in cells if c['object_id']==oid]
        valid=[c for c in domain if c.get('binding_status')=='VERIFIED' and c.get('cell_status')=='COMPLETE']
        complete=len(domain)==9 and len(valid)==9
        positive=[c for c in valid if isinstance(c.get('teacher_best_raw_positive_gain'),(float,int)) and c['teacher_best_raw_positive_gain']>0]
        denominator=sum(c['teacher_best_raw_positive_gain'] for c in positive)
        row={'object_id':oid,'planned_cells':9,'complete_initial_cells':len(valid),'all_planned_initial_complete':complete,
             'positive_teacher_cells_observed':len(positive),'teacher_positive_gain_sum':denominator if complete else None,
             'scope':'Initial receiver neighborhood only; positive-full-teacher-Q weighted; zero opportunities excluded from ratio weights'}
        for field,label in [('balanced12_best_raw_positive_gain','balanced12_capture'),('full_dictionary_anchor_top1_noop_capture','anchor_top1_raw_noop_capture'),('full_dictionary_anchor_top1_verified_noop_gain','anchor_top1_significance_verified_noop_capture')]:
            values=[c.get(field) for c in positive]
            known=all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
            numerator=sum(v*c['teacher_best_raw_positive_gain'] if field.endswith('_capture') else v for c,v in zip(positive,values)) if known else None
            row[label]=numerator/denominator if complete and known and denominator>0 else None
        rows.append(row)
    return rows

def path_final(cells):
    return [{**{key:c.get(key) for key in ['object_id','phase','k','state_status','binding_status','teacher_path_status','teacher_path_length']},
             'actual_online_final_full_gap':c.get('actual_online_final_full_gap'),
             'local_teacher_final_full_gap':c.get('local_teacher_final_full_gap') if c.get('teacher_path_status') in {'LOCAL_3ACTION_BUDGET_COMPLETE','NUMERICAL_LOCAL_1SWAP_STOP_NO_DETERMINISTIC_BOUND'} else None,
             'actual_minus_local_teacher_risk':c.get('actual_minus_local_teacher_risk') if c.get('teacher_path_status') in {'LOCAL_3ACTION_BUDGET_COMPLETE','NUMERICAL_LOCAL_1SWAP_STOP_NO_DETERMINISTIC_BOUND'} else None,
             'scope':'Bounded local teacher trajectory; not global oracle; candidate/ranking/search combined. Partial path final maximum/comparison omitted.'} for c in cells]

def number(v):
 if v in (None,'','null','None'):return None
 x=float(v)
 if not math.isfinite(x):raise ValueError('nonfinite numeric field')
 return x

def boolean(v):
 if v in (True,'True','true'):return True
 if v in (False,'False','false'):return False
 raise ValueError('invalid boolean')

def csv_bytes(rows):
 s=io.StringIO(newline='');fields=sorted(set().union(*(set(r) for r in rows)))
 w=csv.DictWriter(s,fieldnames=fields);w.writeheader()
 w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
 return s.getvalue().encode()

def prepare(initial_rows,path_rows,labels):
 if len(initial_rows)!=11 or {int(r['object_id']) for r in initial_rows}!=set(OBJECTS):raise ValueError('initial denominator must retain 11 objects')
 if len(path_rows)!=99 or {(int(r['object_id']),r['phase'],int(r['k'])) for r in path_rows}!={(o,p,k) for o in OBJECTS for p in PHASES for k in KS}:raise ValueError('path denominator must retain all 99 cells')
 out_initial=[];out_path=[]
 for r in initial_rows:
  x=dict(r);x['object_id']=int(r['object_id']);n=int(r['complete_initial_cells']);planned=int(r['planned_cells']);complete=boolean(r['all_planned_initial_complete'])
  if planned!=9 or not 0<=n<=9 or complete!=(n==9):raise ValueError('inconsistent initial completeness')
  for k in (*CAPTURES,'teacher_positive_gain_sum'):x[k]=number(r.get(k))
  if not complete and any(x[k] is not None for k in (*CAPTURES,'teacher_positive_gain_sum')):raise ValueError('partial initial aggregate promoted')
  if complete and x['teacher_positive_gain_sum']==0 and any(x[k] is not None for k in CAPTURES):raise ValueError('undefined zero opportunity capture')
  for k in CAPTURES:
   if x[k] is not None and not -1e-10<=x[k]<=1+1e-10:raise ValueError('capture outside unit range')
  x['reason']='partial '+str(n)+'/9' if not complete else ('no positive opportunity' if x['teacher_positive_gain_sum']==0 else 'null metric')
  out_initial.append(x)
 for r in path_rows:
  x=dict(r);x['object_id']=int(r['object_id']);x['k']=int(r['k'])
  for k in RISKS:x[k]=number(r.get(k))
  eligible=r.get('binding_status')=='VERIFIED' and r.get('teacher_path_status') in PATH_TERMINAL
  if not eligible and any(x[k] is not None for k in RISKS[1:]):raise ValueError('partial/unverified teacher endpoint promoted')
  if r.get('binding_status')!='VERIFIED' and x[RISKS[0]] is not None:raise ValueError('unverified actual endpoint')
  if all(x[k] is not None for k in RISKS) and not math.isclose(x[RISKS[0]]-x[RISKS[1]],x[RISKS[2]],rel_tol=1e-8,abs_tol=1e-10):raise ValueError('risk difference inconsistent')
  if any(x[k] is not None and x[k]<0 for k in RISKS[:2]):raise ValueError('negative squared-risk metric')
  x['null_code']='P' if 'PARTIAL' in str(r.get('teacher_path_status')) or 'PARTIAL' in r.get('state_status','') else ('I' if 'INVALID' in r.get('binding_status','') else ('NR' if r.get('state_status','').startswith('NOT_RUN') else 'NA'))
  out_path.append(x)
 return {'initial':out_initial,'path':out_path,'labels':labels}
def validate(authority_path,authority_sha,initial_csv,path_csv,receipt_sha,synthetic=False):
 S.check(S.sha(authority_path)==authority_sha,'public authority SHA differs');a=S.read(authority_path)
 S.check(a.get('schema')=='a17.saved.coverage.evidence.v1' and a.get('synthetic_fixture',False)==synthetic,'wrong/synthetic public authority')
 S.check(a.get('snapshot_receipt_sha256')==receipt_sha,'receipt SHA binding differs')
 for key in ['snapshot_receipt_sha256','config_sha256','input_manifest_sha256','collector_sha256']:S.check(isinstance(a.get(key),str) and len(a[key])==64 and all(c in '0123456789abcdef' for c in a[key]),'missing identity SHA '+key)
 for file,key in [(initial_csv,'object_initial_opportunity'),(path_csv,'path_final_risk')]:S.check(Path(file).read_bytes()==csv_bytes(a[key]),'public CSV/authority bytes differ: '+key)
 with Path(initial_csv).open(newline='') as f:initial=list(csv.DictReader(f))
 with Path(path_csv).open(newline='') as f:path=list(csv.DictReader(f))
 prepared=prepare(initial,path,{})
 rechecked=False
 if 'cells' in a:
  cells=a['cells'];S.check(len(cells)==99 and {(c['object_id'],c['phase'],c['k']) for c in cells}=={(o,p,k) for o in OBJECTS for p in PHASES for k in KS},'public cell denominator differs')
  S.check(aggregate_initial(cells)==a['object_initial_opportunity'],'initial positive-Q aggregate differs')
  S.check(path_final(cells)==a['path_final_risk'],'bounded path view differs');rechecked=True
 return {'schema':'a17.public.coverage.aggregate.validation.v1','identity':'NEW_PUBLIC_REANALYSIS_NOT_HISTORICAL_RUNTIME','status':'PUBLIC_AUTHORITY_AND_CSV_VALIDATED','source_authority_sha256':authority_sha,'snapshot_receipt_sha256':receipt_sha,'input_csv_sha256':{'initial':S.sha(initial_csv),'path':S.sha(path_csv)},'planned_objects':11,'planned_cells':99,'initial_aggregation_recomputed_from_public_cells':rechecked,'missing_public_cells_reason':None if rechecked else 'PUBLIC_AUTHORITY_OMITS_CELL_LEVEL_RECORDS_VIEW_BINDING_ONLY','initial_scope':'Initial receiver-neighborhood opportunity only; positive full-teacher-Q weighted; zero opportunities have no ratio','path_scope':'Separate bounded local path final risk; no global oracle or deterministic certificate','private_source_bound_provenance_reaudited':False,'initial_complete_objects':sum(r['all_planned_initial_complete'] in [True,'True','true'] for r in initial),'path_null_cells':sum(any(number(r.get(k)) is None for k in RISKS) for r in path),'port_source_sha256':S.sha(__file__)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--authority',required=True);p.add_argument('--authority-sha256',required=True);p.add_argument('--initial-csv',required=True);p.add_argument('--path-csv',required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--out');a=p.parse_args();result=validate(a.authority,a.authority_sha256,a.initial_csv,a.path_csv,a.receipt_sha256)
 if a.out:
  with Path(a.out).open('xb') as f:f.write(S.canonical(result))
 print(json.dumps(result))
if __name__=='__main__':main()
