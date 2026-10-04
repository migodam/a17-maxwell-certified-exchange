"""Local saved-result analysis only. No runtime imports, solves, SSH, or gate decisions."""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics, collections, re, time
VERSION='a17.efficiency.local_analysis.v3'
NONLINEAR_DRIVER_FILES={
 'a17.efficiency.nonlinear_transfer.v1':'code/run_nonlinear_efficient.py',
 'a17.efficiency.nonlinear_reuse_transfer.v1':'code/run_nonlinear_reuse.py',
 'a17.efficiency.nonlinear_top1cap3_transfer.v1':'code/run_nonlinear_top1cap3_v1.py',
}
EFF=Path(__file__).resolve().parents[1]
POLICIES=['original_top2_cap3','top1_cap3','top2_cap1','top1_cap1','top1_cap2']
STAGES=('receiver_wall_s','Jx_initialization_wall_s','endpoint_wall_s','anchor_wall_s','Jd_wall_s','acceptance_wall_s')
def digest(raw): return hashlib.sha256(raw).hexdigest()
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def div(a,b): return a/b if finite(a) and finite(b) and b!=0 else None
def stats(values):
 v=[x for x in values if finite(x)]
 if not v:return dict(n=0,median=None,mad=None,min=None,max=None,std=None)
 m=statistics.median(v)
 return dict(n=len(v),median=m,mad=statistics.median(abs(x-m) for x in v),min=min(v),max=max(v),std=statistics.stdev(v) if len(v)>1 else None)
def counter_delta(a,b):
 a=(a or {}).get('totals',{});b=(b or {}).get('totals',{})
 return {k:b.get(k,0)-a.get(k,0) for k in a.keys()|b.keys()}
def risk_error(r):
 values=[r.get(k) for k in ('full_gap','half_H_step_energy','reference_gradient_cross')]
 return values[0]-values[1]-values[2] if all(finite(x)for x in values)else None
def safe_json(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:safe_json(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe_json(v)for v in x]
 return x
def nonlinear_identity(a,cfg,scope,authorized_locks):
 """Receipt identity only; no scientific acceptance or metric changes."""
 lock=cfg.get('lock_receipt',{})
 candidates=[lock.get('config_sha256'),lock.get('config',{}).get('sha256'),cfg.get('config_sha256'),cfg.get('closure_config_sha256')]
 supplied=[x for x in candidates if x is not None]
 configsha=supplied[0] if supplied else None
 reasons=[]
 if len(set(supplied))>1:reasons.append('conflicting_config_sha256_fields')
 version=a.get('version','UNKNOWN');pol=a.get('policy');gc=a.get('geometry_cache');rep=a.get('repetition')
 method=pol+(':geometry_cache'if gc is True else ':uncached'if gc is False else '')
 sources=lock.get('source_hashes',{})
 declared=a.get('runtime_end',{}).get('declared_efficiency',{})
 driver=declared.get('nonlinear_driver_sha256')
 source_sha=digest(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()) if sources else None
 strict=scope=='eff_nonlinear'
 if strict:
  driver_file=NONLINEAR_DRIVER_FILES.get(version)
  if driver_file is None:reasons.append('unknown_nonlinear_driver_version')
  if not isinstance(configsha,str) or not re.fullmatch('[0-9a-f]{64}',configsha):reasons.append('missing_or_invalid_config_binding')
  expected=authorized_locks.get(configsha)
  if expected is None:reasons.append('config_not_in_explicit_authorized_locks')
  else:
   if version!=expected.get('nonlinear_version'):reasons.append('driver_version_differs_from_lock')
   if cfg.get('version')!=version:reasons.append('run_config_version_differs_from_result')
   if a.get('object_id') not in expected.get('nonlinear_objects',[]):reasons.append('object_outside_lock')
   if dict(policy=pol,geometry_cache=gc,repetition=rep) not in expected.get('authorized_routes',[]):reasons.append('method_cache_rep_not_authorized')
   if not sources or sources!=expected.get('source_hashes'):reasons.append('source_hashes_differ_from_lock')
   if not driver or driver!=expected.get('source_hashes',{}).get(driver_file):reasons.append('driver_hash_differs_from_lock')
   if cfg.get('numerical_environment')!=expected.get('numerical_environment'):reasons.append('numerical_environment_differs_from_lock')
  cli=cfg.get('cli',{})
  for key,value in [('object',a.get('object_id')),('policy',pol),('geometry_cache',gc),('repetition',rep)]:
   if cli.get(key)!=value:reasons.append('cli_result_mismatch_'+key)
  if declared.get('geometry_cache')!=gc:reasons.append('declared_cache_result_mismatch')
 # Legacy historical strata retain their schema and cannot enter EFF pairings.
 stratum=version+'|'+str(configsha)
 if strict:stratum+='|sources='+str(source_sha)+'|driver='+str(driver)
 return dict(config_sha256=configsha,comparison_stratum=stratum,method=method,
             match_identity_eligible=not reasons,match_identity_errors=reasons,
             match_identity_source_sha256=source_sha,match_identity_driver_sha256=driver,
             match_identity_scope='explicit_lock_source_driver_method_cache_rep' if strict else 'legacy_saved_schema_separate_scope')

def pair_identity(row):
 return (row['scope'],row['object'],row['comparison_stratum'],row['repetition'])

class Reader:
 def __init__(self):self.manifest=[];self.cache={};self.aliases=[];self.errors=[]
 def read(self,p,optional=False):
  p=Path(p)
  if not p.exists():return None
  key=str(p.resolve())
  if key in self.cache:return self.cache[key]
  try:
   raw=p.read_bytes();data=json.loads(raw);sha=digest(raw)
   self.manifest.append(dict(path=key,sha256=sha,bytes=len(raw)))
   self.cache[key]=(data,sha);return data,sha
  except (OSError,ValueError)as e:
   self.errors.append(dict(path=key,error=str(e)));return None

def emit(out,name,rows):
 out.mkdir(parents=True,exist_ok=True);rows=safe_json(rows)
 (out/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
 if isinstance(rows,list):
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (out/(name+'.csv')).open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   w.writerows({k:json.dumps(v,sort_keys=True)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)

def run(args):
 started=time.perf_counter();out=args.out.resolve();read=Reader();frozen=[];nl=[];cost=[];profiles=[];checks=[];cache_records=[];block_controls=[];occupation=[];availability=[];seen_result={};seen_cell=set();conflicts=[]
 authorized_locks={}
 expected_locks=args.expected_lock if args.expected_lock else sorted((EFF/'configs').glob('*.json'))
 for lockpath in expected_locks:
  loaded=read.read(lockpath)
  if loaded and loaded[0].get('schema')=='a17.efficiency.nonlinear.lock.v1':authorized_locks[loaded[1]]=loaded[0]
 def cache_fields(report,scope,oid,version,sha,p):
  if not isinstance(report,dict):return dict(cache_acquisitions=None,cache_hits=None,cache_hit_observed=None,cache_actual_branches=None,reported_physical_admission=None)
  records=report.get('records',[])
  for index,record in enumerate(records):
   cache_records.append(dict(scope=scope,object=oid,version=version,run_id=sha,record_index=index,actual_branch=record.get('branch'),status=record.get('status'),wall_s=record.get('wall_s'),exclusive_stage_wall_s=record.get('exclusive_stage_wall_s'),unattributed_wall_s=record.get('unattributed_wall_s'),hash_read_bytes=record.get('hash_read_bytes'),bank_storage_copy_bytes=record.get('bank_storage_copy_bytes'),return_copy_bytes=record.get('return_copy_bytes'),audited_provider_sha256=record.get('audited_provider_sha256'),source=str(p.resolve()),phase_label_is_branch_evidence=False))
  hits=report.get('hits')
  return dict(cache_acquisitions=report.get('acquisitions'),cache_hits=hits,cache_invalidations=report.get('invalidations'),cache_hit_observed=hits>0 if finite(hits)else None,cache_actual_branches=[r.get('branch')for r in records],reported_physical_admission=report.get('physical_admission'),cache_actual_callables=report.get('actual_callables'),phase_label_is_branch_evidence=False)
 def result_files(root):
  return sorted(p for p in root.rglob('*.json') if p.name in ('result.json','profile_result.json','verification_result.json','failure.json')) if root.exists() else []
 def frozen_row(a,p,sha,scope,online=None):
  oid=a.get('object_id');phase=a.get('phase');k=a.get('k');policy=a.get('policy','original_top2_cap3');cellhash=digest(json.dumps(a,sort_keys=True,separators=(',',':')).encode());identity=(scope,oid,phase,k,policy)
  if (identity,cellhash)in seen_cell:return
  if any(r['scope']==scope and r['object']==oid and r['phase']==phase and r['k']==k and r['policy']==policy for r in frozen):conflicts.append(dict(kind='different_frozen_cell_payloads',identity=identity,source=str(p)))
  seen_cell.add((identity,cellhash));pc=a.get('pilot_cost',{});rr=a.get('receiver_risk',{});fr=a.get('final_risk',{});gain=a.get('receiver_to_final_full_gap_reduction')
  if gain is None and finite(rr.get('full_gap'))and finite(fr.get('full_gap')):gain=rr['full_gap']-fr['full_gap']
  online=online or {};stage=online.get('stage_wall_s',{});accepted_gain=a.get('sum_positive_accepted_gain')
  if accepted_gain is None and online:accepted_gain=sum(x.get('Q_true',0)for x in online.get('accepted',[])if x.get('Q_true',0)>0)
  total_rhs=a.get('direction_cost',online.get('direction_cost',{})).get('tangent_rhs');initial_rhs=pc.get('Jx_physical_rhs',online.get('_receiver_cost',{}).get('guarded_Jx_initialization_rhs'));jd_rhs=pc.get('Jd_physical_rhs')
  if jd_rhs is None and finite(total_rhs)and finite(initial_rhs):jd_rhs=total_rhs-initial_rhs
  row=dict(scope=scope,object=oid,phase=phase,k=k,policy=policy,source=str(p.resolve()),result_sha256=sha,cell_payload_sha256=cellhash,status='MEASURED',gain= gain,sum_positive_accepted_gain=accepted_gain,gain_retention=None,baseline_gain=None,Jd_directions=a.get('Jd_directions',online.get('efficiency_policy',{}).get('full_Jd_directions',sum(len(t.get('finalists',[]))for t in online.get('rounds',[]))if 'rounds'in online else None)),Jx_directions=a.get('Jx_directions',1),Jd_physical_rhs=jd_rhs,Jx_physical_rhs=initial_rhs,total_tangent_rhs=a.get('direction_cost',online.get('direction_cost',{})).get('tangent_rhs'),incremental_online_wall_s=pc.get('online_wall_with_engine_setup_s',online.get('wall_online_total_s')),cost_scope=pc.get('cache_scope','ORIGINAL_SHARED_ACQUISITION'),standalone_cold_wall_s=pc.get('standalone_cold_wall_s'),accepted_count=a.get('accepted_count',len(online.get('accepted',[]))),receiver_full_gap=rr.get('full_gap'),final_full_gap=fr.get('full_gap'),receiver_risk_identity_error=risk_error(rr),final_risk_identity_error=risk_error(fr),telescoping_gain_identity_error=gain-accepted_gain if finite(gain)and finite(accepted_gain)else None,reference_gradient_relative_residual=a.get('original_reference_relative_residual',a.get('reference_relative_residual')),gradient_scope='g(0) only; g(x) NOT_SAVED',gradient_at_x_available=False,quality_material_error=a.get('offline_one_step_quality',a.get('one_step_truth',{})).get('relative_material_error'),constraints_violated=a.get('offline_one_step_quality',a.get('one_step_truth',{})).get('constraints_violated'),gate_decision='OWNER_NOT_ASSIGNED')
  frozen.append(row)
  for label,v in stage.items():
   if label in STAGES and finite(v):cost.append(dict(scope=scope,object=oid,phase=phase,k=k,policy=policy,run_id=sha,label=label,wall_s=v,time_kind='within_online_disjoint_stage',source=str(p.resolve())))
 def nonlinear_result(a,p,sha,scope):
  traj=read.read(p.parent/'trajectory.json');cfg=read.read(p.parent/'run_config.json');traj=traj[0]if traj else [];cfg=cfg[0]if cfg else {};pol=a['policy'];rep=a.get('repetition');gc=a.get('geometry_cache');method=pol+(':geometry_cache'if gc is True else ':uncached'if gc is False else '')
  identity=nonlinear_identity(a,cfg,scope,authorized_locks);configsha=identity['config_sha256'];stratum=identity['comparison_stratum']
  phase_totals=collections.defaultdict(float);stage_totals=collections.defaultdict(float);jd=[];trial_accept=0
  for rec in traj:
   for label,x in rec.get('phase_costs',{}).items():
    if finite(x.get('wall_s')):phase_totals[label]+=x['wall_s']
   pr=rec.get('policy_result',{});ep=pr.get('efficiency_policy',pr.get('reuse_policy',{}));n=ep.get('full_Jd_directions')
   if n is None and 'rounds'in pr:n=sum(len(t.get('finalists',[]))for t in pr['rounds'])
   if n is not None:jd.append(n)
   for label,v in pr.get('stage_wall_s',{}).items():
    if label in STAGES and finite(v):stage_totals[label]+=v
   trial_accept+=int(bool(rec.get('accepted')))
  counters=a.get('full_physical_counters',{}).get('totals',{});metrics=a.get('metrics',{});record_count=a.get('records',len(traj));complete=a.get('status')in ('completed_max_updates','stopped_small_step','COMPLETED')
  row=dict(scope=scope,object=a['object_id'],policy=pol,method=method,geometry_cache=gc,repetition=rep,version=a.get('version'),comparison_stratum=stratum,config_sha256=configsha,result_sha256=sha,source=str(p.resolve()),status=a.get('status'),measurement_status='MEASURED'if complete else 'PARTIAL_OR_STOPPED',child_wall_total_s=a.get('wall_total_s'),trajectory_records=len(traj),declared_records=record_count,accepted_updates=trial_accept if traj else None,gradient_fallbacks=sum(bool(r.get('gradient_fallback'))for r in traj)if traj else None,outer_phase_exclusive_sum_s=sum(phase_totals.values())if traj else None,online_stage_exclusive_sum_s=sum(stage_totals.values())if stage_totals else None,online_driver_policy_sum_s=sum(v for k,v in phase_totals.items()if k in ('receiver_only_acquisition_and_endpoint','exchange_anchor_pool_workspace_acquisition','verified_online_exchange'))if traj else None,material_complex_relative=metrics.get('material_complex_relative'),material_real_relative=metrics.get('material_real_relative'),material_imag_relative=metrics.get('material_imag_relative'),measured_data_relative=metrics.get('measured_data_relative'),heldout_data_relative=metrics.get('heldout_data_relative'),full_forward_rhs=counters.get('solve_rhs_full_forward'),full_adjoint_rhs=counters.get('solve_rhs_full_adjoint'),full_tangent_rhs=counters.get('solve_rhs_full_tangent',0 if pol=='receiver_only'and counters else None),full_LU_count=counters.get('lu_factorizations_full'),Jd_directions=sum(jd)if jd else (0 if pol=='receiver_only'else None),avg_Jd_per_outer=div(sum(jd),len(traj))if jd else (0 if pol=='receiver_only'and traj else None),full_gradient_scope='outer g(0); not verification g(x)'if a.get('full_gradient_outer_only')else 'NOT_DECLARED',gradient_at_x_available=None,full_GN_risk='NOT_MEASURED_IN_NONLINEAR_RUN',risk_identity_error=None,matched_receiver_wall_ratio=None,matched_original_wall_ratio=None,gate_decision='OWNER_NOT_ASSIGNED')
  declared=a.get('runtime_end',{}).get('declared_efficiency',{})
  row.update(cache_fields(declared.get('cache_report'),scope,a['object_id'],a.get('version'),sha,p))
  row['declared_controller_version']=declared.get('controller_version');row['nonlinear_driver_sha256']=declared.get('nonlinear_driver_sha256')
  row['child_wall_minus_outer_phases_s']=row['child_wall_total_s']-row['outer_phase_exclusive_sum_s'] if finite(row['child_wall_total_s'])and finite(row['outer_phase_exclusive_sum_s'])else None
  row.update(identity)
  if not identity['match_identity_eligible']:availability.append(dict(scope=scope,object=a['object_id'],status='MATCH_IDENTITY_REJECTED',source=str(p.resolve()),reason=identity['match_identity_errors']))
  nl.append(row)
  for label,v in phase_totals.items():cost.append(dict(scope=scope,object=a['object_id'],policy=pol,method=method,repetition=rep,run_id=sha,label=label,wall_s=v,time_kind='outer_driver_disjoint_phase',source=str(p.resolve())))
  for label,v in stage_totals.items():cost.append(dict(scope=scope,object=a['object_id'],policy=pol,method=method,repetition=rep,run_id=sha,label=label,wall_s=v,time_kind='within_online_disjoint_stage',source=str(p.resolve())))
 def parse_result(p,scope):
  loaded=read.read(p)
  if not loaded:return
  a,sha=loaded
  if sha in seen_result:read.aliases.append(dict(sha256=sha,duplicate_path=str(p.resolve()),kept_path=seen_result[sha]));return
  version=a.get('version',a.get('schema',''));oid=a.get('object_id')
  if oid is None and p.name=='failure.json':
   config=read.read(p.parent/'run_config.json');cli=config[0].get('cli',{})if config else {};oid=cli.get('object')
   if oid is None:
    match=re.search(r'object_(\d+)',str(p));oid=int(match[1])if match else None
  if oid not in args.objects:return
  seen_result[sha]=str(p.resolve())
  if p.name=='failure.json':availability.append(dict(scope=scope,object=oid,status='FAILED_RECEIPT',source=str(p),result_sha256=sha,reason=a.get('error')));return
  if 'frozen_pilot'in version or ('evaluation'in a and 'online_cells'in a):
   for cell in a.get('evaluation',[]):
    if cell.get('object_id')not in args.objects:continue
    phase=cell['phase'];b=p.parent if a.get('phase')else p.parent/phase;online=read.read(b/f"k{cell['k']}"/cell['policy']/'online_result.json');frozen_row(cell,p,sha,'eff_frozen',online[0]if online else None)
  elif p.name=='profile_result.json'and 'profile'in a:
   pro=a['profile'];profiles.append(dict(scope='eff_profile',object=oid,version=version,result_sha256=sha,source=str(p.resolve()),actual_wall_s=pro.get('actual_wall_s'),exclusive_partition_wall_s=pro.get('exclusive_partition_wall_s'),unattributed_wall_s=pro.get('unattributed_wall_s'),gradient_scope='shared full g(0)',gradient_at_x_available=False,FP64=a.get('FP64'),policy_physical_counter_delta=counter_delta(a.get('initial_counters'),a.get('policy_end_counters'))))
   for label,v in pro.get('groups',{}).items():cost.append(dict(scope='eff_profile',object=oid,run_id=sha,label=label,wall_s=v.get('exclusive_wall_s'),inclusive_wall_s=v.get('inclusive_wall_s'),calls=v.get('calls'),time_kind='profile_tree_EXCLUSIVE',source=str(p.resolve())))
   for d in a.get('diagnostic_checks',[]):checks.append(dict(scope='eff_profile',object=oid,run_id=sha,check='original_directional_output_equivalence',relative_error=d.get('relative_error'),absolute_error=d.get('absolute_error'),detail=d.get('cost'),source=str(p.resolve())))
  elif p.name=='verification_result.json'and 'block_controls'in a:
   row=dict(scope='eff_verification_control',object=oid,version=version,result_sha256=sha,source=str(p.resolve()),actual_wall_s=a.get('actual_wall_s'),exclusive_partition_wall_s=a.get('exclusive_phase_wall_s'),unattributed_wall_s=a.get('unattributed_wall_s'),FP64=a.get('FP64'));row.update(cache_fields(a.get('cache_report'),'eff_verification_control',oid,version,sha,p));profiles.append(row)
   for v in a.get('phases',[]):cost.append(dict(scope='eff_verification_control',object=oid,version=version,run_id=sha,label=v['label'],wall_s=v['wall_s'],counter_delta=v.get('counter_delta'),time_kind='verification_control_disjoint_phase',phase_label_is_branch_evidence=False,source=str(p.resolve())))
   for label in ('bank_check','first_hit_check','metadata_exact_equal','block_controls','displacement_QR_diagnostic','cache_report','cache_branch_check'):checks.append(dict(scope='eff_verification_control',object=oid,version=version,run_id=sha,check=label,detail=a.get(label),source=str(p.resolve())))
   for control in a.get('block_controls',[]):block_controls.append(dict(scope='eff_verification_control',object=oid,version=version,run_id=sha,source=str(p.resolve()),**control))
  elif 'nonlinear'in version and 'policy'in a and 'metrics'in a:nonlinear_result(a,p,sha,'closure_saved_nonlinear'if version.startswith('a17.closure.')else scope)
 # Discover every local future snapshot and classify by actual payload schema/version.
 candidates=[(p,'eff_nonlinear')for p in result_files(args.snapshots)]
 if not args.no_saved:
  old=args.closure/'remote_snapshots/nonlinear_phase_closed_v1/source/results/nonlinear_transfer_v1'
  candidates.extend((p,'closure_saved_nonlinear')for p in result_files(old))
 # Among identical bytes only, choose the source with the richest local companions.
 candidates.sort(key=lambda item:(-sum((item[0].parent/n).exists()for n in ('trajectory.json','run_config.json')),str(item[0])))
 for p,scope in candidates:parse_result(p,scope)
 # External execution occupation is a distinct cost scope, never added to child wall.
 seen_terminal=set()
 for p in sorted(args.execution_receipts.rglob('*TERMINAL.json'))if args.execution_receipts.exists()else []:
  loaded=read.read(p)
  if not loaded:continue
  terminal,sha=loaded;end=terminal.get('last_end',{})
  if sha in seen_terminal or not end:continue
  seen_terminal.add(sha);occupation.append(dict(scope='eff_external_execution_occupation',unit=end.get('unit'),attempt=end.get('attempt'),occupation_s=end.get('occupation_s'),exit_code=end.get('exit_code'),stop_reason=end.get('stop_reason'),monitor_cpu_s=end.get('monitor_cpu_s'),terminal_child_cpu_s=end.get('terminal_child_cpu_s'),terminal_child_cpu_missing_reason=end.get('terminal_child_cpu_missing_reason'),peak_device_used_mib=end.get('peak_device_used_mib'),utc=end.get('utc'),identity_sha256=end.get('identity_sha256'),result_sha256=sha,source=str(p.resolve()),time_kind='external_occupation_includes_child_and_guard; DO_NOT_ADD_TO_CHILD_WALL'))
 if not args.no_saved:
  prim=args.closure/'remote_snapshots/primary_complete_v1/results'
  for p in sorted(prim.rglob('offline_evaluation.json'))if prim.exists()else []:
   match=re.search(r'object_(\d+)/(early|middle|late)/k(4|8|16)/offline_evaluation.json',p.as_posix())
   if not match or int(match[1])not in args.objects:continue
   a,sha=read.read(p);online=read.read(p.parent/'online_result.json');a=dict(a,object_id=int(match[1]),phase=match[2],policy='original_top2_cap3');receiver_cost=read.read(p.parent/'receiver_cost.json');online_payload=dict(online[0],_receiver_cost=receiver_cost[0]if receiver_cost else {})if online else None;frozen_row(a,p,sha,'closure_saved_frozen',online_payload)
 # Baselines and aggregations are within identical experiment scope/object/state/k only.
 for r in frozen:
  baseline=[b for b in frozen if (b['scope'],b['object'],b['phase'],b['k'],b['policy'])==(r['scope'],r['object'],r['phase'],r['k'],'original_top2_cap3')]
  if len(baseline)==1:r['baseline_gain']=baseline[0]['sum_positive_accepted_gain'];r['gain_retention']=div(r['sum_positive_accepted_gain'],r['baseline_gain'])
 agg=[]
 for key in sorted(set((r['scope'],r['object'],r['policy'])for r in frozen)):
  rows=[r for r in frozen if(r['scope'],r['object'],r['policy'])==key];usable=[r for r in rows if finite(r['baseline_gain'])and finite(r['sum_positive_accepted_gain'])];den=sum(r['baseline_gain']for r in usable);num=sum(r['sum_positive_accepted_gain']for r in usable);wall=stats([r['incremental_online_wall_s']for r in rows]);dirs=stats([r['Jd_directions']for r in rows])
  agg.append(dict(scope=key[0],object=key[1],policy=key[2],cells=len(rows),expected_cells=9,coverage_status='COMPLETE_MEASUREMENT_GRID'if len(rows)==9 else 'PARTIAL',sum_positive_gain=num if usable else None,sum_original_positive_gain=den if usable else None,object_gain_retention=div(num,den),avg_Jd=statistics.mean(r['Jd_directions']for r in rows if finite(r['Jd_directions']))if dirs['n']else None,max_Jd=dirs['max'],incremental_online_wall_n=wall['n'],incremental_online_wall_sum_s=sum(r['incremental_online_wall_s']for r in rows if finite(r['incremental_online_wall_s']))if wall['n']else None,incremental_online_wall_median_s=wall['median'],max_abs_risk_identity_error=max([abs(r[x])for r in rows for x in ('receiver_risk_identity_error','final_risk_identity_error')if finite(r[x])],default=None),max_abs_telescoping_gain_error=max([abs(r['telescoping_gain_identity_error'])for r in rows if finite(r['telescoping_gain_identity_error'])],default=None),gate_decision='OWNER_NOT_ASSIGNED'))
 # Never select the fastest run. Conflicting result hashes for one route/rep are ambiguous.
 route=collections.defaultdict(list)
 for r in nl:route[(r['scope'],r['object'],r['comparison_stratum'],r['method'],r['repetition'])].append(r)
 for key,rows in route.items():
  if len(rows)>1:conflicts.append(dict(kind='nonlinear_route_rep_has_multiple_result_hashes',identity=key,hashes=[r['result_sha256']for r in rows]))
 for r in nl:
  if not r['match_identity_eligible'] or len(route[(r['scope'],r['object'],r['comparison_stratum'],r['method'],r['repetition'])])!=1:continue
  peers=[b for b in nl if b['match_identity_eligible'] and pair_identity(b)==pair_identity(r)]
  for policy,field in [('receiver_only','matched_receiver_wall_ratio'),('original_top2_cap3','matched_original_wall_ratio')]:
   b=[x for x in peers if x['policy']==policy and (policy=='receiver_only'or x['geometry_cache']is False)]
   if len(b)==1:r[field]=div(r['child_wall_total_s'],b[0]['child_wall_total_s'])
 nl_agg=[]
 for key in sorted(set((r['scope'],r['object'],r['comparison_stratum'],r['method'])for r in nl)):
  rr=[r for r in nl if r['match_identity_eligible'] and (r['scope'],r['object'],r['comparison_stratum'],r['method'])==key and len(route[(r['scope'],r['object'],r['comparison_stratum'],r['method'],r['repetition'])])==1];row=dict(scope=key[0],object=key[1],comparison_stratum=key[2],method=key[3],unique_repetitions=[r['repetition']for r in rr],measurement_status='PARTIAL'if len(rr)<3 or any(r['measurement_status']!='MEASURED'for r in rr)else 'COMPLETE_MEASUREMENT_GRID',gate_decision='OWNER_NOT_ASSIGNED')
  for f in ('child_wall_total_s','online_stage_exclusive_sum_s','online_driver_policy_sum_s','material_complex_relative','heldout_data_relative','Jd_directions','accepted_updates','matched_receiver_wall_ratio','matched_original_wall_ratio'):
   for k,v in stats([r[f]for r in rr]).items():row[f+'_'+k]=v
  nl_agg.append(row)
 # Explicit authorized-route locks supply future missing-cell expectations, never gates.
 expected_locks=args.expected_lock if args.expected_lock else sorted((EFF/'configs').glob('*.json'))
 for lockpath in expected_locks:
  loaded=read.read(lockpath)
  if not loaded:continue
  lock,locksha=loaded
  if lock.get('schema')!='a17.efficiency.nonlinear.lock.v1':continue
  for o in lock.get('nonlinear_objects',args.objects):
   if o not in args.objects:continue
   for rt in lock.get('authorized_routes',[]):
    found=[r for r in nl if r['scope']=='eff_nonlinear'and r['object']==o and r['policy']==rt.get('policy')and r['geometry_cache']==rt.get('geometry_cache')and r['repetition']==rt.get('repetition')and r['config_sha256']==locksha and r['match_identity_eligible']]
    availability.append(dict(scope='eff_nonlinear_authorized_route',object=o,policy=rt.get('policy'),geometry_cache=rt.get('geometry_cache'),repetition=rt.get('repetition'),config_sha256=locksha,status='MEASURED'if len(found)==1 and found[0]['measurement_status']=='MEASURED'else 'PARTIAL_OR_NOT_RUN',reason='Explicit lock expectation; no best-run selection'))
 for o in args.objects:
  for phase in ('early','middle','late'):
   for k in (4,8,16):
    for pol in POLICIES:
     rr=[r for r in frozen if(r['scope'],r['object'],r['phase'],r['k'],r['policy'])==('eff_frozen',o,phase,k,pol)];availability.append(dict(scope='eff_frozen',object=o,phase=phase,k=k,policy=pol,status='MEASURED'if len(rr)==1 else 'MISSING_OR_AMBIGUOUS',reason=None if len(rr)==1 else 'NOT_RUN or untransferred snapshot; no interpolation'))
  for scope in ('eff_profile','eff_verification_control'):
   availability.append(dict(scope=scope,object=o,status='MEASURED'if any(r['scope']==scope and r['object']==o for r in profiles)else 'MISSING_NOT_RUN',reason='No gate assigned'))
  if not any(r['scope']=='eff_nonlinear'and r['object']==o for r in nl):availability.append(dict(scope='eff_nonlinear',object=o,status='NOT_RUN_OR_NOT_TRANSFERRED',reason='No closed result; proposed policies are not assumed authorized'))
 state='PARTIAL'if read.errors or conflicts or any(r['status']not in ('MEASURED',)for r in availability)or any(r['measurement_status']=='PARTIAL'for r in nl_agg if r['scope']=='eff_nonlinear')else 'DERIVED_MEASUREMENTS_COMPLETE_OWNER_REVIEW_PENDING'
 for name,rows in [('frozen_cells',frozen),('frozen_object_policy',agg),('nonlinear_runs',nl),('nonlinear_identity_audit',[{k:r.get(k) for k in ('scope','object','method','geometry_cache','repetition','source','result_sha256','config_sha256','comparison_stratum','match_identity_eligible','match_identity_errors','match_identity_source_sha256','match_identity_driver_sha256','match_identity_scope')} for r in nl]),('nonlinear_matched_summary',nl_agg),('profile_runs',profiles),('cost_components',cost),('numerical_checks',checks),('cache_branch_records',cache_records),('verification_block_controls',block_controls),('occupation_receipts',occupation),('availability',availability),('conflicts',conflicts),('input_manifest',read.manifest),('duplicate_snapshots',read.aliases),('parse_errors',read.errors)]:emit(out,name,rows)
 emit(out,'analysis_receipt',dict(schema=VERSION,status=state,cpu_analysis_wall_s=time.perf_counter()-started,objects=args.objects,snapshots=str(args.snapshots.resolve()),scope_rules='EFF cached frozen/profile/verification/cold nonlinear separated from CLOSURE saved evidence; matched NL strata bind version+configSHA+locked sources+driver; explicit method/cache/rep authorization required',missing_rule='null/blank; no interpolation; zero denominator retention is null',duplicate_rule='result byte SHA-256; identical bytes prefer complete local trajectory/config companions only; nested identical cells canonical payload SHA; distinct logical duplicates exported as conflicts',median_rule='all unambiguous matched repetitions; median/MAD/std/min/max; never choose fastest rep',risk_rule='full_gap-half_H_step_energy-reference_gradient_cross; telescoping gap reduction minus accepted gain',gate_decision='OWNER_NOT_ASSIGNED',physics_launched=False,network=False,source_sha256=digest(Path(__file__).read_bytes())))
 print(json.dumps(dict(status=state,frozen_cells=len(frozen),nonlinear_runs=len(nl),profiles=len(profiles),conflicts=len(conflicts),out=str(out)),indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,default=EFF/'remote_snapshots');p.add_argument('--closure',type=Path,default=EFF.parent/'CLOSURE_R1');p.add_argument('--out',type=Path,default=EFF/'analysis/derived_v3');p.add_argument('--execution-receipts',type=Path,default=EFF/'research');p.add_argument('--objects',type=int,nargs='+',default=[2002,2009]);p.add_argument('--no-saved',action='store_true');p.add_argument('--expected-lock',type=Path,action='append');run(p.parse_args())
if __name__=='__main__':main()
