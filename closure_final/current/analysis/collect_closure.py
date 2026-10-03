"""Mechanical arriving-result collection; no numerical kernels, mutations or judgment.
Only terminal, receipt-audited, uniquely paired monitored runs enter risk statistics.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,collections
import numpy as np
SCHEMA='a17.closure.collect.v1';SEED=20261002;BOOTSTRAPS=10000
DIRECT_ROLES={'run_closure.py':'closure_state.py','run_closure_v2.py':'closure_state_v2.py'}
BATCH_ROLES={'run_locked_batch_v2.py':('run_closure.py','closure_state.py'),
             'run_locked_batch_v3.py':('run_closure_v2.py','closure_state_v2.py')}
CAVEAT='Locked A17-rule validation on historically studied corpus; NOT confirmed blind validation.'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def invalid_constant(value):raise ValueError('nonfinite JSON constant '+value)
 return json.loads(Path(p).read_text(),parse_constant=invalid_constant)
def normpath(p):return str(p).replace('\\','/').rstrip('/').lower()
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def write_csv(p,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def paired_attempts(path):
 if not Path(path).exists():return [],[dict(kind='MISSING_MONITOR_LEDGER',path=str(path))],[]
 starts={};ends={};issues=[];rows=[]
 for line,text in enumerate(Path(path).read_text().splitlines(),1):
  if not text.strip():continue
  try:r=json.loads(text)
  except Exception as e:issues.append(dict(kind='LEDGER_PARSE_ERROR',line=line,error=repr(e)));continue
  event=r.get('event');a=r.get('attempt')
  if event not in ('start','end'):continue
  target=starts if event=='start' else ends
  if not isinstance(a,str) or a in target:issues.append(dict(kind='DUPLICATE_OR_INVALID_LEDGER_ID',attempt=a,event=event));continue
  target[a]=r
 for a in sorted(set(starts)|set(ends)):
  s=starts.get(a);e=ends.get(a)
  if s is None or e is None:issues.append(dict(kind='UNPAIRED_ATTEMPT',attempt=a));continue
  if not finite(e.get('occupation_s')) or e['occupation_s']<0:issues.append(dict(kind='UNKNOWN_ATTEMPT_FEE',attempt=a));continue
  if e.get('identity_sha256')!=s.get('identity',{}).get('identity_sha256') or not e.get('identity_sha256'):
   issues.append(dict(kind='ATTEMPT_IDENTITY_MISMATCH',attempt=a));continue
  rows.append(dict(attempt=a,start=s,end=e))
 fees=[dict(attempt=x['attempt'],occupation_s=x['end']['occupation_s'],exit_code=x['end'].get('exit_code'),stop_reason=x['end'].get('stop_reason'),identity_sha256=x['end']['identity_sha256'],scope='Whole monitored occupied interval; do not add nested stage walls',ledger_path=str(path),ledger_sha256=sha(path)) for x in rows]
 # Ambiguous ledger means no run may be advertised as audited yet.
 return (rows if not issues else []),issues,fees

def command_option(command,option):
 if command.count(option)!=1:return None
 i=command.index(option);return command[i+1] if i+1<len(command) else None

def command_out(start):return command_option(start.get('command',[]),'--out')
def source_digest(mapping,key):return next((v for k,v in mapping.items() if normpath(k)==key),None)
def batch_binding(folder,result,config,attempt):
 """Exactly one direct parent receipt and registered named child, never any ancestor."""
 parent=Path(folder).parent;receipt=parent/'batch_receipt.json'
 if not receipt.is_file():raise ValueError('direct parent batch receipt missing')
 batch=read(receipt);start=attempt['start'];identity=start.get('identity',{});command=start.get('command',[])
 script=[x for x in command if normpath(x).rsplit('/',1)[-1] in BATCH_ROLES]
 if len(script)!=1:raise ValueError('monitored parent not one of exact reviewed serial runner names')
 runner=normpath(script[0]).rsplit('/',1)[-1];driver,adapter=BATCH_ROLES[runner]
 reviewed=Path(__file__).resolve().parents[1]/'code'/runner;expected=source_digest(identity.get('sources',{}),'code/'+runner)
 childsource=identity.get('child_source',{})
 if not expected or sha(reviewed)!=expected or childsource.get('sha256')!=expected or not normpath(childsource.get('path','')).endswith('/code/'+runner):
  raise ValueError('reviewed parent source hash not bound to monitor child source')
 mode=result['mode'];oid=result['object_id'];cli=config.get('cli',{})
 if mode not in ('replay','validation','noise'):raise ValueError('unknown batch mode')
 command_script=normpath(script[0]);root=normpath(childsource['path']).rsplit('/code/',1)[0]
 if command_script in ('code/'+runner,runner):command_script=root+'/'+('code/' if command_script==runner else '')+command_script
 if command_script!=normpath(childsource['path']):raise ValueError('monitored command source path differs from reviewed child source')
 if command_option(command,'--mode')!=mode or command_option(command,'--object')!=str(oid) or batch.get('mode')!=mode or batch.get('object')!=oid:
  raise ValueError('parent command/batch mode-object mismatch')
 if normpath(command_option(command,'--config'))!=normpath(cli.get('config')) or batch.get('config_sha256')!=config.get('closure_config_sha256'):
  raise ValueError('parent command/batch config binding mismatch')
 expected_tasks=[(p,0,0) for p in ('early','middle','late')] if mode in ('replay','validation') else [('middle',n,j) for n in (100,300) for j in range(3)]
 tasks=batch.get('tasks',[])
 if [(r.get('phase'),r.get('noise_basis_points'),r.get('realization')) for r in tasks]!=expected_tasks:
  raise ValueError('batch rows not exactly complete registered serial task order')
 parentout=command_out(start)
 if not parentout:raise ValueError('parent output missing')
 for row,(phase,noise,rep) in zip(tasks,expected_tasks):
  name=phase if noise==0 else f'{phase}_noise{noise}_rep{rep}';remote=normpath(parentout)+'/'+name
  if normpath(row.get('path'))!=remote or row.get('returncode')!=0 or row.get('status')!='COMPLETED':raise ValueError('batch row path/returncode/status invalid')
 target=(result['phase'],cli.get('noise_basis_points',0),cli.get('realization_index',0))
 if target not in expected_tasks:raise ValueError('child phase/noise outside parent declared serial tasks')
 phase,noise,rep=target;name=phase if noise==0 else f'{phase}_noise{noise}_rep{rep}'
 if Path(folder).name!=name or normpath(cli.get('out'))!=normpath(parentout)+'/'+name:raise ValueError('child is not the exact named direct descendant')
 if cli.get('mode')!=mode or cli.get('object')!=oid or cli.get('phase')!=result['phase']:raise ValueError('child actual run_config identity mismatch')
 if normpath(cli.get('baseline_root'))!=normpath(command_option(command,'--baseline-root')):raise ValueError('parent/child baseline root mismatch')
 if mode!='replay':
  root=normpath(childsource['path']).rsplit('/code/',1)[0]
  if normpath(cli.get('inputs_root'))!=root+'/inputs' or normpath(cli.get('input_manifest'))!=root+'/inputs/input_manifest.json':raise ValueError('child new input paths differ from serial runner contract')
 command_sha=hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest()
 declared_command_sha=start.get('command_sha256',identity.get('command_sha256'))
 if declared_command_sha is not None and declared_command_sha!=command_sha:raise ValueError('parent declared command SHA mismatch')
 return dict(kind='EXACT_SERIAL_BATCH_CHILD',batch_receipt_path=str(receipt),batch_receipt_sha256=sha(receipt),parent_command_sha256=command_sha,parent_command_hash_status='MATCHED_DECLARED_MONITOR_HASH' if declared_command_sha else 'DERIVED_FROM_HASHED_LEDGER_PLUS_EXACT_ARGS_VALIDATION_NO_SEPARATE_MONITOR_FIELD',parent_runner_sha256=expected,parent_runner_name=runner,required_driver=driver,required_adapter=adapter,parent_remote_out=parentout,child_name=name,registered_tasks=len(tasks))


def validate_v2_fixture(result,config,receipt,identity,folder=None):
 """Two exact reviewed config epochs and source roles; no wildcard versions."""
 root=Path(__file__).resolve().parents[1]
 filename=normpath(config.get('cli',{}).get('config','')).rsplit('/',1)[-1]
 reviewed={'closure_replay_remote_v2.json':('CLOSURE_REPLAY_REMOTE_V2.json',None),
           'a17_final_validation_lock.json':('A17_FINAL_VALIDATION_LOCK.json','a214e38954be1438a7b3e5a84cc6ea595d07544ceff652d6fd84995765a00ae5')}
 if filename not in reviewed:raise ValueError('v3 requires an exact reviewed config name')
 name,pinned=reviewed[filename];cfgpath=root/'configs'/name;cfg=read(cfgpath);digest=sha(cfgpath)
 if config.get('closure_config_sha256')!=digest or (pinned and digest!=pinned):raise ValueError('v3 exact reviewed config digest mismatch')
 mode=result.get('mode')
 if mode not in cfg.get('permitted_modes',[]) or (not pinned and mode!='replay'):raise ValueError('mode outside reviewed config epoch')
 for key,expected in cfg['deployment_sources'].items():
  if source_digest(config.get('lock_receipt',{}).get('source_hashes',{}),key)!=expected or source_digest(identity.get('sources',{}),key)!=expected:raise ValueError('exact eight-source deployment lock mismatch: '+key)
 aliases=[]
 for epoch,rt in (('initial',receipt.get('runtime',{})),('final',result.get('final_runtime',{}))):
  for filename in ('run_closure_v2.py','closure_state_v2.py','verified_exchange.py','online_tolerance.py'):
   key='code/'+filename;matches=[dict(module=name,**row) for name,row in rt.get('actual_imports',{}).items() if normpath(row.get('actual_path','')).endswith('/code/'+filename)]
   identities={(normpath(row['actual_path']),row.get('sha256')) for row in matches}
   if len(identities)!=1 or next(iter(identities))[1]!=cfg['deployment_sources'][key]:raise ValueError('required actual imported role not bound: '+filename)
   aliases.append(dict(epoch=epoch,role=filename,unique_source_identities=len(identities),aliases=matches))
 if mode!='replay':
  oid=result['object_id'];phase=result['phase'];cli=config.get('cli',{})
  if oid not in cfg[{'validation':'validation_objects','noise':'noise_objects'}[mode]] or phase not in cfg['phases']:raise ValueError('object/phase outside final validation lock')
  if mode=='noise' and (phase!='middle' or cli.get('noise_basis_points') not in (100,300) or cli.get('realization_index') not in (0,1,2)):raise ValueError('noise task outside final validation lock')
  if mode=='validation' and (cli.get('noise_basis_points',0)!=0 or cli.get('realization_index',0)!=0):raise ValueError('primary validation has unexpected noise')
  manifestpath=root/'inputs/input_manifest.json';holdoutpath=root/'HOLDOUT_OBJECT_MANIFEST.json'
  if sha(manifestpath)!=cfg['input_manifest_sha256'] or sha(holdoutpath)!=cfg['holdout_manifest_sha256'] or receipt.get('input_manifest_sha256')!=cfg['input_manifest_sha256']:raise ValueError('final validation input/holdout manifest binding mismatch')
  known={normpath(r['path']):r['sha256'] for r in read(manifestpath)['inputs']};remote=normpath(cli.get('inputs_root',''));iteration=cfg['phases'][phase]
  required=['scenes.json',f'object_{oid}/common_data.npz',f'object_{oid}/anchor_{iteration:02d}.npz','original_resolved_config.json']
  expected_inputs={remote+'/'+name:known['inputs/'+name] for name in required};actual_inputs={normpath(r['path']):r['sha256'] for r in receipt.get('inputs',[])}
  if not remote or len(receipt.get('inputs',[]))!=4 or actual_inputs!=expected_inputs:raise ValueError('new four-file online input receipts mismatch')
  if receipt.get('preparation_reference_access') is not False or receipt.get('persisted_hashes') is not None:raise ValueError('new input preparation must not use old persisted fixture')
  if not folder:raise ValueError('coefficient pool file required for new-input audit')
  poolpath=Path(folder)/'public_workspace.npz'
  with np.load(poolpath,allow_pickle=False) as bank:poolhash=hashlib.sha256(np.ascontiguousarray(bank['pool']).tobytes()).hexdigest()
  if receipt.get('coefficient_pool_hash')!=poolhash:raise ValueError('saved coefficient pool hash does not match receipt')
  return dict(config_sha256=digest,input_manifest_sha256=sha(manifestpath),holdout_manifest_sha256=sha(holdoutpath),new_input_receipts=receipt['inputs'],coefficient_pool_hash=poolhash,public_workspace_sha256=sha(poolpath),actual_import_aliases=aliases)
 fixturepath=root/'research/REPLAY_FIXTURE_MANIFEST.json'
 if cfg.get('replay_fixture_manifest_sha256')!=sha(fixturepath):raise ValueError('prehashed replay fixture manifest differs from config')
 oid=result['object_id'];phase=result['phase'];pair=read(fixturepath)['fixtures'][f'object_{oid}_{phase}'];actual=receipt.get('persisted_hashes',{})
 for name,expected in pair.items():
  matching=[value for path,value in actual.items() if normpath(path).endswith('/'+name)]
  if matching!=[expected]:raise ValueError('child actual persisted file hash differs: '+name)
 if not receipt.get('coefficient_pool_hash'):raise ValueError('v2 coefficient hash semantic receipt missing')
 # Manifest bound by immutable config digest; monitor config input already checked.
 return dict(manifest_sha256=sha(fixturepath),fixture_key=f'object_{oid}_{phase}',config_sha256=sha(cfgpath),actual_import_aliases=aliases)

def audit_terminal(folder,attempts):
 errors=[];folder=Path(folder)
 try:
  result=read(folder/'result.json');status=read(folder/'status.json');receipt=read(folder/'runtime_receipt.json');config=read(folder/'run_config.json')
 except Exception as e:return None,None,[dict(kind='RESULT_OR_RECEIPT_MISSING',path=str(folder),error=repr(e))]
 if result.get('status')!='COMPLETED' or status.get('status')!='COMPLETED':errors.append('NOT_TERMINAL_COMPLETED')
 cli=config.get('cli',{});out=cli.get('out');matches=[];bindings=[];binding_errors=[]
 for candidate in attempts:
  if out and normpath(command_out(candidate['start']))==normpath(out):
   sourcepath=candidate['start'].get('identity',{}).get('child_source',{}).get('path','');driver=normpath(sourcepath).rsplit('/',1)[-1]
   if driver not in DIRECT_ROLES:
    binding_errors.append(dict(attempt=candidate['attempt'],error='direct child is not exact reviewed driver name'));continue
   matches.append(candidate);bindings.append(dict(kind='DIRECT_MONITORED_CHILD',required_driver=driver,required_adapter=DIRECT_ROLES[driver],parent_command_sha256=hashlib.sha256(json.dumps(candidate['start'].get('command',[]),separators=(',',':')).encode()).hexdigest()))
  elif (folder.parent/'batch_receipt.json').is_file():
   try:binding=batch_binding(folder,result,config,candidate)
   except Exception as exc:
    binding_errors.append(dict(attempt=candidate['attempt'],error=repr(exc)));continue
   matches.append(candidate);bindings.append(binding)
 if len(matches)!=1:
  errors.append('NO_UNIQUE_DIRECT_OR_EXACT_BATCH_PAIRED_BINDING')
  if binding_errors:errors.append('BATCH_BINDING_ERRORS:'+json.dumps(binding_errors))
  attempt=None
 else:
  attempt={**matches[0],'collector_binding':bindings[0]};s=attempt['start'];e=attempt['end'];identity=s.get('identity',{})
  if e.get('exit_code')!=0 or e.get('stop_reason') is not None:errors.append('MONITORED_ATTEMPT_NOT_SUCCESSFUL')
  if identity.get('config',{}).get('sha256')!=config.get('closure_config_sha256'):errors.append('CONFIG_HASH_NOT_BOUND_TO_ATTEMPT')
  source=receipt.get('runtime',{}).get('closure_sources',{});finalsource=result.get('final_runtime',{}).get('closure_sources',{})
  expected=identity.get('sources',{})
  for name in ('closure_runtime.py','closure_state.py','verified_exchange.py','run_closure.py','online_tolerance.py'):
   key='code/'+name;get=lambda d:next((v for k,v in d.items() if normpath(k)==key),None)
   if not get(expected) or get(source)!=get(expected) or get(finalsource)!=get(expected):errors.append('SOURCE_HASH_NOT_BOUND:'+key)
  if attempt['collector_binding']['kind']=='DIRECT_MONITORED_CHILD':
   binding=attempt['collector_binding']
   for name in (binding['required_driver'],binding['required_adapter']):
    key='code/'+name;wanted=source_digest(expected,key)
    if not wanted or source_digest(source,key)!=wanted or source_digest(finalsource,key)!=wanted:errors.append('DIRECT_CHILD_EXACT_ROLE_HASH_MISMATCH:'+key)
   if identity.get('child_source',{}).get('sha256')!=source_digest(expected,'code/'+binding['required_driver']):errors.append('DIRECT_DRIVER_SOURCE_IDENTITY_MISMATCH')
   if binding['required_driver']=='run_closure_v2.py':
    try:binding['v2_fixture_validation']=validate_v2_fixture(result,config,receipt,identity,folder)
    except Exception as exc:errors.append('V2_FIXTURE_BINDING:'+repr(exc))
  if attempt['collector_binding']['kind']=='EXACT_SERIAL_BATCH_CHILD':
   binding=attempt['collector_binding'];declared=config.get('lock_receipt',{}).get('source_hashes',{})
   for name in (binding['parent_runner_name'],binding['required_driver'],binding['required_adapter']):
    key='code/'+name;wanted=source_digest(expected,key)
    if not wanted or source_digest(source,key)!=wanted or source_digest(finalsource,key)!=wanted:errors.append('CHILD_RECEIPT_EXACT_ROLE_HASH_MISMATCH:'+key)
    if binding['parent_runner_name']=='run_locked_batch_v3.py' and source_digest(declared,key)!=wanted:errors.append('CONFIG_LOCK_EXACT_ROLE_HASH_MISMATCH:'+key)
   if binding['parent_runner_name']=='run_locked_batch_v3.py':
    try:binding['v2_fixture_validation']=validate_v2_fixture(result,config,receipt,identity,folder)
    except Exception as exc:errors.append('V2_FIXTURE_BINDING:'+repr(exc))
  inputsha={r.get('sha256') for r in identity.get('inputs',[])}
  manifestsha=receipt.get('input_manifest_sha256')
  # replay inputs may be individual files; other modes monitor the input manifest.
  if result.get('mode')!='replay' and (not manifestsha or manifestsha not in inputsha):errors.append('INPUT_MANIFEST_NOT_BOUND_TO_ATTEMPT')
 if receipt.get('online_controller_reference_access') is not False or receipt.get('online_fullJ_cache') is not False:errors.append('ONLINE_REFERENCE_ACCESS_RECEIPT_NOT_FALSE')
 if receipt.get('actual_dtype')!='complex128/float64':errors.append('FP64_RECEIPT_MISSING')
 if config.get('closure_config_sha256')!=receipt.get('config',{}).get('closure_config_sha256'):errors.append('CONFIG_RECEIPT_MISMATCH')
 initial_lock=receipt.get('runtime',{}).get('baseline_lock_sha256');final_lock=result.get('final_runtime',{}).get('baseline_lock_sha256');declared_lock=config.get('lock_receipt',{}).get('baseline_lock_sha256')
 if declared_lock and (initial_lock!=declared_lock or final_lock!=declared_lock):errors.append('BASELINE_RUNTIME_EPOCH_LOCK_MISMATCH')
 if result.get('mode') in ('validation','noise') and not config.get('lock_receipt',{}).get('source_hashes'):errors.append('NONREPLAY_PRE_RESULT_SOURCE_LOCK_MISSING')
 if errors:return None,None,[dict(kind='TERMINAL_AUDIT_REJECTED',path=str(folder),reasons=errors)]
 return result,attempt,[]

def expected_keys(manifest):
 objects=[r['object_id'] if isinstance(r,dict) else r for r in manifest['eligible_objects']];noise=manifest['noise_objects'];keys=[]
 for mode,obs,phases,levels,reps in [('replay',[2001,2007,2012,2014],['early','middle','late'],[0],[0]),('validation',objects,['early','middle','late'],[0],[0]),('noise',noise,['middle'],[100,300],range(3))]:
  keys.extend((mode,o,p,k,n,r) for o in obs for p in phases for k in (4,8,16) for n in levels for r in reps)
 return keys

def array_comparison(old,new,prefix):
 if old.shape!=new.shape:return {prefix+'_shape_equal':False,prefix+'_old_shape':list(old.shape),prefix+'_new_shape':list(new.shape)}
 if not np.isfinite(old).all() or not np.isfinite(new).all():return {prefix+'_shape_equal':True,prefix+'_finite':False}
 delta=new-old;den=max(float(np.linalg.norm(old)),float(np.linalg.norm(new)),np.finfo(float).tiny)
 return {prefix+'_shape_equal':True,prefix+'_finite':True,prefix+'_exact_equal':bool(np.array_equal(old,new)),prefix+'_max_abs_error':float(np.max(np.abs(delta),initial=0)),prefix+'_relative_norm_error':float(np.linalg.norm(delta)/den)}

def replay_comparison(baseline,folder,online,evaluation,k):
 oid=online['object_id'];phase=online['phase'];old=Path(baseline)/'results'/f'object_{oid}_{phase}'
 if oid==2007 and phase=='early':old=Path(baseline)/'results/object_2007_early_portable'
 kd=old/f'k{k}';oldpolicy=read(kd/'directed_verified.json');base=read(kd/'receiver_no_swap.json');oldreceipt=read(old/'runtime_receipt.json');runtime=oldreceipt.get('runtime',oldreceipt)
 trajectories=lambda j:[dict(round=x['round'],drop=x['drop'],add=x['add'],kind=x['kind']) for x in j['accepted']]
 short=lambda j:[[x['move_index'] for x in r['finalists']] for r in j['rounds']]
 oldshort=short(oldpolicy);newshort=short(online)
 oldq=[[r.get('Qhat') for r in x['finalists']] for x in oldpolicy['rounds']];newq=[[r.get('Qhat') for r in x['finalists']] for x in online['rounds']]
 out=dict(object_id=oid,phase=phase,k=k,old_source_directory=str(old),fresh_source_directory=str(folder),old_policy_sha256=sha(kd/'directed_verified.json'),old_runtime_receipt_sha256=sha(old/'runtime_receipt.json'),historical_first_state_variant=oid==2007 and phase=='early',old_run_exchange_sha256=next((v for p,v in runtime.get('a17_sources',{}).items() if normpath(p)=='code/run_exchange.py'),None),accepted_trajectory_equal=trajectories(oldpolicy)==trajectories(online),old_accepted_trajectory=trajectories(oldpolicy),new_accepted_trajectory=trajectories(online),shortlist_order_equal=oldshort==newshort,old_shortlist_order=oldshort,new_shortlist_order=newshort,shortlist_Qhat_exact_equal=oldq==newq,terminal_selected_ids_equal=oldpolicy['selected_ids']==online['selected_ids'],old_terminal_selected_ids=oldpolicy['selected_ids'],new_terminal_selected_ids=online['selected_ids'],old_final_risk=oldpolicy['risk']['full_gap'],new_final_risk=evaluation['final_risk']['full_gap'],old_receiver_risk=base['risk']['full_gap'],new_receiver_risk=evaluation['receiver_risk']['full_gap'],comparison_status='MECHANICAL_VALUES_ONLY_NO_NEW_ARRAY_TOLERANCE')
 if oldshort==newshort:
  diffs=[abs(a-b) for u,v in zip(oldq,newq) for a,b in zip(u,v) if finite(a) and finite(b)];out['shortlist_Qhat_max_abs_error']=max(diffs,default=0.)
 out.update(actual_rank_equal=oldpolicy.get('actual_rank')==online.get('actual_rank'),stop_reason_equal=oldpolicy.get('stop_reason')==online.get('stop_reason'),pool_hash_equal=oldpolicy.get('pool_hash')==online.get('pool_hash'),state_hash_equal=oldpolicy.get('state_hash')==online.get('state_hash'),old_accepted_values=oldpolicy['accepted'],new_accepted_values=online['accepted'])
 round_comparisons=[]
 for roundno in sorted({a['round'] for a in oldpolicy['accepted']}|{a['round'] for a in online['accepted']}):
  oldarray=kd/f'directed_verified_round_{roundno}.npz';newarray=Path(folder)/f'k{k}'/f'round_{roundno}.npz'
  row=dict(round=roundno,old_array_exists=oldarray.exists(),new_array_exists=newarray.exists())
  if oldarray.exists() and newarray.exists():
   row.update(old_array_sha256=sha(oldarray),new_array_sha256=sha(newarray))
   with np.load(oldarray,allow_pickle=False) as a,np.load(newarray,allow_pickle=False) as b:
    for name in ('step','Js'):row.update(array_comparison(a[name],b[name],name))
    row['selected_ids_equal']=bool(np.array_equal(a['selected_ids'],b['selected_ids']))
  round_comparisons.append(row)
 out['round_array_comparisons']=round_comparisons
 with np.load(kd/'directed_verified_endpoint.npz',allow_pickle=False) as a,np.load(Path(folder)/f'k{k}'/'endpoint.npz',allow_pickle=False) as b:
  for name in ('step','Js'):out.update(array_comparison(a[name],b[name],name))
 oldrows=[json.loads(x) for x in (kd/'moves.jsonl').read_text().splitlines() if x.strip()];oldrows=[x for x in oldrows if x.get('policy_id')=='directed_verified']
 newrows=[json.loads(x) for x in (Path(folder)/f'k{k}'/'moves.jsonl').read_text().splitlines() if x.strip()]
 om={(r['round'],r['move_index']):r for r in oldrows};nm={(r['round'],r['move_index']):r for r in newrows};common=set(om)&set(nm)
 out.update(old_candidate_rows=len(om),new_candidate_rows=len(nm),candidate_identity_set_equal=set(om)==set(nm),candidate_score_matched_rows=len(common),candidate_Qhat_exact_equal=all(om[x].get('Qhat')==nm[x].get('Qhat') for x in common),candidate_score_max_abs_error=max((abs(om[x]['Qhat']-nm[x]['Qhat']) for x in common if finite(om[x].get('Qhat')) and finite(nm[x].get('Qhat'))),default=0.),candidate_status_equal=all(om[x].get('status')==nm[x].get('status') for x in common),old_initial_receiver_arrays='NOT_SAVED_AS_DEDICATED_BASELINE_ENDPOINT',pool_hash_comparison_scope='Historical pool_hash is physical; v2 persisted adapter returns coefficient-pool hash. Exact fixture file binding, not raw hash equality, establishes reuse.')
 return out

def object_statistics(cells,expected):
 groups=collections.defaultdict(list)
 for r in cells:groups[(r['mode'],r['noise_basis_points'])].append(r)
 objects=[];summaries=[]
 for mode,level in sorted({(k[0],k[4]) for k in expected}):
  rows=groups.get((mode,level),[]);wanted=[k for k in expected if k[0]==mode and k[4]==level];complete=len(rows)==len(wanted);ratios=[]
  for oid in sorted({k[1] for k in wanted}):
   a=[r for r in rows if r['object_id']==oid];n=sum(r['final_full_gap'] for r in a);d=sum(r['receiver_full_gap'] for r in a);floor=sum(r['evaluation_floor'] for r in a);required=sum(k[1]==oid for k in wanted)
   ratio=n/d if a and d>floor else None
   objects.append(dict(mode=mode,noise_basis_points=level,object_id=oid,observed_cells=len(a),expected_cells=required,complete=len(a)==required,numerator_sum=n if a else None,denominator_sum=d if a else None,evaluation_floor_sum=floor if a else None,ratio=ratio,ratio_missing_reason='NO_CELLS' if not a else ('DENOMINATOR_NOT_ABOVE_SUM_EVALUATION_FLOORS' if ratio is None else None),all_cells_signed_gain_sum=d-n if a else None,aggregation='sum paired absolute GN gaps within object before ratio; cells/realizations not independent',status='DESCRIPTIVE_PARTIAL' if not complete else 'COMPLETE_MECHANICAL'))
   if ratio is not None and len(a)==required:ratios.append(ratio)
  summary=dict(mode=mode,noise_basis_points=level,expected_cells=len(wanted),audited_cells=len(rows),complete=complete,objects_with_resolved_complete_ratio=len(ratios),bootstrap_seed=SEED,bootstrap_samples=BOOTSTRAPS,statistical_unit='object',status='PARTIAL_NO_BOOTSTRAP' if not complete else 'COMPLETE_MECHANICAL_NOT_SCIENTIFIC_VERDICT',median_object_ratio=None,quartiles_object_ratio=None,bootstrap_median_ratio_interval95=None)
  if complete and len(ratios)==len({k[1] for k in wanted}):
   x=np.asarray(ratios);rng=np.random.default_rng(SEED);samples=np.median(x[rng.integers(0,len(x),size=(BOOTSTRAPS,len(x)))],axis=1)
   summary.update(median_object_ratio=float(np.median(x)),quartiles_object_ratio=np.quantile(x,[.25,.5,.75]).tolist(),bootstrap_median_ratio_interval95=np.quantile(samples,[.025,.975]).tolist())
  summaries.append(summary)
 return objects,summaries

def collect(closure,baseline,results,ledger,manifest_path,out):
 closure=Path(closure).resolve();baseline=Path(baseline).resolve();results=Path(results).resolve();out=Path(out).resolve()
 if out.exists():raise ValueError('userselected NEW output must not already exist')
 if out.is_relative_to(baseline) or out.is_relative_to(results):raise ValueError('derivative output cannot be inside source baseline/results')
 manifest=read(manifest_path);expected=expected_keys(manifest);expectedset=set(expected);attempts,issues,fees=paired_attempts(ledger)
 for fee in fees:
  if fee['exit_code']!=0 or fee['stop_reason'] is not None:issues.append(dict(kind='CLOSED_FAILED_MONITORED_JOB_NO_SCIENTIFIC_RESULT',severity='retained_history',attempt=fee['attempt'],exit_code=fee['exit_code'],stop_reason=fee['stop_reason'],occupation_s=fee['occupation_s']))
 cells=[];comparisons=[];costs=[];sources=[];observed={};candidates=[]
 for folder in sorted({p.parent for p in results.rglob('status.json')}|{p.parent for p in results.rglob('result.json')}):
  state_start=len(candidates)
  try:
   status=read(folder/'status.json')
   if status.get('status')!='COMPLETED':
    known_preselection=status.get('status')=='FAILED' and 'persisted pool hash mismatch' in status.get('error','') and not list(folder.glob('k*/online_result.json'))
    issues.append(dict(kind='NONCOMPLETED_STATE_RETAINED',severity='retained_history' if known_preselection else 'critical',known_historical_preselection_hash_failure=known_preselection,path=str(folder),status=status));continue
   result,attempt,errors=audit_terminal(folder,attempts);issues.extend(errors)
   if result is None:continue
   bound_manifest=attempt.get('collector_binding',{}).get('v2_fixture_validation',{}).get('holdout_manifest_sha256')
   if bound_manifest and bound_manifest!=sha(manifest_path):raise ValueError('selected collection manifest differs from terminal final lock')
   modes=result.get('mode');oid=result.get('object_id');phase=result.get('phase');online=result.get('online',[]);evs={r['k']:r for r in result.get('evaluation',[])}
   if len(evs)!=len(result.get('evaluation',[])):raise ValueError('duplicate evaluation k')
   costs.append(dict(source_directory=str(folder),attempt=attempt['attempt'],mode=modes,object_id=oid,phase=phase,job_wall_s=result.get('wall_total_s'),offline_evaluation_wall_s=result.get('offline_evaluation_wall_s'),setup=result.get('setup'),online_stage_records=[dict(k=r['requested_k'],wall_online_total_s=r.get('wall_online_total_s'),stage_wall_s=r.get('stage_wall_s'),direction_cost=r.get('direction_cost'),reconstructed_standalone_fee_wall_s=r.get('reconstructed_standalone_fee_wall_s'),standalone_cold_wall_s=r.get('standalone_cold_wall_s'),cost_scope=r.get('cost_scope')) for r in online],scope='setup/online/offline/stage fields retain overlaps; no naive stage sum; monitored attempt fee exported once separately',cold_missing_reason='not independently measured when null'))
   sources.append(dict(path=str(folder/'result.json'),sha256=sha(folder/'result.json'),runtime_receipt_sha256=sha(folder/'runtime_receipt.json'),run_config_sha256=sha(folder/'run_config.json'),paired_attempt=attempt['attempt'],attempt_binding=attempt.get('collector_binding'),baseline_runtime_lock_sha256=read(folder/'runtime_receipt.json').get('runtime',{}).get('baseline_lock_sha256'),baseline_manifest_sha256=read(folder/'runtime_receipt.json').get('runtime',{}).get('baseline_manifest_sha256')))
   for r in online:
    k=r['requested_k'];level=r.get('noise_basis_points',0);realization=r.get('realization_index',0);key=(modes,oid,phase,k,level,realization)
    if key not in expectedset:issues.append(dict(kind='UNREGISTERED_CELL',key=list(key),path=str(folder)));continue
    e=evs[k];rr=e['receiver_risk'];fr=e['final_risk'];floor=e['evaluation_floor']
    if not all(finite(x) for x in (rr.get('full_gap'),fr.get('full_gap'),floor)) or floor<0:raise ValueError('nonfinite risk/floor')
    if r.get('object_id')!=oid or r.get('phase')!=phase or r.get('mode')!=modes or r.get('actual_rank')!=k:raise ValueError('online/result identity/rank mismatch')
    cellpath=folder/f'k{k}';savedonline=read(cellpath/'online_result.json');savedeval=read(cellpath/'offline_evaluation.json')
    if savedonline!=r or savedeval!=e:raise ValueError('terminal/cell journal mismatch')
    if not (cellpath/'endpoint.npz').is_file():raise ValueError('missing endpoint array')
    with np.load(cellpath/'endpoint.npz',allow_pickle=False) as endpoint:
     for field in ('step','Js','initial_step','initial_Js'):
      value=endpoint[field]
      if value.dtype!=np.float64 or value.ndim!=1 or not np.isfinite(value).all():raise ValueError('nonfinite/nonFP64 endpoint '+field)
     if endpoint['selected_ids'].tolist()!=r['selected_ids']:raise ValueError('endpoint selected identity mismatch')
    gain=rr['full_gap']-fr['full_gap'];classification='improved' if gain>floor else ('worse' if gain<-floor else 'equal_within_evaluation_floor')
    row=dict(mode=modes,object_id=oid,phase=phase,k=k,noise_basis_points=level,realization_index=realization,source_directory=str(folder),paired_attempt=attempt['attempt'],receiver_full_gap=rr['full_gap'],final_full_gap=fr['full_gap'],signed_full_gap_improvement=gain,evaluation_floor=floor,evaluation_classification=classification,classification_scope='evaluation floor ONLY; not online acceptance tolerance',receiver_H_step_error=rr.get('H_step_error'),final_H_step_error=fr.get('H_step_error'),receiver_normalized_H_step_error=rr.get('relative_H_step_error'),final_normalized_H_step_error=fr.get('relative_H_step_error'),full_reference_H_energy=fr.get('full_reference_H_energy'),receiver_half_H_step_energy=rr.get('half_H_step_energy'),final_half_H_step_energy=fr.get('half_H_step_energy'),reference_relative_residual=e.get('reference_relative_residual'),accepted_moves=len(r['accepted']),stop_reason=r['stop_reason'],selected_ids=r['selected_ids'],one_step_truth=e.get('one_step_truth'),final_nonlinear_status=e.get('final_nonlinear_reconstruction'),result_sha256=sha(folder/'result.json'),endpoint_sha256=sha(cellpath/'endpoint.npz'))
    candidates.append((key,row,folder,r,e))
  except Exception as exc:
   del candidates[state_start:]
   issues.append(dict(kind='COLLECTION_STATE_ERROR',path=str(folder),error=repr(exc)))
 counts=collections.Counter(x[0] for x in candidates)
 for key,row,folder,r,e in candidates:
  if counts[key]!=1:issues.append(dict(kind='DUPLICATE_REGISTERED_CELL_NO_SELECTION',key=list(key),path=str(folder)));continue
  if key[0]=='replay':
   try:comparisons.append(replay_comparison(baseline,folder,r,e,key[3]))
   except Exception as exc:issues.append(dict(kind='REPLAY_COMPARE_ERROR',path=str(folder),key=list(key),error=repr(exc)))
  cells.append(row);observed[key]=row
 coverage=[dict(mode=k[0],object_id=k[1],phase=k[2],k=k[3],noise_basis_points=k[4],realization_index=k[5],status='AUDITED_TERMINAL' if k in observed else 'MISSING_OR_NOT_AUDITED') for k in expected]
 objects,summaries=object_statistics(cells,expected);expectedcounts=dict(collections.Counter(k[0] for k in expected));registered_counts_match=expectedcounts=={'replay':36,'validation':99,'noise':72}
 if not registered_counts_match:issues.append(dict(kind='EXPECTED_DENOMINATOR_MISMATCH',actual=expectedcounts,required=dict(replay=36,validation=99,noise=72)))
 for issue in issues:issue.setdefault('severity','critical')
 critical_issues=[r for r in issues if r['severity']=='critical'];retained_history=[r for r in issues if r['severity']=='retained_history']
 counts=dict(collections.Counter(r['mode'] for r in cells));phase_status={mode:('COMPLETE_MECHANICAL_OWNER_REVIEW_REQUIRED' if counts.get(mode,0)==count and not critical_issues else 'INCOMPLETE_OR_CRITICAL_AUDIT_ISSUES') for mode,count in expectedcounts.items()}
 status=dict(schema=SCHEMA,status='INCOMPLETE_MECHANICAL_COLLECTION' if len(cells)!=len(expected) or critical_issues else 'COMPLETE_MECHANICAL_COLLECTION_OWNER_REVIEW_REQUIRED',phase_status=phase_status,critical_issues=len(critical_issues),retained_historical_issues=len(retained_history),expected_counts=expectedcounts,required_counts=dict(replay=36,validation=99,noise=72),audited_counts=dict(collections.Counter(r['mode'] for r in cells)),replay_comparisons=len(comparisons),missing_cells=len(expected)-len(observed),issues=len(issues),holdout_caveat=CAVEAT,driver_mode_to_user_phase=dict(validation='holdout_validation',replay='old36_replay',noise='noise_validation'),manifest_primary_scope=manifest.get('primary_scope'),confirmed_blind_objects=manifest.get('strict_new_blind_holdout_count_confirmed'),manifest_sha256=sha(manifest_path),scientific_verdict=None,bootstrap_is_object_cluster_only=True,old36_cells_are_not_independent=True)
 epochs=[]
 for filename in ('A17_BASELINE_LOCK.json','A17_REMOTE_EXECUTION_BASELINE_LOCK.json'):
  path=closure/'research'/filename
  if path.is_file():
   epoch=read(path);epochs.append(dict(path=str(path),sha256=sha(path),read_only_code_count=sum(k.startswith('code/') for k in epoch.get('read_only_files',{})),epoch_scope=epoch.get('epoch_scope'),local_only_nonruntime_files=epoch.get('local_only_nonruntime_files',{}),prior_paid_occupation_s=epoch.get('prior_paid_occupation_s')))
 out.mkdir(parents=True)
 (out/'BASELINE_EPOCHS.json').write_text(json.dumps(dict(epochs=epochs,scope='13 remote executable files vs14 local code files; local-only audit_first_state.py is nonruntime; preserve separate lock hashes rather than equate epochs'),indent=2)+'\n')
 for name,rows in [('cells.csv',cells),('coverage.csv',coverage),('replay_comparisons.csv',comparisons),('object_summed_risks.csv',objects),('attempt_costs.csv',fees),('state_stage_costs.csv',costs)]:write_csv(out/name,rows)
 for name,data in [('STATUS.json',status),('ISSUES.json',issues),('OBJECT_SUMMARY.json',summaries),('SOURCE_BINDINGS.json',sources)]:
  (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 return status

def main():
 p=argparse.ArgumentParser();p.add_argument('--closure-root',required=True);p.add_argument('--baseline-root',required=True);p.add_argument('--results-root');p.add_argument('--ledger');p.add_argument('--manifest');p.add_argument('--out',required=True);a=p.parse_args();c=Path(a.closure_root)
 result=collect(c,a.baseline_root,a.results_root or c/'results',a.ledger or c/'execution/gpu_attempts.jsonl',a.manifest or c/'HOLDOUT_OBJECT_MANIFEST.json',a.out)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
