"""Authorized read-only provenance binding. Frozen trainer/config untouched."""
import csv,json,hashlib,sys,zipfile,math,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];R=ROOT/'Gaussian/A17/EXCHANGE_R1';A=ROOT/'research/delegated/a17_pilot_analysis/analysis_output_final'
sys.path.insert(0,str(HERE));import tiny_proposer as tp
start=time.perf_counter();authpath=R/'research/NN_OWNER_AUTHORIZATION.json';auth=tp.authorize(authpath)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
frozen=json.loads((HERE/'SOURCE_MANIFEST.json').read_text());expected={x['path']:x['sha256'] for x in frozen['files']};require(all(sha(HERE/p)==expected[p] for p in ['FROZEN_CONFIG.json','tiny_proposer.py']),'frozen source/config drift')
require(not (HERE/'DATASET_MANIFEST_FINAL.json').exists(),'manifest already exists; no implicit repeat')
for name,h in auth['evidence_sha256'].items():require(sha(A/name)==h,'owner gate evidence hash mismatch '+name)
cases=[r for r in csv.DictReader((A/'case_coverage.csv').open()) if r['main_eligible_complete_state']=='True'];require(len(cases)==12,'exactly12 completed frozen states required');require({(int(r['object_id']),r['phase']) for r in cases}=={(o,p) for o in [2001,2007,2012,2014] for p in ['early','middle','late']},'four objects x3 phase identity mismatch')
# Verify actual generator sources against receipt ZIP members, not current filename alone.
archives=[]
for zp in (R/'research/deployments').glob('*.zip'):
 with zipfile.ZipFile(zp) as z:
  for member in z.namelist():
   if member.endswith('/run_exchange.py') or member.endswith('/endpoint_batch.py'):archives.append((zp.name,member,sha(zp),z.read(member)))
checks=[];canary_count=0;candidate_count=0;sourcebindings=[]
for case in cases:
 d=Path(case['directory']);state=json.loads((d/'result.json').read_text());receipt=json.loads((d/'runtime_receipt.json').read_text());status=json.loads((d/'status.json').read_text());require(state['status']==status['status']=='COMPLETED','terminal mismatch');require(state['object_id']==int(case['object_id']) and state['phase']==case['phase'],'case/state mismatch')
 for name in ['run_exchange.py','endpoint_batch.py']:
  h=receipt['runtime']['a17_sources']['code\\'+name];matches=[dict(zip=zn,member=member,zip_sha256=zh) for zn,member,zh,raw in archives if hashlib.sha256(raw).hexdigest()==h];require(matches,'actual generator source not found in receipt ZIP '+name);sourcebindings.append(dict(state=d.name,source=name,receipt_sha256=h,archive_matches=matches))
  raw=next(raw for zn,member,zh,raw in archives if hashlib.sha256(raw).hexdigest()==h).decode()
  if name=='run_exchange.py':require("information_trace=float(evaluated['info_trace'][j])" in raw and 'qhat=gains_on_anchor' in raw,'feature generator provenance mismatch')
  else:require('trace=2*' in raw and 'np.sum(np.abs(Z)**2' in raw and 'Z=(Rc[:,None]@F)' in raw,'reduced trace provenance mismatch')
 with np.load(d/'public_workspace.npz',allow_pickle=False) as data:poolsize=data['pool'].shape[1]
 for k in [4,8,16]:
  kd=d/f'k{k}';p=kd/'teacher_receiver_moves.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];cp=json.loads((kd/'receiver_checkpoint.json').read_text());base=json.loads((kd/'receiver_no_swap.json').read_text());feasible=[x for x in rows if x['status']=='OK'];selected=set(base['selected_ids']);require(len(selected)==k,'base requested rank mismatch')
  pairs={(tuple(x['drop']),tuple(x['add'])) for x in rows};required={((u,),(v,)) for u in selected for v in range(poolsize) if v not in selected};require(pairs==required and len(rows)==len(pairs)==k*(poolsize-k),'not exact full1swap dictionary');require(len({x['move_index'] for x in rows})==len(rows),'duplicate move index');require(cp['declared_neighborhood_count']==len(rows) and cp['feasible_count']==len(feasible) and cp['full_domain_1exchange_coverage'] is True,'coverage mismatch')
  hashes={x.get('base_U_hash') for x in feasible};require(len(hashes)==1 and None not in hashes,'base U hash mismatch');basehash=next(iter(hashes));require(all(x['state_hash']==receipt['state_hash']==cp['state_hash'] and x['pool_hash']==receipt['pool_hash']==cp['pool_hash'] and x['policy_id']=='receiver_neighborhood_teacher' and x['round']==0 and x['object_id']==state['object_id'] and x['phase']==state['phase'] and x['k']==k for x in rows),'per-move frozen identity mismatch')
  x,scale,_=tp.legal_features(feasible);poison=[]
  for row in feasible:
   changed=dict(row)
   for forbidden in tp.CONFIG['forbidden_features']:changed[forbidden]={'POISON':1e99} if 'information' in forbidden else -1e99
   poison.append(changed)
  x2,scale2,_=tp.legal_features(poison);require(np.array_equal(x,x2) and scale==scale2,'forbidden-field canary altered features');canary_count+=1;candidate_count+=len(feasible)
  costs={str(z.relative_to(R)):sha(z) for z in [d/'setup_cost.json',d/'result.json',d/'runtime_receipt.json',kd/'receiver_checkpoint.json']};identity=dict(object_id=state['object_id'],phase=state['phase'],k=k,state_hash=receipt['state_hash'],pool_hash=receipt['pool_hash'],base_U_hash=basehash)
  checks.append(dict(path=str(p.resolve()),sha256=sha(p),total_rows=len(rows),feasible_rows=len(feasible),complete_one_swap=True,result_status='COMPLETED',split=tp.split_of(state['object_id']),identity=identity,paid_feature_cost_source=dict(files=costs,scope='prior-paid setup/whole-checkpoint/endpoints; source binding only, not separately measured all-in feature acquisition',feature_acquisition_wall_s=None,fresh_physical_validation_wall_s=None),binding_files={str(z.relative_to(R)):sha(z) for z in [d/'status.json',d/'runtime_receipt.json',kd/'receiver_no_swap.json',kd/'receiver_checkpoint.json']},legal_information_trace_provenance='endpoint_batch evaluates2||Z_child||F² from reduced Galerkin F and QR(SU); run_exchange stores evaluated.info_trace, excludingLambda; not full/reference/evaluator information'))
require(len(checks)==36,'exactly36 required');manifest=dict(schema='a17.tiny.proposer.final.bound',status='HASH_BOUND_AUTHORIZED_REAL_PILOT',checkpoints=checks,authorization_path=str(authpath.resolve()),authorization_sha256=sha(authpath),source_sha256=sha(HERE/'tiny_proposer.py'),config_sha256=sha(HERE/'FROZEN_CONFIG.json'),generator_source_bindings=sourcebindings,legal_feature_audit=dict(allowlist=tp.CONFIG['features'],forbidden_canary_checkpoints=canary_count,canary_pass=True,feasible_candidate_count=candidate_count,train_normalization_source_only=[2007,2012],heldout_test_object=2014,validation_selection=False,all_in_feature_cost=None,fresh_validation_cost=None,scope='legal features need whole-neighborhood reduced endpoints and anchor computations; legal does not mean cheap'),cpu_binding_wall_s=time.perf_counter()-start)
(HERE/'DATASET_MANIFEST_FINAL.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(checkpoints=36,feasible_candidates=candidate_count,source_unchanged=True,config_unchanged=True,forbidden_canary_pass=True,cpu_wall_s=manifest['cpu_binding_wall_s'])))
