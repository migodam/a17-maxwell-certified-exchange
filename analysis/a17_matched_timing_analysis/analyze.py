from pathlib import Path
import json,hashlib,csv,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).absolute().parents[3];R=ROOT/'Gaussian/A17/EXCHANGE_R1';OUT=Path(__file__).absolute().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
METHODS=['cached_receiver_legacy16_onepass','same_shortlist12_directed_one_decision','a17_conditional12_directed3_matched']
SHORT=['legacy16 fixed one-pass','same12 one decision','conditional12 max3']
summary={'scope':'Read-only persisted analysis; no physical actions, benchmarks, source modifications or scientific adjudication.','objects':{},'failures':[],'limits':[]};table=[];comp=[];provenance={}
def resolve(p):
 s=p.replace('\\','/');s='/'.join(x for x in s.split('/') if x)
 for key in ('results/','inputs/','vendor/','code/'):
  if key in s:return R/(key+s.split(key,1)[1])
 return ROOT/'research/delegated/a17_cached_onepass'/s.split('/')[-1]
def quant(a):return dict(zip(['q1','median','q3'],map(float,np.percentile(a,[25,50,75]))))
def walkstats(stats):
 for st in stats:
  yield st.get('statistics',st)
for oid in (2007,2012):
 folder=R/f'results/timing_{oid}_early_v2';original=R/('results/object_'+str(oid)+'_early'+('_portable' if oid==2007 else ''))
 d=read(folder/'TIMINGS.json');s=d['setup'];start=read(folder/'IMPORTS_START.json');rows=d['samples'];warm=d['prewarm_charged_separately'];allrows=warm+rows;journal=[json.loads(x) for x in (folder/'SAMPLE_JOURNAL.jsonl').read_text().splitlines()]
 assert d['status']=='COMPLETE' and len(rows)==45 and len(warm)==9 and len(journal)==54
 assert start['actual_imports']==d['final_imports']['actual_imports'] and start['driver']==d['final_imports']['driver']
 checks=[]
 for name,v in start['actual_imports'].items():
  p=resolve(v['actual_path']);checks.append({'name':name,'path':str(p.relative_to(ROOT)),'expected':v['sha256'],'current':sha(p),'match':sha(p)==v['sha256']})
 for p,h in s['input_sha256'].items():assert sha(resolve(p))==h
 assert s['is_actual_physical_callback'] and not s['test_callback_is_not_physical'] and not s['offlinefullJ_constructed'] and not s['full_reference_step_optimization']
 assert s['physical_objects_constructed']==s['shared_physical_object_cache_count']==1
 for rep in range(5):
  order=METHODS[rep%3:]+METHODS[:rep%3]
  for k in (4,8,16):
   z=sorted([x for x in rows if x['repetition']==rep and x['requested_k']==k],key=lambda x:x['order_position']);assert [x['method'] for x in z]==order
 for k in (4,8,16):assert {x['method'] for x in warm if x['requested_k']==k}==set(METHODS)
 sums={scope:{key:sum(x['physical_'+scope+'_counter_delta']['totals'].get(key,0) for x in allrows) for key in d['physical_all_trials_counter_delta']['totals']} for scope in ('selection','offline_evaluation','all')}
 assert sums['all']==d['physical_all_trials_counter_delta']['totals']
 assert all(sums['selection'][key]+sums['offline_evaluation'][key]==sums['all'][key] for key in sums['all'])
 stats=[];arraychecks=[];accepted_errors=[]
 for j in journal:
  p=folder/j['endpoint_array_path'].replace('\\','/').split('/')[-1];assert sha(p)==j['endpoint_array_sha256'];a=np.load(p);x=a['step'];U=a['U'];assert ah(x)==j['step_sha256'] and ah(U)==j['basis_sha256'];assert x.dtype==np.float64 and U.dtype==np.complex128 and U.shape[1]==j['actual_rank']==j['requested_k'];assert np.isfinite(x).all() and np.isfinite(U).all()
  arraychecks.append({'npz':p.name,'sha256':sha(p),'orthogonality_F':float(np.linalg.norm(U.conj().T@U-np.eye(U.shape[1])))})
  assert (j['device'],j['anchor_device'],j['dtype'])==('cuda','cpu','complex128/float64');assert j['cold_wall_s'] is None and j['cold_speedup'] is None
  assert j['direction_selection']['new_physical_rhs']==j['physical_selection_counter_delta']['totals'].get('jvp_rhs',0)
  assert j['direction_offline_evaluation']['new_physical_rhs']==j['physical_offline_evaluation_counter_delta']['totals'].get('jvp_rhs',0)==12
  stats+=list(walkstats(j['endpoint_statistics']))
  if j['method']!=METHODS[0]:
   positives=[]
   for t in j['trace']:
    qs=[f['Q_true'] for f in t.get('finalists',[]) if f['Q_true']>f['tau']]
    if t.get('accepted'):positives.append(max(qs))
   accepted_errors.append(abs(sum(positives)-j['evaluation']['true_gain']))
 failure_stats=[v for v in stats if v.get('factorization_failures',0) or v.get('successful_units',v.get('candidate_count',0))<v.get('candidate_count',0)]
 incomplete_current=[v for v in stats if v.get('current_factorizations',0)>0 and v.get('successful_units',0)==0 and 'current_full_coupling_solve' not in v.get('stage_wall_s',{})]
 incomplete_schur=[v for v in stats if v.get('core_reuse_solve_calls',0)>0 and v.get('successful_units',0)==0 and 'batched_Schur_child_coefficients' not in v.get('stage_wall_s',{})]
 for k in (4,8,16):
  for mi,m in enumerate(METHODS):
   z=[x for x in rows if x['requested_k']==k and x['method']==m];sel=quant([x['warm_selection_wall_s'] for x in z]);ev=quant([x['offline_evaluation_wall_s'] for x in z]);gain=quant([x['evaluation']['true_gain'] for x in z]);acc=[sum(bool(t.get('accepted')) for t in x['trace']) for x in z]
   record=dict(object_id=oid,k=k,method=m,n=5,selection_median_s=sel['median'],selection_q1_s=sel['q1'],selection_q3_s=sel['q3'],selection_IQR_s=sel['q3']-sel['q1'],offline_eval_median_s=ev['median'],offline_eval_q1_s=ev['q1'],offline_eval_q3_s=ev['q3'],true_gain_median=gain['median'],true_gain_min=min(x['evaluation']['true_gain'] for x in z),true_gain_max=max(x['evaluation']['true_gain'] for x in z),accepted_min=min(acc),accepted_max=max(acc),selection_physical_RHS_min=min(x['direction_selection']['new_physical_rhs'] for x in z),selection_physical_RHS_max=max(x['direction_selection']['new_physical_rhs'] for x in z),offline_physical_RHS=12,cold_wall_s=None,cold_speedup=None)
   for stage in ('receiver_basis_wall_s','receiver_endpoint_wall_s','proposal_basis_wall_s','candidate_endpoint_wall_s','anchor_screen_wall_s','Jx_wall_s','Jd_wall_s','proposal_basis_endpoint_wall_s'):record[stage+'_median']=float(np.median([x['stage_wall_s'].get(stage,0) for x in z]))
   for fee in z[0]['endpoint_fee_totals']:record[fee+'_median']=float(np.median([x['endpoint_fee_totals'][fee] for x in z]))
   table.append(record)
  for m,target in [(METHODS[2],'directed_verified_endpoint.npz'),(METHODS[1],'directed_verified_round_0.npz')]:
   a=np.load(original/f'k{k}'/target);b=np.load(folder/f'r0_k{k}_p{METHODS.index(m)}_{m}.npz');u=a['U_coeff'];v=b['U'];x=a['step'];y=b['step'];span=float(max(np.linalg.norm(u-v@(v.conj().T@u)),np.linalg.norm(v-u@(u.conj().T@v))));err=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-300))
   comp.append(dict(object_id=oid,k=k,matched_method=m,production_array=target,step_relative_error=err,step_max_abs_error=float(np.max(np.abs(x-y))),basis_span_F_error=span,production_endpoint_device='cpu' if oid==2007 else 'cuda',matched_endpoint_device='cuda',comparison_scope='cached output compatibility; not production runtime speed comparison'))
 out=dict(status=d['status'],formal_samples=45,paid_prewarm_samples=9,journal_arrays_verified=54,rotations_verified=True,imports_start_end_equal=True,local_import_hash_checks=checks,physical_callback=s['direction_backend'],dtype='complex128/float64',common_device='CUDA endpoints and physical callback; CPU anchor',physical_counter_totals=sums,physical_setup=s['physical_model_setup_counters'],fresh_shared_physical_setup_wall_s=s['fresh_physical_state_setup_wall_s'],cache_load_wall_s=s['public_cache_load_wall_s'],adapter_wall_s=s['adapter_wall_s'],historical_receipt=s['historical_public_acquisition_receipt'],historical_setup_fields=read(original/'setup_cost.json'),cold_wall_s=None,cold_speedup=None,paid_prewarm_selection_s=sum(x['warm_selection_wall_s'] for x in warm),paid_prewarm_evaluation_s=sum(x['offline_evaluation_wall_s'] for x in warm),failures=d['failures'],endpoint_stat_records=len(stats),factorization_or_unit_failure_records=len(failure_stats),stage_timer_all_current_observed=len(incomplete_current),stage_timer_all_schur_observed=len(incomplete_schur),stage_timer_limit='No triggering failed unit in complete persisted sample/stat records. This is not proof branch gap absent generally; phase timers remain incomplete if all-current/all-Schur fail.',directed_telescoping_gain_max_abs_error=max(accepted_errors),array_orthogonality_max_F=max(a['orthogonality_F'] for a in arraychecks),deps={k:start[k] for k in ('python','numpy','scipy','torch','cuda_version')})
 summary['objects'][str(oid)]=out
 provenance[str(oid)]={'receipts':{p.name:sha(p) for p in folder.glob('*.json')},'journal_sha256':sha(folder/'SAMPLE_JOURNAL.jsonl'),'arrays':arraychecks,'original_arrays':{str(p.relative_to(original)):sha(p) for p in original.glob('k*/directed_verified*.npz')}}
for name,data in [('method_by_k.csv',table),('comparison.csv',comp)]:
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary['limits']=['Legacy control is cached receiver + fixed at-most16 one-pass, not entire old S8; methods deliberately have different physical direction budgets.','Historical public workspace/anchor/pool acquisition is reused and not a newly measured standalone cold run; no cold speedup is available.','Gaussian original production uses CPU small-material endpoint; matched methods all use CUDA.','Physical model byte-transfer details are not fully exposed; endpoint upload/download estimates and callback/solve RHS remain explicit.','Sparse native counters have no adjoint action records; j-only callback and tangent RHS observed. No independent instrumentation proves unlogged hardware work.','Final chosen logical IDs and all intermediate endpoint arrays are not persisted; finalist/Q acceptance and final span/steps can be checked, full move replay cannot.','Offline reference matrices are read only as optional audit evidence, never used to rank matched methods.']
summary['comparison_max_step_relative_error']=max(x['step_relative_error'] for x in comp);summary['comparison_max_span_error']=max(x['basis_span_F_error'] for x in comp)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(OUT/'INPUT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
colors=['#555555','#2274a5','#a63d40']
for oi,oid in enumerate((2007,2012)):
 for mi,m in enumerate(METHODS):
  z=[x for x in table if x['object_id']==oid and x['method']==m];xs=np.arange(3)+(mi-1)*.18
  for ci,prefix in enumerate(('selection','offline_eval')):
   med=np.array([x[prefix+'_median_s'] for x in z]);lo=np.array([x[prefix+'_q1_s'] for x in z]);hi=np.array([x[prefix+'_q3_s'] for x in z]);axes[oi,ci].errorbar(xs,med,yerr=[med-lo,hi-med],fmt='o-',color=colors[mi],label=SHORT[mi],capsize=3)
 for ci in range(2):axes[oi,ci].set(title=f'{oid} early: '+('warm selection' if ci==0 else 'separate offline endpoint evaluation'),ylabel='seconds (median, Q1–Q3)',xticks=range(3),xticklabels=[4,8,16],xlabel='actual k');axes[oi,ci].grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures/warm_wall.png',dpi=160);fig.savefig(OUT/'figures/warm_wall.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for oi,oid in enumerate((2007,2012)):
 for mi,m in enumerate(METHODS):
  z=[x for x in table if x['object_id']==oid and x['method']==m];axes[oi].plot([4,8,16],[x['true_gain_median'] for x in z],'o-',label=SHORT[mi],color=colors[mi])
 axes[oi].axhline(0,color='black',lw=.7);axes[oi].set(title=f'{oid} early: persisted full objective gain',xlabel='actual k',ylabel='initial minus final objective');axes[oi].grid(alpha=.2)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures/objective_gain.png',dpi=160);plt.close(fig)
print(json.dumps({'max_step_relative_error':summary['comparison_max_step_relative_error'],'max_span_error':summary['comparison_max_span_error'],'all_local_hash_match':all(x['match'] for s in summary['objects'].values() for x in s['local_import_hash_checks']),'gain_telescoping_error':{o:s['directed_telescoping_gain_max_abs_error'] for o,s in summary['objects'].items()},'out':str(OUT)}))
