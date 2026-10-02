import json,hashlib,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'real_runs_final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
config=json.loads((HERE/'FROZEN_CONFIG.json').read_text());manifest=json.loads((HERE/'DATASET_MANIFEST_FINAL.json').read_text());idx=json.loads((OUT/'SIX_RUN_INDEX.json').read_text());baseline=json.loads((OUT/'QHAT_TOP2_BASELINE.json').read_text());base={r['object_id']:r for r in baseline['object_summaries']};runs=[];tab=[]
assert len(idx)==6 and {(r['architecture'],r['seed']) for r in idx}=={(a,s) for a in config['architectures'] for s in config['seeds']}
for item in idx:
 r=json.loads((OUT/f'{item["architecture"]}_seed{item["seed"]}.json').read_text());assert r['training_epochs']==100 and r['dtype']=='float64' and r['device']=='cpu' and r['cpu_threads']==1 and r['source_sha256']==manifest['source_sha256'] and r['configuration_sha256']==manifest['config_sha256'];assert sha(OUT/f'{r["architecture"]}_seed{r["seed"]}.pt')==r['model_sha256'];assert all(np.isfinite(r['loss_history']))
 allmetrics=[m for ss in r['metrics'].values() for m in ss];assert len(allmetrics)==36 and all(m['all_candidate_physical_features_wall_s'] is None and m['fresh_validation_wall_s'] is None and m['all_in_speedup'] is None for m in allmetrics)
 costs=dict(training_cpu_s=r['cpu_train_wall_s'],inference_cpu_s_all36=sum(m['inference_cpu_s'] for m in allmetrics),normalization_tensor_cpu_s_all36=sum(m['feature_normalization_tensor_cpu_s'] for m in allmetrics),all_candidate_physical_feature_cost=None,fresh_physical_validation_cost=None,all_in_speedup=None)
 objects=[]
 for x in r['object_summaries']:
  y=dict(**x,Qhat_top2_baseline_capture=base[x['object_id']]['object_summed_capture'],capture_minus_Qhat_baseline=x['object_summed_capture']-base[x['object_id']]['object_summed_capture']);objects.append(y);tab.append(dict(architecture=r['architecture'],seed=r['seed'],**y))
 runs.append(dict(architecture=r['architecture'],seed=r['seed'],training_epochs=100,cpu_threads=1,parameter_count=r['parameter_count'],costs=costs,object_summaries=objects,model_sha256=r['model_sha256']))
norm=json.loads((OUT/'normalization.json').read_text());assert norm['object_ids']==[2007,2012]
summary=dict(status='COMPLETED_ONCE_FIXED_SIX_RUNS',checkpoint_count=36,feasible_candidate_count=manifest['legal_feature_audit']['feasible_candidate_count'],total_dictionary_rows=sum(c['total_rows'] for c in manifest['checkpoints']),split_checkpoint_counts={s:sum(c['split']==s for c in manifest['checkpoints']) for s in config['split']},split=config['split'],architectures=config['architectures'],seeds=config['seeds'],epochs=100,device='CPU',dtype='float64',cpu_threads=1,runs=runs,Qhat_top2_baseline_objects=baseline['object_summaries'],legal_no_leakage_checks=dict(allowlist_only=True,forbidden_field_injection_canary_36_pass=True,information_trace_endpoint_reduced=True,train_only_normalization=True,normalization_candidate_count=norm['candidate_count'],validation_not_used_for_selection=True,test_after_model_serialization=True,no_tuning_no_retry_all_six_retained=True),costs=dict(sum_six_training_cpu_s=sum(r['costs']['training_cpu_s'] for r in runs),sum_six_inference_cpu_s=sum(r['costs']['inference_cpu_s_all36'] for r in runs),feature_array_construction_cpu_s_shared_once=sum(m['feature_construction_cpu_s'] for ss in baseline['metrics'].values() for m in ss),normalization_fit_cpu_s_shared_once=json.loads((OUT/f'{idx[0]["architecture"]}_seed{idx[0]["seed"]}.json').read_text())['normalization_fit_cpu_s_shared_once'],manifest_binding_cpu_s=manifest['cpu_binding_wall_s'],all_in_feature_acquisition_cost=None,fresh_physical_validation_cost=None,all_in_speedup=None),hashes=dict(dataset_manifest=sha(HERE/'DATASET_MANIFEST_FINAL.json'),source=sha(HERE/'tiny_proposer.py'),config=sha(HERE/'FROZEN_CONFIG.json'),authorization_binding=sha(OUT/'AUTHORIZATION_BINDING.json')),scope='one receiver round0 top2 priority pilot on four frozen objects; offline oracle labels choose top2/no-op; not fresh physical acceptance, trajectory, nonlinear reconstruction, generalized safety or speedup',new_full_actions=0)
(HERE/'REAL_PILOT_SUMMARY.json').write_text(json.dumps(summary,indent=2))
with (OUT/'OBJECT_TOP2_CAPTURE.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(tab[0]));w.writeheader();w.writerows(tab)
text='''# Fixed six-run real priority pilot

COMPLETED once with unchanged frozen trainer/config: two architectures × three seeds ×100 epochs, CPU FP64 one thread. Exactly36 complete receiver-neighborhood round0 dictionary traces;43,108 feasible candidates. Split train2007/2012 (18checkpoints), validation2001 (9), heldout test2014 (9). Validation did not select model/epoch; test only evaluated after each trained model was serialized. All six receipts/models/loss curves retained, no tuning/retry.

| Architecture | Seed | Train2007 | Train2012 | Validation2001 | Test2014 |
|---|---:|---:|---:|---:|---:|
'''
for r in runs:
 captures={x['object_id']:x['object_summed_capture'] for x in r['object_summaries']};text+=f"|{r['architecture']}|{r['seed']}|"+'|'.join(f'{captures[o]:.6f}' for o in [2007,2012,2001,2014])+"|\n"
text+='|Qhat top2 baseline|fixed|'+'|'.join(f'{base[o]["object_summed_capture"]:.6f}' for o in [2007,2012,2001,2014])+'|\n'
text+='''
Capture is sum best(full-dictionary Q,no-op) across an object's nine common receiver checkpoints in the denominator, and sum best(predicted top2 Q,no-op) in the numerator. Finalist labels were already-paid offline labels, not fresh physical acceptance. Candidate counts are not independent generalization samples; a single heldout object does not establish broad safety or universality. No seed or architecture is selected here. Root owns interpretation and gates.

The baseline retains higher heldout capture than every learned run in this fixed pilot. This descriptive outcome does not establish that all learned proposals fail or that a different method would improve; no tuning or additional model was attempted.

Feature provenance: actual runtime receipt generator sources were matched to deployment ZIP SHA. The endpoint batch forms reduced Z from QR(SU) and Galerkin material injection/propagation F, evaluates2||Z||F² excludingLambda, and run_exchange stores evaluated.info_trace. This is reduced-model information_trace, not evaluator full/projected information. Qhat is independently richer-anchor gain; child step norm/stability/trace are reduced endpoint quantities. FullQ labels are targets/evaluation only. Thirty-six canary injections of all forbidden fields left features unchanged; normalization fits only train objects. Source/config SHA still match pre-real frozen manifest.

DATASET_MANIFEST_FINAL.json binds each full trace's SHA, complete Cartesian1swap coverage, original round0/baseU/state/pool identity, terminal result/status, runtime/source and prior-paid setup/checkpoint cost receipts. Full dictionary includes preserved non-feasible records. Paid-feature source binding does not imply acquisition cost was measured separately or was free.

'''
c=summary['costs'];text+=f"Measured CPU: six training intervals sum {c['sum_six_training_cpu_s']:.6f}s; six runs' inference on all36checkpoints sum {c['sum_six_inference_cpu_s']:.6f}s; shared feature-array construction {c['feature_array_construction_cpu_s_shared_once']:.6f}s; shared train normalization fit {c['normalization_fit_cpu_s_shared_once']:.6f}s; manifest binding {c['manifest_binding_cpu_s']:.6f}s. These are component timings, not whole-job/all-in elapsed time. Feature array construction excludes prior anchor/child physics acquisition. Full feature acquisition, fresh physical finalist validation and all-in speedup remain null/NOT_MEASURED. No speedup or generalized safety claim.\n\nOutputs: real_runs_final/SIX_RUN_INDEX.json, six .pt/.json receipts, QHAT_TOP2_BASELINE.json, normalization.json, AUTHORIZATION_BINDING.json and OBJECT_TOP2_CAPTURE.csv. Detailed legal/count/cost/hash checks in REAL_PILOT_SUMMARY.json. No new Maxwell/fulladjoint/tangent actions, GPU, remote operations or changes to frozen source/config/data.\n"
(HERE/'REAL_PILOT_SUMMARY.md').write_text(text);print(json.dumps(summary['costs']))
