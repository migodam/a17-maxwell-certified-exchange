"""Read analyzer CSV + persisted teacher/info records; no fullJ or physics."""
import argparse,csv,json,hashlib,time,math
from pathlib import Path
from itertools import combinations
import numpy as np
from ranking_core import summarize
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
METRICS=['receiver_score','delta_logdet_volume','delta_effective_dim','Qhat','information_fidelity_improvement']
def num(x):
 try:y=float(x);return y if math.isfinite(y) else None
 except (ValueError,TypeError):return None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def readcsv(p):return list(csv.DictReader(Path(p).open()))
def field(ev):
 if ev is None:return None
 for key in ['full_information_fidelity','projected_information_fidelity']:
  if ev.get(key):return num(ev[key].get('info_fidelity_spectral'))
 return None

def run(csvpath,coveragepath,out):
 start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True);raw=readcsv(csvpath);coverage=readcsv(coveragepath);cov={ (r['source_directory'],r['k']):r for r in coverage};groups={}
 for r in raw:groups.setdefault((r['source_directory'],r['k']),[]).append(r)
 summaries=[];witnesses=[];linked=[];missing=[]
 for (directory,k),rows in groups.items():
  try:
   d=Path(directory);kp=d/f'k{k}';teacherpath=kp/'teacher_receiver_moves.jsonl';infopath=kp/'information_rankings_by_move.jsonl';cp=kp/'receiver_checkpoint.json';icp=kp/'information_coverage.json';bp=kp/'receiver_no_swap.json'
   tc=lines(teacherpath);teacher={r['move_index']:r for r in tc};info={r['move_index']:r for r in lines(infopath)};check=json.loads(cp.read_text());ic=json.loads(icp.read_text());base=json.loads(bp.read_text());basef=field(base.get('information',{}).get('evaluator_information'));tau=num(check.get('tau'))
   if tau is None:raise ValueError('missing persisted numerical Q tolerance')
   c=cov.get((directory,k));
   if c is None or int(c['information_moves'])!=len(info) or int(c['teacher_moves'])!=len(teacher):raise ValueError('diagnostic coverage mismatch/missing')
   fixed=set(ic['fixed_incoming_ids']);basehash=set();joined=[]
   for csvrow in rows:
    idx=int(csvrow['move_index']);t=teacher.get(idx);i=info.get(idx)
    if not t or not i or csvrow['teacher_identity_match']!='True' or t['ids']!=i['ids']:raise ValueError('move identity not paired')
    if any(csvrow[key]!=t[key] or t[key]!=check[key] for key in ['state_hash','pool_hash']):raise ValueError('state/pool mismatch')
    if csvrow['base_U_hash']!=t['base_U_hash']:raise ValueError('CSV base U hash mismatch')
    if any(num(csvrow.get(key))!=num(t.get(key)) for key in ['Qhat','receiver_score']):raise ValueError('CSV task/receiver score differs from original teacher')
    if any(num(csvrow.get(key))!=num(i['selector_information']['candidate'].get(key)) for key in ['delta_logdet_volume','delta_effective_dim']):raise ValueError('CSV spectrum delta differs from original information record')
    if num(csvrow['Q_true'])!=num(t.get('Q_true')) or num(i.get('Q_true'))!=num(t.get('Q_true')):raise ValueError('original Q differs from CSV/info record')
    if t.get('status')!='OK' or i['selector_information'].get('spectrum_evaluated') is not True:continue
    basehash.add(t['base_U_hash']);j={key:num(csvrow.get(key)) for key in ['Q_true','Qhat','receiver_score','delta_logdet_volume','delta_effective_dim','delta_info_trace']};f=field(i.get('evaluator_information'));j['information_fidelity_improvement']=basef-f if basef is not None and f is not None else None
    j.update(object_id=int(csvrow['object_id']),phase=csvrow['phase'],k=int(k),move_index=idx,state_hash=t['state_hash'],pool_hash=t['pool_hash'],base_U_hash=t['base_U_hash'],base_logger_U_hash=base.get('information',{}).get('U_hash'),U_hash_scope='teacher base hash raw array bytes; logger child/base hash includes shape and dtype, not directly interchangeable',child_U_hash=i.get('U_hash'),step_hash=t.get('step_hash'),original_ids=t['ids'],drop=t['drop'],add=t['add'],fixed_subset=all(x in fixed for x in t['add']),teacher_json_path=str(teacherpath),teacher_record_sha256=digest(t),teacher_json_sha256=sha(teacherpath),information_json_path=str(infopath),information_record_sha256=digest(i),information_json_sha256=sha(infopath),csv_source_path=str(Path(csvpath).resolve()),checkpoint_json_path=str(cp),base_json_path=str(bp),array_paths={'public_workspace':str(d/'public_workspace.npz'),'stored_delta_b_original':i.get('delta_b_path'),'stored_delta_b_local':str(kp/'information_rhs'/('delta_b_'+i['delta_b_sha256']+'.npz')) if i.get('delta_b_sha256') else None},information_scope=csvrow['information_scope'],fidelity_base_available=basef is not None,move_hash=digest(dict(state=t['state_hash'],pool=t['pool_hash'],base=t['base_U_hash'],ids=t['ids'],drop=t['drop'],add=t['add'])))
    joined.append(j);linked.append(j)
   if len(basehash)!=1:raise ValueError('requires single common receiver base U hash')
   identity=dict(object_id=int(rows[0]['object_id']),phase=rows[0]['phase'],k=int(k),state_hash=check['state_hash'],pool_hash=check['pool_hash'],base_U_hash=next(iter(basehash)),source_directory=directory,information_scope=rows[0]['information_scope'],fidelity_score_definition='base spectral fidelity minus child spectral fidelity, same full/projected scope; no-op0',teacher_domain_count=len(teacher),persisted_spectrum_subset_count=len(joined),fixed12_incoming_count=len(fixed),teacher_best_added=ic['teacher_best_added_for_evaluation_only'],scope='exact persisted spectrum subset only; not complete teacher domain; descriptive no inference/certificate')
   for regime,subset in [('fixed12_per_removal',[r for r in joined if r['fixed_subset']]),('spectrum_plus_teacher_best',joined)]:
    for metric in METRICS:
     stats=summarize(subset,metric,tau);summaries.append(dict(**identity,regime=regime,selection_bias='predeclared fixed bank; bounded nonrandom subset' if regime.startswith('fixed') else 'teacher-best evaluator augmentation introduces utility-selection bias',**stats))
   # Raw reversals are drawn ONLY from fixed subset; augmentation gets separate role flags in linked rows.
   for metric in ['delta_logdet_volume','delta_effective_dim','delta_info_trace']:
    subset=[r for r in joined if r['fixed_subset'] and r.get(metric) is not None];tol=float(128*np.finfo(float).eps*max(1.,max([abs(r[metric]) for r in subset]+[0.])))
    for r in subset:
     if r[metric]>tol and r['Q_true']< -tau:witnesses.append(dict(kind='positive_information_negative_Q',metric=metric,score_tolerance=tol,Q_tolerance=tau,regime='fixed12_per_removal',move=r))
    # Preserve up to three largest-margin pairs/checkpoint/metric; retain total count.
    inv=[]
    for a,b in combinations(subset,2):
     ds=a[metric]-b[metric];dq=a['Q_true']-b['Q_true']
     if abs(ds)>tol and abs(dq)>tau and ds*dq<0:inv.append((abs(dq),abs(ds),a,b))
    inv.sort(key=lambda x:(-x[0],-x[1]))
    for dq,ds,a,b in inv[:3]:witnesses.append(dict(kind='inverse_information_utility_pair',metric=metric,Q_margin=dq,information_margin=ds,score_tolerance=tol,Q_tolerance=tau,all_inversion_pairs_count=len(inv),selection='top3 by absolute Q separation; illustrative selection, not frequency estimator',regime='fixed12_per_removal',move_a=a,move_b=b))
  except Exception as exc:missing.append(dict(source_directory=directory,k=k,status='UNRESOLVED',reason=str(exc)))
 objects=[]
 for oid in sorted(set(s['object_id'] for s in summaries)):
  for metric in METRICS:
   for regime in ['fixed12_per_removal','spectrum_plus_teacher_best']:
    ss=[s for s in summaries if s['object_id']==oid and s['metric']==metric and s['regime']==regime];vals=[s['regret_with_no_op'] for s in ss if s['regret_with_no_op'] is not None];objects.append(dict(object_id=oid,metric=metric,regime=regime,checkpoint_count=len(ss),median_kendall_tau_b=float(np.median([s['kendall_tau_b'] for s in ss if s['kendall_tau_b'] is not None])) if any(s['kendall_tau_b'] is not None for s in ss) else None,median_spearman=float(np.median([s['spearman'] for s in ss if s['spearman'] is not None])) if any(s['spearman'] is not None for s in ss) else None,sum_checkpoint_regret=float(sum(vals)) if vals else None,interpretation='descriptive correlated checkpoint aggregates; neither trajectory regret nor independent samples; no confidence interval'))
 payload=dict(status='PASS' if not missing else 'PARTIAL',checkpoint_metric_summaries=summaries,object_descriptive_summaries=objects,missing=missing,cpu_wall_s=time.perf_counter()-start,input_hashes={str(Path(x).resolve()):sha(x) for x in [csvpath,coveragepath]},source_hashes={x.name:sha(x) for x in [Path(__file__),HERE/'ranking_core.py']},new_full_actions=0)
 (out/'RESULTS.json').write_text(json.dumps(payload,indent=2));(out/'LINKED_MOVES.json').write_text(json.dumps(linked,indent=2));(out/'REVERSAL_WITNESSES.json').write_text(json.dumps(witnesses,indent=2))
 for name,rr in [('CHECKPOINT_SUMMARIES',summaries),('OBJECT_DESCRIPTIVE_SUMMARIES',objects)]:
  if rr:
   fields=list(dict.fromkeys(k for r in rr for k in r));
   with (out/(name+'.csv')).open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rr])
 summary=dict(status=payload['status'],checkpoints=len(groups),resolved_checkpoints=len(set((s['source_directory'],s['k']) for s in summaries)),linked_spectrum_moves=len(linked),witness_count=len(witnesses),missing=missing,cpu_wall_s=payload['cpu_wall_s'],new_full_actions=0,scope='exact fixed spectrum subset plus separately biased teacher-best augmentation; no full-neighborhood claim')
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return payload
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input-dir',default=str(ROOT/'research/delegated/a17_pilot_analysis/analysis_output_development'));p.add_argument('--out',default=str(HERE/'development_results'));a=p.parse_args();run(Path(a.input_dir)/'move_information.csv',Path(a.input_dir)/'diagnostic_coverage.csv',a.out)
