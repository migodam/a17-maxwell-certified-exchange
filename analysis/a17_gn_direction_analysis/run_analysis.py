"""Read completed terminal caches only. Never import physical state/runtime."""
import sys,json,csv,time,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy import linalg as la
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];R=ROOT/'Gaussian/A17/EXCHANGE_R1'
sys.path.insert(0,str(ROOT/'research/delegated/a17_oldspace_information/newpostprocess'))
from postprocess_oldspaces import galerkin_from_cache
from direction_analysis import fixed_reference,compare,receiver_weak_overlap
POLICIES=['surrogate_exchange','exact_score_teacher','directed_verified','adaptive_leq_k']
TOL=1e-9

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rel(a,b):return float(la.norm(np.asarray(a)-np.asarray(b))/max(la.norm(a),la.norm(b),1e-300))
def load(p):return dict(np.load(p,allow_pickle=False))
def require(x,msg):
 if not x:raise ValueError(msg)
def analyze_state(directory,out):
 start=time.perf_counter();directory=Path(directory).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True);result=dict(state=directory.name,status='MISSING',scope='FULL_REFERENCE_FIXED_BASIS',rows=[],missing_reason=None)
 try:
  terminal=json.loads((directory/'status.json').read_text());require(terminal['status']=='COMPLETED','state is not completed')
  state=json.loads((directory/'result.json').read_text());receipt=json.loads((directory/'runtime_receipt.json').read_text());data=load(directory/'public_workspace.npz');offline=load(directory/'offline_information_reference.npz');oid=state['object_id'];phase=state['phase']
  if 'fullJ' not in offline:
   result.update(status='PROJECTED_ONLY',scope='PROJECTED_ONLY',missing_reason='fullJ absent: no complete material directional risk decomposition authorized');return result
  config=json.loads((R/'A17_RUN_CONFIG.json').read_text());it=config['phases'][phase];paths=[R/f'inputs/object_{oid}/{f}' for f in ['common_data.npz',f'anchor_{it:02d}.npz',f'step_{it:02d}.npz']];common,anchor,saved=[load(p) for p in paths]
  checks={str(p.relative_to(R)):sha(p)==receipt['inputs'].get(str(p.relative_to(R))) for p in paths};require(all(checks.values()),'receipt frozen input SHA binding failed')
  fingerprint=str(common['material_tangent_fingerprint']);require(str(anchor['material_tangent_fingerprint'])==str(saved['material_tangent_fingerprint'])==receipt['material_gauge']['basis_fingerprint']==fingerprint,'material gauge mismatch');require(receipt['material_gauge']['portable_material_gauge'] and not bool(saved['material_tangent_legacy_gauge_unverified']),'legacy/unverified saved reference step')
  require(ah(anchor['chi'])==receipt['state_hash'],'state hash mismatch');require(np.array_equal(anchor['ell'],data['ell']) and float(anchor['lambda_total'])==float(data['lambda_total']),'ell/lambda differs from saved anchor')
  require(receipt['information_metadata']['prior_precision']==1e-5,'prior mismatch');prior_path=R/'inputs/original_resolved_config.json';require(sha(prior_path)==receipt['information_metadata']['prior_source_hash'] and json.loads(prior_path.read_text())['prior']==1e-5,'frozen prior config binding failed');require(ah(common['scale'])==receipt['information_metadata']['data_whitening_hash'],'whitening hash mismatch');require(receipt['full_rhs_layout']=='all illumination share one real material direction','illumination layout mismatch');J=offline['fullJ'];sref=saved['step'];require(J.shape[1]==len(sref) and J.shape[0]==len(data['r']),'fullJ material/data shape mismatch')
  q=len(sref)//2;physical=common['material_tangent_Q']@(sref[:q]+1j*sref[q:]);require(rel(physical,saved['step_physical'])<=TOL,'saved real reference coefficients do not match physical gauge')
  ref=fixed_reference(J,float(data['lambda_total']));Jsr=J@sref;gradient=J.T@(data['r']+Jsr)+data['ell']+ref['lam']*sref;gres=float(la.norm(gradient)/max(la.norm(J.T@data['r']+data['ell']),1e-300));require(gres<=1e-10,'full reference residual fails; do not infer or solve missing sref')
  result.update(binding_checks=checks,object_id=oid,phase=phase,prior_precision=ref['precision'],lm_damping=ref['lm'],lambda_total=ref['lam'],state_hash=receipt['state_hash'],material_gauge_fingerprint=fingerprint,reference_relative_residual=gres,input_hashes={str(p.relative_to(R)):sha(p) for p in paths}|{str(p.relative_to(R)):sha(p) for p in [directory/'public_workspace.npz',directory/'offline_information_reference.npz',directory/'runtime_receipt.json']},reference_basis_scope='Complete Euclidean orthonormal eigenbasis of full I; generalized P-normalized basis equals V/sqrt(P)',eigenvalue_blocks=ref['blocks'],full_band_dimensions={b:ref['bands'].count(b) for b in ['weak','intermediate','strong']},basis_clustering_rtol=ref['cluster_rtol'],runtime_source_receipt=receipt['runtime'].get('a17_sources',{}))
  for kd in sorted(directory.glob('k*')):
   if not kd.is_dir() or not (kd/'receiver_no_swap.json').exists():continue
   receiver=json.loads((kd/'receiver_no_swap.json').read_text());ids=receiver['selected_ids'];U,s,_=la.svd(data['pool'][:,ids],full_matrices=False);base=galerkin_from_cache(data,U[:,s>s[0]*1e-10]);before=base['step'];overlap=receiver_weak_overlap(ref,base['J']);k=int(kd.name[1:]);
   for policy in POLICIES:
    rec=json.loads((kd/f'{policy}.json').read_text());endpoint=load(kd/f'{policy}_endpoint.npz');require(rec['state_hash']==receiver['state_hash']==receipt['state_hash'],'endpoint state mismatch');require(rec['pool_hash']==receiver['pool_hash']==receipt['pool_hash'],'endpoint pool mismatch');require(np.array_equal(endpoint['selected_ids'],rec['selected_ids']),'endpoint selected ids mismatch');require(rel(J@endpoint['step'],endpoint['Js'])<=TOL,'saved endpoint Js differs from persisted fullJ step')
    cmp=compare(ref,data['r'],data['ell'],sref,before,endpoint['step']);actual=receiver['risk']['full_gap']-rec['risk']['full_gap'];error=abs(cmp['sum_band_gain']-actual)/max(abs(actual),abs(cmp['sum_band_gain']),1e-300);require(error<=TOL,'band gain differs from archived actual RGN delta');require(cmp['sum_consistency_abs']<=TOL*max(abs(cmp['actual_RGN_delta']),1e-300),'band gain sum differs from array quadratic gap')
    np.savez(out/f'{directory.name}_k{k}_{policy}_directions.npz',full_eigenvalues=ref['e'],full_eigenvectors=ref['V'],reference_step=sref,receiver_step=before,endpoint_step=endpoint['step'],error_before=ref['V'].T@(before-sref),error_after=ref['V'].T@(endpoint['step']-sref),reference_gradient=gradient)
    result['rows'].append(dict(k=k,policy=policy,endpoint_source_sha256=sha(kd/f'{policy}_endpoint.npz'),policy_record_sha256=sha(kd/f'{policy}.json'),archived_actual_RGN_delta=actual,archived_gain_relative_error=error,receiver_ROM_weak_overlap=overlap,**cmp))
  result['status']='PASS'
 except Exception as exc:result.update(status='MISSING_OR_FAILED_BINDING',missing_reason=str(exc))
 finally:result['cpu_wall_s']=time.perf_counter()-start;result['new_full_tangent_rhs']=0;result['new_full_adjoint_rhs']=0
 return result

def write_tables(results,out):
 for kind in ['bands','directions']:
  rows=[]
  for state in results:
   for r in state['rows']:
    for item in r[kind]:rows.append(dict(state=state['state'],k=r['k'],policy=r['policy'],**item))
  if rows:
   with (out/f'{kind.upper()}.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot(results,out):
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
 for state in results:
  if state['status']!='PASS':continue
  ks=sorted(set(x['k'] for x in state['rows']));fig,axs=plt.subplots(1,len(ks),figsize=(7.08,2.8),sharey=True,squeeze=False)
  colors={'weak':'#8AB2CE','intermediate':'#C9782A','strong':'#246A9B'}
  for ax,k in zip(axs[0],ks):
   rr=[x for x in state['rows'] if x['k']==k];bands=['weak','intermediate','strong'];x=np.arange(len(rr));width=.25
   for bi,b in enumerate(bands):ax.bar(x+(bi-1)*width,[next((g['actual_RGN_gain'] for g in z['bands'] if g['band']==b),0.) for z in rr],width,color=colors[b],label=f'{b} ({state["full_band_dimensions"][b]})')
   ax.axhline(0,color='#666',lw=.6);ax.set_xticks(x,['Surrogate','Teacher','Verified','Adaptive'],rotation=35,ha='right');ax.set_title(f'k = {k}');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.spines[['top','right']].set_visible(False)
  axs[0,0].set_ylabel('Receiver → endpoint RGN gain');axs[0,-1].legend(frameon=False,fontsize=8);fig.suptitle(state['state']+' | fixed full-reference information bands',fontsize=9);fig.tight_layout();fig.savefig(out/f'{state["state"]}_band_gains.svg');fig.savefig(out/f'{state["state"]}_band_gains.pdf');fig.savefig(out/f'{state["state"]}_band_gains.png',dpi=220);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--state-dir',action='append');p.add_argument('--all-completed',action='store_true');p.add_argument('--out',default=str(HERE));a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 paths=[Path(x) for x in a.state_dir] if a.state_dir else ([d for d in (R/'results').glob('object_*') if (d/'status.json').exists()] if a.all_completed else [R/'results/object_2007_early_portable'])
 results=[analyze_state(d,out) for d in paths];payload=dict(states=results,source_hashes={p.name:sha(p) for p in [HERE/'direction_analysis.py',Path(__file__),ROOT/'research/delegated/a17_oldspace_information/newpostprocess/postprocess_oldspaces.py']},threshold=TOL,new_full_actions=0,scope='cached complete full-reference direction decomposition only; no new physics')
 (out/'RESULTS.json').write_text(json.dumps(payload,indent=2));write_tables(results,out);plot(results,out);print(json.dumps([dict(state=r['state'],status=r['status'],rows=len(r['rows']),reason=r['missing_reason']) for r in results]))
