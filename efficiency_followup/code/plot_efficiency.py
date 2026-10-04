"""Plot local derived saved measurements; white-background vector PDF and PNG."""
from pathlib import Path
import argparse,json,hashlib,math,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
EFF=Path(__file__).resolve().parents[1]
COLORS={'original_top2_cap3':'#666666','verified_exchange':'#666666','receiver_only':'#222222','top1_cap3':'#0072B2','top2_cap1':'#E69F00','top1_cap1':'#CC79A7','top1_cap2':'#009E73','reuse_previous':'#D55E00','reuse_reset3':'#56B4E9'}
PHASECOLORS={'State / linearization':'#56B4E9','Full gradient':'#CC79A7','Policy acquisition':'#E69F00','Online exchange':'#009E73','Projection / fallback':'#F0E442','Full Armijo':'#0072B2','Other charged child wall':'#CCCCCC'}
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
def finite(x):return isinstance(x,(int,float))and math.isfinite(x)
def color(policy):return COLORS.get(policy,'#9467BD')
def read(root,name):
 p=root/(name+'.json');return json.loads(p.read_text())if p.exists()else []
def empty(ax,reason):ax.text(.5,.5,reason,ha='center',va='center',transform=ax.transAxes);ax.set_xticks([]);ax.set_yticks([])
def save(fig,out,name,manifest,data_scope):
 fig.tight_layout();paths=[]
 for ext in ('pdf','png'):
  p=out/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight');paths.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 manifest.append(dict(name=name,data_scope=data_scope,artifacts=paths));plt.close(fig)
def plot(args):
 data=args.data.resolve();out=args.out.resolve()if args.out else data/'figures';out.mkdir(parents=True,exist_ok=True);manifest=[];frozen=read(data,'frozen_object_policy');cells=read(data,'frozen_cells');nl=read(data,'nonlinear_runs');cost=read(data,'cost_components');profiles=read(data,'profile_runs');status=read(data,'analysis_receipt').get('status','PARTIAL')
 objects=sorted(set(r['object']for r in frozen if r['scope']=='eff_frozen')|{2002,2009})
 fig,axs=plt.subplots(1,len(objects),figsize=(5*len(objects),4),squeeze=False)
 for ax,obj in zip(axs[0],objects):
  rows=[r for r in frozen if r['scope']=='eff_frozen'and r['object']==obj and finite(r['object_gain_retention'])and r.get('incremental_online_wall_n',r['cells'])>0 and finite(r['incremental_online_wall_sum_s'])]
  if not rows:empty(ax,'NOT_RUN / missing snapshot')
  for r in rows:
   x=r['incremental_online_wall_sum_s']/r.get('incremental_online_wall_n',r['cells']);ax.scatter(x,r['object_gain_retention'],color=color(r['policy']),s=55,label=r['policy']);ax.annotate(r['policy'],(x,r['object_gain_retention']),xytext={'top1_cap1':(6,-14),'top2_cap1':(6,10),'top1_cap2':(6,-14),'top1_cap3':(6,10),'original_top2_cap3':(6,-14)}.get(r['policy'],(4,5)),textcoords='offset points',fontsize=7)
  ax.set(xlabel='Mean per-cell incremental online + engine wall (s)',ylabel='Sum-gain retention vs original policy',title=f'Object {obj} — cached frozen maps');ax.margins(x=.18,y=.18);ax.grid(alpha=.2)
 fig.suptitle(f'Frozen cost–retention measurements · {status}',fontsize=11,y=1.04);save(fig,out,'frozen_pareto',manifest,'eff_frozen only; cached marginal costs, no cold/end-to-end claim')
 # Every repetition is plotted; no best-run choice or interpolation.
 scopes=['eff_nonlinear','closure_saved_nonlinear'];fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 for ax,scope in zip(axs,scopes):
  rows=[r for r in nl if r['scope']==scope and finite(r['child_wall_total_s'])and finite(r['material_complex_relative'])]
  if not rows:empty(ax,'NOT_RUN / no closed local result')
  used=set()
  for r in rows:
   label=r['method']+f" / object {r['object']}";ax.scatter(r['child_wall_total_s'],r['material_complex_relative'],marker='o'if r['object']==2002 else '^',s=45,color=color(r['policy']),label=label if label not in used else None,alpha=.8);used.add(label)
  ax.set(xlabel='Actual total child wall (s)',ylabel='Complex material relative error',title='New matched nonlinear'if scope=='eff_nonlinear'else 'Original saved nonlinear — separate scope');ax.grid(alpha=.2)
  if rows:ax.legend(fontsize=7)
 fig.suptitle(f'Nonlinear wall–error measurements · {status}',fontsize=11,y=1.02);save(fig,out,'nonlinear_wall_error',manifest,'scope-separated actual child walls; all available repetitions')
 # Profile exclusive tree nodes partition work; inclusive fields never summed.
 profile_rows=[r for r in profiles if r['scope']=='eff_profile'];fig,axs=plt.subplots(1,max(1,len(profile_rows)),figsize=(5*max(1,len(profile_rows)),4.5),squeeze=False)
 if not profile_rows:empty(axs[0,0],'NOT_RUN / no exclusive profile')
 for ax,r in zip(axs[0],profile_rows):
  groups=collections.defaultdict(float)
  for c in cost:
   if c['scope']=='eff_profile'and c['run_id']==r['result_sha256']and c['time_kind']=='profile_tree_EXCLUSIVE'and finite(c['wall_s']):groups[c['label'].split('/')[0]]+=c['wall_s']
  if finite(r.get('unattributed_wall_s')):groups['Unattributed']=r['unattributed_wall_s']
  names=list(groups);labels={'pinned_physical_state_load':'Physical state / load','shared_full_gradient':'Shared full gradient','receiver_endpoint_shared_state':'Receiver endpoint','original_public_workspace_acquisition':'Anchor / pool / public maps','exchange_engine_common_maps_setup':'Exchange engine setup','original_frozen_k8_online_exchange':'Online exchange','paid_original_equivalent_J_diagnostic':'Directional equivalence checks','Unattributed':'Unattributed'};palette={'pinned_physical_state_load':'#56B4E9','shared_full_gradient':'#CC79A7','receiver_endpoint_shared_state':'#009E73','original_public_workspace_acquisition':'#E69F00','exchange_engine_common_maps_setup':'#999999','original_frozen_k8_online_exchange':'#0072B2','paid_original_equivalent_J_diagnostic':'#D55E00','Unattributed':'#CCCCCC'};ax.barh([labels.get(n,n)for n in names],[groups[x]for x in names],color=[palette.get(n,'#999999')for n in names]);ax.set(xlabel='Exclusive partition wall (s)',title=f"Object {r['object']} — measured profile");ax.grid(axis='x',alpha=.2)
 save(fig,out,'profile_cost_decomposition',manifest,'new profile exclusive tree only, root+child exclusive sums; diagnostic/setup costs charged')
 # Per-run stacked disjoint outer phases. Difference to total child wall stays explicit.
 for scope in scopes:
  rows=[r for r in nl if r['scope']==scope and finite(r['child_wall_total_s'])];fig,ax=plt.subplots(figsize=(max(8,len(rows)*.8),5))
  if not rows:empty(ax,'NOT_RUN / no closed local nonlinear result')
  totals=[]
  for r in rows:
   groups={k:0. for k in PHASECOLORS}
   for c in cost:
    if c['run_id']!=r['result_sha256']or c['time_kind']!='outer_driver_disjoint_phase':continue
    label=c['label'];target='State / linearization'if label=='full_linearization'else 'Full gradient'if label=='full_gradient'else 'Policy acquisition'if label in ('exchange_anchor_pool_workspace_acquisition','receiver_only_acquisition_and_endpoint')else 'Online exchange'if label=='verified_online_exchange'else 'Projection / fallback'if label=='passive_projection_and_full_gradient_fallback'else 'Full Armijo'if label=='full_nonlinear_armijo'else 'Other charged child wall';groups[target]+=c['wall_s']
   accounted=sum(groups.values());groups['Other charged child wall']+=max(0.,r['child_wall_total_s']-accounted);totals.append(groups)
  bottom=np.zeros(len(rows));positions=np.arange(len(rows))
  for name,col in PHASECOLORS.items():
   heights=np.asarray([r[name]for r in totals]);ax.bar(positions,heights,bottom=bottom,label=name,color=col);bottom+=heights
  if rows:ax.set_xticks(positions,[f"{r['object']}\n{r['method']}\nrep {r['repetition']}"for r in rows],rotation=35,ha='right');ax.legend(fontsize=7,ncol=2)
  ax.set(ylabel='Total child wall decomposed (s)',title=('New nonlinear'if scope=='eff_nonlinear'else 'Original saved nonlinear')+' — disjoint outer phases + remaining child wall');ax.grid(axis='y',alpha=.2);save(fig,out,'nonlinear_cost_'+scope,manifest,scope+' only; nested online stages excluded from outer-phase stack')
 fig,axs=plt.subplots(1,len(objects),figsize=(5*len(objects),4),squeeze=False)
 for ax,obj in zip(axs[0],objects):
  rows=[r for r in frozen if r['scope']=='eff_frozen'and r['object']==obj and finite(r['avg_Jd'])]
  if not rows:empty(ax,'NOT_RUN / missing Jd counts')
  else:ax.bar(np.arange(len(rows)),[r['avg_Jd']for r in rows],color=[color(r['policy'])for r in rows]);ax.set_xticks(np.arange(len(rows)),[r['policy']for r in rows],rotation=35,ha='right')
  ax.set(ylabel='Mean full Jd material directions / frozen cell',title=f'Object {obj} — actual finalist counts');ax.grid(axis='y',alpha=.2)
 save(fig,out,'frozen_Jd_counts',manifest,'eff_frozen only; directions distinct from six-illumination physical RHS')
 (out/'figure_manifest.json').write_text(json.dumps(dict(status=status,style='fixed colors, white background, vector PDF and PNG, no interpolation',figures=manifest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(json.dumps(dict(status=status,figures=len(manifest),out=str(out)),indent=2))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=EFF/'analysis/derived_v1');p.add_argument('--out',type=Path);plot(p.parse_args())
if __name__=='__main__':main()
