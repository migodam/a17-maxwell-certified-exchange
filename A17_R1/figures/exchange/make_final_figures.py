"""Persisted-data publication plotting only. No new physical computation."""
from pathlib import Path
import csv,json,hashlib
csv.field_size_limit(32*1024*1024)
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];R=ROOT/'Gaussian/A17/EXCHANGE_R1';A=ROOT/'research/delegated/a17_pilot_analysis/analysis_output_final';rank=ROOT/'research/delegated/a17_information_rankings/final_results';direction=ROOT/'research/delegated/a17_gn_direction_analysis/final_results'
BLUE='#246A9B';ORANGE='#C9782A';GRAY='#687781';DARK='#213441'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':DARK,'text.color':DARK})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def loadcsv(p):return list(csv.DictReader(Path(p).open()))
def save(fig,name):
 for ext in ['svg','pdf','png']:fig.savefig(HERE/(name+'.'+ext),dpi=300,facecolor='white')
 plt.close(fig)
source_paths=[A/'object_summed_ratios.csv',A/'per_policy_main.csv',rank/'LINKED_MOVES.json',direction/'RESULTS.json'];data=[];provenance=dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in source_paths})
# Figure1: no state/candidate independent uncertainty; actual rank shared across illumination.
policies=['receiver_no_swap','exact_score_teacher','directed_verified'];labels=['Receiver backbone','Exact local teacher','Directed verified'];colors=[GRAY,BLUE,ORANGE];styles=['--','-','-'];objects=[2001,2007,2012,2014];names={2001:'Smooth Gaussian',2007:'Near-contact Gaussian',2012:'Layered voxel',2014:'Asymmetric voxel'}
rows=loadcsv(A/'per_policy_main.csv');ratios=loadcsv(A/'object_summed_ratios.csv');ratio={(int(r['object_id']),r['policy_id']):float(r['ratio']) for r in ratios if not r['k']}
fig,axs=plt.subplots(2,2,figsize=(180/25.4,145/25.4),sharex=True)
for panel,(oid,ax) in enumerate(zip(objects,axs.flat)):
 for policy,label,color,ls in zip(policies,labels,colors,styles):
  y=[]
  for k in [4,8,16]:
   rr=[r for r in rows if int(r['object_id'])==oid and r['policy_id']==policy and int(r['k'])==k];assert len(rr)==3 and {r['phase'] for r in rr}=={'early','middle','late'};assert all(int(r['actual_rank'])==k for r in rr)
   value=sum(float(r['full_gap']) for r in rr);assert value>0;y.append(value);data.append(dict(figure='Figure1',panel=names[oid],object_id=oid,policy=policy,rank=k,metric='object_summed_full_GN_gap_three_states',value=value,units='full frozen quadratic objective',source='per_policy_main.csv'))
  ax.plot([4,8,16],y,marker='o',ms=4,lw=1.3,ls=ls,color=color,label=label)
 ax.set_yscale('log');ax.set_xticks([4,8,16]);ax.set_title(chr(97+panel)+'. '+names[oid],loc='left');ax.grid(axis='y',which='major',color='#E0E5E8',lw=.5)
 ax.text(.97,.95,'Total gap / receiver\nTeacher  '+f'{ratio[(oid,"exact_score_teacher")]:.4f}'+'\nVerified  '+f'{ratio[(oid,"directed_verified")]:.4f}',transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
 for policy in policies[1:]:
  actual=sum(float(r['full_gap']) for r in rows if int(r['object_id'])==oid and r['policy_id']==policy)/sum(float(r['full_gap']) for r in rows if int(r['object_id'])==oid and r['policy_id']=='receiver_no_swap');assert abs(actual-ratio[(oid,policy)])<1e-12
  data.append(dict(figure='Figure1',panel=names[oid],object_id=oid,policy=policy,rank='all',metric='total_gap_ratio_to_receiver',value=actual,units='dimensionless ratio',source='object_summed_ratios.csv'))
for ax in axs[1]:ax.set_xlabel('Actual shared current rank')
for ax in axs[:,0]:ax.set_ylabel('Object-summed full GN gap')
handles,legend=axs[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.005),frameon=False);fig.subplots_adjust(left=.105,right=.975,bottom=.15,top=.945,hspace=.33,wspace=.27);save(fig,'Figure1_object_GN_gap')
# Figure2: linked one-exchange failure plus shared-core pair addition, distinct units.
linked=json.loads((rank/'LINKED_MOVES.json').read_text());match=[x for x in linked if x['object_id']==2007 and x['phase']=='early' and x['k']==4 and x['drop']==[0] and x['add']==[4]];assert len(match)==1;w=match[0]
for field,path in [('teacher_json_sha256','teacher_json_path'),('information_json_sha256','information_json_path')]:assert sha(w[path])==w[field]
origteacher=[json.loads(x) for x in Path(w['teacher_json_path']).read_text().splitlines()];originfo=[json.loads(x) for x in Path(w['information_json_path']).read_text().splitlines()];t=next(x for x in origteacher if x['move_index']==w['move_index']);i=next(x for x in originfo if x['move_index']==w['move_index']);assert canon(t)==w['teacher_record_sha256'] and canon(i)==w['information_record_sha256'];assert t['Q_true']==w['Q_true'] and i['selector_information']['candidate']['delta_logdet_volume']==w['delta_logdet_volume']
base=json.loads(Path(w['base_json_path']).read_text());before=base['risk']['full_gap'];after=before-w['Q_true'];pairpath=R/'results/object_2007_late/k16/pair_forced_dimension.json';pairjson=json.loads(pairpath.read_text());pair=pairjson['Maxwell_singletons_nonpositive_pair_positive_witness'];assert pair['base_rank']==14 and pair['pair_rank']==16 and pair['Q_a']<0 and pair['Q_b']<0 and pair['Q_pair']>0
provenance['information_utility_witness']=w;provenance['information_utility_gap']=dict(base_full_gap=before,child_full_gap_inferred_by_exact_saved_gain_identity=after,actual_Q=w['Q_true'],note='child gap derived by Rchild=Rbase-Q; no new physical evaluation');provenance['paired_addition_witness']=dict(path=str(pairpath.relative_to(ROOT)),file_sha256=sha(pairpath),record_sha256=canon(pair),record=pair,scope=pairjson['scope'],state_hash=json.loads((pairpath.parents[1]/'runtime_receipt.json').read_text())['state_hash'],pool_hash=json.loads((pairpath.parents[1]/'runtime_receipt.json').read_text())['pool_hash'],array_path=str((pairpath.parent/'Maxwell_pair_witness.npz').relative_to(ROOT)))
array=pairpath.parent/'Maxwell_pair_witness.npz';assert array.exists();provenance['paired_addition_witness']['array_sha256']=sha(array)
fig,axs=plt.subplots(2,2,figsize=(180/25.4,140/25.4));
a=axs[0,0];a.bar([0],[w['delta_logdet_volume']],width=.45,color=BLUE);a.set_ylim(0,8.3);a.set_xticks([0],['Same current exchange']);a.set_ylabel('Change in log-det volume');a.set_title('a. Information volume',loc='left');a.text(0,6.7,f'+{w["delta_logdet_volume"]:.10f}',ha='center',fontsize=9)
a=axs[0,1];a.bar([0],[w['delta_effective_dim']],width=.45,color=BLUE);a.set_ylim(0,2.5);a.set_xticks([0],['Same current exchange']);a.set_ylabel('Change in effective dimension');a.set_title('b. Effective information',loc='left');a.text(0,2.,f'+{w["delta_effective_dim"]:.10f}',ha='center',fontsize=9)
a=axs[1,0];a.bar([0],[w['Q_true']],width=.45,color=ORANGE);a.axhline(0,color=GRAY,lw=.7);a.set_ylim(-7.2e-5,4.4e-5);a.set_xticks([0],['Same current exchange']);a.set_ylabel('Actual signed full-quadratic gain Q');a.set_title('c. Positive information, negative utility',loc='left');a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.text(.5,.97,f'Q = {w["Q_true"]:.10e}\nGap before = {before:.10e}\nGap after = {after:.10e}',transform=a.transAxes,ha='center',va='top',fontsize=8)
a=axs[1,1];values=[pair['Q_a'],pair['Q_b'],pair['Q_pair']];a.bar(range(3),values,width=.6,color=[GRAY,GRAY,BLUE]);a.axhline(0,color=GRAY,lw=.7);a.set_xticks(range(3),['Single A\nrank 15','Single B\nrank 15','Pair A+B\nrank 16']);a.set_ylim(-1.2e-7,4.8e-7);a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.set_ylabel('Gain from the same rank-14 core');a.set_title('d. Coupled additions',loc='left');
for x,v in enumerate(values):a.text(x,v+(1.5e-8 if v>=0 else -1.8e-8),f'{v:+.4e}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
fig.text(.5,.025,'Separate scales; fixed prior 10⁻⁵. Paired addition is not a one-swap-trap claim.',ha='center',fontsize=8);fig.subplots_adjust(left=.11,right=.975,bottom=.16,top=.945,hspace=.45,wspace=.35);save(fig,'Figure2_information_and_pair_witnesses')
for metric in ['delta_logdet_volume','delta_effective_dim','Q_true']:data.append(dict(figure='Figure2',panel='near-contact Gaussian fixed exchange',object_id=2007,policy='fixed_move_drop0_add4',rank=4,metric=metric,value=w[metric],units='dimensionless prior-normalized information' if metric!='Q_true' else 'full frozen quadratic objective',source='hash-linked LINKED_MOVES.json'))
for metric,value in [('base_full_gap',before),('child_full_gap_from_saved_Q',after)]:data.append(dict(figure='Figure2',panel='near-contact Gaussian fixed exchange',object_id=2007,policy='fixed_move_drop0_add4',rank=4,metric=metric,value=value,units='full frozen quadratic objective',source='base_json_path and exact saved Q identity'))
for metric,value,rankvalue in [('Q_A',pair['Q_a'],15),('Q_B',pair['Q_b'],15),('Q_pair',pair['Q_pair'],16)]:data.append(dict(figure='Figure2',panel='near-contact Gaussian paired addition',object_id=2007,policy='common_rank14_core_addition',rank=rankvalue,metric=metric,value=value,units='full frozen quadratic objective',source='pair_forced_dimension.json'))
with (HERE/'figure_data.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
full=json.loads((direction/'RESULTS.json').read_text());dims=[dict(state=x['state'],object_id=x['object_id'],phase=x['phase'],**x['full_band_dimensions'],prior_precision=x['prior_precision'],lambda_total=x['lambda_total'],scope='complete full-reference Gaussian information eigenbasis; voxel absent') for x in full['states'] if x['status']=='PASS'];assert len(dims)==6
(HERE/'FULL_REFERENCE_GAUSSIAN_BAND_DIMENSIONS.json').write_text(json.dumps(dims,indent=2));
with (HERE/'FULL_REFERENCE_GAUSSIAN_BAND_DIMENSIONS.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(dims[0]));writer.writeheader();writer.writerows(dims)
provenance['plot_source_sha256']=sha(Path(__file__));(HERE/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));(HERE/'HASHES.json').write_text(json.dumps({p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='HASHES.json'},indent=2));print('PASS: 36 gap points, 8 ratios, two hash-linked witnesses, six Gaussian band summaries')
