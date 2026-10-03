"""Plot collector-bound saved CSVs only; no scientific calculations or runtime access."""
import argparse,csv,hashlib,io,json,math
from pathlib import Path
OBJECTS=(2002,2003,2004,2005,2006,2008,2009,2010,2011,2013,2016)
PHASES=('early','middle','late');KS=(4,8,16)
CAPTURES=('balanced12_capture','anchor_top1_raw_noop_capture','anchor_top1_significance_verified_noop_capture')
RISKS=('actual_online_final_full_gap','local_teacher_final_full_gap','actual_minus_local_teacher_risk')
PATH_TERMINAL={'LOCAL_3ACTION_BUDGET_COMPLETE','NUMERICAL_LOCAL_1SWAP_STOP_NO_DETERMINISTIC_BOUND'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
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
def load_inputs(initial,path,authority,expected_sha,expected_receipt,labels,synthetic=False):
 if sha(authority)!=expected_sha:raise ValueError('authority SHA mismatch')
 a=json.loads(Path(authority).read_text())
 if a.get('schema')!='a17.saved.coverage.evidence.v1':raise ValueError('wrong collector schema')
 if a.get('synthetic_fixture',False)!=synthetic:raise ValueError('synthetic/evidence authority mismatch')
 if a.get('snapshot_receipt_sha256')!=expected_receipt:raise ValueError('snapshot receipt SHA mismatch')
 for k in ('snapshot_receipt_sha256','config_sha256','input_manifest_sha256','collector_sha256'):
  if len(a.get(k,''))!=64 or any(c not in '0123456789abcdef' for c in a[k]):raise ValueError('missing SHA authority '+k)
 for f,k in ((initial,'object_initial_opportunity'),(path,'path_final_risk')):
  if Path(f).read_bytes()!=csv_bytes(a[k]):raise ValueError('CSV differs from collector authority '+k)
 with Path(initial).open(newline='') as f:initial_rows=list(csv.DictReader(f))
 with Path(path).open(newline='') as f:path_rows=list(csv.DictReader(f))
 labels=json.loads(Path(labels).read_text())
 if set(labels)!=set(map(str,OBJECTS)):raise ValueError('geometry/material labels must cover all 11 objects')
 for lab in labels.values():
  if not all(isinstance(lab.get(k),str) and lab[k].strip() for k in ('geometry','material')):raise ValueError('missing English geometry/material label')
 return prepare(initial_rows,path_rows,labels),a

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

def draw(data,out,synthetic=False):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 from matplotlib.colors import ListedColormap,BoundaryNorm
 from matplotlib.patches import Patch
 plt.rcParams.update({'font.size':9,'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 labels=[data['labels'][str(o)]['geometry']+' / '+data['labels'][str(o)]['material'] for o in OBJECTS]
 fig,ax=plt.subplots(figsize=(11,8));colors=('#577590','#43AA8B','#F9C74F');series=('Short pool (12 candidates)','Anchor top-1, raw + no-op','Anchor top-1, significance + no-op')
 by={r['object_id']:r for r in data['initial']}
 for j,k in enumerate(CAPTURES):
  for i,o in enumerate(OBJECTS):
   r=by[o];v=r[k];y=i+(j-1)*.23
   if v is not None:
    ax.barh(y,v,height=.21,color=colors[j],label=series[j] if i==0 else None)
    if j==0:ax.barh(y,1-v,left=v,height=.21,color='#d9d9d9',label='Missed short-pool opportunity' if i==0 else None)
   else:ax.plot(.015,y,marker='x',color=colors[j],ms=4)
 for i,o in enumerate(OBJECTS):
  r=by[o]
  if any(r[k] is None for k in CAPTURES):ax.text(1.025,i,r['reason'],va='center',fontsize=12)
 ax.set_yticks(range(11),labels,fontsize=13);ax.tick_params(axis='x',labelsize=12);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Initial-neighborhood positive-gain capture (weighted by teacher opportunity)',fontsize=13);ax.set_title('Initial short-pool coverage and anchor ranking',fontsize=15);ax.axvline(1,color='#888888',lw=.6);fig.legend(handles=[Patch(color=c,label=l) for c,l in zip((*colors,'#d9d9d9'),(*series,'Missed short-pool opportunity'))],loc='lower center',bbox_to_anchor=(.5,.085),ncol=2,fontsize=11);ax.spines[['top','right']].set_visible(False)
 fig.text(.03,.015,'Missing bars remain null. Missed short-pool opportunity = 1 − capture where defined. Initial receiver neighborhood only.',fontsize=11)
 if synthetic:fig.text(.5,.5,'SYNTHETIC FIXTURE — NOT RESEARCH EVIDENCE',ha='center',rotation=25,alpha=.25,fontsize=22)
 fig.subplots_adjust(left=.36,right=.80,bottom=.25,top=.92)
 paths=[]
 for ext in ('pdf','svg','png'):
  p=out/('initial_opportunity.'+ext);fig.savefig(p,dpi=180);paths.append(str(p))
 plt.close(fig)
 by={(r['object_id'],r['phase'],r['k']):r for r in data['path']};columns=[(p,k) for p in PHASES for k in KS]
 arrays=[np.array([[by[o,p,k][m] if by[o,p,k][m] is not None else np.nan for p,k in columns] for o in OBJECTS]) for m in RISKS]
 finite=np.concatenate([a[np.isfinite(a)] for a in arrays[:2]]);maxrisk=float(finite.max()) if finite.size else 1.;maxrisk=max(maxrisk,1e-12)
 signed=arrays[2][np.isfinite(arrays[2])];limit=max(float(np.abs(signed).max()) if signed.size else 1.,1e-12)
 fig,axs=plt.subplots(1,3,figsize=(12,9));titles=('Actual online\nendpoint risk','Bounded local-path\nendpoint risk','Actual minus bounded\nlocal-path risk')
 for j,(ax,a) in enumerate(zip(axs,arrays)):
  palette=['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594'] if j<2 else ['#2166ac','#67a9cf','#d1e5f0','#f7f7f7','#fddbc7','#ef8a62','#b2182b']
  cm=ListedColormap(palette);cm.set_bad('#eeeeee');bounds=np.linspace(0,maxrisk,9) if j<2 else np.linspace(-limit,limit,8)
  im=ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cm,norm=BoundaryNorm(bounds,cm.N));ax.set_title(titles[j],fontsize=14);ax.set_xticks(range(9),[str(k) for p,k in columns],fontsize=11,rotation=45,ha='right');ax.set_xlabel('Current dimension k',fontsize=13,labelpad=32);ax.set_yticks(range(11),labels if j==0 else ['']*11,fontsize=14)
  for center,phase in zip((1,4,7),PHASES):ax.text(center,-.105,phase,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=12)
  for i,o in enumerate(OBJECTS):
   for t,(p,k) in enumerate(columns):
    if not math.isfinite(a[i,t]):ax.text(t,i,by[o,p,k]['null_code'],ha='center',va='center',fontsize=9,color='#555555')
  for x in (2.5,5.5):ax.axvline(x,color='white',lw=2)
  cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.23);ticks=[0,maxrisk/2,maxrisk] if j<2 else [-limit,0,limit];cb.set_ticks(ticks);cb.set_ticklabels(['0' if v==0 else format(v,'.1e').replace('e-0','e−').replace('e+0','e+') for v in ticks]);cb.ax.get_xticklabels()[0].set_ha('left');cb.ax.get_xticklabels()[-1].set_ha('right');cb.ax.tick_params(labelsize=10);cb.set_label('Frozen full-gap risk' if j<2 else 'Signed risk difference',fontsize=13);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
 fig.suptitle('Bounded local-path terminal comparison (99 planned cells retained)',fontsize=16)
 fig.text(.02,.015,'Grey null cells: P partial, NR not run, I invalid binding, NA unavailable. Partial actual endpoints may appear; partial teacher endpoints/comparisons do not.\nNumerical local stop has no deterministic bound. Three-action paths are local; candidate/ranking/search effects are combined.',fontsize=10)
 if synthetic:fig.text(.5,.5,'SYNTHETIC FIXTURE — NOT RESEARCH EVIDENCE',ha='center',rotation=20,alpha=.25,fontsize=25)
 fig.subplots_adjust(left=.35,right=.97,bottom=.17,top=.88,wspace=.38)
 for ext in ('pdf','svg','png'):
  p=out/('bounded_path_risk.'+ext);fig.savefig(p,dpi=180);paths.append(str(p))
 plt.close(fig);return paths

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('initial-csv','path-csv','authority','authority-sha256','receipt-sha256','labels','out'):p.add_argument('--'+k,required=True)
 a=p.parse_args();data,authority=load_inputs(a.initial_csv,a.path_csv,a.authority,a.authority_sha256,a.receipt_sha256,a.labels)
 paths=draw(data,a.out)
 provenance={'status':'CANDIDATE_OWNER_REVIEW_PENDING','scientific_acceptance':False,'inputs':{k:{'path':str(Path(v).resolve()),'sha256':sha(v)} for k,v in {'initial':a.initial_csv,'path':a.path_csv,'authority':a.authority,'labels':a.labels}.items()},'snapshot_receipt_sha256':authority['snapshot_receipt_sha256'],'config_sha256':authority['config_sha256'],'input_manifest_sha256':authority['input_manifest_sha256'],'collector_sha256':authority['collector_sha256'],'script_sha256':sha(__file__),'outputs':{Path(x).name:sha(x) for x in paths},'planned_objects':11,'planned_cells':99,'scope':'Separate initial receiver-neighborhood captures and bounded local-path endpoints; no global-oracle claim.'}
 (Path(a.out)/'FIGURE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print('Saved two figure families; owner review pending.')
if __name__=='__main__':main()
