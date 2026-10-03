"""Replot audited immutable saved data only; no physics or scientific verdict.

Production needs an explicit immutable file/SHA manifest and completed requested
collector phases. Nonlinear panels are explicitly mapped by the owner; no scans,
implicit truth acquisition, interpolation, inferred costs, or missing-value fill.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'receiver':'#687781','exchange':'#C9782A','truth':'#246A9B'}
RANKS=(4,8,16);POLICIES=('receiver','exchange')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(value):
 if value in (None,'','null','None'):return None
 x=float(value)
 if not math.isfinite(x):raise ValueError('nonfinite plot input')
 return x
def total(rows,key):
 values=[number(r.get(key)) for r in rows]
 return math.fsum(values) if rows and all(x is not None for x in values) else None
def json_read(path):
 return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))
class Inputs:
 def __init__(self,root,manifest,mock=False):
  self.root=Path(root).resolve();self.manifest=Path(manifest).resolve();self.lock=json_read(self.manifest);self.used={};self.missing=[];self.mock=mock
  if not mock and self.lock.get('immutable_complete') is not True:raise ValueError('owner immutable-complete snapshot receipt required')
  if not mock and self.lock.get('mock') is True:raise ValueError('mock snapshot cannot produce evidence figures')
  self.files=self.lock.get('files',{})
 def path(self,name,optional=False):
  path=(self.root/name).resolve()
  if not path.is_relative_to(self.root):raise ValueError('input escapes selected snapshot')
  key=path.relative_to(self.root).as_posix()
  if not path.is_file():
   if optional:self.missing.append(dict(path=key,reason='UNAVAILABLE'));return None
   raise ValueError('required saved input missing: '+key)
  digest=sha(path)
  if self.files.get(key)!=digest:raise ValueError('input not bound to immutable receipt: '+key)
  self.used[key]=digest;return path
 def json(self,name,optional=False):
  p=self.path(name,optional);return json_read(p) if p else None
 def csv(self,name):
  with self.path(name).open(newline='') as f:return list(csv.DictReader(f))
 def arrays(self,name):
  with np.load(self.path(name),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def scene_labels(scenes):
 rows=scenes.get('scenes',[]) if isinstance(scenes,dict) else scenes;labels={};counts={}
 for s in rows:
  oid=int(s['object_id']);family=str(s.get('family','material')).replace('_',' ').capitalize()
  geometry=f"{len(s['components'])} inclusions" if s.get('components') else 'material scene'
  parts=[family,geometry]
  if s.get('partial'):parts.append('partial view')
  if s.get('strength_multiplier',1)!=1:parts.append('contrast ×'+str(s['strength_multiplier']))
  if s.get('representation') and s['representation']!=s.get('family'):parts.append(str(s['representation']).replace('_',' '))
  label=', '.join(parts);counts[label]=counts.get(label,0)+1
  # Stable display variants disambiguate scene metadata; numeric IDs stay in mapping.
  labels[oid]=label+' · '+chr(64+counts[label])
 return labels
def label(labels,oid):
 if int(oid) not in labels:raise ValueError('scene descriptor unavailable for selected object')
 return labels[int(oid)]
def unavailable(ax,message='Unavailable'):
 ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color=COLORS['receiver']);ax.set_xticks([]);ax.set_yticks([])
def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
def save(fig,out,name,mock):
 if mock:fig.text(.5,.995,'MOCK SYNTHETIC — NOT RESEARCH EVIDENCE',ha='center',va='top',color='#B22222',fontsize=12)
 fig.subplots_adjust(top=.86 if mock else .96);fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=250,bbox_inches='tight');plt.close(fig)
def phase_gate(status,mode,mock):
 if mock:return
 required={'validation':99,'noise':72}[mode]
 if status.get('audited_counts',{}).get(mode)!=required or status.get('phase_status',{}).get(mode)!='COMPLETE_MECHANICAL_OWNER_REVIEW_REQUIRED' or status.get('critical_issues')!=0:raise ValueError('requested phase is not complete audited immutable data: '+mode)
def figure_primary(cells,objects,labels,out,mock):
 rows=[r for r in cells if r['mode']=='validation'];oids=sorted({int(r['object_id']) for r in rows})
 fig=plt.figure(figsize=(11.5,8.5));grid=fig.add_gridspec(2,3,height_ratios=[1.8,1]);axes=[fig.add_subplot(grid[0,j]) for j in range(3)];ratioax=fig.add_subplot(grid[1,:])
 for ax,k in zip(axes,RANKS):
  for pi,p in enumerate(POLICIES):
   values=[total([r for r in rows if int(r['object_id'])==oid and int(r['k'])==k],p+'_full_gap' if p=='receiver' else 'final_full_gap') for oid in oids]
   x=[j+(pi-.5)*.32 for j,v in enumerate(values) if v is not None];y=[v for v in values if v is not None]
   ax.scatter(y,x,color=COLORS[p],marker='o' if p=='receiver' else 's',s=22,label='Receiver backbone' if p=='receiver' else 'Verified exchange')
   for j,v in enumerate(values):
    if v is None:ax.text(.02,j+(pi-.5)*.32,'Unavailable',ha='left',fontsize=5,transform=ax.get_yaxis_transform())
  ax.set_title(f'Absolute GN gaps, rank {k}',loc='left');ax.set_xlabel('Sum over early, middle and late states');ax.set_yticks(range(len(oids)),[label(labels,o).replace(', ','\n',1) for o in oids] if k==4 else ['']*len(oids),fontsize=6)
  if oids:ax.set_ylim(-.5,len(oids)-.5);ax.invert_yaxis()
  if rows and all(number(r.get(n)) is not None and number(r[n])>0 for r in rows for n in ('receiver_full_gap','final_full_gap')):ax.set_xscale('log')
  if not oids:unavailable(ax)
 axes[0].legend(fontsize=7)
 for j,oid in enumerate(oids):
  matches=[r for r in objects if r['mode']=='validation' and int(r['object_id'])==oid and int(r['noise_basis_points'])==0]
  v=number(matches[0].get('ratio')) if len(matches)==1 else None
  if v is None:ratioax.text(j,1,'Unavailable',rotation=90,ha='center',fontsize=6)
  else:ratioax.plot(j,v,'o',color=COLORS['exchange'])
 ratioax.axhline(1,color=COLORS['receiver'],lw=.7);ratioax.set_ylabel('Object-summed gap ratio\nexchange / receiver');ratioax.set_xticks(range(len(oids)),[label(labels,o) for o in oids],rotation=25,ha='right',fontsize=7);ratioax.set_title('Sum paired absolute gaps across ranks and states before dividing',loc='left')
 if not oids:unavailable(ratioax)
 fig.tight_layout();save(fig,out,'Figure1_additional_objects',mock)
def figure_noise(cells,objects,labels,out,mock):
 rows=[r for r in cells if r['mode']=='noise'];oids=sorted({int(r['object_id']) for r in rows});fig,axes=plt.subplots(max(1,len(oids)),3,figsize=(11.5,max(3,2.4*len(oids))),squeeze=False)
 for i,oid in enumerate(oids):
  a=axes[i];rr=[r for r in rows if int(r['object_id'])==oid]
  for level,marker in ((100,'o'),(300,'s')):
   for p in POLICIES:
    values=[total([r for r in rr if int(r['noise_basis_points'])==level and int(r['k'])==k], 'receiver_full_gap' if p=='receiver' else 'final_full_gap') for k in RANKS]
    points=[(k,v) for k,v in zip(RANKS,values) if v is not None]
    if points:a[0].plot(*zip(*points),marker=marker,color=COLORS[p],ls='-' if level==100 else '--',label=f"{'Receiver' if p=='receiver' else 'Exchange'}, {level/100:g}%")
    if any(v is None for v in values):a[0].text(.02,.02,'Missing absolute gaps: unavailable',transform=a[0].transAxes,fontsize=6)
   matches=[r for r in objects if r['mode']=='noise' and int(r['object_id'])==oid and int(r['noise_basis_points'])==level];v=number(matches[0].get('ratio')) if len(matches)==1 else None
   if v is not None:a[1].plot(level/100,v,marker,color=COLORS['exchange'])
   else:a[1].text(level/100,1,'Unavailable',fontsize=6,rotation=90)
   for r in rr:
    if int(r['noise_basis_points'])!=level:continue
    moves=number(r.get('accepted_moves'))
    if moves is not None:a[2].scatter(int(r['k'])+(int(r['realization_index'])-1)*.35,moves,marker=marker,facecolors='none' if level==300 else COLORS['exchange'],edgecolors=COLORS['exchange'],s=22)
  a[0].set_title(label(labels,oid),loc='left');a[0].set_ylabel('Absolute GN gap\nsum over three realizations');a[0].set_xticks(RANKS);a[0].set_xlabel('Actual current rank');a[1].axhline(1,color=COLORS['receiver'],lw=.7);a[1].set_xticks([1,3],['1%','3%']);a[1].set_xlabel('Measurement noise');a[1].set_ylabel('Object-summed gap ratio');a[2].set_xticks(RANKS);a[2].set_xlabel('Rank; offset = realization');a[2].set_ylabel('Accepted exchanges');a[2].set_yticks([0,1,2,3])
  if i==0:a[0].legend(fontsize=6)
 if not oids:
  for ax in axes.flat:unavailable(ax)
 fig.tight_layout();save(fig,out,'Figure2_noise_objects',mock)
def slice_grid(points,values):
 points=np.asarray(points);values=np.asarray(values)
 if points.ndim!=2 or points.shape[1]!=3 or values.shape!=(len(points),) or not np.isfinite(points).all() or not np.isfinite(values).all():raise ValueError('invalid saved material/points shape or finite values')
 zs=np.unique(points[:,2]);z=zs[np.argmin(abs(zs-(zs[0]+zs[-1])/2))];mask=points[:,2]==z;p=points[mask];v=values[mask];xs=np.unique(p[:,0]);ys=np.unique(p[:,1])
 if len(xs)*len(ys)!=len(p):raise ValueError('saved central slice is not rectangular; no interpolation')
 grid=np.empty((len(ys),len(xs)));seen=set()
 for pos,val in zip(p,v):
  ij=(int(np.searchsorted(ys,pos[1])),int(np.searchsorted(xs,pos[0])))
  if ij in seen:raise ValueError('duplicate saved slice point')
  seen.add(ij);grid[ij]=val
 return xs,ys,z,grid
def figure_nonlinear(inputs,mapping,costs,labels,out,mock):
 panels=mapping.get('panels',[]) if mapping else [];fig=plt.figure(figsize=(12,max(3,3.6*len(panels))));g=fig.add_gridspec(max(2,2*len(panels)),5,width_ratios=[1,1,1,1.6,1.4])
 if not panels:
  ax=fig.add_subplot(g[:,:]);unavailable(ax,'Constrained nonlinear endpoint artifacts unavailable');save(fig,out,'Figure3_nonlinear_material',mock);return
 for i,panel in enumerate(panels):
  common=inputs.arrays(panel['common_data']);points=common.get('points');arrays={'truth':common.get('truth')};results={}
  for p in POLICIES:
   arrays[p]=inputs.arrays(panel[p]['final'])['chi'];results[p]=inputs.json(panel[p]['result'])
  for component,row in (('real',2*i),('imag',2*i+1)):
   available=[getattr(v,component) for v in arrays.values() if v is not None];vmin=min(np.min(v) for v in available);vmax=max(np.max(v) for v in available)
   line=fig.add_subplot(g[row,3])
   for j,p in enumerate(('truth','receiver','exchange')):
    ax=fig.add_subplot(g[row,j]);v=arrays[p]
    if v is None:unavailable(ax,'Truth unavailable');continue
    x,y,z,image=slice_grid(points,getattr(v,component));im=ax.imshow(image,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],vmin=vmin,vmax=vmax,cmap='viridis',aspect='equal');fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax.set_title(('Truth' if p=='truth' else 'Receiver backbone' if p=='receiver' else 'Verified exchange')+f' · {component}',fontsize=7);ax.set_xlabel(f'x (z={z:g})');ax.set_ylabel('y' if j==0 else '')
    iy=int(np.argmin(abs(y-(y[0]+y[-1])/2)));line.plot(x,image[iy],color=COLORS[p],label=p.capitalize());line.set_xlabel(f'x (y={y[iy]:g}, z={z:g})')
   line.set_ylabel(f'{component.capitalize()} material contrast');line.set_title(label(labels,panel['object_id']) if component=='real' else 'Saved central line profile',fontsize=7,loc='left');line.legend(fontsize=6)
  costax=fig.add_subplot(g[2*i:2*i+2,4]);shown=False
  for j,p in enumerate(POLICIES):
   attempt=panel[p].get('attempt');matches=[r for r in costs if r.get('attempt')==attempt];occupied=number(matches[0].get('occupation_s')) if len(matches)==1 else None;child=number(results[p].get('wall_total_s'))
   for offset,value,hatch in ((-.16,occupied,''),(.16,child,'//')):
    if value is None:costax.text(j+offset,0,'Unavailable',rotation=90,fontsize=6,ha='center')
    else:costax.bar(j+offset,value,width=.28,color=COLORS[p],hatch=hatch);shown=True
  costax.set_xticks([0,1],['Receiver','Exchange']);costax.set_ylabel('Wall time (s)');costax.set_title('All-in monitored / child wall\nsolid / hatched; overlapping scopes',fontsize=7)
  if not shown:unavailable(costax,'All-in costs unavailable')
 fig.tight_layout();save(fig,out,'Figure3_nonlinear_material',mock)
def run(root,manifest,collection,scenes,out,nonlinear=None,costs=None,mock=False,figures=('primary','noise','nonlinear')):
 inputs=Inputs(root,manifest,mock);out=Path(out).resolve()
 if out.exists() or out.is_relative_to(inputs.root):raise ValueError('fresh derivative output outside immutable snapshot required')
 if mock and 'mock' not in str(out).lower():raise ValueError('mock output path must explicitly include mock')
 prefix=collection.rstrip('/');status=inputs.json(prefix+'/STATUS.json');cells=inputs.csv(prefix+'/cells.csv');objects=inputs.csv(prefix+'/object_summed_risks.csv');labels=scene_labels(inputs.json(scenes));style()
 for mode,fig in (('validation','primary'),('noise','noise')):
  if fig in figures:phase_gate(status,mode,mock)
 mapping=inputs.json(nonlinear,optional=True) if nonlinear else None;fees=inputs.csv(costs or prefix+'/attempt_costs.csv') if 'nonlinear' in figures else []
 if mapping and not mock and mapping.get('complete_audited') is not True:raise ValueError('owner completed audited nonlinear mapping required')
 out.mkdir(parents=True)
 if 'primary' in figures:figure_primary(cells,objects,labels,out,mock)
 if 'noise' in figures:figure_noise(cells,objects,labels,out,mock)
 if 'nonlinear' in figures:figure_nonlinear(inputs,mapping,fees,labels,out,mock)
 unavailable_metrics=[dict(mode=r.get('mode'),object_id=r.get('object_id'),k=r.get('k'),metric=key) for r in cells for key in ('receiver_full_gap','final_full_gap','accepted_moves') if number(r.get(key)) is None]
 config=dict(schema='a17.closure.replot.v1',root=str(inputs.root),manifest=str(inputs.manifest),collection=collection,scenes=scenes,nonlinear=nonlinear,costs=costs,figures=list(figures),mock=mock,source_sha256=sha(__file__),immutable_manifest_sha256=sha(inputs.manifest),used_input_sha256=inputs.used,missing=inputs.missing,unavailable_metrics=unavailable_metrics,palette=COLORS,nonlinear_slice='nearest saved z to midrange; nearest saved y profile; no interpolation',cost_scope='whole monitored occupied wall and child wall shown separately, never added',scientific_verdict=None)
 (out/'REPLOT_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');(out/'SCENE_SOURCE_MAPPING.json').write_text(json.dumps(labels,indent=2)+'\n');(out/'OUTPUT_HASHES.json').write_text(json.dumps({p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},indent=2)+'\n');return config
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--collection',required=True);p.add_argument('--scenes',required=True);p.add_argument('--out',required=True);p.add_argument('--nonlinear');p.add_argument('--costs');p.add_argument('--mock',action='store_true');p.add_argument('--figures',nargs='+',choices=['primary','noise','nonlinear'],default=['primary','noise','nonlinear']);a=p.parse_args();run(a.root,a.manifest,a.collection,a.scenes,a.out,a.nonlinear,a.costs,a.mock,a.figures)
if __name__=='__main__':main()
