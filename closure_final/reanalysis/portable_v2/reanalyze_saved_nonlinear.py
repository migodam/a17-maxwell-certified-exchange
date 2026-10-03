"""NEW_PUBLIC_REANALYSIS: saved-array metrics and plots only; no kernel, optimizer, ledger or transport."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
import public_support as S
from materialize_evidence import relative,ordinary,digest
OBJECTS=(2002,2005,2009,2016);POLICIES=('receiver_only','verified_exchange')
STATUS={'COMPLETED','FAILED_PREFIX','NOT_RUN'}
SOURCE_REFERENCE={'collect_saved_nonlinear_v2.py': 'f09fc95f40c7fb557ca43165d8a289efeaba3cd85a8c81f5f752b27a04f5aec4', 'build_figures.py': '0dcab78ada5aab167ccd8b67440232dfe1367161da28b7b31b0c017cdaf447b8'}

def load_arrays(path):
 with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
def finite(x):return bool(np.isfinite(x).all())
def coefficients(common,chi):
    dc=chi-common['init'];v=float(common['material_tangent_volume']);q=common.get('material_tangent_Q')
    z=dc*np.sqrt(v) if q is None else v*(q.T@dc)
    return np.concatenate([z.real,z.imag])
def objective(common,chi,field):
    reg=coefficients(common,chi)
    return float(.5*np.linalg.norm((field-common['data0'])/float(common['scale']))**2+5e-6*(reg@reg))
def metrics(common,state):
 values={};missing={};chi=state.get('chi');truth=common.get('truth')
 if chi is None:missing['material_metrics']='MISSING_SAVED_CHI'
 else:
  S.check(chi.dtype==np.complex128 and chi.ndim==1 and finite(chi),'saved chi FP64/finite/vector required')
  if truth is not None:S.check(truth.shape==chi.shape and finite(truth),'truth shape/finite')
 for component in ['complex','real','imag']:
  key='material_'+component+'_relative';values[key]=None
  if chi is None:missing[key]='MISSING_SAVED_CHI';continue
  if truth is None:missing[key]='MISSING_TRUTH_ARRAY';continue
  a=chi if component=='complex' else getattr(chi,component);b=truth if component=='complex' else getattr(truth,component);den=float(np.linalg.norm(b))
  if den==0:missing[key]='ZERO_TRUTH_COMPONENT_NORM'
  else:values[key]=float(np.linalg.norm(a-b)/den)
 values['measured_data_relative']=None;values['final_objective']=None;field=state.get('field');keys=['data0','scale']
 if field is None:missing['measured_data_relative']='MISSING_SAVED_FIELD'
 elif any(k not in common for k in keys):missing['measured_data_relative']='MISSING_MEASURED_DATA_OR_SCALE'
 else:
  S.check(field.dtype==np.complex128 and finite(field) and field.shape==common['data0'].shape and finite(common['data0']),'saved measured field shape/FP64/finite');scale=float(common['scale']);S.check(math.isfinite(scale) and scale>0,'positive data scale required');values['measured_data_relative']=float(np.linalg.norm(field-common['data0'])/scale)
 requirements=['init','material_tangent_volume'];missing_objective=[k for k in requirements if k not in common]
 if chi is None or values['measured_data_relative'] is None or missing_objective:missing['final_objective']='MISSING_SAVED_OBJECTIVE_PREREQUISITES'
 else:
  S.check(common['init'].shape==chi.shape and finite(common['init']),'initial material shape/finite');v=float(common['material_tangent_volume']);S.check(math.isfinite(v) and v>0,'positive material volume');q=common.get('material_tangent_Q')
  if q is not None:S.check(q.ndim==2 and q.shape[0]==len(chi) and np.isrealobj(q) and finite(q),'material tangent Q shape/real/finite')
  values['final_objective']=objective(common,chi,field)
 held=state.get('held');actual=common.get('held_truth');values['heldout_data_relative']=None
 if held is None or actual is None:missing['heldout_data_relative']='MISSING_SAVED_HELD_FIELD_OR_HELD_TRUTH'
 else:
  S.check(held.dtype==np.complex128 and finite(held) and held.shape==actual.shape and finite(actual),'held field shape/FP64/finite');den=float(np.linalg.norm(actual))
  if den==0:missing['heldout_data_relative']='ZERO_HELD_TRUTH_NORM'
  else:values['heldout_data_relative']=float(np.linalg.norm(held-actual)/den)
 return values,missing

def grid(points,values,expected_shape):
 S.check(points.ndim==2 and points.shape[1]==3 and finite(points) and values.shape==(len(points),),'saved grid data shape');axes=[np.unique(points[:,j]) for j in range(3)];S.check([len(a) for a in axes]==expected_shape and int(np.prod(expected_shape))==len(points),'saved grid shape differs');idx=tuple(np.searchsorted(a,points[:,j]) for j,a in enumerate(axes));S.check(len(set(zip(*idx)))==len(points),'duplicate grid coordinates');out=np.empty(expected_shape,dtype=values.dtype);out[idx]=values;S.check(np.array_equal(out[idx],values),'grid mapping drift');return axes,out
def nearest0(axis):return int(np.flatnonzero(np.isclose(np.abs(axis),np.abs(axis).min(),rtol=0,atol=1e-14))[0])
def curve_rows(doc,expected_updates):
 rows=doc.get('objective_curve',doc) if isinstance(doc,dict) else doc;S.check(isinstance(rows,list),'saved objective curve list required');S.check(len(rows)==expected_updates,'saved update count differs from declared case scope');previous=None
 for i,row in enumerate(rows):
  S.check(row['iteration']==i and all(isinstance(row[k],(float,int)) and not isinstance(row[k],bool) and math.isfinite(row[k]) and row[k]>=0 for k in ['objective_before','objective_after']),'saved objective curve invalid')
  if previous is not None:S.check(math.isclose(row['objective_before'],previous,rel_tol=2e-12,abs_tol=2e-13),'objective trajectory continuity')
  previous=row['objective_after']
 return rows

def analyze(package_root,manifest_path,manifest_sha,require_full=False,synthetic=False):
 root=Path(package_root).resolve();mp=ordinary(manifest_path);S.check(S.sha(mp)==digest(manifest_sha),'curated manifest SHA differs');manifest=S.read(mp);S.check(manifest.get('schema')=='a17.public.saved.nonlinear.inputs.v1' and manifest.get('synthetic_fixture',False)==synthetic,'wrong/synthetic nonlinear authority');S.check(manifest.get('objective_regularization_lambda2')==1e-5,'frozen objective lambda2 differs');S.check(manifest.get('counts')=={'planned':8,'completed':6,'failed_prefix':1,'not_run':1},'all-eight outcome denominator differs');shape=manifest.get('grid_shape');S.check(isinstance(shape,list) and len(shape)==3 and all(isinstance(n,int) and n>0 for n in shape) and (synthetic or shape==[12,12,12]),'grid shape authority differs')
 cases=manifest['cases'];S.check(len(cases)==8 and {(c['object_id'],c['policy']) for c in cases}=={(o,p) for o in OBJECTS for p in POLICIES},'planned jobs differ');used={};records=[];plot_data=[];missing=[];shared_points=None;reference=manifest.get('source_reference_sha256',{});S.check(reference==SOURCE_REFERENCE,'source reference SHA identities required')
 def path(name):
  rel=relative(name);p=ordinary(root/rel);S.check(p.resolve().is_relative_to(root),'package input escape');S.check(name in manifest['files'] and S.sha(p)==digest(manifest['files'][name]),'public input hash drift');used[name]=manifest['files'][name];return p
 for c in cases:
  oid,policy=c['object_id'],c['policy'];status=c['status'];expected='FAILED_PREFIX' if oid==2016 and policy=='verified_exchange' else 'NOT_RUN' if oid==2016 else 'COMPLETED';S.check(status==expected and status in STATUS,'known case outcome altered');updates=18 if status=='COMPLETED' else 17 if status=='FAILED_PREFIX' else 0;S.check(c.get('saved_updates')==updates,'known completed/failed/not-run update scope differs');record={'object_id':oid,'policy':policy,'status':status,'saved_updates':updates,'endpoint_scope':'COMPLETE_ENDPOINT' if status=='COMPLETED' else 'LAST_SAFE_FAILED_PREFIX_NOT_ENDPOINT' if status=='FAILED_PREFIX' else 'NOT_RUN','metrics':None,'missing_prerequisites':{},'objective_curve':[]}
  if status=='NOT_RUN':S.check(not c.get('state_file') and not c.get('curve_file'),'not-run case cannot have runtime arrays/curves');records.append(record);continue
  if c.get('curve_file'):record['objective_curve']=curve_rows(S.read(path(c['curve_file'])),updates)
  else:record['missing_prerequisites']['objective_curve']='MISSING_SAVED_OBJECTIVE_CURVE'
  common=load_arrays(path(c['common_file'])) if c.get('common_file') else {};state=load_arrays(path(c['state_file'])) if c.get('state_file') else {};values,gaps=metrics(common,state);record['metrics']=values;record['missing_prerequisites'].update(gaps)
  compared={}
  for name,expected_metric in c.get('audited_metrics',{}).items():
   S.check(name in values,'unknown audited metric field')
   if values[name] is None:compared[name]='UNAVAILABLE_SAVED_PREREQUISITE'
   else:S.check(isinstance(expected_metric,(int,float)) and not isinstance(expected_metric,bool) and math.isfinite(expected_metric) and math.isclose(values[name],expected_metric,rel_tol=2e-12,abs_tol=2e-13),'saved metric target differs: '+name);compared[name]='PASS_SAVED_ARRAY_VALUE'
  record['audited_metric_comparison']=compared
  if gaps:missing.append({'object_id':oid,'policy':policy,'scope':record['endpoint_scope'],'missing':gaps})
  if 'points' in common and 'chi' in state:
   if shared_points is not None:S.check(np.array_equal(shared_points,common['points']),'shared saved grid coordinates differ')
   shared_points=common['points'].copy()
   axes,chi=grid(common['points'],state['chi'],shape);truth=None
   if 'truth' in common:axes,t=grid(common['points'],common['truth'],shape);truth=t
   plot_data.append({'object_id':oid,'policy':policy,'status':status,'axes':axes,'chi':chi,'truth':truth})
  else:record['missing_prerequisites']['slice_plot']='MISSING_POINTS_OR_CHI_ARRAY'
  records.append(record)
 if require_full:
  curve_gaps=[{'object_id':c['object_id'],'policy':c['policy'],'missing':'MISSING_SAVED_OBJECTIVE_CURVE'} for c in records if c['status']!='NOT_RUN' and not c['objective_curve']]
  S.check(not missing and not curve_gaps,'MISSING_FULL_SAVED_METRICS_PREREQUISITES: '+json.dumps(missing+curve_gaps,sort_keys=True))
 return {'schema':'a17.public.saved.nonlinear.reanalysis.v1','identity':'NEW_PUBLIC_REANALYSIS_NOT_HISTORICAL_RUNTIME','manifest_sha256':manifest_sha,'source_reference_sha256':reference,'port_source_sha256':S.sha(__file__),'used_input_sha256':used,'counts':manifest['counts'],'cases':records,'missing_metric_prerequisites':missing,'no_new_physics':True,'private_monitor_reaudited':False,'scope':'Metrics use saved chi/field/truth only; no forward/held operator is applied; failed last-safe metrics are not terminal recovery.'},plot_data

def draw(report,data,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for oid,ax in zip(OBJECTS,axes.flat):
  for c in report['cases']:
   if c['object_id']!=oid:continue
   curve=c['objective_curve'];label=c['policy']+(' (failed prefix)' if c['status']=='FAILED_PREFIX' else '')
   if curve:ax.plot([0]+[r['iteration']+1 for r in curve],[curve[0]['objective_before']]+[r['objective_after'] for r in curve],marker='o',markersize=2,label=label)
   else:ax.text(.02,.1 if c['policy']=='receiver_only' else .2,c['policy']+': '+c['status']+' / curve unavailable',transform=ax.transAxes,fontsize=8)
  ax.set_title('Object '+str(oid));ax.set_xlabel('Completed saved updates');ax.set_ylabel('Saved objective');ax.legend(fontsize=7)
 fig.suptitle('8 planned: 6 complete / 1 failed prefix / 1 not run');fig.savefig(out/'saved_objective_curves.png',dpi=160);fig.savefig(out/'saved_objective_curves.svg');plt.close(fig)
 if data:
  fig,axes=plt.subplots(len(data),2,figsize=(9,2.7*len(data)),squeeze=False,layout='constrained')
  for d,axrow in zip(data,axes):
   x,y,z=d['axes'];iz=nearest0(z);iy=nearest0(y);im=axrow[0].imshow(d['chi'].real[:,:,iz].T,origin='lower',extent=[x.min(),x.max(),y.min(),y.max()],aspect='equal');fig.colorbar(im,ax=axrow[0]);axrow[0].set_title(f"{d['object_id']} {d['policy']} {d['status']} / z={z[iz]:g}")
   axrow[1].plot(x,d['chi'].real[:,iy,iz],label='Saved real chi');axrow[1].plot(x,d['chi'].imag[:,iy,iz],label='Saved imaginary chi')
   if d['truth'] is not None:axrow[1].plot(x,d['truth'].real[:,iy,iz],label='Truth real',linestyle='--')
   axrow[1].set_title(f'Saved profile y={y[iy]:g}; no interpolation');axrow[1].legend(fontsize=7)
  fig.savefig(out/'supported_saved_slices.png',dpi=160);fig.savefig(out/'supported_saved_slices.svg');plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--package-root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--require-full-metrics',action='store_true');p.add_argument('--write-derived',action='store_true');p.add_argument('--plots',action='store_true');p.add_argument('--out');a=p.parse_args();report,data=analyze(a.package_root,a.manifest,a.manifest_sha256,a.require_full_metrics)
 if a.write_derived:
  S.check(a.out,'new output required');out=Path(a.out).absolute();S.check(not out.exists() and not out.is_symlink() and out.parent.is_dir() and not any(q.is_symlink() for q in out.parents),'safe new output required');S.check(not out.resolve().is_relative_to(Path(a.package_root).resolve()),'outputs outside immutable package root required');out.mkdir();(out/'SAVED_NONLINEAR_REANALYSIS.json').write_bytes(S.canonical(report))
  fields=['object_id','policy','status','saved_updates','endpoint_scope','material_complex_relative','material_real_relative','material_imag_relative','measured_data_relative','heldout_data_relative','final_objective','missing_prerequisites']
  with (out/'SAVED_ARRAY_METRICS.csv').open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for row in report['cases']:writer.writerow({k:json.dumps(row[k],sort_keys=True) if k=='missing_prerequisites' else row.get(k,(row.get('metrics') or {}).get(k)) for k in fields})
  if a.plots:draw(report,data,out)
 print(json.dumps({'status':'PUBLIC_SAVED_ARRAY_REANALYSIS','counts':report['counts'],'missing_metric_cases':len(report['missing_metric_prerequisites']),'written':a.write_derived,'no_physics':True}))
if __name__=='__main__':main()
