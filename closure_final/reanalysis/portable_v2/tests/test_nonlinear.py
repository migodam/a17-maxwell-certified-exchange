import sys,json,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reanalyze_saved_nonlinear as N
import public_support as S
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.package=self.root/'package';self.package.mkdir();self.manifest=self.root/'manifest.json';self.files={};self.cases=[]
  points=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]]);truth=np.ones(8,dtype=np.complex128)*(1+.2j);self.common={'points':points,'truth':truth,'init':np.zeros(8,dtype=np.complex128),'data0':np.ones(2,dtype=np.complex128),'scale':np.array(2.),'material_tangent_volume':np.array(.5),'held_truth':np.ones(2,dtype=np.complex128)};self.state={'chi':truth*.8,'field':self.common['data0']*.9,'held':self.common['held_truth']*.9}
  self.array('closure_final/inputs/common.npz',self.common)
  for oid in N.OBJECTS:
   for policy in N.POLICIES:
    status='COMPLETED' if oid!=2016 else 'NOT_RUN' if policy=='receiver_only' else 'FAILED_PREFIX';n=18 if status=='COMPLETED' else 17 if status=='FAILED_PREFIX' else 0;c={'object_id':oid,'policy':policy,'status':status,'saved_updates':n}
    if n:
     name=f'closure_final/results/{oid}_{policy}';self.array(name+'/state.npz',self.state);curve=[{'iteration':i,'objective_before':float(20-i),'objective_after':float(19-i)} for i in range(n)];self.json(name+'/curve.json',curve);c.update(common_file='closure_final/inputs/common.npz',state_file=name+'/state.npz',curve_file=name+'/curve.json')
    self.cases.append(c)
  self.data={'schema':'a17.public.saved.nonlinear.inputs.v1','synthetic_fixture':True,'objective_regularization_lambda2':1e-5,'grid_shape':[2,2,2],'counts':{'planned':8,'completed':6,'failed_prefix':1,'not_run':1},'files':self.files,'cases':self.cases,'source_reference_sha256':N.SOURCE_REFERENCE};self.save()
 def tearDown(self):self.temp.cleanup()
 def array(self,name,values):
  p=self.package/name;p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,**values);self.files[name]=S.sha(p)
 def json(self,name,value):
  p=self.package/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(S.canonical(value));self.files[name]=S.sha(p)
 def save(self):self.manifest.write_bytes(S.canonical(self.data))
 def runcheck(self,full=False):return N.analyze(self.package,self.manifest,S.sha(self.manifest),full,True)
 def test_saved_metrics_and_all_eight_scope(self):
  report,plots=self.runcheck(True);self.assertEqual(report['counts']['planned'],8);self.assertEqual(len(report['cases']),8);self.assertAlmostEqual(report['cases'][0]['metrics']['material_complex_relative'],.2);self.assertEqual(len(plots),7);self.assertEqual(report['cases'][-1]['endpoint_scope'],'LAST_SAFE_FAILED_PREFIX_NOT_ENDPOINT');self.assertFalse(report['private_monitor_reaudited'])
 def test_audited_metric_comparison(self):
  self.cases[0]['audited_metrics']={'material_complex_relative':.2};self.save();report,_=self.runcheck();self.assertEqual(report['cases'][0]['audited_metric_comparison']['material_complex_relative'],'PASS_SAVED_ARRAY_VALUE')
  self.cases[0]['audited_metrics']={'material_complex_relative':.7};self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_synthetic_saved_plot_render(self):
  report,data=self.runcheck();out=self.root/'plots';out.mkdir();N.draw(report,data,out);self.assertEqual(len(list(out.iterdir())),4);self.assertTrue(all(p.stat().st_size>0 for p in out.iterdir()))
 def test_missing_truth_supported_null_strict_fails(self):
  c=dict(self.common);del c['truth'];self.array('closure_final/inputs/common.npz',c);self.save();report,_=self.runcheck();self.assertIsNone(report['cases'][0]['metrics']['material_complex_relative']);self.assertEqual(report['cases'][0]['missing_prerequisites']['material_complex_relative'],'MISSING_TRUTH_ARRAY')
  with self.assertRaisesRegex(ValueError,'MISSING_FULL'):self.runcheck(True)
 def test_missing_saved_field_never_new_solve(self):
  c=dict(self.state);del c['field'];self.array(self.cases[0]['state_file'],c);self.save();r,_=self.runcheck();self.assertIsNone(r['cases'][0]['metrics']['final_objective']);self.assertEqual(r['cases'][0]['missing_prerequisites']['measured_data_relative'],'MISSING_SAVED_FIELD')
 def test_source_reference_sha_drift(self):
  self.data['source_reference_sha256']={**N.SOURCE_REFERENCE,'build_figures.py':'0'*64};self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_source_input_hash_drift(self):
  p=self.package/self.cases[0]['state_file'];p.write_bytes(p.read_bytes()+b'changed')
  with self.assertRaises(ValueError):self.runcheck()
 def test_notrun_and_failed_status_cannot_promote(self):
  self.cases[-1]['status']='COMPLETED';self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_curve_update_count_and_continuity(self):
  self.json(self.cases[0]['curve_file'],[{'iteration':0,'objective_before':2.,'objective_after':1.}]);self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_grid_duplicate_and_nearest_tie(self):
  self.assertEqual(N.nearest0(np.array([-1.,1.])),0);points=self.common['points'].copy();points[1]=points[0]
  with self.assertRaises(ValueError):N.grid(points,self.state['chi'],[2,2,2])
 def test_pickle_or_fp32_state_rejected(self):
  self.array(self.cases[0]['state_file'],{'chi':self.state['chi'].astype(np.complex64)});self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_synthetic_cannot_be_research(self):
  with self.assertRaises(ValueError):N.analyze(self.package,self.manifest,S.sha(self.manifest))
if __name__=='__main__':unittest.main()
