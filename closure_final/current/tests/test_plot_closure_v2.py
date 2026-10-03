"""Tiny synthetic plotting contracts only; every rendered output is MOCK."""
from pathlib import Path
import csv,importlib.util,json,tempfile,unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'analysis/plot_closure_v2.py';s=importlib.util.spec_from_file_location('plot_closure',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def fixture(root):
 root.mkdir();c=root/'collection';c.mkdir()
 def dump(p,v):p.write_text(json.dumps(v))
 def table(p,rows):
  with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 dump(c/'STATUS.json',dict(audited_counts=dict(validation=99,noise=72),phase_status=dict(validation='COMPLETE_MECHANICAL_OWNER_REVIEW_REQUIRED',noise='COMPLETE_MECHANICAL_OWNER_REVIEW_REQUIRED'),critical_issues=0))
 cells=[];objects=[]
 for mode in ('validation','noise'):
  for level in ([0] if mode=='validation' else [100,300]):
   objects.append(dict(mode=mode,object_id=1,noise_basis_points=level,ratio=.5))
   for k in (4,8,16):
    for rep in ([0] if mode=='validation' else range(3)):cells.append(dict(mode=mode,object_id=1,phase='middle',k=k,noise_basis_points=level,realization_index=rep,receiver_full_gap=2.,final_full_gap=1.,accepted_moves=rep))
 table(c/'cells.csv',cells);table(c/'object_summed_risks.csv',objects);table(c/'attempt_costs.csv',[dict(attempt='r',occupation_s=3.),dict(attempt='e',occupation_s=5.)])
 dump(root/'scenes.json',dict(scenes=[dict(object_id=1,family='synthetic',representation='synthetic',components=[{},{}])]))
 points=np.array([[x,y,z] for z in (-1.,0.,1.) for y in (-1.,0.,1.) for x in (-1.,0.,1.)]);truth=points[:,0]+1j*points[:,1];np.savez(root/'common.npz',points=points,truth=truth)
 panel=dict(object_id=1,common_data='common.npz')
 for p,a in (('receiver','r'),('exchange','e')):
  np.savez(root/(p+'.npz'),chi=truth*(.8 if p=='receiver' else .9));dump(root/(p+'.json'),dict(wall_total_s=2. if p=='receiver' else 4.));panel[p]=dict(final=p+'.npz',result=p+'.json',attempt=a)
 dump(root/'nonlinear.json',dict(complete_audited=True,panels=[panel]));manifest=root/'IMMUTABLE.json';dump(manifest,dict(immutable_complete=True,mock=True,files={p.relative_to(root).as_posix():m.sha(p) for p in root.rglob('*') if p.is_file()}));return manifest
class PlotTests(unittest.TestCase):
 def test_missing_is_never_zero(self):
  for v in (None,'','null','None'):self.assertIsNone(m.number(v))
  self.assertIsNone(m.total([dict(gap='')],'gap'));self.assertEqual(m.total([dict(gap=0.)],'gap'),0.)
  with self.assertRaises(ValueError):m.number('nan')
 def test_hash_drift_and_escape_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'mock_source';manifest=fixture(root);i=m.Inputs(root,manifest,True);(root/'scenes.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'bound'):i.json('scenes.json')
   with self.assertRaisesRegex(ValueError,'escapes'):i.path('../outside')
 def test_complete_phase_and_immutable_gate(self):
  with self.assertRaisesRegex(ValueError,'coverage'):m.phase_gate(dict(audited_counts=dict(validation=98)), 'validation',False)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'mock_source';manifest=fixture(root);j=m.json_read(manifest);j['immutable_complete']=False;manifest.write_text(json.dumps(j))
   with self.assertRaisesRegex(ValueError,'immutable'):m.Inputs(root,manifest)
 def test_scene_labels_do_not_show_ids(self):
  labels=m.scene_labels(dict(scenes=[dict(object_id=2002,family='gaussian',components=[{}])]))
  self.assertNotIn('2002',labels[2002]);self.assertIn('Gaussian',labels[2002])
 def test_slice_and_profile_use_saved_grid_without_interpolation(self):
  points=np.array([[x,y,z] for z in (-1.,0.,1.) for y in (-1.,0.,1.) for x in (-1.,0.,1.)]);x,y,z,v=m.slice_grid(points,points[:,0]);self.assertEqual(z,0.);np.testing.assert_array_equal(v[1],x)
  with self.assertRaisesRegex(ValueError,'rectangular'):m.slice_grid(np.array([[0.,0.,0.],[1.,1.,0.]]),np.ones(2))
 def test_three_mock_figures_and_replot_receipts(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'mock_source';manifest=fixture(root);out=Path(t)/'mock_plot_output';config=m.run(root,manifest,'collection','scenes.json',out,'nonlinear.json',mock=True)
   self.assertTrue(config['mock']);self.assertIsNone(config['scientific_verdict']);self.assertEqual(len(list(out.glob('Figure*.png'))),4)
   self.assertIn('MOCK SYNTHETIC', (out/'Figure1_additional_objects.svg').read_text());self.assertTrue((out/'OUTPUT_HASHES.json').exists())
 def test_mock_path_and_new_output_required(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'source';manifest=fixture(root)
   with self.assertRaisesRegex(ValueError,'mock output'):m.run(root,manifest,'collection','scenes.json',Path(t)/'production',mock=True)
   with self.assertRaisesRegex(ValueError,'outside'):m.run(root,manifest,'collection','scenes.json',root/'mock',mock=True)
 def test_missing_nonlinear_stays_unavailable(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'mock_source';manifest=fixture(root);out=Path(t)/'mock_missing_output';m.run(root,manifest,'collection','scenes.json',out,mock=True,figures=['nonlinear'])
   self.assertIn('artifacts unavailable',(out/'Figure3_nonlinear_material.svg').read_text())
def partial_fixture(parent,extra_issue=False,noise_missing=False,extra_primary_missing=False):
 root=parent/'mock_partial_source';root.mkdir();c=root/'collection';c.mkdir()
 def dump(p,v):p.write_text(json.dumps(v))
 def table(p,rows):
  with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 coverage=[];cells=[];objects=[]
 for mode,oids,phases,levels,reps in [('validation',list(range(1,11))+[2016],['early','middle','late'],[0],[0]),('noise',[1,2,3,2016],['middle'],[100,300],range(3))]:
  for oid in oids:
   for level in levels:
    objects.append(dict(mode=mode,object_id=oid,noise_basis_points=level,complete=oid!=2016 or mode=='noise',observed_cells=6 if oid==2016 and mode=='validation' else 9,expected_cells=9,ratio=.75))
    for phase in phases:
     for k in m.RANKS:
      for rep in reps:
       row=dict(mode=mode,object_id=oid,phase=phase,k=k,noise_basis_points=level,realization_index=rep);missing=m.cell_key(row) in m.KNOWN_MISSING or (noise_missing and mode=='noise' and oid==1 and level==100 and k==4 and rep==0) or (extra_primary_missing and mode=='validation' and oid==1 and phase=='early' and k==4)
       coverage.append(dict(**row,status='MISSING_OR_NOT_AUDITED' if missing else 'AUDITED_TERMINAL'))
       if not missing:cells.append(dict(**row,receiver_full_gap=2.,final_full_gap=1.5,accepted_moves=0))
 status=dict(expected_counts=dict(validation=99,noise=72),audited_counts=dict(validation=95 if extra_primary_missing else 96,noise=71 if noise_missing else 72),phase_status=dict(validation='INCOMPLETE_OR_CRITICAL_AUDIT_ISSUES',noise='INCOMPLETE_OR_CRITICAL_AUDIT_ISSUES'),critical_issues=1)
 issue=dict(kind='KNOWN_PRIMARY_PREREQUISITE_FAILURE',object_id=2016,phase='late',error='NORMAL_RESIDUAL_FAILED',severity='critical');issues=[issue]+([dict(kind='UNKNOWN_OTHER_AUDIT_FAILURE',severity='critical')] if extra_issue else [])
 dump(c/'STATUS.json',status);dump(c/'ISSUES.json',issues);table(c/'coverage.csv',coverage);table(c/'cells.csv',cells);table(c/'object_summed_risks.csv',objects)
 failure=root/'mock_results/object_2016/late/status.json';failure.parent.mkdir(parents=True);dump(failure,dict(status='FAILED',error='NORMAL_RESIDUAL_FAILED',mock=True));dump(root/'scenes.json',dict(scenes=[dict(object_id=o,family='synthetic',components=[{}]) for o in list(range(1,11))+[2016]]))
 manifest=root/'IMMUTABLE.json';dump(manifest,dict(mock=True,immutable_complete=True,files={p.relative_to(root).as_posix():m.sha(p) for p in root.rglob('*') if p.is_file()}))
 authority=parent/'MOCK_PRIMARY_AUTHORITY.json';a=dict(schema='a17.primary.plot.authority.v1',mock=True,owner_decision='AUTHORIZE_DESCRIPTIVE_KNOWN_PRIMARY_PREREQUISITE_FAILURE',planned_primary_cells=99,audited_primary_cells=96,immutable_manifest_sha256=m.sha(manifest),collector_status_sha256=m.sha(c/'STATUS.json'),collector_issues_sha256=m.sha(c/'ISSUES.json'),collector_coverage_sha256=m.sha(c/'coverage.csv'),collector_cells_sha256=m.sha(c/'cells.csv'),allowed_missing_cells=[dict(mode=k[0],object_id=k[1],phase=k[2],k=k[3],noise_basis_points=k[4],realization_index=k[5]) for k in sorted(m.KNOWN_MISSING)],failed_state=dict(object_id=2016,phase='late',status='FAILED',prerequisite_error='NORMAL_RESIDUAL_FAILED',status_path='mock_results/object_2016/late/status.json',status_sha256=m.sha(failure)),allowed_issue_sha256=[m.issue_sha(issue)]);dump(authority,a)
 return root,manifest,authority,status,cells
class PartialGateTests(unittest.TestCase):
 def call(self,setup,mode='validation',authority=True):
  root,manifest,a,status,cells=setup;i=m.Inputs(root,manifest,True)
  return m.phase_gate(status,mode,False,i,'collection',cells,str(a) if authority else None,m.sha(a) if authority else None)
 def test_known_three_missing_requires_exact_root_authority(self):
  with tempfile.TemporaryDirectory() as t:
   setup=partial_fixture(Path(t))
   with self.assertRaisesRegex(ValueError,'authority required'):self.call(setup,authority=False)
   g=self.call(setup);self.assertEqual(g['planned_primary_cells'],99);self.assertEqual(g['phase_audited_cells'],96)
   root,manifest,a,status,cells=setup;i=m.Inputs(root,manifest,True)
   with self.assertRaisesRegex(ValueError,'authority SHA'):m.phase_gate(status,'validation',False,i,'collection',cells,str(a),'wrong')
 def test_unknown_issue_rejected_even_if_global_binding_updated(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaisesRegex(ValueError,'other audit issues'):self.call(partial_fixture(Path(t),extra_issue=True))
 def test_noise_uses_full_phase_coverage_with_known_primary_failure(self):
  with tempfile.TemporaryDirectory() as t:
   g=self.call(partial_fixture(Path(t)),'noise');self.assertEqual(g['phase_audited_cells'],72)
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaisesRegex(ValueError,'full72'):self.call(partial_fixture(Path(t),noise_missing=True),'noise')
 def test_wrong_failed_hash_and_extra_missing_keys_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   setup=partial_fixture(Path(t));a=setup[2];j=m.json_read(a);j['failed_state']['status_sha256']='wrong';a.write_text(json.dumps(j))
   with self.assertRaisesRegex(ValueError,'status SHA'):self.call(setup)
   j['failed_state']['status_sha256']=m.sha(setup[0]/'mock_results/object_2016/late/status.json');j['allowed_missing_cells'][0]['k']=99;a.write_text(json.dumps(j))
   with self.assertRaisesRegex(ValueError,'known three'):self.call(setup)
 def test_partial_figure_open_marker_count_annotation_and_planned_denominator(self):
  with tempfile.TemporaryDirectory() as t:
   root,manifest,a,status,cells=partial_fixture(Path(t));out=Path(t)/'mock_partial_plot';cfg=m.run(root,manifest,'collection','scenes.json',out,mock=True,figures=['primary']);svg=(out/'Figure1_additional_objects.svg').read_text()
   for text in ('partial 6/9','2/3 states','Partial; observed cells only','Planned 99 cells retained','MOCK SYNTHETIC'):self.assertIn(text,svg)
   self.assertEqual(cfg['planned_primary_cells'],99)
 def test_additional_actual_primary_missing_cell_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaisesRegex(ValueError,'actual primary missing'):self.call(partial_fixture(Path(t),extra_primary_missing=True))
 def test_overview_uses_collector_ratio_and_raw_object_sums(self):
  with tempfile.TemporaryDirectory() as t:
   root,manifest,a,status,cells=partial_fixture(Path(t));objects=m.Inputs(root,manifest,True).csv('collection/object_summed_risks.csv');objects[0]['ratio']='0.123';data=m.primary_overview_data(cells,objects)
   self.assertEqual(len(data['object_ratios']),11);self.assertEqual(data['object_ratios'][0]['ratio'],.123);self.assertIsNone(data['statistical_summary'])
   partial=[r for r in data['object_ratios'] if r['object_id']==2016][0];self.assertFalse(partial['complete']);self.assertEqual(partial['observed_cells'],6)
   for r in data['absolute_object_gaps_by_rank']:
    self.assertEqual(r['receiver_absolute_gap_sum'],4. if r['object_id']==2016 else 6.);self.assertEqual(r['exchange_absolute_gap_sum'],3. if r['object_id']==2016 else 4.5)
 def test_overview_export_retains_hollow_partial_and_supplement(self):
  with tempfile.TemporaryDirectory() as t:
   root,manifest,a,status,cells=partial_fixture(Path(t));out=Path(t)/'mock_overview_plot';m.run(root,manifest,'collection','scenes.json',out,mock=True,figures=['primary']);svg=(out/'Figure1_primary_overview.svg').read_text()
   for text in ('(6/9)','Hollow: partial','Partial: 2/3 states','Raw object sums by rank','MOCK SYNTHETIC'):self.assertIn(text,svg)
   self.assertTrue((out/'Figure1_additional_objects.pdf').exists());self.assertEqual(len(m.json_read(out/'PRIMARY_OVERVIEW_DATA.json')['object_ratios']),11)
if __name__=='__main__':unittest.main(verbosity=2)
