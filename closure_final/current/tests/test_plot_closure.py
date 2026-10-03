"""Tiny synthetic plotting contracts only; every rendered output is MOCK."""
from pathlib import Path
import csv,importlib.util,json,tempfile,unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'analysis/plot_closure.py';s=importlib.util.spec_from_file_location('plot_closure',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
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
  with self.assertRaisesRegex(ValueError,'complete'):m.phase_gate(dict(audited_counts=dict(validation=98)), 'validation',False)
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
   self.assertTrue(config['mock']);self.assertIsNone(config['scientific_verdict']);self.assertEqual(len(list(out.glob('Figure*.png'))),3)
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
if __name__=='__main__':unittest.main(verbosity=2)
