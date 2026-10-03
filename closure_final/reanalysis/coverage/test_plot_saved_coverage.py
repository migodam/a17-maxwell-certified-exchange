import copy,csv,json,tempfile,unittest
from pathlib import Path
import plot_saved_coverage as m

def fixture():
 initial=[];path=[];labels={}
 for i,o in enumerate(m.OBJECTS):
  labels[str(o)]={'geometry':'Fixture geometry '+str(i+1),'material':'Fixture material '+str(i+1)}
  initial.append({'object_id':o,'planned_cells':9,'complete_initial_cells':9,'all_planned_initial_complete':True,'positive_teacher_cells_observed':9,'teacher_positive_gain_sum':2.,'balanced12_capture':.3,'anchor_top1_raw_noop_capture':.6,'anchor_top1_significance_verified_noop_capture':.5,'scope':'SYNTHETIC FIXTURE'})
  for p in m.PHASES:
   for k in m.KS:
    path.append({'object_id':o,'phase':p,'k':k,'state_status':'COMPLETED','binding_status':'VERIFIED','teacher_path_status':'LOCAL_3ACTION_BUDGET_COMPLETE','teacher_path_length':3,'actual_online_final_full_gap':.6,'local_teacher_final_full_gap':.4,'actual_minus_local_teacher_risk':.2,'scope':'SYNTHETIC FIXTURE'})
 # Partial complete initial opportunity cannot be aggregated.
 initial[-1].update(complete_initial_cells=6,all_planned_initial_complete=False,teacher_positive_gain_sum=None,**{k:None for k in m.CAPTURES})
 path[-1].update(state_status='NOT_RUN_PRIMARY_INCOMPLETE',binding_status='FIXED_PREREQUISITE_EXCLUSION_VERIFIED',teacher_path_status=None,teacher_path_length=None,**{k:None for k in m.RISKS})
 path[-2].update(state_status='PARTIAL_OFFLINE_BUDGET',teacher_path_status='PARTIAL_OFFLINE_BUDGET',teacher_path_length=1,local_teacher_final_full_gap=None,actual_minus_local_teacher_risk=None)
 path[-3].update(teacher_path_status='NUMERICAL_LOCAL_1SWAP_STOP_NO_DETERMINISTIC_BOUND',teacher_path_length=1)
 initial[1].update(teacher_positive_gain_sum=0.,positive_teacher_cells_observed=0,**{k:None for k in m.CAPTURES})
 path[0].update(actual_online_final_full_gap=.3,local_teacher_final_full_gap=.4,actual_minus_local_teacher_risk=-.1)
 return initial,path,labels

def string_rows(rows):
 return list(csv.DictReader(m.csv_bytes(rows).decode().splitlines()))
def prepared():
 a,b,l=fixture();return m.prepare(string_rows(a),string_rows(b),l)
class Tests(unittest.TestCase):
 def test_denominators_and_null(self):
  d=prepared();self.assertEqual(len(d['initial']),11);self.assertEqual(len(d['path']),99);self.assertIsNone(d['initial'][-1]['balanced12_capture']);self.assertEqual(d['path'][-1]['null_code'],'NR');self.assertEqual(d['path'][-2]['null_code'],'P')
 def test_missing_cell_fails(self):
  a,b,l=fixture()
  with self.assertRaisesRegex(ValueError,'99'):m.prepare(string_rows(a),string_rows(b[:-1]),l)
 def test_duplicate_object_fails(self):
  a,b,l=fixture();a[-1]['object_id']=a[0]['object_id']
  with self.assertRaisesRegex(ValueError,'11'):m.prepare(string_rows(a),string_rows(b),l)
 def test_partial_aggregate_fails(self):
  a,b,l=fixture();a[-1]['balanced12_capture']=.5
  with self.assertRaisesRegex(ValueError,'promoted'):m.prepare(string_rows(a),string_rows(b),l)
 def test_partial_teacher_endpoint_fails(self):
  a,b,l=fixture();b[-2]['local_teacher_final_full_gap']=.3
  with self.assertRaisesRegex(ValueError,'promoted'):m.prepare(string_rows(a),string_rows(b),l)
 def test_zero_opportunity_undefined(self):
  d=prepared();self.assertEqual(d['initial'][1]['reason'],'no positive opportunity')
 def test_capture_out_of_range_fails(self):
  a,b,l=fixture();a[0]['balanced12_capture']=1.2
  with self.assertRaisesRegex(ValueError,'unit range'):m.prepare(string_rows(a),string_rows(b),l)
 def test_difference_signed_and_checked(self):
  self.assertLess(prepared()['path'][0]['actual_minus_local_teacher_risk'],0)
  a,b,l=fixture();b[1]['actual_minus_local_teacher_risk']=.9
  with self.assertRaisesRegex(ValueError,'inconsistent'):m.prepare(string_rows(a),string_rows(b),l)
 def test_nonfinite_fails(self):
  with self.assertRaisesRegex(ValueError,'nonfinite'):m.number('NaN')
 def test_authority_csv_and_fixture_isolation(self):
  a,b,l=fixture()
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);f=t/'a.csv';g=t/'b.csv';h=t/'authority.json';lab=t/'labels.json'
   f.write_bytes(m.csv_bytes(a));g.write_bytes(m.csv_bytes(b));lab.write_text(json.dumps(l))
   authority={'schema':'a17.saved.coverage.evidence.v1','synthetic_fixture':True,'object_initial_opportunity':a,'path_final_risk':b,**{k:'a'*64 for k in ('snapshot_receipt_sha256','config_sha256','input_manifest_sha256','collector_sha256')}};h.write_text(json.dumps(authority))
   with self.assertRaisesRegex(ValueError,'synthetic/evidence'):m.load_inputs(f,g,h,m.sha(h),'a'*64,lab)
   data,_=m.load_inputs(f,g,h,m.sha(h),'a'*64,lab,synthetic=True);self.assertEqual(len(data['path']),99)
   f.write_bytes(f.read_bytes()+b'\n')
   with self.assertRaisesRegex(ValueError,'CSV differs'):m.load_inputs(f,g,h,m.sha(h),'a'*64,lab,synthetic=True)
 def test_render_fixture_only(self):
  with tempfile.TemporaryDirectory() as t:
   paths=m.draw(prepared(),Path(t)/'SYNTHETIC_ONLY',synthetic=True)
   self.assertEqual(len(paths),6);self.assertTrue(all(Path(p).stat().st_size>1000 for p in paths))
   self.assertIn('SYNTHETIC FIXTURE',(Path(t)/'SYNTHETIC_ONLY/initial_opportunity.svg').read_text())
   self.assertIn('SYNTHETIC FIXTURE',(Path(t)/'SYNTHETIC_ONLY/bounded_path_risk.svg').read_text())
if __name__=='__main__':unittest.main()
