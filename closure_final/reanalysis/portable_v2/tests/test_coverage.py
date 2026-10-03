import sys,tempfile,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_public_coverage as C
import public_support as S
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.authority=self.root/'authority.json';self.initial=self.root/'initial.csv';self.path=self.root/'path.csv';self.receipt='a'*64;cells=[]
  for o in C.OBJECTS:
   for p in C.PHASES:
    for k in C.KS:cells.append({'object_id':o,'phase':p,'k':k,'state_status':'COMPLETED','binding_status':'VERIFIED','cell_status':'COMPLETE','teacher_best_raw_positive_gain':float(k),'balanced12_best_raw_positive_gain':k/2,'full_dictionary_anchor_top1_noop_capture':.25,'full_dictionary_anchor_top1_verified_noop_gain':k/4,'teacher_path_status':'LOCAL_3ACTION_BUDGET_COMPLETE','teacher_path_length':3,'actual_online_final_full_gap':.6,'local_teacher_final_full_gap':.4,'actual_minus_local_teacher_risk':.2})
  self.data={'schema':'a17.saved.coverage.evidence.v1','synthetic_fixture':True,'snapshot_receipt_sha256':self.receipt,'config_sha256':'b'*64,'input_manifest_sha256':'c'*64,'collector_sha256':'d'*64,'cells':cells};self.views()
 def tearDown(self):self.temp.cleanup()
 def views(self):self.data['object_initial_opportunity']=C.aggregate_initial(self.data['cells']);self.data['path_final_risk']=C.path_final(self.data['cells']);self.save()
 def save(self):self.authority.write_bytes(S.canonical(self.data));self.initial.write_bytes(C.csv_bytes(self.data['object_initial_opportunity']));self.path.write_bytes(C.csv_bytes(self.data['path_final_risk']))
 def runcheck(self):return C.validate(self.authority,S.sha(self.authority),self.initial,self.path,self.receipt,True)
 def test_positive_Q_weighted_and_separate_scope(self):
  r=self.runcheck();self.assertTrue(r['initial_aggregation_recomputed_from_public_cells']);self.assertEqual(r['planned_cells'],99);self.assertFalse(r['private_source_bound_provenance_reaudited']);self.assertEqual(self.data['object_initial_opportunity'][0]['balanced12_capture'],.5)
 def test_partial_and_notrun_null_retained(self):
  for c in self.data['cells'][-3:]:c.update(cell_status='NOT_RUN',state_status='NOT_RUN_PRIMARY_INCOMPLETE',binding_status='FIXED_PREREQUISITE_EXCLUSION_VERIFIED',teacher_path_status=None,teacher_path_length=None,actual_online_final_full_gap=None,local_teacher_final_full_gap=None,actual_minus_local_teacher_risk=None)
  self.views();r=self.runcheck();self.assertEqual(r['initial_complete_objects'],10);self.assertIsNone(self.data['object_initial_opportunity'][-1]['balanced12_capture']);self.assertEqual(r['path_null_cells'],3)
 def test_public_views_without_cells_not_claim_reaudit(self):
  del self.data['cells'];self.save();r=self.runcheck();self.assertFalse(r['initial_aggregation_recomputed_from_public_cells']);self.assertIn('VIEW_BINDING_ONLY',r['missing_public_cells_reason'])
 def test_partial_initial_promoted_rejected(self):
  c=self.data['cells'][-1];c['cell_status']='PARTIAL_OFFLINE_BUDGET';self.views();self.data['object_initial_opportunity'][-1]['balanced12_capture']=.5;self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_no_positive_has_null_capture(self):
  for c in self.data['cells'][:9]:c['teacher_best_raw_positive_gain']=0.
  self.views();self.assertIsNone(self.data['object_initial_opportunity'][0]['balanced12_capture']);self.runcheck()
 def test_csv_tamper_and_sha_stale(self):
  old=S.sha(self.authority);self.path.write_bytes(self.path.read_bytes()+b'bad')
  with self.assertRaises(ValueError):self.runcheck()
  self.save();self.authority.write_text('{}')
  with self.assertRaises(ValueError):C.validate(self.authority,old,self.initial,self.path,self.receipt,True)
 def test_path_partial_promoted_rejected(self):
  self.data['cells'][0]['teacher_path_status']='PARTIAL_OFFLINE_BUDGET';self.views();self.data['path_final_risk'][0]['local_teacher_final_full_gap']=.4;self.save()
  with self.assertRaises(ValueError):self.runcheck()
 def test_denominator_cell_duplicate_rejected(self):
  self.data['cells'][-1]=dict(self.data['cells'][0]);self.views()
  with self.assertRaises(ValueError):self.runcheck()
 def test_receipt_mismatch_and_synthetic_not_real(self):
  with self.assertRaises(ValueError):C.validate(self.authority,S.sha(self.authority),self.initial,self.path,'0'*64,True)
  with self.assertRaises(ValueError):C.validate(self.authority,S.sha(self.authority),self.initial,self.path,self.receipt)
if __name__=='__main__':unittest.main()
