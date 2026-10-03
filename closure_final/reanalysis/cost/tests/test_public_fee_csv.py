import sys,csv,json,math,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_public_fee_csv as V
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.at=self.root/'attempt.csv';self.pr=self.root/'preflight.csv';self.au=self.root/'authority.json';self.source='a'*64
  self.attempts=[{'attempt':'noise_failed','partition':'noise','exit_code':1,'occupation_s':7,'charge_kind':'ADDITIVE_WHOLE_ATTEMPT_OCCUPATION','source_json_sha256':self.source,'child_process_wall_sum_s':None},{'attempt':'noise_ok','partition':'noise','exit_code':0,'occupation_s':3,'charge_kind':'ADDITIVE_WHOLE_ATTEMPT_OCCUPATION','source_json_sha256':self.source,'child_process_wall_sum_s':2}];self.rejects=[{'attempt':'noise_reject','partition':'noise','wall_s':2,'charge_kind':'ADDITIVE_REJECTED_PREFLIGHT','source_json_sha256':self.source}]
  phases=[{'partition':p,'occupation_s':10 if p=='noise' else 0,'rejected_preflight_s':2 if p=='noise' else 0,'closure_phase_fee_s':12 if p=='noise' else 0} for p in sorted(V.PHASES)];self.authority={'schema':'a17.public.cost.derivation.v1','raw_monitor_reaudit_supported':False,'source_authoritative_json_sha256':self.source,'attempts':self.attempts,'rejected_preflights':self.rejects,'phase_totals':phases,'totals':{'old_carry_s':100,'closure_monitored_occupation_s':10,'rejected_preflight_wall_s':2,'cumulative_charged_s':112,'closure_paired_attempts':2,'rejected_preflight_count':1}};self.write()
 def tearDown(self):self.tmp.cleanup()
 def csv(self,p,rows):
  with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 def write(self):self.csv(self.at,self.attempts);self.csv(self.pr,self.rejects);self.au.write_text(json.dumps(self.authority))
 def runcheck(self):return V.validate(self.at,self.pr,self.au,V.sha(self.au))
 def test_all_failure_charges_retained(self):
  r=self.runcheck();self.assertEqual(r['campaign_s'],112);self.assertEqual(r['failed_attempts_retained'],1);self.assertFalse(r['nested_stages_added']);self.assertFalse(r['raw_private_monitor_reaudited'])
 def test_stale_authority(self):
  old=V.sha(self.au);self.au.write_text('{}')
  with self.assertRaises(ValueError):V.validate(self.at,self.pr,self.au,old)
 def test_csv_hash_value_tamper(self):
  self.attempts[0]['occupation_s']=8;self.csv(self.at,self.attempts)
  with self.assertRaises(ValueError):self.runcheck()
 def test_missing_additive_fee_not_zero(self):
  self.attempts[0]['occupation_s']=None;self.write()
  with self.assertRaises(ValueError):self.runcheck()
 def test_unknown_command_header(self):
  self.attempts[0]['command']='unknown';self.attempts[1]['command']='unknown';self.write()
  with self.assertRaises(ValueError):self.runcheck()
 def test_exact_total_mismatch(self):
  self.authority['totals']['cumulative_charged_s']=112.00000000001;self.write()
  with self.assertRaises(ValueError):self.runcheck()
 def test_overlapping_reject(self):
  self.rejects[0]['attempt']='noise_failed';self.write()
  with self.assertRaises(ValueError):self.runcheck()
 def test_null_nested_detail_not_added(self):
  self.authority['nested_actual_physics_and_null_reasons']=[{'wall_s':100000}];self.write();self.assertEqual(self.runcheck()['campaign_s'],112)
if __name__=='__main__':unittest.main()
