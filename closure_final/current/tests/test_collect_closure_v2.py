"""Synthetic completed-prefix/failed-parent audit; no kernels or remote actions."""
from pathlib import Path
import importlib.util,json,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('collector_v2',ROOT/'analysis/collect_closure_v2.py')
old=load('collector_fixture_helpers',ROOT/'tests/test_collect_closure.py')
class PartialParentTests(unittest.TestCase):
 def fixture(self,t):
  f=old.CollectTests();c,b,r,l,h,d=f.batch_fixture(t)
  batch=m.read(r/'batch_receipt.json');batch['tasks'][-1].update(returncode=1,status='FAILED');batch['total_wall_s']=3.;f.put(r/'batch_receipt.json',batch);f.put(r/'late/status.json',dict(status='FAILED',error='synthetic receiver failure'))
  rows=[json.loads(x) for x in l.read_text().splitlines()];rows[1].update(exit_code=1,occupation_s=4.);l.write_text('\n'.join(json.dumps(x) for x in rows)+'\n');return c,b,r,l,h,d,f
 def audit(self,d,l):return m.audit_terminal(d,m.paired_attempts(l)[0])
 def test_completed_prefix_admitted_failed_cost_preserved(self):
  with tempfile.TemporaryDirectory() as t:
   c,b,r,l,h,d,f=self.fixture(t);result,a,issues=self.audit(d,l);self.assertEqual(issues,[]);self.assertEqual(result['status'],'COMPLETED');self.assertEqual(a['end']['exit_code'],1);self.assertEqual(a['collector_binding']['closed_failed_parent_prefix']['completed_prefix_rows'],2)
   status=m.collect(c,b,r,l,h,c/'new');self.assertEqual(status['audited_counts'],{'validation':1});self.assertTrue(any(x['kind']=='NONCOMPLETED_STATE_RETAINED' for x in m.read(c/'new/ISSUES.json')));self.assertTrue(any(x['kind']=='CLOSED_FAILED_MONITORED_JOB_NO_SCIENTIFIC_RESULT' for x in m.read(c/'new/ISSUES.json')))
 def test_unpaired_parent_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   c,b,r,l,h,d,f=self.fixture(t);l.write_text(l.read_text().splitlines()[0]+'\n');self.assertIsNone(self.audit(d,l)[0])
 def test_unknown_parent_runner_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   c,b,r,l,h,d,f=self.fixture(t);l.write_text(l.read_text().replace('run_locked_batch_v2.py','run_unknown.py'));self.assertIsNone(self.audit(d,l)[0])
 def test_wrong_row_path_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   c,b,r,l,h,d,f=self.fixture(t);v=m.read(r/'batch_receipt.json');v['tasks'][0]['path']+='wrong';f.put(r/'batch_receipt.json',v);self.assertIsNone(self.audit(d,l)[0])
 def test_source_and_child_import_hash_rejected(self):
  for field in ('parent','child'):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
    c,b,r,l,h,d,f=self.fixture(t)
    if field=='parent':l.write_text(l.read_text().replace(m.sha(ROOT/'code/run_locked_batch_v2.py'),'wronghash'))
    else:
     q=m.read(d/'runtime_receipt.json');q['runtime']['closure_sources']['code/run_locked_batch_v2.py']='wronghash';f.put(d/'runtime_receipt.json',q)
    self.assertIsNone(self.audit(d,l)[0])
 def test_target_status_failed_or_row_failed_rejected(self):
  for field in ('status','row'):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
    c,b,r,l,h,d,f=self.fixture(t)
    if field=='status':f.put(d/'status.json',dict(status='FAILED'))
    else:
     q=m.read(r/'batch_receipt.json');q['tasks'][0].update(status='FAILED',returncode=1);f.put(r/'batch_receipt.json',q)
    self.assertIsNone(self.audit(d,l)[0])
 def test_guard_kill_identity_and_fee_rejected(self):
  for field in ('stop','identity','fee'):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
    c,b,r,l,h,d,f=self.fixture(t);v=[json.loads(x) for x in l.read_text().splitlines()]
    if field=='stop':v[1]['stop_reason']='memory_stop'
    elif field=='identity':v[1]['identity_sha256']='wrong'
    else:v[1]['occupation_s']=2.
    l.write_text('\n'.join(json.dumps(x) for x in v)+'\n');self.assertIsNone(self.audit(d,l)[0])
 def test_failed_tail_missing_or_nonfailed_status_rejected(self):
  for value in (None,'RUNNING','COMPLETED'):
   with self.subTest(value=value),tempfile.TemporaryDirectory() as t:
    c,b,r,l,h,d,f=self.fixture(t);p=r/'late/status.json'
    if value is None:p.unlink()
    else:f.put(p,dict(status=value))
    self.assertIsNone(self.audit(d,l)[0])
 def test_complete_parent_regression(self):
  with tempfile.TemporaryDirectory() as t:
   c,b,r,l,h,d=old.CollectTests().batch_fixture(t);result,a,issues=self.audit(d,l);self.assertEqual(issues,[]);self.assertIsNone(a['collector_binding']['closed_failed_parent_prefix'])
 def test_registered_99_denominator_unchanged(self):
  from collections import Counter
  self.assertEqual(Counter(x[0] for x in m.expected_keys(dict(eligible_objects=list(range(11)),noise_objects=[0,1,2,3])))['validation'],99)
if __name__=='__main__':unittest.main(verbosity=2)
