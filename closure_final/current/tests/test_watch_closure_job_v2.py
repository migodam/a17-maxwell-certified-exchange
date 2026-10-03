"""Static/mock only: no subprocess child, SSH, numerical or GPU action."""
from pathlib import Path
import importlib.util,json,tempfile,unittest
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'code/watch_closure_job_v2.py'
s=importlib.util.spec_from_file_location('monitor_under_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class MonitorTests(unittest.TestCase):
 def setup_roots(self,t):
  root=Path(t)/'CLOSURE_R1';old=Path(t)/'EXCHANGE_R1'
  for r in (root,old):(r/'execution').mkdir(parents=True)
  (root/'research').mkdir();(root/'research/A17_BASELINE_LOCK.json').write_text(json.dumps(dict(prior_paid_occupation_s=m.PRIOR_S,cumulative_limit_s=m.TOTAL_S)))
  self.ledger(old,17,m.PRIOR_S);return root,old
 def ledger(self,root,n,charged):
  with (root/'execution/gpu_attempts.jsonl').open('w') as f:
   for i in range(n):
    f.write(json.dumps(dict(event='start',attempt=str(i)))+'\n');f.write(json.dumps(dict(event='end',attempt=str(i),occupation_s=charged/n))+'\n')
 def test_budget_carry_and_job_cap(self):
  with tempfile.TemporaryDirectory() as t:
   r,b=self.setup_roots(t);a=m.accounting(r,b);self.assertAlmostEqual(a['remaining_s'],38310.485);self.assertEqual(a['limit_s'],7200.)
   self.ledger(r,1,37000.);a=m.accounting(r,b);self.assertAlmostEqual(a['limit_s'],1310.485)
 def test_exhausted_budget(self):
  with tempfile.TemporaryDirectory() as t:
   r,b=self.setup_roots(t);self.ledger(r,1,38311.)
   with self.assertRaises(RuntimeError):m.accounting(r,b)
 def test_old_carry_mismatch(self):
  with tempfile.TemporaryDirectory() as t:
   r,b=self.setup_roots(t);self.ledger(b,16,m.PRIOR_S)
   with self.assertRaises(RuntimeError):m.accounting(r,b)
   self.ledger(b,17,1.)
   with self.assertRaises(RuntimeError):m.accounting(r,b)
 def test_old_and_new_locks_retained(self):
  with tempfile.TemporaryDirectory() as t:
   r,b=self.setup_roots(t)
   for p in (b/'execution/GPU_JOB_LOCK.json',r/'execution/GPU_JOB_LOCK.json'):
    p.write_text('{}')
    with self.assertRaises(RuntimeError):m.accounting(r,b)
    self.assertTrue(p.exists());p.unlink() # fixture cleanup only
 def test_unpaired_duplicate_or_unknown_fee_fail(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'gpu_attempts.jsonl'
   for rows in ([dict(event='start',attempt='x')],[dict(event='end',attempt='x',occupation_s=1.)],[dict(event='start',attempt='x'),dict(event='start',attempt='x')],[dict(event='start',attempt='x'),dict(event='end',attempt='x')]):
    p.write_text('\n'.join(json.dumps(x) for x in rows))
    with self.assertRaises(RuntimeError):m.ledger_state(p)
 def test_unknown_lock_not_removed(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'GPU_JOB_LOCK.json';p.write_text(json.dumps(dict(attempt='foreign')))
   with self.assertRaises(RuntimeError):m.release_owned_lock(p,'mine')
   self.assertTrue(p.exists());m.release_owned_lock(p,'foreign');self.assertFalse(p.exists())
 def test_guard_overhead_charged_not_zero(self):
  with tempfile.TemporaryDirectory() as t:
   r,b=self.setup_roots(t);p=r/'execution/preflight_checks.jsonl';p.write_text(json.dumps(dict(event='preflight_rejected',wall_s=7.25)))
   self.assertAlmostEqual(m.accounting(r,b)['remaining_s'],38303.235)
   p.write_text(json.dumps(dict(event='preflight_rejected',wall_s=None)))
   with self.assertRaises(RuntimeError):m.accounting(r,b)
 def test_other_python_and_unavailable_failclosed(self):
  class Proc:
   pid=999999
   def name(self):return 'python.exe'
   def cmdline(self):return ['python','A18/matched_queue.py']
  class API:
   class NoSuchProcess(Exception):pass
   class AccessDenied(Exception):pass
   @staticmethod
   def process_iter():return [Proc()]
  with self.assertRaises(RuntimeError):m.inspect_processes(API)
  with patch.object(Proc,'cmdline',side_effect=PermissionError('unavailable')):
   with self.assertRaises(RuntimeError):m.inspect_processes(API)
 def test_unknown_occupation_not_zero(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'ledger';p.write_text('\n'.join(json.dumps(x) for x in [dict(event='start',attempt='x'),dict(event='end',attempt='x',occupation_s=float('nan'))]))
   with self.assertRaises(RuntimeError):m.ledger_state(p)
 def test_external_or_inline_child_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'child.py').write_text('pass')
   self.assertTrue(m.validate_child_command(r,['python','child.py'])['sha256'])
   for c in (['python','-c','pass'],['python','/unknown/old.py'],['python','-m','old']):
    with self.assertRaises(RuntimeError):m.validate_child_command(r,c)
 def test_root_and_constants(self):
  self.assertEqual(m.ROOT,P.parents[1]);self.assertEqual(m.PRIOR_S,4889.515);self.assertEqual(m.TOTAL_S,43200.);self.assertEqual(m.MAX_JOB_S,7200.)
if __name__=='__main__':unittest.main(verbosity=2)
