"""Pure contract/source tests only; do not run Maxwell or remote jobs."""
from pathlib import Path
import ast
import copy
import importlib.util
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
spec = importlib.util.spec_from_file_location('nonlinear_contract_subject', ROOT/'code/run_nonlinear.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def valid_config():
    return dict(permitted_modes=['nonlinear'], nonlinear_objects=subject.OBJECTS,
        nonlinear_k=8, nonlinear_noise_percent=0, nonlinear_outer_cap=18,
        nonlinear_initialization='existing_scene_init', nonlinear_policies=subject.POLICIES,
        nonlinear_version=subject.VERSION, nonlinear_outer_rules=subject.OUTER_RULES,
        online_incoming_count=12, max_finalists_per_round=2, max_actions=3,
        anchor_rank=64, pool_count=32, residual_up_count=16, random_pool_count=10,
        pool_and_anchor_seed=20260930, cuda_candidate_batch=16,
        online_tolerance_schema='a17.online.numerical_significance.v1',
        online_tolerance_safety=32, prior_paid_occupation_s=4889.515,
        cumulative_limit_s=43200, max_job_s=7200, memory_warning=.8, memory_stop=.9)


class NonlinearContract(unittest.TestCase):
    def test_fixed_matrix(self):
        for obj in subject.OBJECTS:
            for policy in subject.POLICIES:
                subject.validate_contract(valid_config(), obj, policy)

    def test_replay_prelock_cannot_launch(self):
        config = valid_config(); config['permitted_modes'] = ['replay']
        with self.assertRaises(ValueError):
            subject.validate_contract(config, 2002, 'receiver_only')

    def test_frozen_parameters_reject_drift(self):
        for key, value in [('nonlinear_k', 4), ('nonlinear_noise_percent', 1),
                           ('nonlinear_outer_cap', 19), ('nonlinear_initialization', 'zero'),
                           ('max_actions', 4), ('nonlinear_objects', [2002]),
                           ('nonlinear_version', 'different')]:
            config = valid_config(); config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_contract(config, 2002, 'receiver_only')
        config = copy.deepcopy(valid_config()); config['nonlinear_outer_rules']['armijo'] = 1e-3
        with self.assertRaises(ValueError):
            subject.validate_contract(config, 2002, 'receiver_only')

    def test_original_outer_rules_present(self):
        rules = subject.OUTER_RULES
        self.assertEqual((rules['prior'], rules['lm0'], rules['lm_decay'], rules['lm_decay_period']),
                         (1e-5, .01, .3, 3))
        self.assertEqual((rules['line_trials'], rules['armijo'], rules['small_step'], rules['small_step_first_iteration']),
                         (24, 1e-4, 1e-6, 9))

    def test_no_reference_solver_or_teacher_import(self):
        tree = ast.parse((ROOT/'code/run_nonlinear.py').read_text())
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertNotIn('run_reference', modules)
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        self.assertTrue({'online_exchange', 'CoreEndpointBatch', 'project_step'} <= imports)
        self.assertFalse({'solve_reference', 'EvaluatorContext', 'TeacherContext'} & imports)

    def test_receiver_branch_does_not_acquire_proposed_pool(self):
        tree = ast.parse((ROOT/'code/run_nonlinear.py').read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.policy == 'receiver_only'")
        receiver = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
        exchange = ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
        self.assertNotIn('public_workspace(', receiver)
        self.assertNotIn('online_exchange(', receiver)
        self.assertIn('CoreEndpointBatch(', receiver)
        self.assertIn('public_workspace(', exchange)
        self.assertIn('online_exchange(', exchange)


if __name__ == '__main__':
    unittest.main()
