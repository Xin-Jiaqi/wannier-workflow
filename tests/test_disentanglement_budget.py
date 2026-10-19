import unittest
from w90flow.attempts import plan_candidate
from w90flow.orchestrator import choose_retry_reason

class DisentanglementBudgetTests(unittest.TestCase):
    def plans(self, **kwargs):
        return plan_candidate([[-2., -1., 0., 1.], [-1.9, -.8, .2, 1.2]],
            [{'orbital_scores': {'X:p': 1.}} for _ in range(4)],
            {'projection': ['X:p'], 'num_wann': 3}, -1.1, .3,
            isolation_threshold_eV=10., max_window_attempts=1, **kwargs)

    def test_opt_in_adds_one_single_factor_control(self):
        baseline = self.plans(max_localization_retries=1)
        planned = self.plans(max_localization_retries=1, disentanglement_budget_iterations=2000)
        self.assertEqual(len(planned), len(baseline) + 1)
        control = next(p for p in planned if p.get('retry_class') == 'disentanglement_budget')
        for key, value in baseline[0].items():
            if key != 'dis_num_iter':
                self.assertEqual(control[key], value)
        self.assertEqual(control['dis_num_iter'], 2000)
        self.assertNotIn('num_iter', control)

    def test_only_explicit_dis_failure_can_select_budget(self):
        available = {'disentanglement_budget', 'convergence_retry'}
        self.assertEqual(choose_retry_reason(['DISENTANGLEMENT_NOT_CONVERGED'], set(), available), 'disentanglement_budget')
        self.assertEqual(choose_retry_reason(['DISENTANGLEMENT_NOT_CONVERGED'], {'disentanglement_budget_only'}, {'convergence_retry'}), 'convergence_retry')
        self.assertEqual(choose_retry_reason(['LOCALIZATION_NOT_CONVERGED'], set(), available), 'convergence_retry')
        self.assertIsNone(choose_retry_reason(['CONVERGENCE_UNVERIFIED'], set(), available))
        self.assertIsNone(choose_retry_reason(['QUALITY_GATE_FAILED'], set(), available))

    def test_invalid_budget_rejected(self):
        for value in [True, 500, 10001, 2000.0, '2000']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plans(disentanglement_budget_iterations=value)

    def test_exact_subspace_does_not_gain_disentanglement_trial(self):
        plans = plan_candidate([[-2., -1., 0., 5.], [-1.9, -.8, .2, 5.2]],
            [{'orbital_scores': {'X:p': 1.}} for _ in range(4)],
            {'projection': ['X:p'], 'num_wann': 3}, -1.1, .3,
            disentanglement_budget_iterations=2000)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]['mode'], 'exact')
