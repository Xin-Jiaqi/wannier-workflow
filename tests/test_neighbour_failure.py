import tempfile
import unittest
from pathlib import Path
from w90flow.attempts import classify_failure, FAILURE_TRANSITIONS
from w90flow.orchestrator import choose_retry_reason, WorkflowBlocked

class NeighbourFailureTests(unittest.TestCase):
    def test_actual_c8f4_library_error_is_not_fitting_nonconvergence(self):
        for name in ['runner.log','wannier90.wout','wannier90.node_00000.werr']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/name).write_text('kmesh_get_bvector: Not enough bvectors found\n')
                self.assertEqual(classify_failure(root,1),'NEIGHBOUR_SHELL_FAILURE')

    def test_retry_does_not_expand_energy_window_or_iterations(self):
        with self.assertRaisesRegex(WorkflowBlocked,'search_shells'):
            choose_retry_reason(['NEIGHBOUR_SHELL_FAILURE'],set())
        self.assertIn('preprocessing',FAILURE_TRANSITIONS['NEIGHBOUR_SHELL_FAILURE'])

    def test_other_numerical_failures_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runner.log').write_text('unrelated numerical error')
            self.assertEqual(classify_failure(root,1),'NUMERIC_FAILURE')
