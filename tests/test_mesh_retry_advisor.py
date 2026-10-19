import math
import unittest
from w90flow.mesh_retry_advisor import propose_mesh_retry

class MeshRetryTests(unittest.TestCase):
    def kwargs(self):
        return dict(converged=True,
                    metrics=dict(rmse_mev=6.7,p95_mev=10.7,max_mev=57.3,gap_error_mev=-.8),
                    gates=dict(rmse_meV=10,p95_meV=20,max_meV=50,gap_error_meV=10),
                    training_max_mev=.01,mesh=[9,9,1],periodic_axes=[True,True,False],retries_used=0)
    def test_isolated_error_preserves_nonperiodic_axis_and_acceptance(self):
        k=self.kwargs();r=propose_mesh_retry(**k)
        self.assertEqual(r['mesh'],[11,11,1]);self.assertFalse(r['acceptance_changed'])
        self.assertEqual(k['mesh'],[9,9,1]);self.assertEqual(k['metrics']['max_mev'],57.3)
    def test_unconverged_model_is_not_sent_to_mesh_refinement(self):
        k=self.kwargs();k['converged']=False
        self.assertEqual(propose_mesh_retry(**k)['reason'],'OPTIMIZATION_NOT_CONVERGED')
    def test_no_retry_if_already_passed_or_gap_fails(self):
        k=self.kwargs();k['metrics']['max_mev']=28.4
        self.assertEqual(propose_mesh_retry(**k)['reason'],'BAND_GATES_ALREADY_PASS')
        k['metrics']['max_mev']=57.3;k['metrics']['gap_error_mev']=-11
        self.assertEqual(propose_mesh_retry(**k)['reason'],'ERROR_NOT_ISOLATED_TO_MAXIMUM')
    def test_budget_and_training_reproduction(self):
        k=self.kwargs();k['retries_used']=1
        self.assertEqual(propose_mesh_retry(**k)['reason'],'MESH_RETRY_BUDGET_EXHAUSTED')
        k['retries_used']=0;k['training_max_mev']=2
        self.assertEqual(propose_mesh_retry(**k)['reason'],'TRAINING_SPECTRUM_NOT_REPRODUCED')
    def test_nonfinite_metrics_fail_closed(self):
        k=self.kwargs();k['metrics']['max_mev']=math.nan
        self.assertEqual(propose_mesh_retry(**k)['reason'],'NONFINITE_DIAGNOSTIC')
    def test_arbitrary_periodic_axes(self):
        k=self.kwargs();k['mesh']=[1,9,9];k['periodic_axes']=[False,True,True]
        self.assertEqual(propose_mesh_retry(**k)['mesh'],[1,11,11])
    def test_invalid_inputs(self):
        for change in [{'mesh':[9.0,9,1]},{'periodic_axes':[1,1,0]}, {'converged':'false'}, {'retries_used':-1}]:
            with self.subTest(change=change),self.assertRaises(ValueError):propose_mesh_retry(**(self.kwargs()|change))
    def test_finite_mesh_budget(self):
        k=self.kwargs();k['mesh']=[15,15,1]
        self.assertEqual(propose_mesh_retry(**k)['reason'],'MESH_SIZE_BUDGET_EXHAUSTED')

if __name__=='__main__':unittest.main()
