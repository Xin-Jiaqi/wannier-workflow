import unittest
from w90flow.neighbours import select_stable_neighbour_proposal

class NeighbourProposalTests(unittest.TestCase):
    def probe(self,tol,topology=None,passed=True):
        return {'kmesh_tol':tol,'shells':36,'passed':passed,'nnkp_topology':topology or ['1','1','1','0','0','1']}

    def test_actual_c8f4_sweep_selects_first_of_two_stable_successes(self):
        p=[self.probe(1e-6,passed=False),self.probe(1e-7),self.probe(1e-8)]
        a=select_stable_neighbour_proposal(p)
        self.assertEqual(a['kmesh_tol'],1e-7)
        self.assertEqual(a['verification_tolerance'],1e-8)
        self.assertEqual(a['search_shells'],36)

    def test_single_success_or_changed_topology_is_not_a_stable_proposal(self):
        self.assertIsNone(select_stable_neighbour_proposal([self.probe(1e-7)]))
        self.assertIsNone(select_stable_neighbour_proposal([self.probe(1e-7),self.probe(1e-8,['1','1','1','1','0','0'])]))

    def test_failure_breaks_adjacency_and_repeated_tolerance_does_not_verify(self):
        self.assertIsNone(select_stable_neighbour_proposal([self.probe(1e-7),self.probe(1e-8,passed=False),self.probe(1e-9)]))
        self.assertIsNone(select_stable_neighbour_proposal([self.probe(1e-7),self.probe(1e-7)]))

    def test_malformed_success_is_rejected(self):
        with self.assertRaises(ValueError):
            select_stable_neighbour_proposal([self.probe(1e-7,['1','1','1'])])
