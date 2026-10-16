import unittest
from w90flow.attempts import target_occupation_coverage
from w90flow.windows import suggest_occupation_complete_window

class WindowProposalTests(unittest.TestCase):
    def test_missing_empty_sector_has_explicit_separate_protocol_proposal(self):
        energies=[[-10.]*21+[-.235815,5.858886]]
        result=target_occupation_coverage(energies,44,False,-6,4)
        self.assertFalse(result['passed'])
        self.assertEqual(result['target_energy_eV'],[-6,4])
        proposal=result['separate_window_proposal']
        self.assertTrue(proposal['requires_new_protocol'])
        self.assertAlmostEqual(proposal['target_energy_eV'][1],6.358886)
        low,high=proposal['target_energy_eV']
        self.assertTrue(target_occupation_coverage(energies,44,False,low,high)['passed'])

    def test_already_valid_window_does_not_change(self):
        self.assertIsNone(target_occupation_coverage([[-2.,-.2,.8,5.]],4,False,-1,2)['separate_window_proposal'])

    def test_proposal_is_energy_translation_covariant(self):
        e=[[-3.,-.2,5.8],[-2.9,-.1,6.]]
        a=suggest_occupation_complete_window(e,2,-6,4)['target_energy_eV']
        b=suggest_occupation_complete_window([[v+100 for v in row] for row in e],2,94,104)['target_energy_eV']
        for x,y in zip(a,b):self.assertAlmostEqual(x+100,y)

    def test_no_integer_band_edge_proposal_for_metals_or_missing_conduction(self):
        for e in [[[-2.,1.],[0.,-.1]],[[-2.],[-1.]]]:
            self.assertIsNone(suggest_occupation_complete_window(e,1,-6,4))
        self.assertIsNone(target_occupation_coverage([[-2.,-.1,5.]],3,False,-6,4)['separate_window_proposal'])
