import unittest
from w90flow.windows import suggest_fixed_deep_band_exclusion


class DeepBandProposalTests(unittest.TestCase):
    def energies(self):
        row = [-26 + i for i in range(14)] + [-10.3 + .35*i for i in range(16)] + [.06 + .4*i for i in range(30)]
        return [row, [e + .01 for e in row]]

    def test_kpo_like_block_preserves_target_and_constant_occupation(self):
        rows = self.energies()
        p = suggest_fixed_deep_band_exclusion(rows, 30, 40, -10.27, 3.23)
        self.assertEqual(p['exclude_bands_1_based'], list(range(1, 15)))
        self.assertEqual(p['retained_occupied_bands'], 16)
        self.assertEqual(p['input_target_energy_eV'], [-10.27, 3.23])
        for row in rows:
            self.assertTrue(all(e < -10.27 for e in row[:14]))
            lo, hi = p['outer_window_eV']
            self.assertGreaterEqual(sum(lo <= e <= hi for e in row[14:]), 40)
            lo, hi = p['frozen_window_eV']
            self.assertTrue(all(lo <= e <= hi for e in row[14:30]))
        self.assertTrue(p['requires_new_protocol'])

    def test_energy_origin_shift_is_covariant(self):
        rows = self.energies(); p = suggest_fixed_deep_band_exclusion(rows, 30, 40, -10.27, 3.23)
        q = suggest_fixed_deep_band_exclusion([[e+5 for e in row] for row in rows], 30, 40, -5.27, 8.23)
        self.assertEqual(p['exclude_bands_1_based'], q['exclude_bands_1_based'])
        for a,b in zip(p['outer_window_eV'], q['outer_window_eV']):self.assertAlmostEqual(a+5,b)

    def test_unsafe_models_and_inconsistent_inputs_decline(self):
        rows = self.energies()
        self.assertIsNone(suggest_fixed_deep_band_exclusion(rows, 30, 10, -10.27, 3.23))
        self.assertIsNone(suggest_fixed_deep_band_exclusion(rows, 30, 40, -100, 3.23))
        self.assertIsNone(suggest_fixed_deep_band_exclusion([rows[0],rows[1][:-1]], 30, 40, -10.27, 3.23))
        rows[1][30] = -20
        self.assertIsNone(suggest_fixed_deep_band_exclusion(rows, 30, 40, -10.27, 3.23))
