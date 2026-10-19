from __future__ import annotations
import unittest
from w90flow.candidates import candidate_dimension_budget, suggest_projection_candidates


class CandidateBudgetTests(unittest.TestCase):
    def test_sip2_complete_sp_is_excluded_by_fraction_not_band_count(self):
        scores = {'Si:s': .04, 'Si:p': .29, 'P:s': .12, 'P:p': .55}
        all_candidates = suggest_projection_candidates(scores, {'Si':4,'P':8}, False)
        report = candidate_dimension_budget(all_candidates, 56, .85)
        self.assertEqual(report['maximum_num_wann'], 47)
        full = next(c for c in report['excluded_candidates'] if c['num_wann']==48)
        self.assertEqual(full['reason'], 'fraction_cap_only')
        self.assertEqual(full['minimum_nbands_at_current_fraction'], 57)
        capped = suggest_projection_candidates(scores, {'Si':4,'P':8}, False, max_num_wann=47)
        self.assertTrue(all(c['num_wann']<=47 for c in capped))
        self.assertTrue(any(c['num_wann']==48 for c in all_candidates))

    def test_insufficient_dft_bands_is_separate_from_optional_cap(self):
        c=[{'candidate':'full','projection':['X:p'],'num_wann':60}]
        self.assertEqual(candidate_dimension_budget(c,56,.85)['excluded_candidates'][0]['reason'], 'insufficient_dft_bands')
        self.assertEqual(candidate_dimension_budget(c,80,.85)['excluded_candidates'], [])

    def test_invalid_budgets_are_rejected(self):
        for bands, fraction in [(0,.85),(True,.85),(56,0),(56,1.1),(56,float('nan'))]:
            with self.subTest(bands=bands,fraction=fraction), self.assertRaises(ValueError):
                candidate_dimension_budget([],bands,fraction)
