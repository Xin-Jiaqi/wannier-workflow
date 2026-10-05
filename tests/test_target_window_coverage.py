import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from w90flow.attempts import create_attempts, target_occupation_coverage


class TargetWindowCoverageTests(unittest.TestCase):
    def test_wide_gap_missing_empty_sector_is_not_convergence_failure(self):
        # Measured CaClF band edges, padded to its 22 occupied bands.
        energies = [[-10.] * 21 + [-0.235815, 5.858886]]
        result = target_occupation_coverage(energies, 44, False, -6, 4)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failure_class'], 'TARGET_WINDOW_MISSING_OCCUPATION_SECTOR')
        self.assertEqual(result['target_empty_band_count'], 0)
        self.assertEqual(result['target_energy_eV'], [-6, 4])

    def test_valid_semiconductor_and_spinor_counting(self):
        energies = [[-2., -.2, .8, 5.], [-1.8, -.1, 1., 5.1]]
        for soc, nelect in [(False, 4), (True, 2)]:
            result = target_occupation_coverage(energies, nelect, soc, -1, 2)
            self.assertTrue(result['passed'])
            self.assertEqual(result['target_occupied_band_count'], 1)
            self.assertEqual(result['target_empty_band_count'], 1)

    def test_empty_only_and_no_bands_are_rejected(self):
        energies = [[-2., -.2, .8, 5.]]
        for bounds in [(0, 2), (8, 9)]:
            self.assertFalse(target_occupation_coverage(energies, 4, False, *bounds)['passed'])

    def test_create_attempts_stops_before_procar_or_attempt_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'structures' / 'widegap'
            (base / '02_band').mkdir(parents=True)
            (base / '02_band/EIGENVAL').write_text('source fixture')
            config = {'structures': [{'id': 'widegap'}], 'physics': {'soc': False}}
            spec = {'target_energy_eV': [-6, 4], 'energy_reference': 'fermi_relative',
                    'ensemble_mode': 'independent'}
            eig = {'nelect': 2, 'nbands': 2, 'energies_eV': [[-5.2, .9]]}
            with patch('w90flow.attempts._load_snapshot', return_value=config), \
                 patch('w90flow.attempts.read_json', return_value=spec), \
                 patch('w90flow.attempts.parse_poscar', return_value={}), \
                 patch('w90flow.attempts.parse_eigenval', return_value=eig), \
                 patch('w90flow.attempts.parse_fermi_outcar', return_value=-5.), \
                 patch('w90flow.attempts.parse_procar_orbital_scores') as procar:
                with self.assertRaisesRegex(ValueError, 'TARGET_WINDOW_MISSING_OCCUPATION_SECTOR'):
                    create_attempts(root)
                procar.assert_not_called()
            diagnostic = json.loads((base / 'target-window-diagnostic.json').read_text())
            self.assertFalse(diagnostic['passed'])
            self.assertEqual(len(diagnostic['band_eigenval_sha256']), 64)
            self.assertFalse((base / '04_wannier_attempts').exists())


if __name__ == '__main__':
    unittest.main()
