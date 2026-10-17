"""Regression for GaAs +0.5/-0.5 representatives, not a cell-axis rotation."""
import tempfile
import unittest
from pathlib import Path
from w90flow.attempts import reconcile_mmn_neighbour_cells


class RepresentativeTests(unittest.TestCase):
    def fixture(self, root):
        mmn, nnkp, win, ibz = [root / n for n in ['x.mmn', 'x.nnkp', 'x.win', 'IBZKPT']]
        mmn.write_text('overlaps\n1 2 1\n1 2 0 0 0\n0.25 -0.75\n2 1 0 0 0\n0.25 0.75\n')
        nnkp.write_text('begin nnkpts\n1\n1 2 1 0 0\n2 1 -1 0 0\nend nnkpts\n')
        win.write_text('begin kpoints\n0 0 0\n-0.5 0 0\nend kpoints\n')
        ibz.write_text('mesh\n2\nReciprocal\n0 0 0 1\n0.5 0 0 1\n')
        return mmn, nnkp, win, ibz

    def test_boundary_representatives_preserve_complex_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            mmn, nnkp, win, ibz = self.fixture(Path(tmp))
            before = mmn.read_text().splitlines()
            result = reconcile_mmn_neighbour_cells(mmn, nnkp, win_path=win, ibzkpt_path=ibz)
            after = mmn.read_text().splitlines()
            self.assertEqual(result['mode'], 'ibzkpt_to_win_integer_translation')
            self.assertEqual(result['translated_kpoints'], 1)
            self.assertEqual(result['changed_headers'], 2)
            self.assertEqual(after[3::2], before[3::2])
            self.assertEqual(tuple(map(int, after[2].split())), (1, 2, 1, 0, 0))
            self.assertEqual(tuple(map(int, after[4].split())), (2, 1, -1, 0, 0))
            self.assertTrue(reconcile_mmn_neighbour_cells(mmn, nnkp)['matched_exactly'])

    def test_no_coordinate_evidence_never_guesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            mmn, nnkp, _, _ = self.fixture(Path(tmp))
            before = mmn.read_bytes()
            with self.assertRaises(ValueError):
                reconcile_mmn_neighbour_cells(mmn, nnkp)
            self.assertEqual(mmn.read_bytes(), before)

    def test_different_indexed_kpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            mmn, nnkp, win, ibz = self.fixture(Path(tmp))
            ibz.write_text('mesh\n2\nReciprocal\n0 0 0 1\n0.25 0 0 1\n')
            before = mmn.read_bytes()
            with self.assertRaisesRegex(ValueError, '整倒格矢'):
                reconcile_mmn_neighbour_cells(mmn, nnkp, win_path=win, ibzkpt_path=ibz)
            self.assertEqual(mmn.read_bytes(), before)

    def test_partial_neighbour_agreement_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            mmn, nnkp, win, ibz = self.fixture(Path(tmp))
            nnkp.write_text('begin nnkpts\n1\n1 2 1 0 0\n2 1 -2 0 0\nend nnkpts\n')
            before = mmn.read_bytes()
            with self.assertRaises(ValueError):
                reconcile_mmn_neighbour_cells(mmn, nnkp, win_path=win, ibzkpt_path=ibz)
            self.assertEqual(mmn.read_bytes(), before)
