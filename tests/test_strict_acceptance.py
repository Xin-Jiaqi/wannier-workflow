"""Reject normal termination without positive, stage-specific convergence evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from w90flow.attempts import import_completed_attempt
from w90flow.parsers import convergence_gate, parse_wout
from w90flow.quality import evaluate_attempt

LOCAL = 'Wannierisation convergence criteria satisfied\n'
DIS = 'Disentanglement convergence criteria satisfied\n'
DONE = 'All done: wannier90 exiting\n'


class StrictAcceptance(unittest.TestCase):
    def check(self, text, expected, dis=True):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'sample.wout'
            path.write_text(text)
            self.assertEqual(convergence_gate(parse_wout(path), require_disentanglement=dis), expected)

    def test_normal_exit_is_not_convergence(self):
        self.check(DONE, 'CONVERGENCE_UNVERIFIED')

    def test_explicit_dis_failure(self):
        self.check('Disentanglement convergence criteria not satisfied\n' + LOCAL + DONE,
                   'DISENTANGLEMENT_NOT_CONVERGED')

    def test_missing_dis_requires_verification(self):
        self.check(LOCAL + DONE, 'CONVERGENCE_UNVERIFIED')

    def test_exact_does_not_need_dis(self):
        self.check(LOCAL + DONE, None, dis=False)

    def test_both_success(self):
        self.check(DIS + LOCAL + DONE, None)

    def test_incomplete_output(self):
        self.check(LOCAL, 'W90_NOT_COMPLETED', dis=False)

    def test_local_failure(self):
        self.check(DIS + 'Wannierization convergence criteria not satisfied\n' + DONE,
                   'LOCALIZATION_NOT_CONVERGED')

    def test_restarted_log_uses_latest_status(self):
        self.check('Disentanglement convergence criteria not satisfied\n' + DIS + LOCAL + DONE, None)
        self.check(DIS + 'Disentanglement convergence criteria not satisfied\n' + LOCAL + DONE,
                   'DISENTANGLEMENT_NOT_CONVERGED')

    def test_quality_rejects_before_reading_any_dft(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / 'structures/X/04_wannier_attempts/p/attempt-001'
            attempt.mkdir(parents=True)
            (root / 'project.snapshot.yaml').write_text('{}')
            (attempt / 'attempt.json').write_text(json.dumps({
                'status': 'PASS', 'plan': {'mode': 'disentanglement'}}))
            (attempt / 'wannier90.wout').write_text('Disentanglement convergence criteria not satisfied\n' + DONE)
            result = evaluate_attempt(root, 'X', 'p', 'attempt-001')
            self.assertFalse(result['passed'])
            self.assertEqual(result['failure_class'], 'DISENTANGLEMENT_NOT_CONVERGED')

    def test_reuse_rejects_unconverged_source_before_copying_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            target = root / 'structures/X/04_wannier_attempts/p/attempt-001'
            source = Path(temporary) / 'source'
            target.mkdir(parents=True)
            source.mkdir()
            (target / 'attempt.json').write_text(json.dumps({
                'status': 'READY', 'num_wann': 2, 'plan': {'mode': 'disentanglement'}}))
            for name in ('POSCAR', 'POTCAR', 'INCAR', 'KPOINTS', 'CHGCAR'):
                (target / name).write_text('synthetic identical input')
                (source / name).write_text('synthetic identical input')
            for name in ('wannier90.win', 'wannier90_hr.dat', 'wannier90_centres.xyz', 'wannier90_band.dat'):
                (source / name).write_text('synthetic output; must never be read or copied')
            (source / 'wannier90.wout').write_text(DONE)
            with self.assertRaisesRegex(ValueError, 'CONVERGENCE_UNVERIFIED'):
                import_completed_attempt(root, 'X', 'p', 'attempt-001', source)
            self.assertFalse((target / 'wannier90_hr.dat').exists())
            self.assertEqual(json.loads((target / 'attempt.json').read_text())['status'], 'READY')


if __name__ == '__main__':
    unittest.main()
