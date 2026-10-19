import tempfile
import unittest
from pathlib import Path
from w90flow.acceptance import REQUIRED_CHECKS, require_audited_sources, strict_quality_passed
from w90flow.state import sha256_file

class AcceptanceContractTests(unittest.TestCase):
    def test_historical_pass_cannot_replace_convergence_audit(self):
        record = {'passed': True, 'checks': {key: True for key in REQUIRED_CHECKS if key != 'w90_converged'}}
        self.assertFalse(strict_quality_passed(record))
        record['checks']['w90_converged'] = False
        self.assertFalse(strict_quality_passed(record))
        record['checks']['w90_converged'] = True
        self.assertTrue(strict_quality_passed(record))
        record['checks']['additional_physics_check'] = False
        self.assertFalse(strict_quality_passed(record))

    def test_export_requires_same_audited_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wannier90.wout'
            path.write_text('audited output')
            record = {'passed': True, 'checks': {key: True for key in REQUIRED_CHECKS},
                      'source_sha256': {'wout': sha256_file(path)}}
            require_audited_sources(record, {'wout': path})
            path.write_text('overwritten output')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                require_audited_sources(record, {'wout': path})
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing audited'):
                require_audited_sources(record, {'wout': path})
