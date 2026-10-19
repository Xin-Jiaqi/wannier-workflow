"""A flat remote deployment must not launch an older installed w90flow."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from w90flow import supervisor


class ImportPathTests(unittest.TestCase):
    def test_flat_baseline_precedes_stale_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve() / 'baseline'
            module = base / 'w90flow/supervisor.py'
            module.parent.mkdir(parents=True)
            module.touch()
            with patch.object(supervisor, '__file__', str(module)), patch.dict(os.environ, {'PYTHONPATH': '/old/baseline'}):
                _, env, _ = supervisor._command(Path(tmp), 'wannier')
            self.assertEqual(env['PYTHONPATH'].split(os.pathsep), [str(base), '/old/baseline'])

    def test_repository_source_layout_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            module = root / 'src/w90flow/supervisor.py'
            module.parent.mkdir(parents=True)
            module.touch()
            with patch.object(supervisor, '__file__', str(module)), patch.dict(os.environ, {}, clear=True):
                _, env, cwd = supervisor._command(root, 'wannier')
            self.assertEqual(env['PYTHONPATH'], str(root / 'src'))
            self.assertEqual(cwd, root)
