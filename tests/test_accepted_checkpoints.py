from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from w90flow.checkpoints import checkpoint_accepted_candidate
from w90flow.orchestrator import _run_candidate_search
from w90flow.state import sha256_file


def accepted(root: Path, sid: str = "sheet", candidate: str = "first") -> Path:
    source = root / "structures" / sid / "04_wannier_attempts" / candidate / "attempt-001"
    source.mkdir(parents=True)
    for name in ["POSCAR", "wannier90.win", "wannier90_hr.dat", "wannier90_centres.xyz", "wannier90.wout"]:
        (source / name).write_text("synthetic " + name)
    band = root / "structures" / sid / "02_band" / "EIGENVAL"
    band.parent.mkdir(exist_ok=True)
    band.write_text("synthetic independent DFT path")
    quality = {"passed": True, "structure": sid, "candidate": candidate, "attempt": "attempt-001",
               "checks": {"w90_completed": True, "hr_centres_match": True, "rmse": True,
                          "p95": True, "max": True, "gap": True},
               "source_sha256": {"EIGENVAL": sha256_file(band),
                                 **{n: sha256_file(source / n) for n in ["wannier90_hr.dat", "wannier90.wout"]}}}
    (source / "quality.json").write_text(json.dumps(quality))
    (source / "attempt.json").write_text(json.dumps({"status": "ACCEPTED"}))
    return source


class AcceptedCheckpointTests(unittest.TestCase):
    def test_later_candidate_crash_preserves_first_without_marking_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted(root)
            config = {"structures": [{"id": "sheet"}], "backend": {"mpi_processes": 1}}
            spec = {"ensemble_mode": "independent", "independent_candidates": {
                "sheet": [{"candidate": "first"}, {"candidate": "second"}]}}
            with patch("w90flow.orchestrator.detect_resources", return_value={"available_cpus": 1}), \
                 patch("w90flow.orchestrator._process_candidate", side_effect=[
                     {"passed": True, "candidate": "first", "attempt": "attempt-001"},
                     RuntimeError("second candidate interrupted")]):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    _run_candidate_search(root, config, spec)
            packages = list((root / "accepted-checkpoints").glob("*/checkpoint.json"))
            self.assertEqual(len(packages), 1)
            manifest = json.loads(packages[0].read_text())
            self.assertFalse(manifest["workflow_complete"])
            self.assertFalse(manifest["ranked_export"])
            self.assertFalse(manifest["nlo_accepted"])
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(sha256_file(packages[0].parent / name), digest)
            self.assertFalse((root / "07_export").exists())

    def test_changed_audited_model_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = accepted(root)
            (source / "wannier90_hr.dat").write_text("different HR")
            with self.assertRaisesRegex(ValueError, "audited source changed"):
                checkpoint_accepted_candidate(root, ["sheet"], "first", "attempt-001")
            self.assertFalse((root / "accepted-checkpoints").exists())

    def test_repeated_checkpoint_is_verified_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted(root)
            first = checkpoint_accepted_candidate(root, ["sheet"], "first", "attempt-001")
            self.assertEqual(first, checkpoint_accepted_candidate(root, ["sheet"], "first", "attempt-001"))
            (Path(first["path"]) / "sheet/POSCAR").write_text("modified package")
            with self.assertRaisesRegex(ValueError, "checkpoint content changed"):
                checkpoint_accepted_candidate(root, ["sheet"], "first", "attempt-001")

    def test_common_basis_requires_each_structure_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted(root, "BL")
            source = accepted(root, "ML")
            (source / "attempt.json").write_text(json.dumps({"status": "REJECTED_QUALITY"}))
            with self.assertRaisesRegex(ValueError, "not quality-accepted"):
                checkpoint_accepted_candidate(root, ["BL", "ML"], "first", "attempt-001")
            self.assertFalse((root / "accepted-checkpoints").exists())
