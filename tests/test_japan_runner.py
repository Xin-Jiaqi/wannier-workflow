from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from w90flow.japan_runner import (
    _campaign_signature,
    campaign_manifest,
    derive_material_inputs,
    load_cases_manifest,
    material_preflight,
    render_stage1_pbs,
    resolve_under_root,
    site_qstat_active,
    site_qsub,
    submit_material,
    validate_material_id,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SITE = REPOSITORY / "tests/fixtures/site"
MANIFEST = SITE / "input-lock.json"
POOL = SITE / "pool"
RELEASE_COMMIT = "f00dcafe00000000000000000000000000000000"


class MaterialAuthTests(unittest.TestCase):
    def _preflight(self, material_id: str, work_root: Path, **kwargs: object) -> dict:
        return material_preflight(
            material_id=material_id,
            manifest_path=MANIFEST,
            pool_root=POOL,
            work_root=work_root,
            release_root=SITE,
            expected_release_commit=RELEASE_COMMIT,
            **kwargs,
        )

    def test_legal_material_derives_controlled_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._preflight("LiAuI4_BL", Path(temporary))
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["material_id"], "LiAuI4_BL")
            config = Path(report["config_path"])
            poscar = Path(report["poscar_path"])
            self.assertEqual(config, (POOL / "coldstart/LiAuI4_BL/stage1.yaml").resolve())
            self.assertEqual(poscar, (POOL / "structures/LiAuI4/bilayer/POSCAR").resolve())
            manifest = load_cases_manifest(MANIFEST)
            self.assertEqual(report["config_sha256"], manifest["LiAuI4_BL"]["config_sha256"])
            self.assertEqual(report["poscar_sha256"], manifest["LiAuI4_BL"]["poscar_sha256"])
            self.assertEqual(report["release_commit"], RELEASE_COMMIT)
            self.assertIsNone(report["existing_campaign"])
            self.assertEqual(report["active_jobs"], [])

    def test_any_manifest_material_passes_without_special_cases(self) -> None:
        for material_id in ("LiAuI4_BL", "PdS2_BL", "ZrI2_BL", "X2Y_BL"):
            with self.subTest(material=material_id):
                with tempfile.TemporaryDirectory() as temporary:
                    report = self._preflight(material_id, Path(temporary))
                    self.assertTrue(report["passed"], report["errors"])

    def test_unknown_material_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._preflight("NotInManifest_BL", Path(temporary))
            self.assertFalse(report["passed"])
            self.assertTrue(any("不在受控 cases manifest" in error for error in report["errors"]))

    def test_invalid_material_id_rejected(self) -> None:
        for bad in ("LiAuI4_BL;rm -rf /", "../LiAuI4_BL", "Li AuI4", "", "LiAuI4$HOME"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_material_id(bad)
                with tempfile.TemporaryDirectory() as temporary:
                    report = self._preflight(bad, Path(temporary))
                    self.assertFalse(report["passed"])

    def test_path_escape_from_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._preflight("EVIL_BL", Path(temporary))
            self.assertFalse(report["passed"])
            self.assertTrue(any("越界" in error for error in report["errors"]))

    def test_resolve_under_root_rejects_escape_and_absolute_foreign_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "越界"):
                resolve_under_root(root, root / ".." / "outside")
            with self.assertRaisesRegex(ValueError, "越界"):
                resolve_under_root(root, "/etc/passwd")
            inside = root / "a/b"
            inside.mkdir(parents=True)
            self.assertEqual(resolve_under_root(root, inside), inside.resolve())

    def test_hash_mismatch_is_rejected(self) -> None:
        manifest = load_cases_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temporary:
            pool = Path(temporary) / "pool"
            config = pool / "coldstart/LiAuI4_BL/stage1.yaml"
            config.parent.mkdir(parents=True)
            original = (POOL / "coldstart/LiAuI4_BL/stage1.yaml").read_text(encoding="utf-8")
            config.write_text(original + "# tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "配置 SHA256 与 manifest 不一致"):
                derive_material_inputs(pool, "LiAuI4_BL", manifest)


class PbsTemplateTests(unittest.TestCase):
    def _render(self, **overrides: object) -> str:
        kwargs = {
            "material_id": "LiAuI4_BL",
            "release_root": SITE,
            "pool_root": POOL,
            "work_root": "/tmp/work",
            "python": "/usr/bin/python3",
            "poscar_path": POOL / "structures/LiAuI4/bilayer/POSCAR",
            "config_path": POOL / "coldstart/LiAuI4_BL/stage1.yaml",
            "dry_run": False,
            "resume": False,
            "poll_seconds": 60.0,
        }
        kwargs.update(overrides)
        return render_stage1_pbs(**kwargs)

    def test_template_invokes_run_material_not_stage1_coldstart(self) -> None:
        script = self._render()
        self.assertIn("run-material", script)
        self.assertNotIn("stage1-coldstart", script)
        self.assertNotIn("qsub", script)

    def test_resources_are_fixed_site_standard(self) -> None:
        script = self._render()
        self.assertIn("select=1:ncpus=32:mpiprocs=32:ompthreads=1:mem=128gb", script)
        self.assertIn("walltime=24:00:00", script)
        self.assertIn('test "$(wc -l < "$PBS_NODEFILE")" -ge 32', script)

    def test_dry_run_and_resume_flags(self) -> None:
        dry = self._render(dry_run=True)
        self.assertIn("DRY_RUN=1", dry)
        self.assertIn("RESUME=0", dry)
        self.assertIn("--dry-run", dry)
        resumed = self._render(resume=True)
        self.assertIn("RESUME=1", resumed)
        self.assertIn("DRY_RUN=0", resumed)
        self.assertIn("--resume", resumed)
        real = self._render()
        self.assertIn("DRY_RUN=0", real)
        self.assertIn("RESUME=0", real)

    def test_rendered_script_passes_bash_syntax_check(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available")
        script = self._render()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage1.pbs"
            path.write_text(script, encoding="utf-8")
            completed = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_manifest_self_check_inside_script_uses_lock_not_whitelist(self) -> None:
        script = self._render()
        self.assertIn("CONTROL/input-lock.json", script)
        self.assertIn('material_id not in lock.get("cases", {})', script)
        self.assertNotIn("BiI3_BL|SnO_BL|ZrI2_BL|NbO2I_BL", script)


class SubmitMaterialTests(unittest.TestCase):
    def _submit(self, work_root: Path, **overrides: object) -> dict:
        kwargs = {
            "material_id": "LiAuI4_BL",
            "release_root": SITE,
            "work_root": work_root,
            "manifest_path": MANIFEST,
            "pool_root": POOL,
            "expected_release_commit": RELEASE_COMMIT,
        }
        kwargs.update(overrides)
        return submit_material(**kwargs)

    def test_dry_run_creates_no_state_and_no_qsub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            result = self._submit(work, dry_run=True)
            self.assertEqual(result["mode"], "DRY_RUN")
            self.assertTrue(result["passed"])
            self.assertIn("run-material", result["script"])
            self.assertIn("DRY_RUN=1", result["script"])
            self.assertIn("--dry-run", result["script"])
            self.assertFalse(work.exists())

    def test_real_submit_writes_script_and_qsubs_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            qsub = mock.Mock(return_value={"job_id": "77777.japan", "output": "77777.japan"})
            result = self._submit(
                work,
                active_check=lambda material: [],
                qsub=qsub,
            )
            self.assertEqual(result["mode"], "SUBMITTED")
            self.assertEqual(result["job_id"], "77777.japan")
            qsub.assert_called_once()
            script_path = Path(result["pbs_path"])
            self.assertTrue(script_path.is_file())
            self.assertTrue(Path(result["spec_path"]).is_file())
            spec = json.loads(Path(result["spec_path"]).read_text(encoding="utf-8"))
            self.assertEqual(spec["material_id"], "LiAuI4_BL")
            self.assertEqual(spec["poscar_sha256"], result["poscar_sha256"])
            script = script_path.read_text(encoding="utf-8")
            self.assertIn("DRY_RUN=0", script)
            self.assertIn("RESUME=0", script)

    def test_active_duplicate_blocks_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            qsub = mock.Mock(return_value={"job_id": "99999.japan"})
            result = self._submit(
                work,
                active_check=lambda material: ["12345.japan"],
                qsub=qsub,
            )
            self.assertEqual(result["mode"], "PREFLIGHT_FAILED")
            self.assertFalse(result["passed"])
            self.assertTrue(any("active duplicate" in error for error in result["errors"]))
            qsub.assert_not_called()

    def test_existing_terminal_campaign_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            campaign = work / "campaigns/LiAuI4_BL"
            campaign.mkdir(parents=True)
            manifest = load_cases_manifest(MANIFEST)
            inputs = derive_material_inputs(POOL, "LiAuI4_BL", manifest)
            signature = _campaign_signature(POOL, "LiAuI4_BL", inputs)
            (campaign / "run-material-manifest.json").write_text(
                json.dumps({
                    "status": "TOP3_READY",
                    "terminal": True,
                    "input_signature_sha256": signature,
                }),
                encoding="utf-8",
            )
            report = material_preflight(
                material_id="LiAuI4_BL",
                manifest_path=MANIFEST,
                pool_root=POOL,
                work_root=work,
                release_root=SITE,
            )
            self.assertFalse(report["passed"])
            self.assertTrue(any("--resume" in error for error in report["errors"]))
            resumed = material_preflight(
                material_id="LiAuI4_BL",
                manifest_path=MANIFEST,
                pool_root=POOL,
                work_root=work,
                release_root=SITE,
                resume=True,
            )
            self.assertTrue(resumed["passed"], resumed["errors"])

    def test_campaign_signature_mismatch_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            campaign = work / "campaigns/LiAuI4_BL"
            campaign.mkdir(parents=True)
            (campaign / "run-material-manifest.json").write_text(
                json.dumps({
                    "status": "STARTED",
                    "terminal": False,
                    "input_signature_sha256": "deadbeef",
                }),
                encoding="utf-8",
            )
            report = material_preflight(
                material_id="LiAuI4_BL",
                manifest_path=MANIFEST,
                pool_root=POOL,
                work_root=work,
                release_root=SITE,
                resume=True,
            )
            self.assertFalse(report["passed"])
            self.assertTrue(any("指纹" in error for error in report["errors"]))
            self.assertIsNotNone(campaign_manifest(work, "LiAuI4_BL"))


class SiteCommandTests(unittest.TestCase):
    def test_site_qstat_active_parses_matching_job_names(self) -> None:
        output = (
            "Job id            Name             User              Time Use S Queue\n"
            "12345.japan       xjq-stage1-ZrI2_BL sunq 00:01:00 R CP_001\n"
            "12346.japan       xjq-stage1-LiAuI4_BL sunq 00:00:30 Q CP_001\n"
        )
        with mock.patch("w90flow.japan_runner.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=output, stderr="")
            self.assertEqual(site_qstat_active("LiAuI4_BL"), ["12346.japan"])
            self.assertEqual(site_qstat_active("ZrI2_BL"), ["12345.japan"])
            self.assertEqual(site_qstat_active("PdS2_BL"), [])

    def test_site_qsub_returns_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "job.pbs"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            with mock.patch("w90flow.japan_runner.subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0, stdout="424242.japan\n", stderr="")
                result = site_qsub(script)
            self.assertEqual(result["job_id"], "424242.japan")
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
