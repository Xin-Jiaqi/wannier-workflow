"""Stage-1 child Python pinning: no bare ``python3`` anywhere in the chain.

Covers supervisor child commands, the Japan PBS template, the run-material /
supervise CLI flags, and the doctor's python-consistency report.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from w90flow import supervisor
from w90flow.cli import build_parser
from w90flow.doctor import run_doctor
from w90flow.japan_runner import render_stage1_pbs, submit_material

REPOSITORY = Path(__file__).resolve().parents[1]
SITE = REPOSITORY / "tests/fixtures/site"
MANIFEST = SITE / "input-lock.json"
POOL = SITE / "pool"
RELEASE_COMMIT = "f00dcafe00000000000000000000000000000000"


class SupervisorChildPythonTests(unittest.TestCase):
    def test_child_command_uses_sys_executable_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command, _environment, _cwd = supervisor._command(Path(temporary), "scf")
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1:4], ["-m", "w90flow", "resume"])
            self.assertNotIn("python3", command)

    def test_child_command_honours_explicit_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command, _environment, _cwd = supervisor._command(
                Path(temporary), "band", python="/opt/controlled/python"
            )
            self.assertEqual(command[0], "/opt/controlled/python")
            self.assertNotIn("python3", command)

    def test_broken_path_python3_never_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trap = Path(temporary)
            decoy = trap / "python3"
            decoy.write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
            decoy.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": str(trap)}):
                command, _environment, _cwd = supervisor._command(Path(temporary) / "run", "wannier")
            self.assertEqual(command[0], sys.executable)


class PbsPythonTests(unittest.TestCase):
    def _render(self, **overrides: object) -> str:
        kwargs = {
            "material_id": "LiAuI4_BL",
            "release_root": SITE,
            "pool_root": POOL,
            "work_root": "/tmp/work",
            "python": sys.executable,
            "poscar_path": POOL / "structures/LiAuI4/bilayer/POSCAR",
            "config_path": POOL / "coldstart/LiAuI4_BL/stage1.yaml",
            "dry_run": False,
            "resume": False,
            "poll_seconds": 60.0,
        }
        kwargs.update(overrides)
        return render_stage1_pbs(**kwargs)

    def test_template_pins_and_verifies_job_python(self) -> None:
        script = self._render()
        self.assertIn(f"PYTHON='{sys.executable}'", script)
        self.assertIn('if [[ ! -x "$PYTHON" ]]', script)
        self.assertIn("ARGS+=(--python \"$PYTHON\")", script)
        self.assertIn("--python", script)
        self.assertNotIn("=python3", script)

    def test_submit_material_default_python_is_sys_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            result = submit_material(
                material_id="LiAuI4_BL",
                release_root=SITE,
                work_root=work,
                manifest_path=MANIFEST,
                pool_root=POOL,
                expected_release_commit=RELEASE_COMMIT,
                dry_run=True,
                python=None,
            )
            self.assertEqual(result["mode"], "DRY_RUN")
            self.assertIn(f"PYTHON='{sys.executable}'", result["script"])
            self.assertIn("ARGS+=(--python \"$PYTHON\")", result["script"])

    def test_explicit_python_flows_into_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            result = submit_material(
                material_id="LiAuI4_BL",
                release_root=SITE,
                work_root=work,
                manifest_path=MANIFEST,
                pool_root=POOL,
                expected_release_commit=RELEASE_COMMIT,
                dry_run=True,
                python="/opt/site/python",
            )
            self.assertIn("PYTHON='/opt/site/python'", result["script"])


class CliPythonFlagTests(unittest.TestCase):
    def test_run_material_accepts_python(self) -> None:
        args = build_parser().parse_args([
            "run-material", "POSCAR", "XY2",
            "--config", "c.yaml", "--work-root", "w",
            "--python", "/opt/controlled/python",
        ])
        self.assertEqual(args.python, "/opt/controlled/python")

    def test_supervise_accepts_python(self) -> None:
        args = build_parser().parse_args([
            "supervise", "run_dir", "--python", "/opt/controlled/python",
        ])
        self.assertEqual(args.python, "/opt/controlled/python")


def _doctor_config(root: Path, launcher: str = "/bin/echo") -> Path:
    document = {
        "project": "python_pinning_probe",
        "structures": [{"id": "XY2", "poscar": "POSCAR"}],
        "physics": {"soc": False},
        "backend": {"vasp": "std", "mpi_processes": 2},
        "preprocess_inputs": {
            "commands": {
                "02_scf": [launcher, "{stage_dir}", "std", "2"],
                "03_band": [launcher, "{stage_dir}", "std", "2"],
                "04_wannier": [launcher, "{stage_dir}", "std", "2"],
                "04_wannier90": [launcher, "{stage_dir}", "wannier90"],
            },
        },
    }
    config = root / "doctor-stage1.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return config


class DoctorPythonConsistencyTests(unittest.TestCase):
    def test_doctor_reports_child_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _doctor_config(root)
            report = run_doctor(config, str(root), platform="linux")
            items = {item["id"]: item for item in report["items"]}
            self.assertIn("python.child", items)
            self.assertEqual(items["python.child"]["status"], "PASS")
            self.assertIn(sys.executable, items["python.child"]["detail"])

    def test_doctor_fails_bare_python3_in_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _doctor_config(root, launcher="python3")
            report = run_doctor(config, str(root), platform="linux")
            items = {item["id"]: item for item in report["items"]}
            for key in ("02_scf", "03_band", "04_wannier", "04_wannier90"):
                item = items[f"cmd.{key}.python"]
                self.assertEqual(item["status"], "FAIL")
                self.assertIn("裸 python3", item["detail"])
            self.assertEqual(report["overall"], "FAIL")


if __name__ == "__main__":
    unittest.main()
