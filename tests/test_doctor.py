from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from w90flow.doctor import doctor_cli, run_doctor

POSCAR = """synthetic XY2 sheet
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 20.0
X Y
1 2
Direct
0.0 0.0 0.5
0.25 0.25 0.5
0.75 0.75 0.5
"""


def _fixture(
    root: Path,
    *,
    wrapper_suffix: str = "",
    executable: bool = True,
    mac_command: str | None = None,
    potcar_ok: bool = True,
    include_commands: bool = True,
    vasp_version: str = "6.4.3",
    soc: bool = True,
    include_capability: bool = True,
) -> Path:
    (root / "POSCAR").write_text(POSCAR, encoding="utf-8")
    templates = root / "templates"
    templates.mkdir()
    for stage in ("scf", "band", "wannier"):
        (templates / f"INCAR.{stage}").write_text("ENCUT = 400\n", encoding="utf-8")
        (templates / f"KPOINTS.{stage}").write_text("synthetic\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")

    component_x = root / "POTCAR.X"
    component_y = root / "POTCAR.Y"
    component_x.write_bytes(b"synthetic X POTCAR\n")
    component_y.write_bytes(b"synthetic Y POTCAR\n")
    x_hash = hashlib.sha256(component_x.read_bytes()).hexdigest()
    y_hash = hashlib.sha256(component_y.read_bytes()).hexdigest()
    combined = hashlib.sha256(component_x.read_bytes() + component_y.read_bytes()).hexdigest()
    lock = root / "POTCAR_LOCK.json"
    lock.write_text(json.dumps({
        "sets": {
            "XY_PBE": {
                "order": ["X", "Y"],
                "component_sha256": {"X": x_hash, "Y": y_hash},
                "combined_sha256": combined,
            },
        },
    }), encoding="utf-8")

    kind = "ncl" if soc else "std"
    vasp_bin = root / (f"fake-vasp.{vasp_version}-ncl" if vasp_version else "fake-vasp-ncl")
    w90_bin = root / "fake-wannier90.x"
    vasp_bin.write_bytes(b"#!/bin/sh\necho vasp.5.4.4\n")
    w90_bin.write_bytes(b"#!/bin/sh\necho wannier90 3.1.0\n")
    vasp_bin.chmod(vasp_bin.stat().st_mode | stat.S_IXUSR)
    w90_bin.chmod(w90_bin.stat().st_mode | stat.S_IXUSR)

    run_vasp = root / f"run-vasp{wrapper_suffix}"
    run_w90 = root / f"run-w90{wrapper_suffix}"
    run_vasp.write_text(f"#!/bin/bash\nVASP_BIN={vasp_bin}\n", encoding="utf-8")
    run_w90.write_text(f"#!/bin/bash\nW90={w90_bin}\n", encoding="utf-8")
    if executable:
        run_vasp.chmod(run_vasp.stat().st_mode | stat.S_IXUSR)
        run_w90.chmod(run_w90.stat().st_mode | stat.S_IXUSR)

    command_vasp = mac_command or str(run_vasp)
    commands = {
        "02_scf": [command_vasp, "{stage_dir}", kind, "4"],
        "03_band": [command_vasp, "{stage_dir}", kind, "4"],
        "04_wannier": [command_vasp, "{stage_dir}", kind, "4"],
        "04_wannier90": [str(run_w90), "{stage_dir}", "wannier90"],
    }
    document = {
        "schema_version": 1,
        "project": "synthetic_doctor",
        "structures": [{"id": "XY2", "poscar": "POSCAR"}],
        "physics": {"soc": soc, "dimensionality": 2},
        "pseudopotentials": {"lock": "POTCAR_LOCK.json", "set": "XY_PBE"},
        "backend": {"kind": "local_mpi", "vasp": kind, "mpi_processes": 4},
        "preprocess_inputs": {
            "templates": {
                stage: {"incar": f"templates/INCAR.{stage}", "kpoints": f"templates/KPOINTS.{stage}"}
                for stage in ("scf", "band", "wannier")
            },
            "potcar": {
                "components": [
                    {"element": "X", "path": "POTCAR.X", "sha256": x_hash},
                    {"element": "Y", "path": "POTCAR.Y", "sha256": y_hash},
                ],
                "combined_sha256": combined,
            },
        },
    }
    if include_capability:
        document["backend"]["spinor_wannier_interface"] = {
            "vasp_launcher": str(run_vasp),
            "vasp_version": vasp_version or "6.4.3",
            "wannier90_launcher": str(run_w90),
            "wannier90_version": "3.1.0",
        }
    if include_commands:
        document["preprocess_inputs"]["commands"] = commands
    if not potcar_ok:
        document["preprocess_inputs"]["potcar"]["components"][0]["path"] = "POTCAR.MISSING"
    config = root / "stage1.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return config


def _items(report: dict) -> dict[str, str]:
    return {item["id"]: item["status"] for item in report["items"]}


def _status(report: dict, item_id: str) -> str:
    return _items(report)[item_id]


def _status_prefix(report: dict, prefix: str) -> str | None:
    for item in report["items"]:
        if item["id"].startswith(prefix):
            return item["status"]
    return None


class DoctorEnvironmentTests(unittest.TestCase):
    def _run(self, config: Path, work_root: str, **kwargs) -> dict:
        return run_doctor(str(config), work_root, **kwargs)

    def test_full_config_pass_on_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary))
            with mock.patch("w90flow.doctor.shutil.which",
                            side_effect=lambda name: {"mpirun": "/bin/true",
                                                      "qsub": "/opt/pbs/bin/qsub",
                                                      "qstat": "/opt/pbs/bin/qstat"}.get(name)):
                work = Path(temporary) / "work"
                work.mkdir()
                report = self._run(config, str(work), platform="linux")
            self.assertEqual(report["overall"], "PASS", report)
            self.assertEqual(_status(report, "commands.present"), "PASS")
            self.assertEqual(_status(report, "cmd.04_wannier90"), "PASS")
            self.assertEqual(_status(report, "wrap.02_scf"), "PASS")
            self.assertEqual(_status_prefix(report, "target.VASP."), "PASS")
            self.assertEqual(_status_prefix(report, "target.W90."), "PASS")
            self.assertEqual(_status(report, "interface.vasp_wannier"), "PASS")
            self.assertEqual(_status(report, "potcar.lock"), "PASS")
            self.assertEqual(_status(report, "pbs"), "PASS")

    def test_soc_wannier_rejects_vasp544_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary), vasp_version="5.4.4")
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "interface.vasp_wannier"), "FAIL")
            self.assertIn("5.4.4", [i["detail"] for i in report["items"] if i["id"] == "interface.vasp_wannier"][0])
            self.assertEqual(report["overall"], "FAIL")

    def test_soc_wannier_requires_capability_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary), include_capability=False)
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "interface.vasp_wannier"), "FAIL")
            self.assertEqual(report["overall"], "FAIL")

    def test_soc_wannier_rejects_wrapper_not_from_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root)
            document = yaml.safe_load(config.read_text(encoding="utf-8"))
            document["preprocess_inputs"]["commands"]["04_wannier"][0] = str(root / "other-vasp-wrapper")
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "interface.vasp_wannier"), "FAIL")
            self.assertIn("04_wannier", [i["detail"] for i in report["items"] if i["id"] == "interface.vasp_wannier"][0])

    def test_soc_wannier_rejects_wrapper_without_declared_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary), vasp_version="")
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "interface.vasp_wannier"), "FAIL")
            self.assertIn("未静态声明", [i["detail"] for i in report["items"] if i["id"] == "interface.vasp_wannier"][0])

    def test_non_soc_allows_vasp544_profile_without_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary), soc=False, vasp_version="5.4.4", include_capability=False)
            report = self._run(config, temporary, platform="linux")
            self.assertNotIn("interface.vasp_wannier", _items(report))
            self.assertEqual(_status(report, "wrap.04_wannier"), "PASS")
            self.assertEqual(_status(report, "cmd.04_wannier"), "PASS")

    def test_missing_pyyaml_is_fail(self) -> None:
        def probe(name: str) -> tuple[str, str]:
            if name == "yaml":
                return "MISSING", "ModuleNotFoundError"
            return "OK", "1.0"

        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary))
            report = self._run(config, temporary, platform="linux",
                               import_probe=probe)
            self.assertEqual(_status(report, "import.yaml"), "FAIL")
            self.assertEqual(report["overall"], "FAIL")

    def test_command_missing_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root)
            document = yaml.safe_load(config.read_text(encoding="utf-8"))
            document["preprocess_inputs"]["commands"]["02_scf"] = [
                str(root / "run-vasp-does-not-exist"), "{stage_dir}", "ncl", "4",
            ]
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "wrap.02_scf"), "FAIL")
            self.assertEqual(report["overall"], "FAIL")

    def test_command_not_executable_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root, executable=False)
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "wrap.02_scf"), "FAIL")
            self.assertIn("不可执行", [i["detail"] for i in report["items"] if i["id"] == "wrap.02_scf"][0])

    def test_absolute_mac_path_fails_on_linux_but_command_filename_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root, mac_command="/Users/jiaqi/bin/run-vasp")
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "platform.mac_path"), "FAIL")
            self.assertEqual(report["overall"], "FAIL")
        with tempfile.TemporaryDirectory() as temporary2:
            config2 = _fixture(Path(temporary2), wrapper_suffix=".command")
            report2 = self._run(config2, temporary2, platform="linux")
            self.assertEqual(_status(report2, "platform.mac_path"), "PASS")
            self.assertEqual(_status(report2, "wrap.02_scf"), "PASS")

    def test_potcar_root_missing_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary), potcar_ok=False)
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "potcar.path.POTCAR.MISSING"), "FAIL")
            self.assertEqual(report["overall"], "FAIL")

    def test_work_root_not_writable_is_fail(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root ignores permission bits")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root)
            locked = root / "locked"
            locked.mkdir()
            locked.chmod(0o555)
            try:
                report = self._run(config, str(locked), platform="linux")
                self.assertEqual(_status(report, "work.root"), "FAIL")
            finally:
                locked.chmod(0o755)

    def test_pbs_missing_classified_by_platform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary))
            report_linux = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report_linux, "pbs"), "FAIL")
            report_mac = self._run(config, temporary, platform="darwin")
            self.assertEqual(_status(report_mac, "pbs"), "WARN")

    def test_missing_commands_forbids_home_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary), include_commands=False)
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(_status(report, "commands.present"), "FAIL")
            self.assertEqual(report["overall"], "FAIL")

    def test_json_schema_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary))
            report = self._run(config, temporary, platform="linux")
            self.assertEqual(sorted(report), [
                "config", "generated_utc", "items", "overall", "platform",
                "schema_version", "summary", "work_root",
            ])
            for item in report["items"]:
                self.assertEqual(sorted(item), ["detail", "id", "name", "status"])
            payload = json.loads(json.dumps(report))
            self.assertEqual(payload["schema_version"], 1)

    def test_doctor_cli_exit_codes_and_json_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root)
            good = types.SimpleNamespace(config=str(config), work_root=temporary, json=True)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = doctor_cli(good)
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn(payload["overall"], {"PASS", "WARN"})
            bad = types.SimpleNamespace(config=str(config), work_root=temporary, json=False)
            document = yaml.safe_load(config.read_text(encoding="utf-8"))
            del document["preprocess_inputs"]["commands"]["04_wannier90"]
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = doctor_cli(bad)
            self.assertEqual(exit_code, 1)

    def test_cli_subprocess_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _fixture(root)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            completed = subprocess.run(
                [sys.executable, "-m", "w90flow", "doctor",
                 "--config", str(config), "--work-root", temporary, "--json"],
                capture_output=True, text=True, env=env, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["overall"], "WARN")  # darwin 无 PBS → WARN


if __name__ == "__main__":
    unittest.main()
