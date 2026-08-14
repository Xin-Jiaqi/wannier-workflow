from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from w90flow import run_material as rm
from w90flow.cli import main

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


def _fixture(root: Path, *, structure_id: str = "XY2") -> tuple[Path, Path, Path]:
    poscar = root / "POSCAR"
    poscar.write_text(POSCAR, encoding="utf-8")
    templates = root / "templates"
    templates.mkdir()
    incar = "ENCUT = 400\n"
    kpoints = "synthetic\n0\nGamma\n1 1 1\n0 0 0\n"
    for stage in ("scf", "band", "wannier"):
        (templates / f"INCAR.{stage}").write_text(incar, encoding="utf-8")
        (templates / f"KPOINTS.{stage}").write_text(kpoints, encoding="utf-8")
    component_x = root / "POTCAR.X"
    component_y = root / "POTCAR.Y"
    component_x.write_bytes(b"synthetic X POTCAR\n")
    component_y.write_bytes(b"synthetic Y POTCAR\n")
    x_hash = hashlib.sha256(component_x.read_bytes()).hexdigest()
    y_hash = hashlib.sha256(component_y.read_bytes()).hexdigest()
    combined = hashlib.sha256(component_x.read_bytes() + component_y.read_bytes()).hexdigest()
    (root / "POTCAR_LOCK.json").write_text(json.dumps({
        "registry": str(root),
        "sets": {
            "XY_PBE": {
                "order": ["X", "Y"],
                "component_sha256": {"X": x_hash, "Y": y_hash},
                "combined_sha256": combined,
            },
        },
    }, indent=2), encoding="utf-8")
    document = {
        "schema_version": 1,
        "project": "synthetic_coldstart",
        "structures": [{"id": structure_id, "poscar": "POSCAR"}],
        "physics": {
            "soc": False,
            "dimensionality": 2,
            "axes": ["x", "y", "z"],
            "energy_reference": "fermi_relative",
            "target_energy_eV": [-6.0, 4.0],
        },
        "pseudopotentials": {"lock": "POTCAR_LOCK.json", "set": "XY_PBE"},
        "backend": {"kind": "local_mpi", "vasp": "std", "mpi_processes": 2},
        "band_path": {"mode": "template", "points_per_segment": 31},
        "meshes": {"scf": [9, 9, 1], "wannier": [9, 9, 1]},
        "templates": {
            stage: {"incar": f"templates/INCAR.{stage}", "kpoints": f"templates/KPOINTS.{stage}"}
            for stage in ("scf", "band", "wannier")
        },
        "candidates": {"max_candidates": 3},
        "ranking": {"output_top_k": 3},
        "execution": {"allow_execute": True},
        "preprocess": {
            "soc": {"scf": False, "band": False},
            "electronic": {"ediff_eV": 1.0e-6, "max_steps": 120},
            "relaxation": {"enabled": False},
            "band": {"path_provider": "template"},
        },
        "preprocess_inputs": {
            "templates": {
                "02_scf": {"incar": "templates/INCAR.scf", "kpoints": "templates/KPOINTS.scf"},
                "03_band": {"incar": "templates/INCAR.band", "kpoints": "templates/KPOINTS.band"},
            },
            "potcar": {
                "components": [
                    {"element": "X", "path": str(component_x), "sha256": x_hash},
                    {"element": "Y", "path": str(component_y), "sha256": y_hash},
                ],
                "combined_sha256": combined,
            },
            "commands": {
                "02_scf": ["launcher", "{stage_dir}", "std", "2"],
                "03_band": ["launcher", "{stage_dir}", "std", "2"],
            },
        },
    }
    config = root / "stage1.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return config, poscar, root / "work"


def _stage(root: Path, name: str, status: str = "PASS", **extra: object) -> None:
    payload: dict[str, object] = {"schema_version": 1, "stage": name, "status": status, **extra}
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "stage.json").write_text(json.dumps(payload), encoding="utf-8")


def _export_summary(run_dir: Path, count: int) -> None:
    output = run_dir / "07_export" / "independent-top-k-v04"
    output.mkdir(parents=True, exist_ok=True)
    (output / "export-summary.json").write_text(json.dumps({
        "schema_version": 1,
        "requested_top_k": 3,
        "packages": [{"candidate": f"c{i}", "attempt": "a1"} for i in range(count)],
    }, indent=2), encoding="utf-8")


def _wannier_run(plan: dict, work: Path) -> Path:
    return work.resolve() / "runs" / f"synthetic_coldstart-coldstart-{plan['input_signature_sha256'][:16]}"


class RunMaterialPlanTests(unittest.TestCase):
    def test_dry_run_is_read_only_and_reports_full_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            result = rm.material_plan(poscar, "XY2", config, work)
            self.assertEqual(result["mode"], "DRY_RUN")
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertFalse(result["terminal"])
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["requested_top_k"], 3)
            self.assertEqual(result["poscar_sha256"], hashlib.sha256(poscar.read_bytes()).hexdigest())
            self.assertFalse(work.exists(), "dry-run 不得创建 work-root")
            self.assertEqual(
                [item["stage"] for item in result["stage_plan"]],
                ["preprocess", "scf", "band", "candidates", "wannier", "quality", "rank", "export"],
            )
            self.assertEqual(result["reuse"]["preprocess"][0]["status"], "CREATE")
            self.assertIsNone(result["reuse"]["wannier"]["run_dir"])
            self.assertIsNotNone(result["input_signature_sha256"])

    def test_dry_run_reports_reuse_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            runs = work / "runs"
            preprocess = runs / "preprocess" / (
                f"synthetic_coldstart-XY2-{plan['input_signature_sha256'][:8]}"
                f"-{plan['poscar_sha256'][:8]}"
            )
            _stage(preprocess, "03_band", "PASS")
            (preprocess / "protocol.json").write_text(json.dumps({"protocol_sha256": "x"}), encoding="utf-8")
            (preprocess / "input-lock.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
            wannier = _wannier_run(plan, work)
            _stage(wannier, "01_scf", "PASS")
            _stage(wannier, "02_band", "PENDING")
            reuse = rm.material_plan(poscar, "XY2", config, work)["reuse"]
            self.assertEqual(reuse["preprocess"][0]["status"], "REUSE")
            self.assertEqual(reuse["wannier"]["run_dir"], str(wannier))
            self.assertEqual(reuse["wannier"]["stages"]["scf"], "REUSE")
            self.assertEqual(reuse["wannier"]["stages"]["band"], "EXECUTE")

    def test_plan_rejects_multi_structure_or_wrong_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            with self.assertRaises(ValueError):
                rm.material_plan(poscar, "NOPE", config, work)
            document = yaml.safe_load(config.read_text(encoding="utf-8"))
            document["structures"] = [dict(document["structures"][0]), dict(document["structures"][0])]
            document["structures"][1]["id"] = "XY2b"
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                rm.material_plan(poscar, "XY2", config, work)


class RunMaterialTerminalTests(unittest.TestCase):
    def _coldstart(self, plan: dict, work: Path, result: dict) -> tuple[Path, dict]:
        wannier = _wannier_run(plan, work)
        config_file = Path(plan["config"])
        poscar = Path(plan["poscar"])
        with mock.patch("w90flow.run_material.run_stage1_coldstart", return_value=result) as fake:
            outcome = rm.run_material(poscar, "XY2", config_file, work, resume=True)
        return wannier, outcome

    def test_top3_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            wannier = _wannier_run(plan, work)
            _export_summary(wannier, 3)
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "COMPLETE", "wannier_run": str(wannier)},
            ) as fake:
                outcome = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(outcome["status"], "TOP3_READY")
            self.assertTrue(outcome["terminal"])
            self.assertEqual(outcome["accepted_count"], 3)
            self.assertEqual(outcome["exit_code"], 0)
            fake.assert_called_once()

    def test_partial_topk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            wannier = _wannier_run(plan, work)
            _export_summary(wannier, 2)
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "COMPLETE", "wannier_run": str(wannier)},
            ):
                outcome = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(outcome["status"], "PARTIAL_TOPK")
            self.assertEqual(outcome["accepted_count"], 2)
            self.assertEqual(outcome["exit_code"], 0)

    def test_no_model_when_candidates_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            wannier = _wannier_run(plan, work)
            _stage(wannier, "04_wannier", "FAIL")
            (wannier / "workflow-summary.json").write_text(json.dumps({
                "reason": "有限候选/能窗尝试结束后仍没有覆盖全部结构的合格 Wannier 表示。",
            }), encoding="utf-8")
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "BLOCKED", "wannier_run": str(wannier)},
            ):
                outcome = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(outcome["status"], "NO_MODEL")
            self.assertEqual(outcome["failure_class"], "CANDIDATES_EXHAUSTED")
            self.assertEqual(outcome["accepted_count"], 0)
            self.assertEqual(outcome["exit_code"], 0)

    def test_technical_failed_on_infrastructure_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            wannier = _wannier_run(plan, work)
            _stage(wannier, "01_scf", "FAIL", failure_class="INFRASTRUCTURE_FAILURE")
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "BLOCKED", "wannier_run": str(wannier)},
            ):
                outcome = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(outcome["status"], "TECHNICAL_FAILED")
            self.assertEqual(outcome["failure_class"], "INFRASTRUCTURE_OR_PROGRAM")
            self.assertEqual(outcome["exit_code"], 2)

    def test_technical_failed_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                side_effect=RuntimeError("preprocess 基础设施恢复预算已用完"),
            ):
                outcome = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(outcome["status"], "TECHNICAL_FAILED")
            self.assertEqual(outcome["failure_class"], "RuntimeError")
            self.assertEqual(outcome["exit_code"], 2)


class RunMaterialResumeTests(unittest.TestCase):
    def test_resume_required_then_short_circuits_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            wannier = _wannier_run(plan, work)
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "BLOCKED", "wannier_run": str(wannier)},
            ):
                first = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(first["status"], "TECHNICAL_FAILED")
            with self.assertRaises(RuntimeError):
                rm.run_material(poscar, "XY2", config, work)
            _export_summary(wannier, 3)
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "COMPLETE", "wannier_run": str(wannier)},
            ) as fake:
                resumed = rm.run_material(poscar, "XY2", config, work, resume=True)
            self.assertEqual(resumed["status"], "TOP3_READY")
            fake.assert_called_once()
            with mock.patch("w90flow.run_material.run_stage1_coldstart") as fake2:
                again = rm.run_material(poscar, "XY2", config, work)
            self.assertEqual(again["status"], "TOP3_READY")
            fake2.assert_not_called()

    def test_input_fingerprint_mismatch_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = rm.material_plan(poscar, "XY2", config, work)
            wannier = _wannier_run(plan, work)
            with mock.patch(
                "w90flow.run_material.run_stage1_coldstart",
                return_value={"status": "BLOCKED", "wannier_run": str(wannier)},
            ):
                rm.run_material(poscar, "XY2", config, work, resume=True)
            poscar.write_text(POSCAR.replace("4.0 0.0 0.0", "5.0 0.0 0.0"), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                rm.run_material(poscar, "XY2", config, work, resume=True)


class RunMaterialCliTests(unittest.TestCase):
    def test_cli_dry_run_exit_zero_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            code = main([
                "run-material", str(poscar), "XY2",
                "--config", str(config), "--work-root", str(work), "--dry-run",
            ])
            self.assertEqual(code, 0)
            self.assertFalse(work.exists())

    def test_cli_rejects_wrong_material_id_with_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            code = main([
                "run-material", str(poscar), "NOPE",
                "--config", str(config), "--work-root", str(work), "--dry-run",
            ])
            self.assertEqual(code, 1)

    def test_cli_requires_config_and_work_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _config, poscar, work = _fixture(Path(temporary))
            with self.assertRaises(SystemExit) as ctx:
                main(["run-material", str(poscar), "XY2", "--work-root", str(work)])
            self.assertEqual(ctx.exception.code, 2)
            with self.assertRaises(SystemExit) as ctx:
                main(["run-material", str(poscar), "XY2", "--config", str(_config)])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
