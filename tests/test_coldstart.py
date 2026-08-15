from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

import yaml

from w90flow.coldstart import plan_stage1_coldstart, run_stage1_coldstart


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


class ColdStartPreflightTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict, Path, Path]:
        poscar = root / "POSCAR"
        poscar.write_text(POSCAR, encoding="utf-8")
        templates = root / "templates"
        templates.mkdir()
        soc_incar = """ENCUT = 400
LSORBIT = .TRUE.
LNONCOLLINEAR = .TRUE.
MAGMOM = 9*0
"""
        kpoints = "synthetic\n0\nGamma\n1 1 1\n0 0 0\n"
        for stage in ("scf", "band", "wannier"):
            (templates / f"INCAR.{stage}").write_text(soc_incar, encoding="utf-8")
            (templates / f"KPOINTS.{stage}").write_text(kpoints, encoding="utf-8")

        component_x = root / "POTCAR.X"
        component_y = root / "POTCAR.Y"
        component_x.write_bytes(b"synthetic X POTCAR\n")
        component_y.write_bytes(b"synthetic Y POTCAR\n")
        x_hash = hashlib.sha256(component_x.read_bytes()).hexdigest()
        y_hash = hashlib.sha256(component_y.read_bytes()).hexdigest()
        combined = hashlib.sha256(component_x.read_bytes() + component_y.read_bytes()).hexdigest()
        lock = root / "POTCAR_LOCK.json"
        lock.write_text(json.dumps({
            "registry": str(root),
            "sets": {
                "XY_PBE": {
                    "order": ["X", "Y"],
                    "component_sha256": {"X": x_hash, "Y": y_hash},
                    "combined_sha256": combined,
                },
            },
        }, indent=2), encoding="utf-8")
        vasp_bin = root / "fake-vasp.6.4.3-ncl"
        vasp_bin.write_bytes(b"#!/bin/sh\n")
        vasp_bin.chmod(vasp_bin.stat().st_mode | stat.S_IXUSR)
        vasp_wrapper = root / "run-vasp.6.4.3-ncl"
        w90_wrapper = root / "run-w90"
        vasp_wrapper.write_text(f"#!/bin/bash\nVASP_BIN={vasp_bin}\n", encoding="utf-8")
        w90_wrapper.write_text("#!/bin/bash\nexec wannier90.x\n", encoding="utf-8")
        vasp_wrapper.chmod(vasp_wrapper.stat().st_mode | stat.S_IXUSR)
        w90_wrapper.chmod(w90_wrapper.stat().st_mode | stat.S_IXUSR)
        document = {
            "schema_version": 1,
            "project": "synthetic_SOC_coldstart",
            "structures": [{"id": "XY2", "poscar": "POSCAR"}],
            "physics": {
                "soc": True,
                "dimensionality": 2,
                "axes": ["x", "y", "z"],
                "energy_reference": "fermi_relative",
                "target_energy_eV": [-6.0, 4.0],
            },
            "pseudopotentials": {"lock": "POTCAR_LOCK.json", "set": "XY_PBE"},
            "backend": {
                "kind": "local_mpi",
                "vasp": "ncl",
                "mpi_processes": 4,
                "spinor_wannier_interface": {
                    "vasp_launcher": str(vasp_wrapper),
                    "vasp_version": "6.4.3",
                    "wannier90_launcher": str(w90_wrapper),
                    "wannier90_version": "3.1.0",
                },
            },
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
                "soc": {"scf": True, "band": True},
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
                    "02_scf": [str(vasp_wrapper), "{stage_dir}", "ncl", "4"],
                    "03_band": [str(vasp_wrapper), "{stage_dir}", "ncl", "4"],
                    "04_wannier": [str(vasp_wrapper), "{stage_dir}", "ncl", "4"],
                    "04_wannier90": [str(w90_wrapper), "{stage_dir}", "wannier90"],
                },
            },
        }
        config = root / "stage1.yaml"
        config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return config, document, root / "campaign", root / "runs"

    @staticmethod
    def _write(config: Path, document: dict) -> None:
        config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def test_complete_dry_run_preflight_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _document, campaign, runs = self._fixture(root)
            result = plan_stage1_coldstart(config, campaign, runs)
            self.assertEqual(result["mode"], "DRY_RUN")
            self.assertTrue(result["preflight"]["passed"])
            structure = result["structures"][0]["preflight"]
            self.assertEqual(structure["execution"]["backend_vasp"], "ncl")
            self.assertEqual(structure["execution"]["mpi_processes"], 4)
            self.assertEqual(structure["incar"]["02_scf"]["magmom_components"], 9)
            self.assertEqual(structure["potcar"]["order"], ["X", "Y"])
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())
            self.assertFalse(any(root.rglob("POTCAR.locked")))

    def test_execution_refuses_cross_layer_soc_mismatch_before_creating_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, document, campaign, runs = self._fixture(root)
            document["preprocess"]["soc"]["band"] = False
            self._write(config, document)
            with self.assertRaisesRegex(ValueError, "preprocess.soc.band"):
                run_stage1_coldstart(config, campaign, runs, execute=True)
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())

    def test_backend_and_launcher_kind_and_mpi_must_agree(self) -> None:
        cases = (
            (lambda doc: doc["backend"].update(vasp="std"), "backend.vasp=ncl"),
            (lambda doc: doc["preprocess_inputs"]["commands"]["02_scf"].__setitem__(-2, "std"), "02_scf 应使用 ncl"),
            (lambda doc: doc["preprocess_inputs"]["commands"]["03_band"].__setitem__(-1, "2"), "MPI 数 2"),
        )
        for mutation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, document, campaign, runs = self._fixture(root)
                mutation(document)
                self._write(config, document)
                with self.assertRaisesRegex(ValueError, message):
                    plan_stage1_coldstart(config, campaign, runs)
                self.assertFalse(campaign.exists())
                self.assertFalse(runs.exists())

    def test_soc_wannier_requires_compatible_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, document, campaign, runs = self._fixture(root)
            del document["backend"]["spinor_wannier_interface"]
            self._write(config, document)
            with self.assertRaisesRegex(ValueError, "spinor_wannier_interface"):
                plan_stage1_coldstart(config, campaign, runs)
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())

    def test_soc_wannier_rejects_vasp544_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, document, campaign, runs = self._fixture(root)
            document["backend"]["spinor_wannier_interface"]["vasp_version"] = "5.4.4"
            self._write(config, document)
            with self.assertRaisesRegex(ValueError, "6.4"):
                plan_stage1_coldstart(config, campaign, runs)
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())

    def test_soc_wannier_rejects_launcher_not_from_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, document, campaign, runs = self._fixture(root)
            document["preprocess_inputs"]["commands"]["04_wannier"][0] = str(root / "other-wrapper")
            self._write(config, document)
            with self.assertRaisesRegex(ValueError, "04_wannier"):
                plan_stage1_coldstart(config, campaign, runs)
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())

    def test_soc_magmom_cardinality_is_checked_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, document, campaign, runs = self._fixture(root)
            (root / "templates/INCAR.scf").write_text(
                "LSORBIT = .TRUE.\nLNONCOLLINEAR = .TRUE.\nMAGMOM = 3*0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "3\\*NIONS=9"):
                plan_stage1_coldstart(config, campaign, runs)
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())

    def test_potcar_order_component_and_combined_hash_are_all_hard_gates(self) -> None:
        cases = (
            (lambda doc: doc["preprocess_inputs"]["potcar"]["components"].reverse(), "组件顺序"),
            (lambda doc: doc["preprocess_inputs"]["potcar"]["components"][0].update(sha256="0" * 64), "组件 X SHA256"),
            (lambda doc: doc["preprocess_inputs"]["potcar"].update(combined_sha256="0" * 64), "combined_sha256"),
        )
        for mutation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, document, campaign, runs = self._fixture(root)
                mutation(document)
                self._write(config, document)
                with self.assertRaisesRegex(ValueError, message):
                    plan_stage1_coldstart(config, campaign, runs)
                self.assertFalse(campaign.exists())
                self.assertFalse(runs.exists())

    def test_preprocess_potcar_recipe_must_also_match_named_lock_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _document, campaign, runs = self._fixture(root)
            lock_path = root / "POTCAR_LOCK.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["sets"]["XY_PBE"]["component_sha256"]["X"] = "f" * 64
            lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lock set XY_PBE"):
                plan_stage1_coldstart(config, campaign, runs)
            self.assertFalse(campaign.exists())
            self.assertFalse(runs.exists())


if __name__ == "__main__":
    unittest.main()
