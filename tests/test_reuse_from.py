"""Focused tests for ``--reuse-from`` technical-recovery cross-run reuse."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from w90flow import coldstart
from w90flow import run_material as rm

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


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, Path]:
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
    x_hash = _sha(component_x.read_bytes())
    y_hash = _sha(component_y.read_bytes())
    combined = _sha(component_x.read_bytes() + component_y.read_bytes())
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
        "structures": [{"id": "XY2", "poscar": "POSCAR"}],
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
                "04_wannier": ["launcher", "{stage_dir}", "std", "2"],
                "04_wannier90": ["launcher", "{stage_dir}", "wannier90"],
            },
        },
    }
    config = root / "stage1.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return config, poscar, root / "work"


def _plan(config: Path, work: Path):
    poscar = config.parent / "POSCAR"
    return rm.material_plan(poscar, "XY2", config, work)


def _prior_preprocess(
    work: Path,
    plan: dict,
    *,
    poscar_sha: str | None = None,
    potcar_sha: str | None = None,
    protocol_sha: str | None = None,
    template_overrides: dict[str, str] | None = None,
    missing_band_files: set[str] | None = None,
) -> Path:
    runs = work / "runs"
    prior = runs / "preprocess" / "prior-run-0001"
    prior.mkdir(parents=True, exist_ok=True)
    item = plan["structures"][0]
    sid = str(item["id"])
    active = sorted(item["preflight"]["incar"])
    templates = plan["immutable_input_sha256"]["preprocess_templates"]
    protocol = protocol_sha or item["preprocess_protocol_sha256"]
    (prior / "protocol.json").write_text(json.dumps({"protocol_sha256": protocol}), encoding="utf-8")
    materialized: dict = {
        "POSCAR": {"path": str(Path(item["poscar"])), "sha256": poscar_sha or item["poscar_sha256"]},
        "POTCAR": {
            "path": "private/POTCAR.locked",
            "sha256": potcar_sha or item["preflight"]["potcar"]["combined_sha256"],
        },
    }
    for stage in active:
        for role in ("incar", "kpoints"):
            key = f"{stage}:{role}"
            materialized[f"{stage}:{role.upper()}"] = {
                "path": "templates",
                "sha256": template_overrides.get(key, templates[key]) if template_overrides else templates[key],
            }
    (prior / "input-lock.json").write_text(json.dumps({
        "schema_version": 1,
        "created_utc": "2026-08-15T00:00:00Z",
        "protocol_sha256": protocol,
        "files": {"POSCAR": {"path": str(Path(item["poscar"])), "sha256": poscar_sha or item["poscar_sha256"]}},
        "materialized_files": materialized,
    }), encoding="utf-8")
    for stage in active:
        (prior / stage).mkdir(parents=True, exist_ok=True)
        (prior / stage / "stage.json").write_text(json.dumps({
            "schema_version": 1, "stage": stage, "status": "PASS",
        }), encoding="utf-8")
    band = prior / "03_band"
    band.mkdir(parents=True, exist_ok=True)
    files: dict = {}
    for name in ("EIGENVAL", "PROCAR", "vasprun.xml"):
        if missing_band_files and name in missing_band_files:
            files[name] = {"path": str(band / name), "sha256": _sha(b"x")}
            continue
        path = band / name
        path.write_bytes(b"band-" + name.encode())
        files[name] = {"path": str(path), "sha256": _sha(path.read_bytes())}
    (prior / "03_band/band-handoff.json").write_text(json.dumps({
        "schema_version": 1,
        "files": files,
    }), encoding="utf-8")
    return prior


def _prior_candidates_spec(overrides: dict | None = None) -> dict:
    spec = {
        "ensemble_mode": "independent",
        "soc": False,
        "energy_reference": "fermi_relative",
        "target_energy_eV": [-6.0, 4.0],
        "candidate_policy": {
            "max_candidates": 3,
            "max_num_wann_fraction_of_bands": 1.0,
            "shell_family_min_weight": 0.08,
            "shell_combination_min_weight": 0.45,
            "chemical_valence_used": False,
            "pyw90_dos": {"enabled": False, "required": False, "lower_bound": 0.1},
        },
    }
    if overrides:
        spec.update(overrides)
    return spec


class ReuseDefaultOffTests(unittest.TestCase):
    def test_reuse_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            with mock.patch("w90flow.run_material.run_stage1_coldstart",
                            return_value={"status": "COMPLETE", "wannier_run": str(work / "runs/x")}) as fake:
                outcome = rm.run_material(poscar, "XY2", config, work, resume=True)
            kwargs = fake.call_args.kwargs
            self.assertIsNone(kwargs.get("reuse"))
            manifest = json.loads((work / "campaigns/XY2/run-material-manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["reuse_from"])
            self.assertIsNone(manifest["reuse_stages"])



class PreprocessReuseTests(unittest.TestCase):
    def _run(self, prior: Path):
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            with mock.patch("w90flow.run_material.run_stage1_coldstart",
                            return_value={"status": "COMPLETE", "wannier_run": str(work / "runs/x")}) as fake:
                rm.run_material(poscar, "XY2", config, work, resume=True, reuse_from=prior)
            return fake.call_args.kwargs["reuse"]

    def test_identical_fingerprint_reuse_accepted_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = _prior_preprocess(work, plan)
            reuse = self._run(prior)
            self.assertEqual(reuse["preprocess_sources"]["XY2"], str(prior.resolve()))
            self.assertEqual(reuse["stages"]["preprocess"]["XY2"]["source_run"], str(prior.resolve()))
            self.assertEqual(reuse["stages"]["scf"]["XY2"]["source_stage"], "02_scf")
            self.assertEqual(reuse["stages"]["band"]["XY2"]["source_stage"], "03_band")
            hashes = reuse["stages"]["preprocess"]["XY2"]["hashes"]
            self.assertEqual(hashes["protocol_sha256"], plan["structures"][0]["preprocess_protocol_sha256"])
            self.assertEqual(hashes["poscar_sha256"], plan["structures"][0]["poscar_sha256"])
            self.assertNotIn("04_wannier", reuse["stages"])

    def test_poscar_change_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = _prior_preprocess(work, plan, poscar_sha="deadbeef" * 8)
            with self.assertRaisesRegex(ValueError, "POSCAR"):
                self._run(prior)

    def test_potcar_change_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = _prior_preprocess(work, plan, potcar_sha="deadbeef" * 8)
            with self.assertRaisesRegex(ValueError, "POTCAR"):
                self._run(prior)

    def test_protocol_soc_change_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = _prior_preprocess(work, plan, protocol_sha="cafebabe" * 8)
            with self.assertRaisesRegex(ValueError, "协议"):
                self._run(prior)

    def test_kpoints_template_change_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = _prior_preprocess(work, plan, template_overrides={"02_scf:kpoints": "cafebabe" * 8})
            with self.assertRaisesRegex(ValueError, "模板"):
                self._run(prior)

    def test_missing_band_artifact_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = _prior_preprocess(work, plan, missing_band_files={"PROCAR"})
            with self.assertRaisesRegex(ValueError, "产物"):
                self._run(prior)


class CandidatesReuseTests(unittest.TestCase):
    def _prior_wannier(self, work: Path, plan: dict, *, spec_overrides: dict | None = None,
                       with_import_manifest: bool = True) -> Path:
        prior_pre = _prior_preprocess(work, plan)
        prior = work / "runs" / "prior-coldstart-0001"
        sid = str(plan["structures"][0]["id"])
        structure = prior / "structures" / sid / "02_scf"
        structure.mkdir(parents=True)
        if with_import_manifest:
            (structure / "import-manifest.json").write_text(json.dumps({
                "results": [{"source": str(prior_pre / "02_scf")}],
            }), encoding="utf-8")
        candidates = prior / "03_candidates"
        candidates.mkdir(parents=True)
        (candidates / "stage.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        spec = _prior_candidates_spec(spec_overrides)
        (candidates / "candidate-spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (candidates / "pyw90-dos-XY2.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (prior / "04_wannier/stage.json").parent.mkdir(parents=True)
        (prior / "04_wannier/stage.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
        return prior

    def test_candidates_reused_but_wannier_failure_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = self._prior_wannier(work, plan)
            reuse = coldstart.verify_reuse_source(prior, config, plan)
            self.assertIn("candidates", reuse["stages"])
            self.assertEqual(reuse["stages"]["candidates"]["source_run"], str(prior.resolve()))
            self.assertEqual(reuse["stages"]["candidates"]["source_stage"], "03_candidates")
            self.assertEqual(reuse["stages"]["candidates"]["stage_status"], "PASS")
            self.assertEqual(len(reuse["stages"]["candidates"]["files"]), 2)
            self.assertNotIn("04_wannier", reuse["stages"])
            self.assertEqual(reuse["stages"]["preprocess"]["XY2"]["source_run"],
                             str((work / "runs/preprocess/prior-run-0001").resolve()))

    def test_candidates_parent_resolution_requires_import_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = self._prior_wannier(work, plan, with_import_manifest=False)
            with self.assertRaisesRegex(ValueError, "解析 SCF parent run"):
                coldstart.verify_reuse_source(prior, config, plan)

    def test_candidates_policy_change_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, poscar, work = _fixture(Path(temporary))
            plan = _plan(config, work)
            prior = self._prior_wannier(work, plan, spec_overrides={
                "candidate_policy": {"max_candidates": 5},
            })
            with self.assertRaisesRegex(ValueError, "候选策略"):
                coldstart.verify_reuse_source(prior, config, plan)

    def test_apply_reuse_candidates_copies_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wannier_run = root / "wannier-run"
            (wannier_run / "03_candidates").mkdir(parents=True)
            (wannier_run / "03_candidates/stage.json").write_text(
                json.dumps({"status": "PENDING"}), encoding="utf-8")
            source = root / "prior-spec.json"
            source.write_bytes(b"candidate spec payload")
            info = {
                "source_run": str(root / "prior-coldstart"),
                "source_stage": "03_candidates",
                "stage_status": "PASS",
                "files": [{"path": str(source), "sha256": _sha(source.read_bytes())}],
            }
            applied = coldstart._apply_reuse_candidates(wannier_run, info)
            self.assertEqual(applied["copied"][0]["sha256"], _sha(source.read_bytes()))
            target = wannier_run / "03_candidates" / source.name
            self.assertTrue(target.is_file())
            self.assertEqual(_sha(target.read_bytes()), _sha(source.read_bytes()))
            stage = json.loads((wannier_run / "03_candidates/stage.json").read_text(encoding="utf-8"))
            self.assertEqual(stage["status"], "PASS")
            self.assertEqual(stage["provenance"]["source_stage"], "03_candidates")

    def test_apply_reuse_candidates_rejects_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wannier_run = root / "wannier-run"
            (wannier_run / "03_candidates").mkdir(parents=True)
            (wannier_run / "03_candidates/stage.json").write_text(
                json.dumps({"status": "PENDING"}), encoding="utf-8")
            source = root / "prior-spec.json"
            source.write_bytes(b"payload")
            info = {
                "source_run": str(root / "prior"),
                "source_stage": "03_candidates",
                "stage_status": "PASS",
                "files": [{"path": str(source), "sha256": _sha(b"different")}],
            }
            with self.assertRaisesRegex(ValueError, "哈希改变"):
                coldstart._apply_reuse_candidates(wannier_run, info)


if __name__ == "__main__":
    unittest.main()
