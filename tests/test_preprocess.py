from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from w90flow.cli import _preprocess_project, build_parser
from w90flow.orchestrator import fit_from_preprocess
from w90flow.preprocess import (
    PreprocessConfigError,
    StageBusyError,
    StageRecoveryRequired,
    analyze_preprocess_outputs,
    begin_stage,
    build_preprocess_plan,
    finish_stage,
    initialize_preprocess_run,
    inspect_preprocess_run,
    materialize_potcar_recipe,
    normalize_preprocess_config,
    patch_incar_contract,
    prepare_preprocess_run,
    recover_interrupted_stage,
    run_preprocess_stage,
    run_preprocess_workflow,
    standard_semilocal_preprocess,
    stage_incar_contract,
)
from w90flow.state import atomic_json, read_json


FIXTURES = Path(__file__).resolve().parent / "fixtures/preprocess"
POSCAR_2D = """synthetic sheet
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 20.0
X
1
Direct
0.0 0.0 0.5
"""


def base_document(*, relaxation: bool = True) -> dict:
    return {
        "preprocess": {
            "soc": {"relaxation": False, "scf": True, "band": True},
            "electronic": {"ediff_eV": 1.0e-6, "max_steps": 80},
            "relaxation": {
                "enabled": relaxation,
                "cell_mode": "ions_only",
                "force_threshold_eV_A": 0.01,
                "max_steps": 100,
                "vacuum_axis": 2,
            },
            "band": {"path_provider": "template"},
        }
    }


def write_relax_outputs(directory: Path, poscar: str = POSCAR_2D, force: float = 0.005) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "CONTCAR").write_text(poscar, encoding="utf-8")
    (directory / "OSZICAR").write_text(
        " DAV:   1  -1.0  0.1\n DAV:   2  -1.1  0.001\n  1 F= -1.1 E0= -1.1 d E =0\n",
        encoding="utf-8",
    )
    (directory / "OUTCAR").write_text(
        f""" EDIFF = 1.0E-6
 NELM = 80
 LSORBIT = F
 NSW = 100
 ISIF = 2
 EDIFFG = -0.01
 aborting loop because EDIFF is reached
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
 0.0 0.0 10.0 {force} 0.0 0.0
 -----------------------------------------------------------------------------------
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
""",
        encoding="utf-8",
    )


def write_scf_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "CHGCAR").write_text("synthetic charge\n", encoding="utf-8")
    (directory / "OSZICAR").write_text(
        " DAV: 1 -1 0.1\n DAV: 2 -1.1 0.001\n 1 F= -1.1 E0= -1.1 d E=0\n",
        encoding="utf-8",
    )
    (directory / "OUTCAR").write_text(
        """ EDIFF = 1.0E-6
 NELM = 80
 LSORBIT = T
 NSW = 0
 aborting loop because EDIFF is reached
 General timing and accounting informations for this job:
""",
        encoding="utf-8",
    )


def write_band_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "OSZICAR").write_text(" DAV: 1 -1 0.0\n 1 F= -1 E0= -1 d E=0\n", encoding="utf-8")
    (directory / "OUTCAR").write_text(
        """ EDIFF = 1.0E-6
 NELM = 80
 LSORBIT = T
 NSW = 0
 ICHARG = 11
 General timing and accounting informations for this job:
""",
        encoding="utf-8",
    )
    (directory / "EIGENVAL").write_text(
        "header\nheader\nheader\nheader\nheader\n2 1 2\n\n0 0 0 1\n1 -1.0\n2 1.0\n",
        encoding="utf-8",
    )
    (directory / "PROCAR").write_text(
        "PROCAR lm decomposed\n # of k-points: 1 # of bands: 2 # of ions: 1\n",
        encoding="utf-8",
    )
    (directory / "vasprun.xml").write_text("<modeling/>\n", encoding="utf-8")


def prepare_synthetic_run(
    root: Path,
    config: dict,
    *,
    commands: dict[str, list[str]] | None = None,
) -> Path:
    poscar = root / "POSCAR"
    poscar.write_text(POSCAR_2D, encoding="utf-8")
    potcar = root / "POTCAR.synthetic"
    potcar.write_text("synthetic POTCAR X\n", encoding="utf-8")
    template_root = root / "templates"
    template_root.mkdir()
    active = ["02_scf", "03_band"]
    if config["relaxation"]["enabled"]:
        active.insert(0, "01_relax")
    templates: dict[str, dict[str, str]] = {}
    for stage in active:
        incar = template_root / f"INCAR.{stage}"
        kpoints = template_root / f"KPOINTS.{stage}"
        incar.write_text("ENCUT = 400\n", encoding="utf-8")
        kpoints.write_text("synthetic\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
        templates[stage] = {"incar": str(incar), "kpoints": str(kpoints)}
    default_commands = {stage: ["vasp-not-executed"] for stage in active}
    return prepare_preprocess_run(
        root / "run",
        poscar,
        config,
        templates=templates,
        potcar={
            "path": str(potcar),
            "element_order": ["X"],
            "sha256": hashlib.sha256(potcar.read_bytes()).hexdigest(),
        },
        commands=commands or default_commands,
    )


def materialized_inputs(run: Path, stage: str) -> dict[str, Path]:
    names = ["POSCAR", "POTCAR", "INCAR", "KPOINTS"]
    if stage == "03_band":
        names.append("CHGCAR")
    return {name: run / stage / name for name in names}


class PreprocessConfigTests(unittest.TestCase):
    def test_cli_exposes_prepare_run_and_recovery_without_execution_default(self) -> None:
        parser = build_parser()
        prepared = parser.parse_args([
            "preprocess-prepare", "POSCAR", "preprocess.yaml", "--run-dir", "runs/x",
        ])
        self.assertEqual(prepared.command, "preprocess-prepare")
        preview = parser.parse_args(["preprocess-run", "runs/x"])
        self.assertFalse(preview.execute)
        self.assertEqual(
            parser.parse_args(["preprocess-recover", "runs/x", "02_scf"]).stage,
            "02_scf",
        )

    def test_cli_project_resolves_file_inputs_relative_to_yaml(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poscar = root / "POSCAR"
            poscar.write_text(POSCAR_2D, encoding="utf-8")
            document = base_document(relaxation=False)
            document["preprocess_inputs"] = {
                "templates": {
                    "02_scf": {"incar": "templates/INCAR.scf", "kpoints": "templates/KPOINTS.scf"},
                    "03_band": {"incar": "templates/INCAR.band", "kpoints": "templates/KPOINTS.band"},
                },
                "potcar": {"path": "private/POTCAR", "element_order": ["X"], "sha256": "0" * 64},
                "commands": {"02_scf": ["vasp"], "03_band": ["vasp"]},
            }
            config = root / "preprocess.yaml"
            config.write_text(yaml.safe_dump(document), encoding="utf-8")
            protocol, inputs = _preprocess_project(str(config), str(poscar))
            self.assertEqual(protocol["dimensionality"], 2)
            self.assertEqual(inputs["templates"]["02_scf"]["incar"], str((root / "templates/INCAR.scf").resolve()))
            self.assertEqual(inputs["potcar"]["path"], str((root / "private/POTCAR").resolve()))

    def test_standard_profile_needs_only_soc_after_geometry_resolution(self) -> None:
        config = standard_semilocal_preprocess(
            soc=True,
            dimensionality=2,
            periodic_axes=[True, True, False],
        )
        self.assertEqual(config["profile"], "standard_nonmagnetic_semilocal_v1")
        self.assertFalse(config["soc"]["relaxation"])
        self.assertTrue(config["soc"]["scf"])
        self.assertEqual(config["relaxation"]["vacuum_axis"], 2)
        self.assertIn("potcar_variant_must_be_locked_separately", config["profile_limitations"])

    def test_soc_and_relaxation_choices_are_explicit(self) -> None:
        document = base_document()
        del document["preprocess"]["soc"]
        with self.assertRaisesRegex(PreprocessConfigError, "soc"):
            normalize_preprocess_config(document, 2)

        document = base_document()
        del document["preprocess"]["relaxation"]["cell_mode"]
        with self.assertRaisesRegex(PreprocessConfigError, "cell_mode"):
            normalize_preprocess_config(document, 2)

    def test_2d_never_relaxes_vacuum_axis_implicitly(self) -> None:
        document = base_document()
        document["preprocess"]["relaxation"].update({
            "cell_mode": "full_cell",
            "stress_threshold_kbar": 1.0,
            "stress_components": ["xx", "yy", "zz", "xy", "yz", "zx"],
        })
        with self.assertRaisesRegex(PreprocessConfigError, "allow_vacuum_axis_relaxation"):
            normalize_preprocess_config(document, 2)

        document["preprocess"]["relaxation"].update({
            "cell_mode": "constrained_cell",
            "custom_isif": 3,
            "stress_components": ["xx", "yy", "xy"],
            "locked_lattice_axes": [],
        })
        with self.assertRaisesRegex(PreprocessConfigError, "vacuum_axis"):
            normalize_preprocess_config(document, 2)
        document["preprocess"]["relaxation"]["locked_lattice_axes"] = [2]
        normalized = normalize_preprocess_config(document, 2)
        self.assertEqual(normalized["relaxation"]["locked_lattice_axes"], [2])

    def test_plan_has_exact_handoffs_and_soc_contract(self) -> None:
        config = normalize_preprocess_config(base_document(), 2)
        plan = build_preprocess_plan(config)
        self.assertEqual([item["stage"] for item in plan["stages"]], ["01_relax", "02_scf", "03_band"])
        self.assertFalse(stage_incar_contract(config, "01_relax")["LSORBIT"])
        self.assertTrue(stage_incar_contract(config, "02_scf")["LSORBIT"])
        self.assertEqual(stage_incar_contract(config, "03_band")["ICHARG"], 11)

    def test_incar_contract_rejects_duplicate_mandatory_tag(self) -> None:
        with self.assertRaisesRegex(PreprocessConfigError, "EDIFF 重复"):
            patch_incar_contract("EDIFF = 1e-4\nEDIFF = 1e-5\n", {"EDIFF": 1e-6})


class ConvergenceTests(unittest.TestCase):
    def test_checked_in_synthetic_relaxation_fixture(self) -> None:
        config = normalize_preprocess_config(base_document(), 2)
        fixture = FIXTURES / "2d-relax-pass"
        result = analyze_preprocess_outputs(
            fixture,
            "01_relax",
            config,
            source_poscar=fixture / "POSCAR",
        )
        self.assertTrue(result["passed"], result["issues"])

    def test_relaxation_requires_electronic_ionic_force_and_contcar(self) -> None:
        config = normalize_preprocess_config(base_document(), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "POSCAR"
            source.write_text(POSCAR_2D, encoding="utf-8")
            write_relax_outputs(root / "relax")
            result = analyze_preprocess_outputs(root / "relax", "01_relax", config, source_poscar=source)
            self.assertTrue(result["passed"], result["issues"])
            self.assertLess(result["evidence"]["outcar"]["max_force_eV_A"], 0.01)

            write_relax_outputs(root / "bad", force=0.02)
            failed = analyze_preprocess_outputs(root / "bad", "01_relax", config, source_poscar=source)
            self.assertFalse(failed["passed"])
            self.assertIn("最大原子力未达到门槛", failed["issues"])

    def test_locked_2d_vacuum_vector_is_hard_checked(self) -> None:
        document = base_document()
        document["preprocess"]["relaxation"].update({
            "cell_mode": "constrained_cell",
            "custom_isif": 3,
            "stress_threshold_kbar": 1.0,
            "stress_components": ["xx", "yy", "xy"],
            "locked_lattice_axes": [2],
        })
        config = normalize_preprocess_config(document, 2)
        changed = POSCAR_2D.replace("0.0 0.0 20.0", "0.0 0.0 21.0")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "POSCAR"
            source.write_text(POSCAR_2D, encoding="utf-8")
            write_relax_outputs(root / "relax", poscar=changed)
            outcar = root / "relax/OUTCAR"
            outcar.write_text(
                outcar.read_text(encoding="utf-8").replace("ISIF = 2", "ISIF = 3").replace(
                    "reached required accuracy", "in kB 0.1 0.1 9.0 0.1 0.0 0.0\n reached required accuracy"
                ),
                encoding="utf-8",
            )
            result = analyze_preprocess_outputs(root / "relax", "01_relax", config, source_poscar=source)
            self.assertFalse(result["passed"])
            self.assertIn("锁定的晶格轴 2 被改变", result["issues"])

    def test_scf_and_fixed_charge_band_are_distinct_contracts(self) -> None:
        config = normalize_preprocess_config(base_document(relaxation=False), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_scf_outputs(root / "scf")
            write_band_outputs(root / "band")
            self.assertTrue(analyze_preprocess_outputs(root / "scf", "02_scf", config)["passed"])
            band = analyze_preprocess_outputs(root / "band", "03_band", config)
            self.assertTrue(band["passed"], band["issues"])
            self.assertEqual(band["evidence"]["eigenval"]["nkpts"], 1)
            self.assertEqual(band["evidence"]["procar"]["nbands"], 2)

            (root / "band/vasprun.xml").write_text("<modeling>\n", encoding="utf-8")
            incomplete = analyze_preprocess_outputs(root / "band", "03_band", config)
            self.assertFalse(incomplete["passed"])
            self.assertIn(
                "vasprun.xml 缺少完整 modeling 根节点结束标记",
                incomplete["issues"],
            )


class PreprocessStateTests(unittest.TestCase):
    def test_prepare_materializes_locked_inputs_without_execution(self) -> None:
        config = normalize_preprocess_config(base_document(relaxation=False), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = prepare_synthetic_run(root, config)
            self.assertTrue((run / "00_input/POSCAR").is_file())
            self.assertTrue((run / "02_scf/POTCAR").is_file())
            self.assertTrue((run / "03_band/INCAR").is_file())
            self.assertIn("LSORBIT = .TRUE.", (run / "02_scf/INCAR").read_text(encoding="utf-8"))
            self.assertEqual(read_json(run / "02_scf/stage.json")["status"], "READY")
            self.assertEqual(read_json(run / "03_band/stage.json")["status"], "WAITING_HANDOFF")
            self.assertFalse((run / "02_scf/runner-attempt-001.log").exists())

    def test_potcar_recipe_never_guesses_order_or_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "POTCAR.X"
            component.write_bytes(b"X-potcar")
            digest = hashlib.sha256(component.read_bytes()).hexdigest()
            with self.assertRaisesRegex(PreprocessConfigError, "顺序"):
                materialize_potcar_recipe({
                    "components": [{"element": "Y", "path": str(component), "sha256": digest}],
                    "combined_sha256": digest,
                }, ["X"], root / "combined")
            with self.assertRaisesRegex(PreprocessConfigError, "sha256"):
                materialize_potcar_recipe({"path": str(component), "element_order": ["X"]}, ["X"], root / "combined")

    def test_default_dry_run_is_read_only_and_input_hash_is_hard(self) -> None:
        config = normalize_preprocess_config(base_document(relaxation=False), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = prepare_synthetic_run(root, config)
            before = read_json(run / "02_scf/stage.json")
            preview = run_preprocess_workflow(run)
            after = read_json(run / "02_scf/stage.json")
            self.assertEqual(preview["mode"], "DRY_RUN")
            self.assertEqual(preview["next_action"]["stage"], "02_scf")
            self.assertEqual(before, after)
            self.assertFalse(any((run / "02_scf").glob("runner-attempt-*.log")))

            (run / "02_scf/INCAR").write_text("changed input\n", encoding="utf-8")
            with self.assertRaisesRegex(StageRecoveryRequired, "输入哈希"):
                run_preprocess_stage(run, "02_scf")

    def test_synthetic_subprocess_advances_scf_to_band(self) -> None:
        config = normalize_preprocess_config(base_document(relaxation=False), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "synthetic_worker.py"
            worker.write_text(
                '''from pathlib import Path
import sys

root = Path.cwd()
stage = sys.argv[1]
if stage == "02_scf":
    (root / "CHGCAR").write_text("synthetic charge\\n", encoding="utf-8")
    (root / "EIGENVAL").write_text("header\\nheader\\nheader\\nheader\\nheader\\n2 1 2\\n\\n0 0 0 1\\n1 -1.0\\n2 1.0\\n", encoding="utf-8")
    (root / "OSZICAR").write_text(" DAV: 1 -1 0.1\\n DAV: 2 -1.1 0.001\\n 1 F= -1.1 E0= -1.1 d E=0\\n", encoding="utf-8")
    (root / "OUTCAR").write_text(" EDIFF = 1.0E-6\\n NELM = 80\\n LSORBIT = T\\n NSW = 0\\n aborting loop because EDIFF is reached\\n General timing and accounting informations for this job:\\n", encoding="utf-8")
elif stage == "03_band":
    (root / "OSZICAR").write_text(" DAV: 1 -1 0.0\\n 1 F= -1 E0= -1 d E=0\\n", encoding="utf-8")
    (root / "OUTCAR").write_text(" EDIFF = 1.0E-6\\n NELM = 80\\n LSORBIT = T\\n NSW = 0\\n ICHARG = 11\\n General timing and accounting informations for this job:\\n", encoding="utf-8")
    (root / "EIGENVAL").write_text("header\\nheader\\nheader\\nheader\\nheader\\n2 1 2\\n\\n0 0 0 1\\n1 -1.0\\n2 1.0\\n", encoding="utf-8")
    (root / "PROCAR").write_text("PROCAR lm decomposed\\n # of k-points: 1 # of bands: 2 # of ions: 1\\n", encoding="utf-8")
    (root / "vasprun.xml").write_text("<modeling/>\\n", encoding="utf-8")
else:
    raise SystemExit(3)
''',
                encoding="utf-8",
            )
            commands = {
                "02_scf": [sys.executable, str(worker), "02_scf"],
                "03_band": [sys.executable, str(worker), "03_band"],
            }
            run = prepare_synthetic_run(root, config, commands=commands)
            result = run_preprocess_workflow(run, execute=True)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(read_json(run / "02_scf/stage.json")["status"], "PASS")
            self.assertEqual(read_json(run / "03_band/stage.json")["status"], "PASS")
            handoff = read_json(run / "03_band/band-handoff.json")
            self.assertEqual(set(handoff["files"]), {"POSCAR", "OUTCAR", "EIGENVAL", "PROCAR", "vasprun.xml"})
            self.assertEqual(len(result["history"]), 2)

            registry = root / "registry"
            (registry / "X").mkdir(parents=True)
            potcar_payload = (run / "02_scf/POTCAR").read_bytes()
            (registry / "X/POTCAR").write_bytes(potcar_payload)
            potcar_sha = hashlib.sha256(potcar_payload).hexdigest()
            lock_path = root / "POTCAR_LOCK.json"
            lock_path.write_text(json.dumps({
                "registry": str(registry),
                "sets": {
                    "SYNTHETIC_X": {
                        "order": ["X"],
                        "component_sha256": {"X": potcar_sha},
                        "combined_sha256": potcar_sha,
                    }
                },
            }), encoding="utf-8")
            wannier_incar = root / "INCAR.wannier"
            wannier_incar.write_text(
                'LWANNIER90 = .TRUE.\nWANNIER90_WIN = "\nmp_grid = 3 3 1\nnum_wann = 1\nnum_bands = 2\n"\n',
                encoding="utf-8",
            )

            stage1 = {
                "project": "synthetic-handoff",
                "structures": [{"id": "X", "poscar_resolved": str(run / "02_scf/POSCAR")}],
                "physics": {"soc": True, "dimensionality": "auto", "target_energy_eV": [-2.0, 2.0]},
                "backend": {},
                "ranking": {"output_top_k": 1},
                "candidates": {"max_candidates": 1},
                "ensemble": {"mode": "independent"},
                "meshes": {"scf": [1, 1, 1], "wannier": [1, 1, 1]},
                "pseudopotentials": {
                    "set": "SYNTHETIC_X",
                    "lock_resolved": str(lock_path),
                    "registry": str(registry),
                },
                "templates": {
                    "scf": {},
                    "band": {},
                    "wannier": {"incar_resolved": str(wannier_incar), "kpoints_resolved": str(run / "02_scf/KPOINTS")},
                },
            }
            handoff_preview = fit_from_preprocess(
                stage1,
                root / "wannier-runs",
                {"X": run},
                execute=False,
            )
            self.assertEqual(handoff_preview["mode"], "DRY_RUN")
            self.assertEqual(handoff_preview["preprocess_handoffs"]["X"]["structure_analysis"]["dimensionality"], 2)

            bootstrapped = fit_from_preprocess(
                stage1,
                root / "wannier-runs",
                {"X": run},
                bootstrap_only=True,
            )
            self.assertEqual(bootstrapped["mode"], "BOOTSTRAPPED")
            exact_run = Path(bootstrapped["run_dir"])
            self.assertTrue(exact_run.is_dir())
            self.assertEqual(read_json(exact_run / "01_scf/stage.json")["status"], "PASS")
            self.assertEqual(read_json(exact_run / "02_band/stage.json")["status"], "PASS")
            self.assertEqual(read_json(exact_run / "03_candidates/stage.json")["status"], "PENDING")

    def test_lock_refuses_duplicate_and_pass_refuses_resubmit(self) -> None:
        config = normalize_preprocess_config(base_document(relaxation=False), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = prepare_synthetic_run(root, config)
            inputs = materialized_inputs(run, "02_scf")
            lease = begin_stage(run, "02_scf", inputs, ["vasp_std"])
            with self.assertRaises(StageBusyError):
                begin_stage(run, "02_scf", inputs, ["vasp_std"])
            finish_stage(run, "02_scf", {"passed": True, "evidence": {}}, lease)
            with self.assertRaises(StageBusyError):
                begin_stage(run, "02_scf", inputs, ["vasp_std"])
            self.assertEqual(inspect_preprocess_run(run)["next_action"]["stage"], "03_band")

    def test_failure_is_blocked_and_stale_pid_recovery_is_explicit(self) -> None:
        config = normalize_preprocess_config(base_document(relaxation=False), 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = prepare_synthetic_run(root, config)
            inputs = materialized_inputs(run, "02_scf")
            lease = begin_stage(run, "02_scf", inputs, ["vasp_std"])
            owner = read_json(lease.path)
            if lease.fd is not None:
                os.close(lease.fd)
                lease.fd = None
            owner["pid"] = 999_999_999
            atomic_json(lease.path, owner)
            recovered = recover_interrupted_stage(run, "02_scf")
            self.assertEqual(recovered["status"], "INTERRUPTED")
            same_inputs = begin_stage(run, "02_scf", inputs, ["vasp_std"])
            finish_stage(run, "02_scf", {"passed": False, "issues": ["not converged"]}, same_inputs)
            self.assertEqual(inspect_preprocess_run(run)["next_action"]["action"], "SCIENTIFIC_REVIEW_REQUIRED")
            with self.assertRaises(StageRecoveryRequired):
                begin_stage(run, "02_scf", inputs, ["vasp_std"])


if __name__ == "__main__":
    unittest.main()
