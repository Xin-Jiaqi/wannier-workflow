from __future__ import annotations

import json
import os
import signal
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from w90flow.config import load_project
from w90flow.attempts import classify_failure, import_completed_attempt, materialize_plan_energy_reference, plan_candidate, recorded_or_classified_failure
from w90flow.candidates import normalize_projection, num_wann_from_projection, suggest_projection_candidates
from w90flow.dataset import make_attempt_record
from w90flow.material_pool import screen_c2db
from w90flow.orchestrator import RunLock, WorkflowBlocked, choose_retry_reason, fit_from_precomputed, group_has_runnable_member, inspect_workflow, run_workflow
from w90flow.parsers import parse_eigenval, parse_fermi_outcar, parse_hr_header, parse_poscar, parse_procar_orbital_scores
from w90flow.quality import IncompleteTargetSubspace, _best_contiguous_w90_slice, _common_attempt_ranking, _is_complete_sp, band_metrics, export_ranked_attempts
from w90flow.pipeline import audit_reference, plan_project
from w90flow.pyw90_adapter import element_projection_envelope, parse_eig_suggest_output, parse_pre_dos_output
from w90flow.ranking import Thresholds, audit_ranking_document, rank_common_candidates
from w90flow.state import initialize_run, read_status, update_stage
from w90flow.supervisor import ProgressWatch, SupervisorLock, inspect_supervisor, progress_fingerprint, supervise_workflow
from w90flow.structure import generate_ase_band_path, patch_wannier_uniform_grid, resolve_periodicity, resolve_uniform_mesh
from w90flow.windows import minimum_outer_max, validate_frozen_window, validate_outer_window, validate_window_containment


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY / "tests/fixtures"
REFERENCE_ROOT_VALUE = os.environ.get("W90FLOW_REFERENCE_ROOT")
EXAMPLES = Path(REFERENCE_ROOT_VALUE).expanduser().resolve() if REFERENCE_ROOT_VALUE else None
HAS_REFERENCE_FIXTURES = EXAMPLES is not None and EXAMPLES.is_dir()
CONFIG = FIXTURES / "project/pair-example.yaml"


class ParserTests(unittest.TestCase):
    def test_pyw90_eig_suggest_table_parser(self) -> None:
        output = """
Calculated Energy Range: -3.6, 0.5 with Fermi level 1.000000
There are at most 4 states and at least 2 states in [-4.6, -0.5].
`dis_win_min` and `dis_win_max` Table:
   dis_win_min  dis_win_max   i+1_min     i_max  Nleast  Nmost
3    -0.135904     5.826912  0.096635 -0.368443       2      4
`dis_froz_min` and `dis_froz_max` Table:
0    -2.0     0.2  8
"""
        parsed = parse_eig_suggest_output(output)
        self.assertEqual(parsed["calculated_energy_range_eV"], [-3.6, 0.5])
        self.assertEqual(parsed["state_count"]["minimum"], 2)
        self.assertEqual(parsed["outer_window_suggestions"][0]["nmost"], 4)
        self.assertEqual(parsed["frozen_window_suggestions"][0]["states"], 8)

    def test_pyw90_pre_dos_projection_parser(self) -> None:
        parsed = parse_pre_dos_output(
            """\x1b[31mFermi level : -3.27340 eV\x1b[0m
DOS Gap     : 1.31030 eV
Number of WFs selected: 28 (with degeneracy 2)
Wannier90 Projection:
In:l=0;l=1
Se:l=1

pyw90 --extra input:
In,0-1,0-3;Se,2-3,1-3
"""
        )
        self.assertEqual(parsed["projection"], ["In:s", "In:p", "Se:p"])
        self.assertEqual(parsed["num_wann_selected"], 28)
        self.assertEqual(parsed["degeneracy"], 2)
        self.assertAlmostEqual(parsed["dos_gap_eV"], 1.3103)
        represented = element_projection_envelope(parsed, {"In": 2, "Se": 2})
        self.assertEqual(represented["projection"], ["In:s", "In:p", "Se:p"])
        self.assertEqual(represented["representation"], "exact_element_level")

    def test_pyw90_pre_dos_component_projection_parser(self) -> None:
        parsed = parse_pre_dos_output(
            """Number of WFs selected: 20 (with degeneracy 2)
Wannier90 Projection:
In:l=0;px;py
Se:py;px

pyw90 --extra input:
In,0-1,0-1|3;Se,2-3,1|3
"""
        )
        self.assertEqual(parsed["projection"], ["In:s", "In:px", "In:py", "Se:py", "Se:px"])
        self.assertEqual(
            num_wann_from_projection(parsed["projection"], {"In": 2, "Se": 2}, True),
            20,
        )

    def test_pyw90_site_specific_advice_becomes_audited_element_envelope(self) -> None:
        parsed = parse_pre_dos_output(
            """Number of WFs selected: 8 (with degeneracy 1)
Wannier90 Projection:
f=0.000000,0.500000,0.572750:py;px
f=0.500000,0.000000,0.427250:l=1
Si:l=1

pyw90 --extra input:
S,1,1|3;S,2,1-3;Si,0,1-3
"""
        )
        represented = element_projection_envelope(parsed, {"Si": 1, "S": 2})
        self.assertEqual(represented["projection"], ["Si:p", "S:p"])
        self.assertEqual(represented["representation"], "element_level_envelope")
        self.assertFalse(represented["site_uniform"])
        self.assertEqual(num_wann_from_projection(represented["projection"], {"Si": 1, "S": 2}, False), 9)

    def test_final_fermi_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outcar = Path(temporary) / "OUTCAR"
            outcar.write_text(" E-fermi : 1.25 eV\n E-fermi : 2.75 eV\n", encoding="utf-8")
            self.assertEqual(parse_fermi_outcar(outcar), 2.75)

    @unittest.skipUnless(HAS_REFERENCE_FIXTURES, "set W90FLOW_REFERENCE_ROOT for licensed/large reference fixtures")
    def test_real_eigenval_and_hr(self) -> None:
        assert EXAMPLES is not None
        example = EXAMPLES / "01_BiTeI_nSOC"
        eig = parse_eigenval(example / "02_band/EIGENVAL")
        hr = parse_hr_header(example / "04_wannier/wannier90_hr.dat")
        self.assertEqual((eig["nkpts"], eig["nbands"]), (123, 32))
        self.assertEqual(hr["num_wann"], 9)

    @unittest.skipUnless(HAS_REFERENCE_FIXTURES, "set W90FLOW_REFERENCE_ROOT for licensed/large reference fixtures")
    def test_real_procar_orbital_scores(self) -> None:
        assert EXAMPLES is not None
        example = EXAMPLES / "01_BiTeI_nSOC"
        poscar = parse_poscar(example / "00_input/POSCAR")
        scores = parse_procar_orbital_scores(example / "02_band/PROCAR", poscar, -4.6, -0.5)
        self.assertGreater(scores["selected_state_count"], 0)
        self.assertGreater(sum(value for key, value in scores["orbital_scores"].items() if key.endswith(":p")), 0.8)


class StructureTests(unittest.TestCase):
    def test_2d_and_3d_periodicity_and_mesh(self) -> None:
        sheet = parse_poscar(FIXTURES / "project/POSCAR.L1")
        two_d = resolve_periodicity(sheet, {"dimensionality": "auto"})
        self.assertEqual(two_d["dimensionality"], 2)
        self.assertEqual(two_d["periodic_axes"], [True, True, False])
        mesh = resolve_uniform_mesh(
            {"spacing_Ainv": 0.5, "maximum": 15},
            sheet["lattice_angstrom"],
            two_d["periodic_axes"],
        )
        self.assertEqual(mesh[2], 1)
        self.assertGreater(mesh[0], 1)

        with tempfile.TemporaryDirectory() as temporary:
            bulk_path = Path(temporary) / "POSCAR"
            bulk_path.write_text(
                "bulk\n1.0\n4 0 0\n0 4 0\n0 0 4\nX\n1\nDirect\n0 0 0\n",
                encoding="utf-8",
            )
            bulk = parse_poscar(bulk_path)
            three_d = resolve_periodicity(bulk, {"dimensionality": "auto"})
            self.assertEqual(three_d["dimensionality"], 3)
            self.assertEqual(three_d["periodic_axes"], [True, True, True])
            self.assertTrue(all(value > 1 for value in resolve_uniform_mesh(
                {"spacing_Ainv": 0.6}, bulk["lattice_angstrom"], three_d["periodic_axes"]
            )))

    def test_uniform_grid_patch_and_native_2d_path(self) -> None:
        incar = 'LWANNIER90 = T\nWANNIER90_WIN = "\nmp_grid = 1 1 1\nbegin projections\nX:p\nend projections\n"\n'
        patched = patch_wannier_uniform_grid(incar, [2, 2, 1])
        self.assertIn("mp_grid = 2 2 1", patched)
        self.assertEqual(patched.count("begin kpoints"), 1)
        self.assertEqual(patched.count("end kpoints"), 1)
        try:
            path = generate_ase_band_path(
                parse_poscar(FIXTURES / "project/POSCAR.L1"),
                [True, True, False],
                20,
            )
        except RuntimeError:
            self.skipTest("ASE optional dependency is not installed")
        self.assertTrue(path["segments"])
        self.assertTrue(all(abs(float(value)) < 1e-12 for line in path["wannier_kpoint_path"] for value in [line.split()[-1]]))

    def test_negative_poscar_scale_volume_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "POSCAR"
            path.write_text(
                "volume form\n-64\n1 0 0\n0 1 0\n0 0 1\nX\n1\nCartesian\n0.5 0.5 0.5\n",
                encoding="utf-8",
            )
            parsed = parse_poscar(path)
            self.assertAlmostEqual(parsed["lattice_angstrom"][0][0], 4.0)
            self.assertEqual(parsed["fractional_positions"][0], [0.5, 0.5, 0.5])


class WindowTests(unittest.TestCase):
    def test_state_constraints(self) -> None:
        energies = [[-2.0, -1.0, 0.0, 1.0], [-1.9, -0.8, 0.2, 1.2]]
        self.assertTrue(validate_outer_window(energies, -2.1, 0.3, 3)["valid"])
        self.assertFalse(validate_outer_window(energies, -1.5, 0.3, 3)["valid"])
        self.assertTrue(validate_frozen_window(energies, -2.1, 0.3, 3)["valid"])
        self.assertAlmostEqual(minimum_outer_max(energies, -2.1, 3, 0.02), 0.22)
        self.assertFalse(validate_window_containment(-2.0, 2.0, -2.1, 1.0)["valid"])

    def test_fermi_relative_plan_materialization(self) -> None:
        relative = {
            "mode": "disentanglement",
            "dis_win_min_eV": -6.0,
            "dis_win_max_eV": 4.0,
            "dis_froz_min_eV": -5.0,
            "dis_froz_max_eV": 2.0,
        }
        absolute = materialize_plan_energy_reference(relative, 3.25, "fermi_relative")
        self.assertEqual(absolute["dis_win_min_eV"], -2.75)
        self.assertEqual(absolute["dis_froz_max_eV"], 5.25)
        self.assertEqual(relative["dis_win_min_eV"], -6.0)


class CandidateTests(unittest.TestCase):
    def test_component_projection_dimension(self) -> None:
        self.assertEqual(
            num_wann_from_projection(
                ["In:s", "In:px", "In:py", "Se:px", "Se:py"],
                {"In": 2, "Se": 2},
                True,
            ),
            20,
        )
        self.assertEqual(
            normalize_projection(["In:p", "In:px", "In:py", "Se:px"]),
            ["In:p", "Se:px"],
        )
        self.assertEqual(
            num_wann_from_projection(["In:p", "In:px", "Se:px"], {"In": 2, "Se": 2}, True),
            16,
        )

    def test_soc_dimension_and_nested_candidates(self) -> None:
        counts = {"Bi": 1, "Te": 1, "I": 1}
        self.assertEqual(num_wann_from_projection(["Bi:p", "Te:p", "I:p"], counts, True), 18)
        candidates = suggest_projection_candidates(
            {"Bi:p": 0.4, "Te:p": 0.3, "I:p": 0.2, "Bi:s": 0.06, "Te:s": 0.03, "I:s": 0.01},
            counts,
            True,
            ["Bi:p", "Te:p", "I:p"],
            max_num_wann=24,
        )
        self.assertGreaterEqual(len(candidates), 3)
        self.assertTrue(all(item["num_wann"] % 2 == 0 for item in candidates))
        self.assertTrue(all(item["num_wann"] <= 24 for item in candidates))
        self.assertEqual(candidates[0]["candidate"], "weight-80")

        without_answer = suggest_projection_candidates(
            {"In:s": 0.18, "In:p": 0.42, "Se:s": 0.03, "Se:p": 0.37},
            {"In": 2, "Se": 2},
            True,
            chemical_valence=None,
            max_num_wann=40,
        )
        p_only = next(item for item in without_answer if item["candidate"] == "p-only")
        self.assertEqual(p_only["projection"], ["In:p", "Se:p"])
        self.assertEqual(p_only["num_wann"], 24)

    def test_dp_shell_family_is_generated_without_material_rules(self) -> None:
        candidates = suggest_projection_candidates(
            {"Pt:d": 0.44, "O:p": 0.41, "Pt:s": 0.08, "O:s": 0.07},
            {"Pt": 1, "O": 2},
            True,
            max_num_wann=40,
            max_candidates=8,
        )
        names = {item["candidate"] for item in candidates}
        self.assertIn("d-only", names)
        self.assertIn("p-only", names)
        self.assertTrue(any(name.startswith("complete-") and "d" in name and "p" in name for name in names))

    def test_external_projection_baseline_is_quality_gated_candidate_not_answer(self) -> None:
        candidates = suggest_projection_candidates(
            {"In:s": 0.24, "In:p": 0.24, "In:d": 0.05, "Se:s": 0.03, "Se:p": 0.44},
            {"In": 2, "Se": 2},
            True,
            max_num_wann=40,
            external_candidates=[{
                "candidate": "pyw90-dos",
                "projection": ["In:s", "Se:p"],
                "candidate_family": "external_baseline",
            }],
        )
        baseline = next(item for item in candidates if item["candidate"] == "pyw90-dos")
        self.assertEqual(baseline["num_wann"], 16)
        self.assertEqual(baseline["candidate_family"], "external_baseline")

        components = suggest_projection_candidates(
            {"In:s": 0.2, "In:p": 0.4, "Se:p": 0.4},
            {"In": 2, "Se": 2},
            True,
            max_num_wann=40,
            external_candidates=[{
                "candidate": "pyw90-components",
                "projection": ["In:s", "In:px", "In:py", "Se:px", "Se:py"],
            }],
        )
        component = next(item for item in components if item["candidate"] == "pyw90-components")
        self.assertEqual(component["num_wann"], 20)
        self.assertEqual(component["integrated_weight_estimate"], "equal_component_split")
        self.assertAlmostEqual(component["integrated_weight_fraction"], 11 / 15)

    def test_iteration_budget_controls_disentanglement_plans(self) -> None:
        energies = [[-2.0, -1.0, 0.0, 1.0], [-1.9, -0.8, 0.2, 1.2]]
        features = [{"orbital_scores": {"X:p": 1.0}} for _ in range(4)]
        candidate = {"projection": ["X:p"], "num_wann": 3}
        plans = plan_candidate(
            energies,
            features,
            candidate,
            -1.1,
            0.3,
            isolation_threshold_eV=10.0,
            max_window_attempts=1,
            max_localization_retries=0,
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["mode"], "disentanglement")

        expanded = plan_candidate(
            energies,
            features,
            candidate,
            -1.1,
            0.3,
            isolation_threshold_eV=10.0,
            max_window_attempts=3,
            max_localization_retries=2,
        )
        self.assertEqual(len(expanded), 5)
        self.assertEqual(
            [plan.get("retry_reason", "seed") for plan in expanded],
            [
                "seed",
                "outer_window_expansion_001",
                "outer_window_expansion_002",
                "convergence_retry_001",
                "convergence_retry_002",
            ],
        )
        self.assertEqual(expanded[2]["retry_class"], "outer_window_expansion")
        self.assertEqual(expanded[4]["retry_class"], "convergence_retry")

    def test_candidate_smaller_than_target_manifold_is_rejected_before_vasp(self) -> None:
        energies = [
            [-5.0, -3.0, -2.0, -1.0, 0.0, 2.0],
            [-4.9, -2.9, -1.9, -0.9, 0.1, 2.1],
        ]
        features = [{"orbital_scores": {"X:s": 1.0}} for _ in range(6)]
        candidate = {"projection": ["X:s"], "num_wann": 3}
        self.assertEqual(plan_candidate(energies, features, candidate, -3.2, 0.2), [])

    def test_dataset_failure_label_required(self) -> None:
        record = make_attempt_record(
            material_id="local:BiTeI",
            structure_hash="abc",
            candidate="p-only",
            projection=["Bi:p", "Te:p", "I:p"],
            num_wann=18,
            soc=True,
            status="FAIL",
            failure_class="localization_not_converged",
        )
        self.assertEqual(record["status"], "FAIL")

    def test_exact_band_block_planner(self) -> None:
        energies = [
            [-5.0, -3.0, -2.0, -1.0, 1.0, 2.0],
            [-4.9, -3.1, -2.1, -0.9, 1.1, 2.1],
        ]
        features = [
            {"orbital_scores": {"X:s": 0.9}},
            {"orbital_scores": {"X:p": 0.9}},
            {"orbital_scores": {"X:p": 0.9}},
            {"orbital_scores": {"X:p": 0.9}},
            {"orbital_scores": {"X:d": 0.9}},
            {"orbital_scores": {"X:d": 0.9}},
        ]
        candidate = {"projection": ["X:p"], "num_wann": 3}
        plans = plan_candidate(energies, features, candidate, -3.2, -0.8)
        self.assertEqual(plans[0]["mode"], "exact")
        self.assertEqual(plans[0]["retained_vasp_bands_1_based"], [2, 4])
        disentangled = plan_candidate(energies, features, candidate, -3.2, -0.8, isolation_threshold_eV=10.0)[0]
        self.assertEqual(disentangled["mode"], "disentanglement")
        self.assertLessEqual(disentangled["dis_win_min_eV"], disentangled["dis_froz_min_eV"])
        self.assertTrue(disentangled["containment_check"]["valid"])

    def test_attempt_import_rejects_any_input_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            target = root / "structures/X/04_wannier_attempts/p-only/attempt-001"
            source = Path(temporary) / "source"
            target.mkdir(parents=True)
            source.mkdir()
            (target / "attempt.json").write_text(
                json.dumps({"status": "READY", "num_wann": 3}), encoding="utf-8"
            )
            for name in ("POSCAR", "POTCAR", "KPOINTS", "CHGCAR"):
                (target / name).write_text("same", encoding="utf-8")
                (source / name).write_text("same", encoding="utf-8")
            (target / "INCAR").write_text("new", encoding="utf-8")
            (source / "INCAR").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "INCAR SHA256"):
                import_completed_attempt(root, "X", "p-only", "attempt-001", source)

    def test_mpi_socket_denial_is_infrastructure_not_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "runner.log").write_text(
                "bind() failed for port 0: Operation not permitted\nNo sockets were able to be opened",
                encoding="utf-8",
            )
            self.assertEqual(classify_failure(path, 213), "INFRASTRUCTURE_FAILURE")

    def test_explicit_orphan_diagnosis_overrides_truncated_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "runner.log").write_text("DAV: 4 incomplete local run\n", encoding="utf-8")
            record = {"status": "FAIL", "returncode": -9, "failure_class": "INFRASTRUCTURE_FAILURE"}
            self.assertEqual(recorded_or_classified_failure(record, path), "INFRASTRUCTURE_FAILURE")


class RankingTests(unittest.TestCase):
    def test_declared_target_window_excludes_outside_band_segments_from_fit_error(self) -> None:
        dft = [
            [(0.0, -1.0), (1.0, -0.8)],
            [(0.0, 1.0), (1.0, 5.0)],
        ]
        w90 = [
            [(0.0, -1.001), (1.0, -0.801)],
            [(0.0, 1.001), (1.0, 6.0)],
        ]
        full, _ = band_metrics(dft, w90, 1)
        target, _ = band_metrics(
            dft,
            w90,
            1,
            error_energy_window_eV=(-2.0, 2.0),
        )
        self.assertGreater(full["rmse_mev"], 400.0)
        self.assertLess(target["rmse_mev"], 2.0)
        self.assertEqual(target["compared_point_count"], 3)
        self.assertEqual(target["excluded_point_count"], 1)
        self.assertEqual(target["error_energy_window_eV"], [-2.0, 2.0])
        self.assertAlmostEqual(target["gap_error_mev"], 2.0)

    def test_export_independent_top_k_without_potcar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "structures/SnS2/04_wannier_attempts/complete-sp/attempt-001"
            source.mkdir(parents=True)
            for name in ("POSCAR", "wannier90_hr.dat", "VASP_W90_cmp.pdf", "quality.json", "wannier90.win"):
                (source / name).write_text(name, encoding="utf-8")
            (source / "wannier90_centres.xyz").write_text(
                "2\ncentres\nX 0 0 0\nX 1 1 1\n",
                encoding="utf-8",
            )
            metrics = {"mae_mev": 1, "rmse_mev": 2, "p95_mev": 3, "max_mev": 4, "gap_error_mev": 1}
            ranking = {
                "ensemble_mode": "independent",
                "structures": {
                    "SnS2": {
                        "top_k": [{
                            "rank": 1,
                            "candidate": "complete-sp",
                            "attempt": "attempt-001",
                            "projection": ["Sn:s", "Sn:p"],
                            "num_wann": 2,
                            "nocc_model": 1,
                            "metrics": metrics,
                            "quality": str(source / "quality.json"),
                        }]
                    }
                },
            }
            rank_dir = root / "06_rank"
            rank_dir.mkdir()
            (rank_dir / "candidate-ranking-v04.json").write_text(json.dumps(ranking), encoding="utf-8")
            (root / "project.snapshot.yaml").write_text(
                "physics:\n  soc: true\n  dimensionality: 2\n  axes: [x, y, z]\n",
                encoding="utf-8",
            )
            result = export_ranked_attempts(root)
            package = Path(result["packages"][0]["path"])
            self.assertTrue((package / "SnS2/wannier90-wcenter.dat").is_file())
            self.assertFalse(any(path.name == "POTCAR" for path in package.rglob("*")))
            self.assertEqual(result["ensemble_mode"], "independent")
            manifest = json.loads((package / "package-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["physics"]["dimensionality"], 2)
            self.assertTrue(manifest["physics"]["soc"])

    def test_target_mapping_and_complete_shell_tie_break(self) -> None:
        dft = [[(0.0, -1.0), (1.0, -0.8)], [(0.0, 1.0), (1.0, 1.2)]]
        w90 = [
            [(0.0, -5.0), (1.0, -5.0)],
            [(0.0, -1.001), (1.0, -0.801)],
            [(0.0, 1.001), (1.0, 1.201)],
        ]
        start, metrics = _best_contiguous_w90_slice(dft, w90, 1)
        self.assertEqual(start, 1)
        self.assertLess(metrics["rmse_mev"], 2.0)
        self.assertTrue(_is_complete_sp(["X:s", "X:p", "Y:s", "Y:p"]))
        self.assertFalse(_is_complete_sp(["X:p", "Y:s", "Y:p"]))

    def test_target_window_controls_disentangled_band_mapping(self) -> None:
        dft = [
            [(0.0, -1.0), (1.0, -0.8)],
            [(0.0, 1.0), (1.0, 5.0)],
        ]
        w90 = [
            [(0.0, -1.001), (1.0, -0.801)],
            [(0.0, 1.001), (1.0, 6.0)],
            [(0.0, 8.0), (1.0, 8.0)],
        ]
        start, metrics = _best_contiguous_w90_slice(
            dft,
            w90,
            1,
            error_energy_window_eV=(-2.0, 2.0),
        )
        self.assertEqual(start, 0)
        self.assertLess(metrics["rmse_mev"], 2.0)

    def test_postfit_target_subspace_rejection_keeps_auditable_dimensions(self) -> None:
        dft = [[(0.0, -1.0)], [(0.0, 1.0)]]
        w90 = [[(0.0, -1.0)]]
        with self.assertRaises(IncompleteTargetSubspace) as captured:
            _best_contiguous_w90_slice(dft, w90, 1)
        self.assertEqual(captured.exception.target_band_count, 2)
        self.assertEqual(captured.exception.model_dimension, 1)

    def test_worst_member_ranking_and_rejection(self) -> None:
        good = {"mae_mev": 1, "rmse_mev": 2, "p95_mev": 3, "max_mev": 5, "gap_error_mev": 1}
        candidates = {
            "A": {"projection": ["X:p"], "num_wann": 6, "structures": {"L1": {"metrics": good}, "L2": {"metrics": {**good, "p95_mev": 4}}}},
            "B": {"projection": ["X:s", "X:p"], "num_wann": 8, "structures": {"L1": {"metrics": {**good, "p95_mev": 2}}, "L2": {"metrics": {**good, "p95_mev": 3}}}},
            "C": {"projection": ["X:d"], "num_wann": 10, "structures": {"L1": {"metrics": good}}},
        }
        result = rank_common_candidates(candidates, ["L1", "L2"], Thresholds())
        self.assertEqual([item["candidate"] for item in result["ranking"]], ["B", "A"])
        self.assertEqual(result["rejected"][0]["reason"], "missing_structures")

    def test_common_attempt_ranking_never_mixes_retry_windows(self) -> None:
        def result(candidate: str, attempt: str, p95: float, passed: bool = True) -> dict:
            return {
                "candidate": candidate,
                "attempt": attempt,
                "projection": ["X:s", "X:p"],
                "num_wann": 8,
                "nocc_model": 4,
                "metrics": {"mae_mev": 1, "rmse_mev": 2, "p95_mev": p95, "max_mev": 5, "gap_error_mev": 1},
                "passed": passed,
                "quality": f"{candidate}-{attempt}.json",
            }

        ranked = _common_attempt_ranking({
            "L1": [result("A", "attempt-001", 2), result("A", "attempt-002", 1)],
            "L2": [result("A", "attempt-001", 4), result("A", "attempt-002", 9, passed=False)],
        }, ["L1", "L2"])
        self.assertEqual(len(ranked["ranking"]), 1)
        self.assertEqual(ranked["ranking"][0]["attempt"], "attempt-001")
        self.assertEqual(ranked["ranking"][0]["pair_worst_metrics"]["p95_mev"], 4)

    @unittest.skipUnless(HAS_REFERENCE_FIXTURES, "set W90FLOW_REFERENCE_ROOT for prepublication reference fixtures")
    def test_real_ranking_documents(self) -> None:
        assert EXAMPLES is not None
        for relative in (
            "02_BiTeI_SOC_pair/projection-ranking.json",
            "03_InSe_SOC_pair/projection-ranking.json",
            "03_InSe_SOC_pair/03_disentanglement-v04/candidate-ranking-v04.json",
        ):
            document = json.loads((EXAMPLES / relative).read_text(encoding="utf-8"))
            self.assertTrue(audit_ranking_document(document)["passed"])


class WorkflowTests(unittest.TestCase):
    def test_config_plan_and_immutable_state(self) -> None:
        config = load_project(CONFIG)
        plan = plan_project(config)
        self.assertEqual(plan["mode"], "PLAN_ONLY")
        self.assertEqual(len(plan["structures"]), 2)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            statuses = read_status(run)["stages"]
            self.assertEqual(len(statuses), 8)
            self.assertTrue(all(item["status"] == "PENDING" for item in statuses))

    def test_explicit_run_dir_is_unique_and_must_stay_inside_runs_root(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            fixed = root / "deterministic-coldstart"
            run = initialize_run(config, root, run_dir=fixed)
            self.assertEqual(run, fixed.resolve())
            with self.assertRaises(FileExistsError):
                initialize_run(config, root, run_dir=fixed)
            with self.assertRaises(ValueError):
                initialize_run(config, root, run_dir=Path(temporary) / "outside")

    def test_fit_from_bands_dry_run_is_single_entrypoint(self) -> None:
        config = load_project(CONFIG)
        result = fit_from_precomputed(
            config,
            "unused",
            {"L1": "/scf/L1", "L2": "/scf/L2"},
            {"L1": "/band/L1", "L2": "/band/L2"},
            execute=False,
        )
        self.assertEqual(result["mode"], "DRY_RUN")
        self.assertIn("CHGCAR", result["contract"]["band_required"])

    def test_fit_from_bands_rejects_execute_and_bootstrap_together(self) -> None:
        config = load_project(CONFIG)
        with self.assertRaises(ValueError):
            fit_from_precomputed(
                config,
                "unused",
                {"L1": "/scf/L1", "L2": "/scf/L2"},
                {"L1": "/band/L1", "L2": "/band/L2"},
                execute=True,
                bootstrap_only=True,
            )

    @unittest.skipUnless(HAS_REFERENCE_FIXTURES, "set W90FLOW_REFERENCE_ROOT for prepublication reference fixtures")
    def test_all_reference_examples(self) -> None:
        assert EXAMPLES is not None
        for name in ("01_BiTeI_nSOC", "02_BiTeI_SOC_pair", "03_InSe_SOC_pair"):
            with self.subTest(name=name):
                self.assertTrue(audit_reference(EXAMPLES / name)["passed"])

    def test_supervisor_dry_run_does_not_mutate_stages(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            before = read_status(run)
            preview = run_workflow(run, execute=False, until="export")
            after = read_status(run)
            self.assertEqual(preview["mode"], "DRY_RUN")
            self.assertEqual(preview["next_action"]["stage"], "00_validate")
            self.assertEqual(before, after)
            self.assertFalse((run / "workflow-summary.json").exists())

    def test_supervisor_refuses_unprepared_run(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            result = run_workflow(run, execute=True, until="scf")
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("prepare", result["reason"])
            self.assertEqual(read_status(run)["stages"][1]["status"], "PENDING")

    def test_run_lock_prevents_duplicate_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with RunLock(root):
                self.assertTrue((root / ".w90flow.lock").is_file())
                with self.assertRaises(WorkflowBlocked):
                    with RunLock(root):
                        pass
            self.assertFalse((root / ".w90flow.lock").exists())

    def test_retry_policy_is_finite_and_does_not_relax_quality(self) -> None:
        self.assertEqual(choose_retry_reason(["INFRASTRUCTURE_FAILURE"], set()), "same")
        self.assertEqual(
            choose_retry_reason(["LOCALIZATION_NOT_CONVERGED"], {"seed"}),
            "convergence_retry",
        )
        self.assertEqual(
            choose_retry_reason(["QUALITY_GATE_FAILED"], {"seed"}),
            "outer_window_expansion",
        )
        self.assertIsNone(
            choose_retry_reason(
                ["QUALITY_GATE_FAILED"],
                {"seed", "outer_window_expansion"},
            )
        )
        with self.assertRaises(WorkflowBlocked):
            choose_retry_reason(["NUMERIC_FAILURE"], set())
        self.assertEqual(
            choose_retry_reason(
                ["LOCALIZATION_NOT_CONVERGED"],
                {"seed", "convergence_retry_001"},
                {"outer_window_expansion", "convergence_retry"},
            ),
            "convergence_retry",
        )
        self.assertIsNone(
            choose_retry_reason(
                ["LOCALIZATION_NOT_CONVERGED"],
                {"seed", "convergence_retry_001", "convergence_retry_002"},
                {"outer_window_expansion"},
            )
        )

    def test_common_attempt_resume_only_runs_missing_members(self) -> None:
        self.assertTrue(group_has_runnable_member({"ACCEPTED", "READY"}))
        self.assertTrue(group_has_runnable_member({"PASS", "READY"}))
        self.assertFalse(group_has_runnable_member({"ACCEPTED"}))
        self.assertFalse(group_has_runnable_member({"REJECTED_QUALITY", "READY"}))

    def test_inspect_reports_first_incomplete_stage(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            state = inspect_workflow(run)
            self.assertEqual(state["next_action"], {
                "action": "advance_stage",
                "stage": "00_validate",
                "status": "PENDING",
            })

    def test_long_supervisor_dry_run_is_read_only(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            before = read_status(run)
            preview = supervise_workflow(
                run,
                execute=False,
                until="quality",
                poll_seconds=0.1,
                heartbeat_seconds=0.2,
                no_progress_seconds=1.0,
                stage_timeout_seconds=2.0,
            )
            self.assertEqual(preview["mode"], "DRY_RUN")
            self.assertEqual(preview["targets"][-1], "quality")
            self.assertEqual(before, read_status(run))
            self.assertFalse((run / "supervisor-state.json").exists())

    def test_supervisor_lock_prevents_duplicate_poller(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with SupervisorLock(root):
                with self.assertRaises(WorkflowBlocked):
                    with SupervisorLock(root):
                        pass

    def test_progress_watch_reports_once_without_mutating_thresholds(self) -> None:
        watch = ProgressWatch(
            started=0.0,
            last_change=0.0,
            fingerprint=(("runner.log", 1, 1),),
            no_progress_seconds=10.0,
            stage_timeout_seconds=20.0,
        )
        self.assertEqual(watch.observe((("runner.log", 2, 2),), 9.0), [])
        self.assertEqual(watch.observe((("runner.log", 2, 2),), 19.0), ["NO_PROGRESS_TIMEOUT"])
        self.assertEqual(watch.observe((("runner.log", 2, 2),), 21.0), ["STAGE_TIMEOUT"])
        self.assertEqual(watch.observe((("runner.log", 2, 2),), 30.0), [])

    def test_progress_fingerprint_and_offline_status(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            first = progress_fingerprint(run)
            log = run / "structures/L1/01_scf/runner.log"
            log.write_text("first\n", encoding="utf-8")
            second = progress_fingerprint(run)
            self.assertNotEqual(first, second)
            status = inspect_supervisor(run)
            self.assertEqual(status["supervisor"]["status"], "NOT_STARTED")
            self.assertFalse(status["supervisor_alive"])

    def test_supervisor_polls_fake_child_then_crosses_boundary(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            update_stage(run, "00_validate", status="PASS")

            class FakeChild:
                pid = 987654321
                returncode = None

                def __init__(self, *args, **kwargs) -> None:
                    self.poll_count = 0

                def poll(self):
                    self.poll_count += 1
                    if self.poll_count >= 2:
                        update_stage(run, "01_scf", status="PASS")
                        self.returncode = 0
                    return self.returncode

            with mock.patch("w90flow.supervisor.subprocess.Popen", FakeChild), mock.patch(
                "w90flow.supervisor.time.sleep", return_value=None
            ):
                result = supervise_workflow(
                    run,
                    execute=True,
                    until="scf",
                    poll_seconds=0.01,
                    heartbeat_seconds=0.01,
                    no_progress_seconds=10.0,
                    stage_timeout_seconds=20.0,
                )
            self.assertTrue(result["passed"])
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(read_status(run)["stages"][1]["status"], "PASS")

    def test_supervisor_waits_orphan_worker_before_launching_child(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            update_stage(run, "00_validate", status="PASS")
            events: list[str] = []

            def fake_workers(root):
                if "worker_wait" not in events:
                    return [{"pid": 424242, "alive": True, "state_file": "01_scf/stage.json"}]
                return []

            def fake_sleep(seconds):
                if "worker_wait" not in events:
                    events.append("worker_wait")

            class FakeChild:
                pid = 987654320
                returncode = None

                def __init__(self, *args, **kwargs) -> None:
                    events.append("child_launch")
                    self.poll_count = 0

                def poll(self):
                    self.poll_count += 1
                    if self.poll_count >= 2:
                        update_stage(run, "01_scf", status="PASS")
                        self.returncode = 0
                    return self.returncode

            with mock.patch("w90flow.supervisor._active_workers", side_effect=fake_workers), mock.patch(
                "w90flow.supervisor.subprocess.Popen", FakeChild
            ), mock.patch("w90flow.supervisor.time.sleep", side_effect=fake_sleep):
                result = supervise_workflow(
                    run,
                    execute=True,
                    until="scf",
                    poll_seconds=0.01,
                    heartbeat_seconds=0.01,
                    no_progress_seconds=10.0,
                    stage_timeout_seconds=20.0,
                )
            self.assertTrue(result["passed"])
            self.assertLess(events.index("worker_wait"), events.index("child_launch"))
            self.assertIn("no replacement child launched", (run / "supervisor.log").read_text(encoding="utf-8"))

    def test_signal_checkpoints_then_waits_for_fake_stage_boundary(self) -> None:
        config = load_project(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = initialize_run(config, temporary)
            update_stage(run, "00_validate", status="PASS")

            class FakeChild:
                pid = 987654319
                returncode = None

                def __init__(self, *args, **kwargs) -> None:
                    self.poll_count = 0

                def poll(self):
                    self.poll_count += 1
                    if self.poll_count == 1:
                        signal.raise_signal(signal.SIGINT)
                    elif self.poll_count >= 2:
                        update_stage(run, "01_scf", status="PASS")
                        self.returncode = 0
                    return self.returncode

            with mock.patch("w90flow.supervisor.subprocess.Popen", FakeChild), mock.patch(
                "w90flow.supervisor.time.sleep", return_value=None
            ):
                result = supervise_workflow(
                    run,
                    execute=True,
                    until="scf",
                    poll_seconds=0.01,
                    heartbeat_seconds=0.01,
                    no_progress_seconds=10.0,
                    stage_timeout_seconds=20.0,
                )
            self.assertTrue(result["passed"])
            self.assertEqual(result["status"], "PAUSED_BOUNDARY")
            self.assertEqual(result["pause_reason"], "signal_SIGINT")
            self.assertEqual(result["checkpoint"]["last_passed"], "scf")


class MaterialPoolTests(unittest.TestCase):
    def test_read_only_c2db_screen_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "mini.db"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE systems (id INTEGER PRIMARY KEY, unique_id TEXT, natoms INTEGER, key_value_pairs TEXT)"
            )
            metadata = {
                "folder": "/source/SnS2/SnS2-hash",
                "uid": "SnS2-1",
                "has_inversion_symmetry": False,
                "gap": 1.4,
                "ehull": 0.02,
                "dyn_stab": "Unknown",
            }
            connection.execute(
                "INSERT INTO systems VALUES (?, ?, ?, ?)", (1, "ase-id", 3, json.dumps(metadata))
            )
            connection.commit()
            connection.close()
            result = screen_c2db(database, {"limit": 10})
            self.assertTrue(result["read_only"])
            self.assertEqual(result["selected_count"], 1)
            self.assertEqual(result["candidates"][0]["formula"], "SnS2")
            self.assertEqual(result["candidates"][0]["c2db_uid"], "SnS2-1")


if __name__ == "__main__":
    unittest.main()
