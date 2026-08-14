from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from w90flow.attempts import (
    build_standalone_w90_input,
    classify_failure,
    run_attempt,
    validate_standalone_w90_kpoint_order,
)
from w90flow.pyw90_adapter import _resolve_executable


POSCAR = """Synthetic L1 fixture
1.0
  3.000000  0.000000  0.000000
  0.000000  3.000000  0.000000
  0.000000  0.000000 20.000000
X
1
Direct
  0.000000  0.000000  0.500000
"""

INCAR = '''NBANDS = 8
NUM_WANN = 2
WANNIER90_WIN = "
begin projections
X:s
end projections
mp_grid = 2 1 1
begin kpoints
  0.000000 0.000000 0.000000
  0.500000 0.000000 0.000000
end kpoints
"
'''

EIGENVAL = """    1
EIGENVAL
    X
  1.0 1.0 1.0 1.0 1.0
  1.0
 10 2 8
  0.0000000E+00  0.0000000E+00  0.0000000E+00  1.0000000E+00
  1 -5.0
  2 -4.5
  3 -4.0
  4 -3.5
  5 -3.0
  6 -2.5
  7 -2.0
  8 -1.5
  5.0000000E-01  0.0000000E+00  0.0000000E+00  1.0000000E+00
  1 -5.0
  2 -4.5
  3 -4.0
  4 -3.5
  5 -3.0
  6 -2.5
  7 -2.0
  8 -1.5
"""

SNAPSHOT = """backend:
  wannier_input_mode: standalone_win
  mpi_processes: 2
execution:
  allow_execute: true
preprocess_inputs:
  commands:
    04_wannier: [/bin/echo, "{stage_dir}", std, "2"]
    04_wannier90: [/bin/echo, "{stage_dir}", wannier90]
"""


def _win_text() -> str:
    return build_standalone_w90_input(
        INCAR,
        num_wann=2,
        total_vasp_bands=8,
        plan={"mode": "exact", "exclude_bands_1_based": []},
        poscar_text=POSCAR,
    )


class PoscarCellBlockTests(unittest.TestCase):
    @staticmethod
    def _cell_rows(text: str) -> list[list[float]]:
        match = re.search(r"(?ms)^begin unit_cell_cart\n(.*?)^end unit_cell_cart", text)
        assert match is not None
        return [[float(value) for value in line.split()] for line in match.group(1).splitlines()]

    @staticmethod
    def _atom_rows(text: str) -> list[tuple[str, list[float]]]:
        match = re.search(r"(?ms)^begin atoms_cart\n(.*?)^end atoms_cart", text)
        assert match is not None
        return [
            (line.split()[0], [float(value) for value in line.split()[1:4]])
            for line in match.group(1).splitlines()
        ]

    def test_direct_coordinates_convert_to_scaled_cartesian(self) -> None:
        text = build_standalone_w90_input(
            INCAR,
            num_wann=2,
            total_vasp_bands=8,
            plan={"mode": "exact", "exclude_bands_1_based": []},
            poscar_text=POSCAR,
        )
        self.assertIn("begin unit_cell_cart", text)
        self.assertIn("end unit_cell_cart", text)
        self.assertIn("begin atoms_cart", text)
        self.assertIn("end atoms_cart", text)
        cell = self._cell_rows(text)
        self.assertEqual(len(cell), 3)
        self.assertEqual(cell[0], [3.0, 0.0, 0.0])
        self.assertEqual(cell[2], [0.0, 0.0, 20.0])
        self.assertEqual(self._atom_rows(text), [("X", [0.0, 0.0, 10.0])])

    def test_cartesian_poscar_scales_by_lattice_constant(self) -> None:
        poscar = """Cartesian fixture
2.0
  2.000000  0.000000  0.000000
  0.000000  2.000000  0.000000
  0.000000  0.000000  4.000000
Y
1
Cartesian
  0.500000  0.250000  1.000000
"""
        text = build_standalone_w90_input(
            INCAR,
            num_wann=2,
            total_vasp_bands=8,
            plan={"mode": "exact", "exclude_bands_1_based": []},
            poscar_text=poscar,
        )
        cell = self._cell_rows(text)
        self.assertEqual(cell[0], [4.0, 0.0, 0.0])
        self.assertEqual(cell[2], [0.0, 0.0, 8.0])
        self.assertEqual(self._atom_rows(text), [("Y", [1.0, 0.5, 2.0])])

    def test_truncated_poscar_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "POSCAR 内容不完整"):
            build_standalone_w90_input(
                INCAR,
                num_wann=2,
                total_vasp_bands=8,
                plan={"mode": "exact", "exclude_bands_1_based": []},
                poscar_text="only\none\n",
            )


class KpointIndexValidationTests(unittest.TestCase):
    def _write_ibzkpt(self, path: Path, points: list[list[float]]) -> None:
        body = "\n".join(
            f"  {x:.7E} {y:.7E} {z:.7E}  1.0000000E+00" for x, y, z in points
        )
        path.write_text(
            f"    2\n    {len(points)}\n  1   1   1\n{body}\n",
            encoding="utf-8",
        )

    def test_identical_mesh_passes_and_reports_max_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "wannier90.win").write_text(_win_text(), encoding="utf-8")
            self._write_ibzkpt(directory / "IBZKPT", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
            result = validate_standalone_w90_kpoint_order(
                directory / "wannier90.win",
                directory / "IBZKPT",
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["kpoint_count"], 2)
            self.assertEqual(result["maximum_wrapped_fractional_delta"], 0.0)

    def test_equivalent_wrapped_coordinates_still_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            win = _win_text().replace("0.500000 0.000000 0.000000", "-0.500000 0.000000 0.000000")
            (directory / "wannier90.win").write_text(win, encoding="utf-8")
            self._write_ibzkpt(directory / "IBZKPT", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
            result = validate_standalone_w90_kpoint_order(
                directory / "wannier90.win",
                directory / "IBZKPT",
            )
            self.assertTrue(result["passed"])

    def test_reordered_mesh_is_rejected_with_mismatch_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "wannier90.win").write_text(_win_text(), encoding="utf-8")
            self._write_ibzkpt(directory / "IBZKPT", [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
            with self.assertRaisesRegex(ValueError, "KPOINT_INDEX_ORDER_MISMATCH"):
                validate_standalone_w90_kpoint_order(
                    directory / "wannier90.win",
                    directory / "IBZKPT",
                )

    def test_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "wannier90.win").write_text(_win_text(), encoding="utf-8")
            self._write_ibzkpt(directory / "IBZKPT", [[0.0, 0.0, 0.0]])
            with self.assertRaisesRegex(ValueError, "KPOINT_INDEX_ORDER_MISMATCH.*k 点数不同"):
                validate_standalone_w90_kpoint_order(
                    directory / "wannier90.win",
                    directory / "IBZKPT",
                )

    def test_classify_failure_keeps_kpoint_mismatch_out_of_numeric_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "runner.log").write_text(
                "W90FLOW_VASP54_COMPATIBILITY_ERROR: KPOINT_INDEX_ORDER_MISMATCH: 第 2 个 VASP/Wannier k 点不同",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_failure(directory, 1),
                "KPOINT_INDEX_ORDER_MISMATCH",
            )


class RunAttemptKpointGuardTests(unittest.TestCase):
    def _make_run(self, temporary: str) -> tuple[Path, Path]:
        root = Path(temporary) / "run"
        attempt = root / "structures/X/04_wannier_attempts/p-only/attempt-001"
        (root / "structures/X/02_band").mkdir(parents=True)
        attempt.mkdir(parents=True)
        (root / "project.snapshot.yaml").write_text(SNAPSHOT, encoding="utf-8")
        (root / "structures/X/02_band/EIGENVAL").write_text(EIGENVAL, encoding="utf-8")
        (attempt / "attempt.json").write_text(
            json.dumps({
                "status": "READY",
                "num_wann": 2,
                "plan": {"mode": "exact", "exclude_bands_1_based": []},
            }),
            encoding="utf-8",
        )
        (attempt / "POSCAR").write_text(POSCAR, encoding="utf-8")
        (attempt / "POTCAR").write_text("synthetic", encoding="utf-8")
        (attempt / "INCAR").write_text(INCAR, encoding="utf-8")
        (attempt / "KPOINTS").write_text("Automatic mesh\n0\nGamma\n2 1 1\n", encoding="utf-8")
        (attempt / "CHGCAR").write_text("synthetic", encoding="utf-8")
        return root, attempt

    def _fake_process(self) -> mock.MagicMock:
        process = mock.MagicMock()
        process.pid = 12345
        process.wait.return_value = 0
        return process

    def test_kpoint_mismatch_fails_closed_and_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, attempt = self._make_run(temporary)
            ibzkpt = attempt / "IBZKPT"
            ibzkpt.write_text(
                "    2\n    2\n  2   1   1\n"
                "  0.0000000E+00  0.0000000E+00  0.0000000E+00  1.0000000E+00\n"
                "  0.0000000E+00  5.0000000E-01  0.0000000E+00  1.0000000E+00\n",
                encoding="utf-8",
            )
            for name in ("wannier90.wout", "wannier90_hr.dat", "wannier90_centres.xyz", "wannier90_band.dat"):
                (attempt / name).write_text("All done\n" if name == "wannier90.wout" else "x", encoding="utf-8")
            with mock.patch("w90flow.attempts.subprocess.Popen", return_value=self._fake_process()) as popen:
                result = run_attempt(root, "X", "p-only", "attempt-001", execute=True)
            popen.assert_called_once()
            self.assertFalse(result["passed"])
            self.assertEqual(result["attempt"]["status"], "FAIL")
            self.assertEqual(result["attempt"]["failure_class"], "KPOINT_INDEX_ORDER_MISMATCH")
            self.assertEqual(
                result["attempt"]["next_action"],
                "rebuild_wannier_inputs_from_one_shared_explicit_mesh",
            )
            self.assertNotIn("standalone_w90_kpoint_order", result["attempt"])
            log = (attempt / "runner.log").read_text(encoding="utf-8", errors="replace")
            self.assertIn("KPOINT_INDEX_ORDER_MISMATCH", log)

    def test_consistent_mesh_records_validation_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, attempt = self._make_run(temporary)
            (attempt / "IBZKPT").write_text(
                "    2\n    2\n  2   1   1\n"
                "  0.0000000E+00  0.0000000E+00  0.0000000E+00  1.0000000E+00\n"
                "  5.0000000E-01  0.0000000E+00  0.0000000E+00  1.0000000E+00\n",
                encoding="utf-8",
            )
            for name in ("wannier90.wout", "wannier90_hr.dat", "wannier90_centres.xyz", "wannier90_band.dat"):
                (attempt / name).write_text("All done\n" if name == "wannier90.wout" else "x", encoding="utf-8")
            with mock.patch("w90flow.attempts.subprocess.Popen", return_value=self._fake_process()) as popen:
                result = run_attempt(root, "X", "p-only", "attempt-001", execute=True)
            popen.assert_called_once()
            self.assertTrue(result["passed"])
            self.assertEqual(result["attempt"]["status"], "PASS")
            self.assertEqual(result["attempt"]["standalone_w90_kpoint_order"]["passed"], True)
            self.assertEqual(result["attempt"]["standalone_w90_kpoint_order"]["kpoint_count"], 2)
            self.assertEqual(result["attempt"]["standalone_w90_post_vasp"]["shifted_components"], 1)


class PyW90ExecutableResolutionTests(unittest.TestCase):
    def test_explicit_path_wins_over_path_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            explicit = Path(temporary) / "pyw90-explicit"
            explicit.write_text("#!/bin/sh\n", encoding="utf-8")
            explicit.chmod(0o755)
            with mock.patch("w90flow.pyw90_adapter.shutil.which", return_value="/other/pyw90"):
                resolved = _resolve_executable(explicit)
            self.assertEqual(resolved, explicit.resolve())

    def test_path_discovery_when_no_explicit_file(self) -> None:
        with mock.patch("w90flow.pyw90_adapter.shutil.which", return_value="/usr/local/bin/pyw90"):
            resolved = _resolve_executable("pyw90")
        self.assertEqual(resolved, Path("/usr/local/bin/pyw90"))

    def test_missing_executable_raises_clear_error(self) -> None:
        with mock.patch("w90flow.pyw90_adapter.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "找不到 PyW90 可执行文件"):
                _resolve_executable("/no/such/pyw90")

    def test_none_raises_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "W90FLOW_PYW90_EXECUTABLE"):
            _resolve_executable(None)


if __name__ == "__main__":
    unittest.main()
