"""Focused tests: Wannier site launchers come only from explicit config fields.

Covers the fixed contract:
  preprocess_inputs.commands.04_wannier   = [<VASP wrapper>, "{stage_dir}", "std|ncl", "<mpi>"]
  preprocess_inputs.commands.04_wannier90 = [<Wannier90 wrapper>, "{stage_dir}", "wannier90"]
with the running mode ("pre"/"run") appended at runtime.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from w90flow.attempts import _commands
from w90flow.coldstart import _validate_execution_contract
from w90flow.executor import _launcher, _wannier_launcher

VASP_WRAPPER = "/site/bin/run-vasp"
W90_WRAPPER = "/site/bin/run-wannier90"


def _config(*, soc: bool = True, wannier_input_mode: str = "embedded") -> dict:
    kind = "ncl" if soc else "std"
    return {
        "physics": {"soc": soc, "dimensionality": 2},
        "backend": {
            "vasp": kind,
            "mpi_processes": 4,
            "wannier_input_mode": wannier_input_mode,
        },
        "execution": {"allow_execute": True},
        "preprocess_inputs": {
            "commands": {
                "02_scf": [VASP_WRAPPER, "{stage_dir}", kind, "4"],
                "03_band": [VASP_WRAPPER, "{stage_dir}", kind, "4"],
                "04_wannier": [VASP_WRAPPER, "{stage_dir}", kind, "4"],
                "04_wannier90": [W90_WRAPPER, "{stage_dir}", "wannier90"],
            }
        },
    }


def _protocol(*, soc: bool = True) -> dict:
    return {
        "soc": {"relaxation": False, "scf": soc, "band": soc},
        "relaxation": {"enabled": False},
    }


class AttemptsCommandResolutionTests(unittest.TestCase):
    def _attempt(self, root: Path) -> Path:
        return root / "structures/LiAuI4_BL/04_wannier_attempts/all-relevant/attempt-001"

    def test_japan_style_commands_build_embedded_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = self._attempt(root)
            chain = _commands(_config(), directory, run_dir=root)
            self.assertEqual(chain[0], [VASP_WRAPPER, str(directory), "ncl", "4"])
            self.assertEqual(chain[1], [W90_WRAPPER, str(directory), "wannier90", "run"])
            self.assertEqual(len(chain), 2)

    def test_standalone_win_appends_pre_then_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = self._attempt(root)
            chain = _commands(_config(wannier_input_mode="standalone_win"), directory, run_dir=root)
            self.assertEqual(chain[0], [VASP_WRAPPER, str(directory), "ncl", "4"])
            self.assertEqual(chain[1], [W90_WRAPPER, str(directory), "wannier90", "pre"])
            self.assertEqual(chain[2], [W90_WRAPPER, str(directory), "wannier90", "run"])

    def test_missing_04_wannier_is_config_error_not_home_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = self._attempt(root)
            cfg = _config()
            del cfg["preprocess_inputs"]["commands"]["04_wannier"]
            with self.assertRaisesRegex(ValueError, "04_wannier"):
                _commands(cfg, directory, run_dir=root)

    def test_missing_04_wannier90_is_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = self._attempt(root)
            cfg = _config()
            del cfg["preprocess_inputs"]["commands"]["04_wannier90"]
            with self.assertRaisesRegex(ValueError, "04_wannier90"):
                _commands(cfg, directory, run_dir=root)

    def test_bad_wannier90_seed_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = self._attempt(root)
            cfg = _config()
            cfg["preprocess_inputs"]["commands"]["04_wannier90"] = [W90_WRAPPER, "{stage_dir}", "wannier90.x"]
            with self.assertRaisesRegex(ValueError, "wannier90"):
                _commands(cfg, directory, run_dir=root)

    def test_placeholder_expansion_covers_stage_and_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = self._attempt(root)
            cfg = _config()
            cfg["preprocess_inputs"]["commands"]["04_wannier"] = [
                VASP_WRAPPER, "{stage_dir}", "{run_dir}", "ncl", "4",
            ]
            chain = _commands(cfg, directory, run_dir=root)
            self.assertEqual(chain[0], [VASP_WRAPPER, str(directory), str(root), "ncl", "4"])

    def test_home_legacy_wrapper_is_never_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            legacy = home / "00_命令入口/计算/运行VASP.command"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("#!/bin/sh\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                root = home / "run"
                directory = self._attempt(root)
                chain = _commands(_config(), directory, run_dir=root)
                self.assertEqual(chain[0][0], VASP_WRAPPER)
                cfg = _config()
                del cfg["preprocess_inputs"]["commands"]["04_wannier"]
                with self.assertRaisesRegex(ValueError, "04_wannier"):
                    _commands(cfg, directory, run_dir=root)


class ExecutorLauncherResolutionTests(unittest.TestCase):
    def test_scf_band_wannier_share_one_site_vasp_wrapper(self) -> None:
        cfg = _config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            commands = [
                _launcher(root, cfg, root / "structures/LiAuI4_BL/01_scf", "scf"),
                _launcher(root, cfg, root / "structures/LiAuI4_BL/02_band", "band"),
                _launcher(root, cfg, root / "structures/LiAuI4_BL/04_wannier", "wannier"),
            ]
            for command in commands:
                self.assertEqual(command[0], VASP_WRAPPER)
            self.assertEqual({tuple(command) for command in commands}, {
                (VASP_WRAPPER, str(root / "structures/LiAuI4_BL/01_scf"), "ncl", "4"),
                (VASP_WRAPPER, str(root / "structures/LiAuI4_BL/02_band"), "ncl", "4"),
                (VASP_WRAPPER, str(root / "structures/LiAuI4_BL/04_wannier"), "ncl", "4"),
            })

    def test_wannier90_launcher_appends_mode_only(self) -> None:
        cfg = _config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = root / "structures/LiAuI4_BL/04_wannier"
            pre = _wannier_launcher(root, cfg, directory, mode="pre")
            run = _wannier_launcher(root, cfg, directory, mode="run")
            self.assertEqual(pre, [W90_WRAPPER, str(directory), "wannier90", "pre"])
            self.assertEqual(run, [W90_WRAPPER, str(directory), "wannier90", "run"])
            with self.assertRaisesRegex(ValueError, "pre 或 run"):
                _wannier_launcher(root, cfg, directory, mode="plot")

    def test_missing_wannier_commands_fail_closed_in_executor(self) -> None:
        cfg = _config()
        del cfg["preprocess_inputs"]["commands"]["04_wannier"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = root / "structures/LiAuI4_BL/04_wannier"
            with self.assertRaisesRegex(ValueError, "04_wannier"):
                _launcher(root, cfg, directory, "wannier")
            cfg2 = _config()
            del cfg2["preprocess_inputs"]["commands"]["04_wannier90"]
            with self.assertRaisesRegex(ValueError, "04_wannier90"):
                _wannier_launcher(root, cfg2, directory)


class ColdstartContractValidationTests(unittest.TestCase):
    def test_contract_accepts_japan_style_commands(self) -> None:
        cfg = _config()
        checked = _validate_execution_contract(cfg, _protocol(), cfg["preprocess_inputs"]["commands"])
        self.assertEqual(checked["preprocess_commands"]["04_wannier"]["vasp"], "ncl")
        self.assertEqual(checked["preprocess_commands"]["04_wannier90"]["seed"], "wannier90")

    def test_contract_rejects_missing_wannier_keys(self) -> None:
        cfg = _config()
        commands = dict(cfg["preprocess_inputs"]["commands"])
        del commands["04_wannier"]
        with self.assertRaisesRegex(ValueError, "04_wannier"):
            _validate_execution_contract(cfg, _protocol(), commands)
        commands = dict(cfg["preprocess_inputs"]["commands"])
        del commands["04_wannier90"]
        with self.assertRaisesRegex(ValueError, "04_wannier90"):
            _validate_execution_contract(cfg, _protocol(), commands)

    def test_contract_rejects_bad_kind_and_seed(self) -> None:
        cfg = _config()
        commands = dict(cfg["preprocess_inputs"]["commands"])
        commands["04_wannier"] = [VASP_WRAPPER, "{stage_dir}", "std", "4"]
        with self.assertRaisesRegex(ValueError, "04_wannier 应使用 ncl"):
            _validate_execution_contract(cfg, _protocol(), commands)
        commands = dict(cfg["preprocess_inputs"]["commands"])
        commands["04_wannier90"] = [W90_WRAPPER, "{stage_dir}", "wannier90.x"]
        with self.assertRaisesRegex(ValueError, "wannier90"):
            _validate_execution_contract(cfg, _protocol(), commands)
        commands = dict(cfg["preprocess_inputs"]["commands"])
        commands["04_wannier"] = [VASP_WRAPPER, "{stage_dir}", "ncl", "2"]
        with self.assertRaisesRegex(ValueError, "backend.mpi_processes=4"):
            _validate_execution_contract(cfg, _protocol(), commands)


if __name__ == "__main__":
    unittest.main()
