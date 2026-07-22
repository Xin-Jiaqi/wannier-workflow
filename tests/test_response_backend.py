from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from w90flow.matlab_backend import command_for_job, preview_plan, run_response_plan
from w90flow.response_plan import ResponsePlanError, build_response_plan


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ResponseBackendTests(unittest.TestCase):
    def make_campaign(self, root: Path, *, quality_nocc_model: int = 3) -> Path:
        export = root / "topk"
        rank_dir = export / "rank-01_p-only"
        model = rank_dir / "L1"
        model.mkdir(parents=True)
        (model / "POSCAR").write_text("test\n1\n1 0 0\n0 1 0\n0 0 10\nX\n1\nDirect\n0 0 0\n", encoding="utf-8")
        (model / "wannier90_hr.dat").write_text("test\n4\n1\n1\n", encoding="utf-8")
        (model / "wannier90-wcenter.dat").write_text(
            "X 0 0 0\nX 0 0 1\nX 0 0 2\nX 0 0 3\n", encoding="utf-8"
        )
        quality = {
            "num_wann": 4,
            "nocc_target": 2,
            "nocc_model": quality_nocc_model,
            "passed": True,
            "checks": {"hr_centres_match": True},
        }
        write_json(model / "quality.json", quality)
        files = {}
        import hashlib

        for path in model.iterdir():
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "rank": 1,
            "candidate": "p-only",
            "attempt": "attempt-001",
            "projection": ["X:p"],
            "num_wann": 4,
            "nocc_model": 3,
            "structures": {"L1": {"files": files, "metrics": {}}},
        }
        write_json(rank_dir / "package-manifest.json", manifest)
        write_json(
            export / "export-summary.json",
            {"schema_version": 1, "packages": [{"path": "/stale/mount/rank-01_p-only", **manifest}]},
        )
        shift_code = root / "matlab/shift"
        shg_code = root / "matlab/shg"
        (shg_code / "common").mkdir(parents=True)
        shift_code.mkdir(parents=True)
        (shift_code / "compute_shift_current_model.m").write_text("function compute_shift_current_model(x)\nend\n")
        (shg_code / "SHG_unified_optimized.m").write_text("function x=SHG_unified_optimized(c)\nend\n")
        spec = {
            "schema_version": 1,
            "system": "X",
            "export_summary": str(export / "export-summary.json"),
            "plan_dir": str(root / "plan"),
            "output_root": str(root / "outputs"),
            "responses": {
                "shift_current": {
                    "code_dir": str(shift_code),
                    "profile": "preflight",
                    "components": ["xxy", "yyy"],
                    "kmesh": [3, 3, 1],
                    "energy_grid_eV": {"min": 0.0, "max": 3.5, "points": 5},
                    "broadening_eV": 0.05,
                    "thickness_A": 5.0,
                    "parallel_workers": 1,
                },
                "shg": {
                    "code_dir": str(shg_code),
                    "profile": "preflight",
                    "components": ["112", "222"],
                    "kmesh": [1, 1, 1],
                    "energy_grid_eV": {"min": 0.0, "max": 3.5, "points": 5},
                    "spectral_broadening_eV": 0.05,
                    "effective_thickness_A": 5.0,
                },
            },
        }
        spec_path = root / "campaign.json"
        write_json(spec_path, spec)
        return spec_path

    def test_build_orders_jobs_and_uses_nocc_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_response_plan(self.make_campaign(root))
            self.assertEqual([job["response_type"] for job in plan["jobs"]], ["shift_current", "shg"])
            for job in plan["jobs"]:
                self.assertEqual(job["nocc_model"], 3)
                self.assertEqual(job["nocc_target_fit_only"], 2)
                self.assertEqual(job["matlab_config"]["nocc"], 3)
                self.assertTrue(Path(job["inputs"]["wannier90_hr.dat"]).is_file())
            preview = preview_plan(root / "plan")
            self.assertEqual(preview["mode"], "DRY_RUN")
            self.assertTrue(all(item["argv"][1] == "-batch" for item in preview["jobs"]))
            self.assertIn("compute_shift_current_model", command_for_job(plan["jobs"][0])[2])

    def test_manifest_quality_nocc_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ResponsePlanError, "nocc_model mismatch"):
                build_response_plan(self.make_campaign(root, quality_nocc_model=2))

    def test_default_run_never_starts_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_response_plan(self.make_campaign(root))
            with mock.patch("w90flow.matlab_backend.subprocess.Popen") as run:
                result = run_response_plan(root / "plan")
            run.assert_not_called()
            self.assertEqual(result["mode"], "DRY_RUN")

    def test_serial_resume_skips_matching_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_response_plan(self.make_campaign(root))
            first, second = plan["jobs"]
            marker = root / "plan/jobs" / first["job_id"] / "backend-success.json"
            write_json(marker, {"fingerprint": first["fingerprint"]})

            def fake_run(command: list[str], **_: object) -> mock.Mock:
                job = second
                output = Path(job["output_dir"]) / "wannier90_shg_batch_1"
                output.mkdir(parents=True, exist_ok=True)
                (output / "batch_summary.mat").write_text("mat")
                (output / "batch_summary.txt").write_text("txt")
                return mock.Mock(pid=987654, wait=mock.Mock(return_value=0))

            with mock.patch("w90flow.matlab_backend.subprocess.Popen", side_effect=fake_run) as run:
                result = run_response_plan(root / "plan", execute=True)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(result["summary"]["SUCCEEDED"], 2)

    def test_resume_attaches_live_matlab_without_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_response_plan(self.make_campaign(root))
            first = plan["jobs"][0]
            status_path = root / "plan/response-status.json"
            status = json.loads(status_path.read_text())
            status["jobs"][first["job_id"]].update(
                state="RUNNING", active_process={"pid": 43210, "state": "RUNNING"}
            )
            write_json(status_path, status)
            alive = iter([True, True, False])

            def pid_alive(pid: int) -> bool:
                if pid == 43210:
                    return next(alive, False)
                return False

            output = Path(first["output_dir"])
            output.mkdir(parents=True)
            (output / "_SUCCESS").write_text("ok")
            (output / "shift-current-all.mat").write_text("mat")
            # Mark second complete so this test isolates the attachment path.
            second = plan["jobs"][1]
            write_json(
                root / "plan/jobs" / second["job_id"] / "backend-success.json",
                {"fingerprint": second["fingerprint"]},
            )
            with mock.patch("w90flow.matlab_backend._pid_is_alive", side_effect=pid_alive), mock.patch(
                "w90flow.matlab_backend.time.sleep"
            ), mock.patch("w90flow.matlab_backend.subprocess.Popen") as popen:
                result = run_response_plan(root / "plan", execute=True, attach_poll_seconds=0.01)
            popen.assert_not_called()
            self.assertEqual(result["summary"]["SUCCEEDED"], 2)

    def test_stale_lock_is_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_response_plan(self.make_campaign(root))
            for job in plan["jobs"]:
                write_json(
                    root / "plan/jobs" / job["job_id"] / "backend-success.json",
                    {"fingerprint": job["fingerprint"]},
                )
            write_json(root / "plan/.response-runner.lock", {"pid": 99999999})
            with mock.patch("w90flow.matlab_backend._pid_is_alive", return_value=False):
                run_response_plan(root / "plan", execute=True)
            self.assertTrue(list((root / "plan").glob(".response-runner.lock.stale-*.json")))

    def test_dead_recorded_pid_with_complete_outputs_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_response_plan(self.make_campaign(root))
            first, second = plan["jobs"]
            status_path = root / "plan/response-status.json"
            status = json.loads(status_path.read_text())
            status["jobs"][first["job_id"]].update(
                state="RUNNING", active_process={"pid": 54321, "state": "RUNNING"}
            )
            write_json(status_path, status)
            output = Path(first["output_dir"])
            output.mkdir(parents=True)
            (output / "_SUCCESS").write_text("ok")
            (output / "shift-current-all.mat").write_text("mat")
            write_json(
                root / "plan/jobs" / second["job_id"] / "backend-success.json",
                {"fingerprint": second["fingerprint"]},
            )
            with mock.patch("w90flow.matlab_backend._pid_is_alive", return_value=False), mock.patch(
                "w90flow.matlab_backend.subprocess.Popen"
            ) as popen:
                result = run_response_plan(root / "plan", execute=True)
            popen.assert_not_called()
            self.assertEqual(result["summary"]["SUCCEEDED"], 2)


if __name__ == "__main__":
    unittest.main()
