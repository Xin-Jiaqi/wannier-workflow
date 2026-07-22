"""Serial, resumable subprocess backend for response plans.

The backend is dry-run unless ``execute=True`` is passed explicitly.  It never
uses a shell, and therefore preserves the exact argv shown by preview mode.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .response_plan import PLAN_SCHEMA, STATUS_SCHEMA, ResponsePlanError, _atomic_json, _read_json


SUCCESS_MARKER = "backend-success.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def command_for_job(job: dict[str, Any]) -> list[str]:
    interface = job["matlab"]
    code_dir = Path(interface["code_dir"])
    config_file = Path(interface["config_file"])
    entry = interface["entry_point"]
    add_paths = [f"addpath('{_matlab_quote(str(code_dir))}')"]
    common_dir = code_dir / "common"
    if common_dir.is_dir():
        add_paths.append(f"addpath('{_matlab_quote(str(common_dir))}')")
    read_config = f"cfg=jsondecode(fileread('{_matlab_quote(str(config_file))}'))"
    if job["response_type"] == "shift_current":
        invoke = f"{entry}(cfg)"
    elif job["response_type"] == "shg":
        invoke = (
            f"batch={entry}(cfg);"
            "assert(batch.n_success==numel(cfg.tensor_components),'response:IncompleteSHG',"
            "'SHG component count mismatch')"
        )
    else:
        raise ResponsePlanError(f"Unsupported response type in plan: {job['response_type']}")
    expression = ";".join([*add_paths, read_config, invoke]) + ";"
    return [str(interface["matlab_executable"]), "-batch", expression]


def _load(plan_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(plan_dir).expanduser().resolve()
    plan = _read_json(root / "response-plan.json")
    status = _read_json(root / "response-status.json")
    if plan.get("schema") != PLAN_SCHEMA or status.get("schema") != STATUS_SCHEMA:
        raise ResponsePlanError("Unknown response plan/status schema")
    if plan.get("plan_sha256") != status.get("plan_sha256"):
        raise ResponsePlanError("Plan and status hashes differ; refusing unsafe resume")
    return root, plan, status


def _marker(root: Path, job: dict[str, Any]) -> Path:
    return root / "jobs" / job["job_id"] / SUCCESS_MARKER


def _matching_marker(root: Path, job: dict[str, Any]) -> bool:
    path = _marker(root, job)
    if not path.is_file():
        return False
    try:
        marker = _read_json(path)
    except ResponsePlanError:
        return False
    return marker.get("fingerprint") == job.get("fingerprint")


def _summarize(status: dict[str, Any]) -> None:
    counts = {name: 0 for name in ("PENDING", "RUNNING", "SUCCEEDED", "FAILED")}
    for record in status["jobs"].values():
        counts[record["state"]] = counts.get(record["state"], 0) + 1
    status["summary"] = counts
    status["updated_at"] = _utc_now()


def _save_status(root: Path, status: dict[str, Any]) -> None:
    _summarize(status)
    _atomic_json(root / "response-status.json", status)


def _pid_is_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reconcile(root: Path, plan: dict[str, Any], status: dict[str, Any]) -> None:
    for job in plan["jobs"]:
        record = status["jobs"].get(job["job_id"])
        if not isinstance(record, dict) or record.get("fingerprint") != job.get("fingerprint"):
            raise ResponsePlanError(f"Status does not match planned job {job['job_id']}")
        if _matching_marker(root, job):
            record["state"] = "SUCCEEDED"
        elif record.get("state") == "RUNNING" and _pid_is_alive(
            record.get("active_process", {}).get("pid")
        ):
            # A previous Python runner may have died while MATLAB survived.
            # Preserve RUNNING so execute mode can attach instead of duplicate.
            continue
        elif record.get("state") == "RUNNING":
            # MATLAB is no longer alive.  Recover a completed run from its
            # scientific outputs before permitting any retry.
            try:
                _mark_success(root, job, record)
            except ResponsePlanError as exc:
                record["state"] = "FAILED"
                record["error"] = f"previous MATLAB pid is gone and outputs are incomplete: {exc}"
            record["reconciled_at"] = _utc_now()
        elif record.get("state") == "SUCCEEDED":
            record["state"] = "PENDING"
            record["reconciled_at"] = _utc_now()


def preview_plan(plan_dir: str | Path) -> dict[str, Any]:
    root, plan, status = _load(plan_dir)
    _reconcile(root, plan, status)
    jobs = []
    for job in plan["jobs"]:
        state = status["jobs"][job["job_id"]]["state"]
        command = command_for_job(job)
        jobs.append(
            {
                "sequence": job["sequence"],
                "job_id": job["job_id"],
                "state": state,
                "response_type": job["response_type"],
                "rank": job["rank"],
                "structure": job["structure"],
                "NOCC": job["nocc_model"],
                "nocc_target_fit_only": job["nocc_target_fit_only"],
                "kmesh": job["kmesh"],
                "energy_grid_eV": job["energy_grid_eV"],
                "components": job["components"],
                "argv": command,
                "shell_preview": shlex.join(command),
            }
        )
    return {"mode": "DRY_RUN", "plan_dir": str(root), "jobs": jobs}


def read_response_status(plan_dir: str | Path) -> dict[str, Any]:
    root, plan, status = _load(plan_dir)
    _reconcile(root, plan, status)
    _summarize(status)
    return status


class _PlanLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "_PlanLock":
        if self.path.is_file():
            try:
                existing = _read_json(self.path)
            except ResponsePlanError:
                existing = {}
            old_pid = existing.get("pid")
            if _pid_is_alive(old_pid):
                raise ResponsePlanError(
                    f"Another live response runner pid={old_pid} holds {self.path}"
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            archived = self.path.with_name(f"{self.path.name}.stale-{stamp}.json")
            self.path.replace(archived)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ResponsePlanError(f"Another response runner holds {self.path}") from exc
        payload = json.dumps({"pid": os.getpid(), "started_at": _utc_now()}) + "\n"
        os.write(self.fd, payload.encode())
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _validate_outputs(job: dict[str, Any]) -> list[str]:
    output = Path(job["output_dir"])
    if job["response_type"] == "shift_current":
        success = output / "_SUCCESS"
        combined = output / "shift-current-all.mat"
        if not success.is_file() or not combined.is_file():
            raise ResponsePlanError(f"MATLAB returned zero but shift-current outputs are incomplete: {output}")
        return [str(success), str(combined)]
    batches = sorted(output.glob("wannier90_shg_batch_*/batch_summary.mat"))
    if not batches:
        raise ResponsePlanError(f"MATLAB returned zero but no SHG batch_summary.mat exists: {output}")
    summary = batches[-1]
    text_summary = summary.with_suffix(".txt")
    if not text_summary.is_file():
        raise ResponsePlanError(f"SHG batch is missing batch_summary.txt: {summary.parent}")
    return [str(summary), str(text_summary)]


def _mark_success(root: Path, job: dict[str, Any], record: dict[str, Any]) -> None:
    evidence = _validate_outputs(job)
    marker = {
        "schema": "w90flow-response-success-v1",
        "job_id": job["job_id"],
        "fingerprint": job["fingerprint"],
        "completed_at": _utc_now(),
        "evidence": evidence,
    }
    _atomic_json(_marker(root, job), marker)
    record["state"] = "SUCCEEDED"
    record["evidence"] = evidence
    record.pop("error", None)


def _attach_existing_process(
    root: Path,
    job: dict[str, Any],
    record: dict[str, Any],
    status: dict[str, Any],
    poll_seconds: float,
) -> None:
    process_info = record.get("active_process", {})
    pid = process_info.get("pid")
    if not _pid_is_alive(pid):
        return
    record["attached_at"] = _utc_now()
    _save_status(root, status)
    while _pid_is_alive(pid):
        time.sleep(poll_seconds)
    process_info["state"] = "EXITED_EXTERNAL"
    process_info["observed_exit_at"] = _utc_now()
    record["finished_at"] = _utc_now()
    try:
        _mark_success(root, job, record)
    except ResponsePlanError as exc:
        # The old process has exited but did not produce complete evidence.
        # Do not relaunch it in this invocation: a later explicit resume may retry.
        record["state"] = "FAILED"
        record["error"] = f"attached MATLAB exited without complete outputs: {exc}"
    _save_status(root, status)


def run_response_plan(
    plan_dir: str | Path,
    *,
    execute: bool = False,
    continue_on_error: bool = False,
    attach_poll_seconds: float = 5.0,
) -> dict[str, Any]:
    """Preview by default, or serially execute and persist resumable state."""

    if not execute:
        return preview_plan(plan_dir)
    root, plan, status = _load(plan_dir)
    with _PlanLock(root / ".response-runner.lock"):
        _reconcile(root, plan, status)
        _save_status(root, status)
        attached: set[str] = set()
        for job in plan["jobs"]:
            record = status["jobs"][job["job_id"]]
            if record["state"] == "RUNNING":
                attached.add(job["job_id"])
                _attach_existing_process(
                    root, job, record, status, max(float(attach_poll_seconds), 0.01)
                )
        for job in plan["jobs"]:
            record = status["jobs"][job["job_id"]]
            if record["state"] == "SUCCEEDED":
                continue
            if job["job_id"] in attached:
                if record["state"] == "FAILED" and not continue_on_error:
                    break
                continue
            command = command_for_job(job)
            entry_file = Path(job["matlab"]["code_dir"]) / f"{job['matlab']['entry_point']}.m"
            config_file = Path(job["matlab"]["config_file"])
            if not entry_file.is_file() or not config_file.is_file():
                record.update(state="FAILED", error="MATLAB entry/config file missing", finished_at=_utc_now())
                _save_status(root, status)
                if not continue_on_error:
                    break
                continue
            output = Path(job["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            log_path = Path(record["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            record["state"] = "RUNNING"
            record["attempts"] = int(record.get("attempts", 0)) + 1
            record["started_at"] = _utc_now()
            record["command_argv"] = command
            _save_status(root, status)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{_utc_now()}] START attempt={record['attempts']}\n")
                log.write(shlex.join(command) + "\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=job["matlab"]["code_dir"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                record["active_process"] = {
                    "pid": process.pid,
                    "state": "RUNNING",
                    "started_at": _utc_now(),
                    "argv": command,
                }
                _save_status(root, status)
                returncode = process.wait()
                log.write(f"[{_utc_now()}] RETURN_CODE={returncode}\n")
            record["returncode"] = returncode
            record["active_process"].update(
                state="EXITED", returncode=returncode, observed_exit_at=_utc_now()
            )
            record["finished_at"] = _utc_now()
            try:
                if returncode != 0:
                    raise ResponsePlanError(f"MATLAB exit code {returncode}")
                _mark_success(root, job, record)
            except ResponsePlanError as exc:
                record["state"] = "FAILED"
                record["error"] = str(exc)
            _save_status(root, status)
            if record["state"] == "FAILED" and not continue_on_error:
                break
    return read_response_status(root)
