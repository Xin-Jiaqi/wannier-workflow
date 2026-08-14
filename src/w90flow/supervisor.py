from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .orchestrator import STAGE_NAME, WORKFLOW_STOPS, WorkflowBlocked, _pid_alive, inspect_workflow
from .state import atomic_json, read_json


STATE_FILE = "supervisor-state.json"
LOG_FILE = "supervisor.log"
LOCK_FILE = ".w90flow-supervisor.lock"
PROGRESS_FILENAMES = {
    "stage.json",
    "attempt.json",
    "runner.log",
    "OUTCAR",
    "OSZICAR",
    "vasprun.xml",
    "EIGENVAL",
    "PROCAR",
    "wannier90.wout",
    "wannier90_hr.dat",
    "quality.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(root: Path, message: str) -> None:
    line = f"[{_utc_now()}] {message}\n"
    with (root / LOG_FILE).open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _unique_stale_path(path: Path, pid: int) -> Path:
    stem = f"{path.name}.stale-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{pid}"
    candidate = path.with_name(stem)
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{stem}-{suffix:02d}")
        suffix += 1
    return candidate


class SupervisorLock:
    """Separate lock for the long-lived poller; the child owns RunLock."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / LOCK_FILE
        self.acquired = False

    def __enter__(self) -> "SupervisorLock":
        if self.path.exists():
            try:
                old = read_json(self.path)
                pid = int(old.get("pid") or 0)
                host = str(old.get("host") or "")
            except (OSError, ValueError, json.JSONDecodeError):
                pid, host = 0, ""
            if host == socket.gethostname() and _pid_alive(pid):
                raise WorkflowBlocked(f"同一 run 已由监督器 PID {pid} 管理，拒绝重复调度。")
            os.replace(self.path, _unique_stale_path(self.path, pid))
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise WorkflowBlocked("另一个监督器刚刚获得了同一 run 的锁。") from exc
        try:
            payload = {
                "schema_version": 1,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_utc": _utc_now(),
            }
            os.write(descriptor, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired or not self.path.exists():
            return
        try:
            current = read_json(self.path)
        except (OSError, json.JSONDecodeError):
            current = {}
        if int(current.get("pid") or 0) == os.getpid():
            self.path.unlink()


@dataclass
class ProgressWatch:
    """Deterministic timeout detector. It reports but never edits science input."""

    started: float
    last_change: float
    fingerprint: tuple[Any, ...]
    no_progress_seconds: float
    stage_timeout_seconds: float
    reported: set[str] = field(default_factory=set)

    def observe(self, fingerprint: tuple[Any, ...], now: float) -> list[str]:
        if fingerprint != self.fingerprint:
            self.fingerprint = fingerprint
            self.last_change = now
        alerts: list[str] = []
        if self.no_progress_seconds > 0 and now - self.last_change >= self.no_progress_seconds:
            alerts.append("NO_PROGRESS_TIMEOUT")
        if self.stage_timeout_seconds > 0 and now - self.started >= self.stage_timeout_seconds:
            alerts.append("STAGE_TIMEOUT")
        fresh = [code for code in alerts if code not in self.reported]
        self.reported.update(fresh)
        return fresh


def progress_fingerprint(run_dir: str | Path) -> tuple[Any, ...]:
    """Small metadata-only fingerprint; never reads large scientific payloads."""
    root = Path(run_dir).expanduser().resolve()
    records: list[tuple[str, int, int]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in PROGRESS_FILENAMES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(records))


def _active_workers(root: Path) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    for path in list(root.glob("??_*/stage.json")) + list(root.glob("structures/*/04_wannier_attempts/*/attempt-*/attempt.json")):
        try:
            document = read_json(path)
            active = document.get("active_process") or {}
            pid = int(active.get("pid") or 0)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if pid:
            workers.append({
                "pid": pid,
                "alive": _pid_alive(pid),
                "state_file": str(path),
                "command": active.get("command"),
                "structure": active.get("structure"),
            })
    return workers


def _stage_status(root: Path, target: str) -> str:
    path = root / STAGE_NAME[target] / "stage.json"
    if not path.is_file():
        return "MISSING"
    try:
        return str(read_json(path).get("status", "UNKNOWN"))
    except (OSError, json.JSONDecodeError):
        return "UNREADABLE"


def _save(root: Path, status: str, **extra: Any) -> dict[str, Any]:
    path = root / STATE_FILE
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = read_json(path)
        except (OSError, json.JSONDecodeError):
            previous = {}
    payload = {
        "schema_version": 1,
        "created_utc": previous.get("created_utc", _utc_now()),
        "updated_utc": _utc_now(),
        "status": status,
        "supervisor_pid": os.getpid(),
        "host": socket.gethostname(),
        "alerts": list(previous.get("alerts", [])),
        **extra,
    }
    atomic_json(path, payload)
    return payload


def inspect_supervisor(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    state_path = root / STATE_FILE
    state: dict[str, Any] = read_json(state_path) if state_path.is_file() else {"status": "NOT_STARTED"}
    supervisor_pid = int(state.get("supervisor_pid") or 0)
    child_pid = int((state.get("active_child") or {}).get("pid") or 0)
    return {
        "run_dir": str(root),
        "supervisor": state,
        "supervisor_alive": _pid_alive(supervisor_pid),
        "child_alive": _pid_alive(child_pid),
        "active_workers": _active_workers(root),
        "workflow": inspect_workflow(root),
    }


@contextmanager
def _signal_checkpoint(callback: Callable[[int], None]) -> Iterator[None]:
    old: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        callback(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    try:
        yield
    finally:
        for restored, previous in old.items():
            signal.signal(restored, previous)


def _command(
    root: Path,
    target: str,
    python: str | None = None,
) -> tuple[list[str], dict[str, str], Path]:
    package_root = Path(__file__).resolve().parents[2]
    source_root = package_root / "src"
    environment = os.environ.copy()
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) if not prior else os.pathsep.join((str(source_root), prior))
    child_python = python or sys.executable
    command = [child_python, "-m", "w90flow", "resume", str(root), "--until", target, "--execute"]
    return command, environment, package_root


def _poll_pid_until_exit(
    root: Path,
    pid: int,
    target: str,
    poll_seconds: float,
    heartbeat_seconds: float,
) -> None:
    last_heartbeat = 0.0
    while _pid_alive(pid):
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_seconds:
            _save(
                root,
                "RECOVERING_EXISTING_CHILD",
                target=target,
                active_child={"pid": pid, "ownership": "previous_supervisor"},
                active_workers=_active_workers(root),
            )
            _append_log(root, f"HEARTBEAT recover child_pid={pid} target={target}")
            last_heartbeat = now
        time.sleep(poll_seconds)


def _recover_live_child(root: Path, poll_seconds: float, heartbeat_seconds: float) -> None:
    state_path = root / STATE_FILE
    if not state_path.is_file():
        return
    try:
        old = read_json(state_path)
        child = old.get("active_child") or {}
        pid = int(child.get("pid") or 0)
        target = str(old.get("target") or child.get("target") or "")
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if not pid or not _pid_alive(pid):
        return
    if target not in WORKFLOW_STOPS:
        raise WorkflowBlocked(f"发现在运行的旧 child PID {pid}，但不能确定其阶段。")
    _append_log(root, f"RECOVER attach existing child_pid={pid} target={target}")
    _poll_pid_until_exit(root, pid, target, poll_seconds, heartbeat_seconds)
    if _stage_status(root, target) != "PASS":
        raise WorkflowBlocked(f"旧 child PID {pid} 结束后 {target} 仍未 PASS；为避免重复作业，安全阻塞。")
    _append_log(root, f"RECOVER completed existing child_pid={pid} target={target}")


def _recover_live_workers(root: Path, poll_seconds: float, heartbeat_seconds: float) -> None:
    """Wait for orphaned engine workers before starting a replacement child.

    A supervisor/child can disappear while its VASP or Wannier launcher remains
    alive. The worker PID recorded in stage/attempt state is authoritative: the
    new supervisor attaches read-only and lets ``resume`` reconcile outputs only
    after every such PID has exited.
    """
    last_heartbeat = 0.0
    announced: set[int] = set()
    while True:
        workers = [item for item in _active_workers(root) if item.get("alive")]
        if not workers:
            break
        pids = {int(item["pid"]) for item in workers}
        for pid in sorted(pids - announced):
            _append_log(root, f"RECOVER attach existing worker_pid={pid}; no replacement child launched")
        announced.update(pids)
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_seconds:
            _save(
                root,
                "RECOVERING_EXISTING_WORKER",
                active_child=None,
                active_workers=workers,
                recovery_policy="wait_then_resume_reconcile",
            )
            _append_log(root, f"HEARTBEAT recover worker_pids={','.join(map(str, sorted(pids)))}")
            last_heartbeat = now
        time.sleep(poll_seconds)
    if announced:
        _append_log(root, f"RECOVER workers exited pids={','.join(map(str, sorted(announced)))}; resume may reconcile")


def supervise_workflow(
    run_dir: str | Path,
    *,
    execute: bool = False,
    until: str = "export",
    poll_seconds: float = 30.0,
    heartbeat_seconds: float = 300.0,
    no_progress_seconds: float = 1800.0,
    stage_timeout_seconds: float = 86400.0,
    max_hours: float | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Advance one boundary at a time and poll without changing science input.

    The first SIGINT/SIGTERM requests a pause at the current complete stage
    boundary. Timeout/no-progress alarms use the same boundary-safe policy.
    """
    root = Path(run_dir).expanduser().resolve()
    if until not in WORKFLOW_STOPS:
        raise ValueError(f"until 必须是：{', '.join(WORKFLOW_STOPS)}")
    for name, value in (
        ("poll_seconds", poll_seconds),
        ("heartbeat_seconds", heartbeat_seconds),
        ("no_progress_seconds", no_progress_seconds),
        ("stage_timeout_seconds", stage_timeout_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0。")
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours 必须大于 0。")
    targets = WORKFLOW_STOPS[: WORKFLOW_STOPS.index(until) + 1]
    preview = {
        "mode": "DRY_RUN" if not execute else "EXECUTE",
        "run_dir": str(root),
        "targets": list(targets),
        "poll_seconds": poll_seconds,
        "heartbeat_seconds": heartbeat_seconds,
        "no_progress_seconds": no_progress_seconds,
        "stage_timeout_seconds": stage_timeout_seconds,
        "max_hours": max_hours,
        "workflow": inspect_workflow(root),
    }
    if not execute:
        return preview

    deadline = None if max_hours is None else time.monotonic() + max_hours * 3600.0
    pause: dict[str, Any] = {"requested": False, "reason": None}
    live_context: dict[str, Any] = {"target": None, "active_child": None}

    def request_pause(signum: int) -> None:
        if not pause["requested"]:
            pause.update({"requested": True, "reason": f"signal_{signal.Signals(signum).name}"})
            _save(
                root,
                "PAUSE_PENDING_BOUNDARY",
                target=live_context["target"],
                active_child=live_context["active_child"],
                active_workers=_active_workers(root),
                pause_reason=pause["reason"],
            )
            _append_log(root, f"PAUSE requested by {pause['reason']}; wait for stage boundary")

    with SupervisorLock(root), _signal_checkpoint(request_pause):
        try:
            _recover_live_child(root, poll_seconds, heartbeat_seconds)
            _recover_live_workers(root, poll_seconds, heartbeat_seconds)
            _append_log(root, f"START targets={','.join(targets)}")
            for target in targets:
                if _stage_status(root, target) == "PASS":
                    _append_log(root, f"SKIP {target}: already PASS")
                    continue
                if pause["requested"] or (deadline is not None and time.monotonic() >= deadline):
                    reason = pause["reason"] or "walltime_deadline"
                    state = _save(root, "PAUSED_BOUNDARY", target=target, pause_reason=reason, active_child=None)
                    _append_log(root, f"PAUSED before {target}: {reason}")
                    return {"passed": True, **state}

                command, environment, cwd = _command(root, target, python=python)
                child_log_path = root / f"supervisor-child-{target}.log"
                _append_log(root, f"LAUNCH {target}: {' '.join(command)}")
                with child_log_path.open("ab") as child_log:
                    child = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdout=child_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    started = time.monotonic()
                    child_started_utc = _utc_now()
                    live_context.update({
                        "target": target,
                        "active_child": {
                            "pid": child.pid,
                            "target": target,
                            "command": command,
                            "started_utc": child_started_utc,
                        },
                    })
                    _save(
                        root,
                        "RUNNING",
                        target=target,
                        active_child=live_context["active_child"],
                        active_workers=_active_workers(root),
                        seconds_since_progress=0.0,
                        seconds_in_stage=0.0,
                    )
                    watch = ProgressWatch(
                        started=started,
                        last_change=started,
                        fingerprint=progress_fingerprint(root),
                        no_progress_seconds=no_progress_seconds,
                        stage_timeout_seconds=stage_timeout_seconds,
                    )
                    last_heartbeat = 0.0
                    alerts: list[dict[str, Any]] = []
                    while child.poll() is None:
                        now = time.monotonic()
                        if deadline is not None and now >= deadline and not pause["requested"]:
                            pause.update({"requested": True, "reason": "walltime_deadline"})
                            _save(
                                root,
                                "PAUSE_PENDING_BOUNDARY",
                                target=target,
                                active_child=live_context["active_child"],
                                active_workers=_active_workers(root),
                                pause_reason=pause["reason"],
                            )
                            _append_log(root, f"PAUSE deadline reached during {target}; wait for boundary")
                        for code in watch.observe(progress_fingerprint(root), now):
                            alert = {
                                "code": code,
                                "target": target,
                                "detected_utc": _utc_now(),
                                "policy": "finish_current_stage_then_block",
                            }
                            alerts.append(alert)
                            pause.update({"requested": True, "reason": code.lower()})
                            _save(
                                root,
                                "PAUSE_PENDING_BOUNDARY",
                                target=target,
                                active_child=live_context["active_child"],
                                active_workers=_active_workers(root),
                                alerts=alerts,
                                pause_reason=pause["reason"],
                            )
                            _append_log(root, f"ALERT {code} target={target}; no scientific input changed")
                        if now - last_heartbeat >= heartbeat_seconds:
                            _save(
                                root,
                                "PAUSE_PENDING_BOUNDARY" if pause["requested"] else "RUNNING",
                                target=target,
                                active_child=live_context["active_child"],
                                active_workers=_active_workers(root),
                                seconds_since_progress=round(now - watch.last_change, 3),
                                seconds_in_stage=round(now - started, 3),
                                alerts=alerts,
                                pause_reason=pause["reason"],
                            )
                            _append_log(
                                root,
                                f"HEARTBEAT target={target} child_pid={child.pid} "
                                f"workers={len(_active_workers(root))} progress_age={now - watch.last_change:.1f}s",
                            )
                            last_heartbeat = now
                        time.sleep(poll_seconds)
                    returncode = int(child.returncode or 0)
                    live_context["active_child"] = None

                if returncode != 0 or _stage_status(root, target) != "PASS":
                    state = _save(
                        root,
                        "BLOCKED_CHILD_FAILURE",
                        target=target,
                        active_child=None,
                        returncode=returncode,
                        child_log=str(child_log_path),
                        alerts=alerts,
                    )
                    _append_log(root, f"BLOCKED {target}: child return={returncode} stage={_stage_status(root, target)}")
                    return {"passed": False, **state}
                _append_log(root, f"PASS boundary {target}")
                if pause["requested"]:
                    reason = str(pause["reason"])
                    status = "BLOCKED_ALERT_BOUNDARY" if alerts else "PAUSED_BOUNDARY"
                    state = _save(
                        root,
                        status,
                        target=target,
                        active_child=None,
                        pause_reason=reason,
                        alerts=alerts,
                        checkpoint={"last_passed": target, "next_action": inspect_workflow(root)["next_action"]},
                    )
                    _append_log(root, f"{status} after {target}: {reason}")
                    return {"passed": not alerts, **state}

            state = _save(
                root,
                "COMPLETE",
                target=until,
                active_child=None,
                checkpoint={"last_passed": until, "next_action": inspect_workflow(root)["next_action"]},
            )
            _append_log(root, f"COMPLETE through {until}")
            return {"passed": True, **state}
        except WorkflowBlocked as exc:
            state = _save(root, "BLOCKED", active_child=None, reason=str(exc))
            _append_log(root, f"BLOCKED: {exc}")
            return {"passed": False, **state}
        except Exception as exc:
            state = _save(
                root,
                "BLOCKED_SUPERVISOR_ERROR",
                active_child=None,
                reason=str(exc),
                error_type=type(exc).__name__,
            )
            _append_log(root, f"BLOCKED_SUPERVISOR_ERROR {type(exc).__name__}: {exc}")
            return {"passed": False, **state}
