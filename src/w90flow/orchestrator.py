from __future__ import annotations

import json
import os
import shutil
import socket
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from .attempts import create_attempts, recorded_or_classified_failure, run_attempt
from .executor import generate_candidates, import_completed_stage, prepare_run, reconcile_stage, run_vasp_stage
from .parsers import parse_poscar, parse_wout
from .structure import resolve_periodicity, resolve_uniform_mesh
from .quality import IncompleteTargetSubspace, evaluate_attempt, export_ranked_attempts, rank_attempts
from .state import atomic_json, read_json, read_status, sha256_file, update_stage
from .resources import candidate_search_plan, detect_resources


WORKFLOW_STOPS = ("scf", "band", "candidates", "wannier", "quality", "rank", "export")
STAGE_NAME = {
    "validate": "00_validate",
    "scf": "01_scf",
    "band": "02_band",
    "candidates": "03_candidates",
    "wannier": "04_wannier",
    "quality": "05_quality",
    "rank": "06_rank",
    "export": "07_export",
}

WINDOW_FAILURES = {
    "NOT_ENOUGH_STATES",
    "TOO_MANY_FROZEN_STATES",
    "INVALID_WINDOW_CONTAINMENT",
    "BAND_CONNECTIVITY_FAILED",
    "QUALITY_GATE_FAILED",
}
CONVERGENCE_FAILURES = {"DISENTANGLEMENT_NOT_CONVERGED", "LOCALIZATION_NOT_CONVERGED"}


class WorkflowBlocked(RuntimeError):
    """Raised when deterministic safe progress is impossible."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    """Single-host lock preventing duplicate heavy workflows for one run."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / ".w90flow.lock"
        self.acquired = False

    def __enter__(self) -> "RunLock":
        if self.path.exists():
            try:
                old = read_json(self.path)
                old_pid = int(old.get("pid") or 0)
                old_host = str(old.get("host") or "")
            except (OSError, ValueError, json.JSONDecodeError):
                old_pid, old_host = 0, ""
            if old_host == socket.gethostname() and _pid_alive(old_pid):
                raise WorkflowBlocked(f"同一 run 已由 PID {old_pid} 执行，拒绝重复重作业。")
            stem = f".w90flow.lock.stale-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{old_pid}"
            stale = self.path.with_name(stem)
            suffix = 1
            while stale.exists():
                stale = self.path.with_name(f"{stem}-{suffix:02d}")
                suffix += 1
            os.replace(self.path, stale)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise WorkflowBlocked("另一个监督器刚刚获得了同一 run 的锁，拒绝重复执行。") from exc
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


def _load_snapshot(root: Path) -> dict[str, Any]:
    with (root / "project.snapshot.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("project.snapshot.yaml 无效。")
    return config


def _stage_status(root: Path, short_name: str) -> str:
    return str(read_json(root / STAGE_NAME[short_name] / "stage.json").get("status", "UNKNOWN"))


def inspect_workflow(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    statuses = read_status(root)["stages"]
    next_action: dict[str, Any] = {"action": "complete"}
    for item in statuses:
        if item["status"] != "PASS":
            stage = str(item["stage"])
            next_action = {
                "action": "advance_stage",
                "stage": stage,
                "status": item["status"],
            }
            break
    return {
        "schema_version": 1,
        "run_dir": str(root),
        "stages": statuses,
        "next_action": next_action,
        "locked": (root / ".w90flow.lock").is_file(),
    }


def fit_from_precomputed(
    config: dict[str, Any],
    runs_root: str | Path,
    scf_sources: Mapping[str, str | Path],
    band_sources: Mapping[str, str | Path],
    *,
    execute: bool = False,
    bootstrap_only: bool = False,
    run_dir: str | Path | None = None,
    until: str = "export",
) -> dict[str, Any]:
    """Bootstrap the deterministic Wannier search from verified SCF/Band data."""
    if execute and bootstrap_only:
        raise ValueError("execute 与 bootstrap_only 不能同时启用。")
    expected = {str(item["id"]) for item in config["structures"]}
    if set(scf_sources) != expected or set(band_sources) != expected:
        raise ValueError("SCF/Band source 映射必须恰好覆盖 project 中的全部结构。")
    contract = {
        "scf_required": ["POSCAR", "POTCAR", "INCAR", "KPOINTS", "OUTCAR", "OSZICAR", "CHGCAR", "EIGENVAL"],
        "band_required": ["POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR", "OUTCAR", "EIGENVAL", "PROCAR", "vasprun.xml"],
        "hash_policy": "SCF verifies four prepared inputs; Band additionally verifies CHGCAR",
        "next": "candidate generation -> finite Wannier attempts -> quality -> rank -> export",
    }
    if not execute and not bootstrap_only:
        return {
            "mode": "DRY_RUN",
            "project": config["project"],
            "until": until,
            "contract": contract,
            "scf_sources": {key: str(value) for key, value in scf_sources.items()},
            "band_sources": {key: str(value) for key, value in band_sources.items()},
        }
    run = prepare_run(config, runs_root, run_dir=run_dir)
    scf_import = import_completed_stage(run, "scf", scf_sources)
    band_import = import_completed_stage(run, "band", band_sources)
    if bootstrap_only:
        return {
            "mode": "BOOTSTRAPPED",
            "run_dir": str(run),
            "contract": contract,
            "scf_import": scf_import,
            "band_import": band_import,
            "passed": False,
            "next": "persist run_dir, then resume this exact run with --execute",
        }
    result = run_workflow(run, execute=True, until=until)
    return {
        "run_dir": str(run),
        "contract": contract,
        "scf_import": scf_import,
        "band_import": band_import,
        "workflow": result,
        "passed": bool(result.get("passed")),
    }


def fit_from_preprocess(
    config: dict[str, Any],
    runs_root: str | Path,
    preprocess_sources: Mapping[str, str | Path],
    *,
    execute: bool = False,
    bootstrap_only: bool = False,
    run_dir: str | Path | None = None,
    until: str = "export",
) -> dict[str, Any]:
    """Verify completed preprocess handoffs and enter the Wannier state machine.

    The optimized POSCAR can change lattice vectors, periodicity evidence and
    reciprocal meshes.  Those derived fields are therefore recomputed here;
    the original project YAML is never edited in place.
    """
    expected = {str(item["id"]) for item in config["structures"]}
    if set(preprocess_sources) != expected:
        raise ValueError("preprocess source 映射必须恰好覆盖 project 中的全部结构。")
    rebound = copy.deepcopy(config)
    scf_sources: dict[str, str] = {}
    band_sources: dict[str, str] = {}
    handoffs: dict[str, Any] = {}
    dimensions: set[int] = set()
    for structure in rebound["structures"]:
        sid = str(structure["id"])
        source = Path(preprocess_sources[sid]).expanduser().resolve()
        protocol_path = source / "protocol.json"
        handoff_path = source / "03_band/band-handoff.json"
        if not protocol_path.is_file() or not handoff_path.is_file():
            raise ValueError(f"{sid} 的 preprocess 尚无完整 protocol/band handoff：{source}")
        protocol = read_json(protocol_path)
        handoff = read_json(handoff_path)
        requested_soc = bool(rebound["physics"].get("soc"))
        if bool(protocol["soc"]["scf"]) != requested_soc or bool(protocol["soc"]["band"]) != requested_soc:
            raise ValueError(f"{sid} 的 preprocess SCF/Band SOC 与 Stage-1 physics.soc 不一致。")
        optimized = source / "02_scf/POSCAR"
        if not optimized.is_file():
            raise ValueError(f"{sid} 缺少优化后/冻结的 SCF POSCAR：{optimized}")
        files = handoff.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"{sid} 的 band-handoff.json 缺少 files 哈希。")
        for name, record in files.items():
            path = source / "03_band" / name
            if not isinstance(record, dict) or record.get("path") != str(path) or not isinstance(record.get("sha256"), str):
                raise ValueError(f"{sid} 的 Band handoff 记录无效：{name}")
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"{sid} 的 Band handoff 文件缺失或哈希改变：{name}")
        parsed = parse_poscar(optimized)
        analysis = resolve_periodicity(parsed, rebound["physics"])
        structure["poscar_resolved"] = str(optimized)
        structure["structure_analysis"] = analysis
        # SCF/Band are immutable, hash-verified handoffs.  Their exact KPOINTS
        # must be copied instead of regenerated from an equivalent mesh.
        structure["preserve_precomputed_inputs"] = True
        dimensions.add(int(analysis["dimensionality"]))
        meshes = rebound.get("meshes", {})
        resolved_meshes: dict[str, list[int]] = {}
        for role, specification in {
            "scf": meshes.get("scf"),
            "wannier": meshes.get("wannier", meshes.get("screening")),
            "production": meshes.get("production"),
        }.items():
            if specification is not None:
                resolved_meshes[role] = resolve_uniform_mesh(
                    specification,
                    parsed["lattice_angstrom"],
                    analysis["periodic_axes"],
                )
        structure["meshes_resolved"] = resolved_meshes
        for stage, directory in (("scf", source / "02_scf"), ("band", source / "03_band")):
            rebound["templates"][stage]["incar_resolved"] = str(directory / "INCAR")
            rebound["templates"][stage]["kpoints_resolved"] = str(directory / "KPOINTS")
        scf_sources[sid] = str(source / "02_scf")
        band_sources[sid] = str(source / "03_band")
        handoffs[sid] = {
            "preprocess_run": str(source),
            "protocol_sha256": protocol["protocol_sha256"],
            "band_handoff": str(handoff_path),
            "optimized_poscar_sha256": sha256_file(optimized),
            "structure_analysis": analysis,
            "meshes_resolved": resolved_meshes,
        }
    if rebound.get("ensemble", {}).get("mode") == "common_basis" and len(dimensions) != 1:
        raise ValueError("共同基底的 preprocess 结构必须具有相同 dimensionality。")
    result = fit_from_precomputed(
        rebound,
        runs_root,
        scf_sources,
        band_sources,
        execute=execute,
        bootstrap_only=bootstrap_only,
        run_dir=run_dir,
        until=until,
    )
    return {**result, "preprocess_handoffs": handoffs}


def _attempt_complete(directory: Path) -> bool:
    required = ("wannier90.wout", "wannier90_hr.dat", "wannier90_centres.xyz", "wannier90_band.dat")
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        return bool(parse_wout(directory / "wannier90.wout")["completed"])
    except (OSError, ValueError):
        return False


def _recover_attempt(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    if record.get("status") != "RUNNING":
        return record
    active = record.get("active_process") or {}
    pid = int(active.get("pid") or 0)
    if _pid_alive(pid):
        raise WorkflowBlocked(f"attempt worker PID {pid} 仍在运行：{directory}")
    if _attempt_complete(directory):
        record.update({"status": "PASS", "reconciled_utc": _utc_now(), "active_process": None})
    else:
        record.update({
            "status": "FAIL",
            "finished_utc": _utc_now(),
            "returncode": -1,
            "failure_class": "INFRASTRUCTURE_FAILURE",
            "next_action": "retry_same_attempt_once_after_environment_check",
            "active_process": None,
            "recovered_orphan": True,
        })
    atomic_json(directory / "attempt.json", record)
    return record


def _attempt_records(root: Path, structure: str, candidate: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    candidate_root = root / "structures" / structure / "04_wannier_attempts" / candidate
    records: dict[str, tuple[Path, dict[str, Any]]] = {}
    for record_path in sorted(candidate_root.glob("attempt-*/attempt.json")):
        directory = record_path.parent
        record = _recover_attempt(directory, read_json(record_path))
        records[str(record["attempt"])] = (directory, record)
    return records


def _retry_reason(record: dict[str, Any]) -> str:
    plan = record.get("plan_reference", record.get("plan", {}))
    if plan.get("mode") == "exact":
        return "exact"
    return str(plan.get("retry_reason") or "seed")


def _retry_class(record: dict[str, Any]) -> str:
    plan = record.get("plan_reference", record.get("plan", {}))
    if plan.get("mode") == "exact":
        return "exact"
    declared = plan.get("retry_class")
    if declared:
        return str(declared)
    reason = str(plan.get("retry_reason") or "seed")
    if reason.startswith("outer_window_expansion"):
        return "outer_window_expansion"
    if reason.startswith("convergence_retry"):
        return "convergence_retry"
    return reason


def _retry_index(record: dict[str, Any]) -> int:
    plan = record.get("plan_reference", record.get("plan", {}))
    return int(plan.get("retry_index") or 0)


def choose_retry_reason(
    failures: list[str | None],
    attempted_reasons: set[str],
    available_classes: set[str] | None = None,
) -> str | None:
    """Choose one finite deterministic branch; never relax quality thresholds."""
    clean = {failure for failure in failures if failure}
    if "INTERFACE_FAILURE" in clean:
        raise WorkflowBlocked(
            "VASP/Wannier90 接口或工具链失败（全局技术故障）：需更换兼容的 site "
            "VASP/Wannier90 组合；不自动放宽科学参数，不继续消耗候选。"
        )
    if "NUMERIC_FAILURE" in clean:
        raise WorkflowBlocked("数值失败不能通过自动放宽科学参数处理。")
    if "INFRASTRUCTURE_FAILURE" in clean:
        return "same"
    if clean & CONVERGENCE_FAILURES and (
        "convergence_retry" in available_classes
        if available_classes is not None
        else "convergence_retry" not in attempted_reasons
    ):
        return "convergence_retry"
    if clean & WINDOW_FAILURES and (
        "outer_window_expansion" in available_classes
        if available_classes is not None
        else "outer_window_expansion" not in attempted_reasons
    ):
        return "outer_window_expansion"
    return None


def group_has_runnable_member(statuses: set[str]) -> bool:
    """Resume a partially completed common attempt without rerunning members."""
    return "READY" in statuses and statuses <= {"READY", "PASS", "ACCEPTED"}


def _evaluate_pass_records(root: Path, structures: list[str], candidate: str, attempt: str) -> None:
    for sid in structures:
        directory, record = _attempt_records(root, sid, candidate)[attempt]
        if record.get("status") == "PASS":
            try:
                evaluate_attempt(root, sid, candidate, attempt)
            except IncompleteTargetSubspace as exc:
                rejection = {
                    "schema_version": 1,
                    "passed": False,
                    "structure": sid,
                    "candidate": candidate,
                    "attempt": attempt,
                    "failure_class": "INCOMPLETE_TARGET_SUBSPACE",
                    "target_band_count": exc.target_band_count,
                    "model_dimension": exc.model_dimension,
                    "reason": str(exc),
                    "quality_gate": "model_dimension_must_cover_protected_target_manifold",
                }
                quality_path = directory / "quality.json"
                atomic_json(quality_path, rejection)
                record.update({
                    "status": "REJECTED_QUALITY",
                    "quality_evaluated_utc": _utc_now(),
                    "quality": str(quality_path),
                    "failure_class": "INCOMPLETE_TARGET_SUBSPACE",
                })
                atomic_json(directory / "attempt.json", record)


def _common_attempt_passed(root: Path, structures: list[str], candidate: str, attempt: str) -> bool:
    records = [_attempt_records(root, sid, candidate).get(attempt) for sid in structures]
    return all(item is not None and item[1].get("status") == "ACCEPTED" for item in records)


def _process_candidate(
    root: Path,
    structures: list[str],
    candidate: str,
    *,
    mpi_processes: int | None = None,
) -> dict[str, Any]:
    attempted_reasons: set[str] = set()
    previous_failures: list[str | None] = []
    while True:
        by_structure = {sid: _attempt_records(root, sid, candidate) for sid in structures}
        common_labels = set.intersection(*(set(records) for records in by_structure.values())) if by_structure else set()
        if not common_labels:
            return {"candidate": candidate, "passed": False, "reason": "no_common_attempt_plan"}
        for label in sorted(common_labels):
            _evaluate_pass_records(root, structures, candidate, label)
            if _common_attempt_passed(root, structures, candidate, label):
                return {"candidate": candidate, "attempt": label, "passed": True}

        terminal: list[tuple[str, list[dict[str, Any]]]] = []
        ready: dict[str, list[dict[str, Any]]] = {}
        for label in sorted(common_labels):
            group = [_attempt_records(root, sid, candidate)[label][1] for sid in structures]
            reason = _retry_reason(group[0])
            if any(_retry_reason(item) != reason for item in group):
                raise WorkflowBlocked(f"{candidate}/{label} 在相关结构间使用了不同计划。")
            retry_class = _retry_class(group[0])
            if any(_retry_class(item) != retry_class for item in group):
                raise WorkflowBlocked(f"{candidate}/{label} 在相关结构间使用了不同重试类别。")
            statuses = {str(item.get("status")) for item in group}
            if group_has_runnable_member(statuses):
                ready[label] = group
            elif statuses & {"FAIL", "REJECTED_QUALITY"}:
                terminal.append((label, group))

        if not terminal and ready:
            selected = min(ready)
        else:
            if terminal:
                last_label, last_group = terminal[-1]
                last_reason = _retry_reason(last_group[0])
                attempted_reasons.add(last_reason)
                previous_failures = [
                    str(item.get("failure_class")) if item.get("failure_class") else None
                    for item in last_group
                ]
                if "INFRASTRUCTURE_FAILURE" in previous_failures:
                    retryable = []
                    for sid in structures:
                        directory, record = _attempt_records(root, sid, candidate)[last_label]
                        history = list(record.get("execution_history", []))
                        if record.get("failure_class") == "INFRASTRUCTURE_FAILURE" and not history:
                            retryable.append(sid)
                    if retryable:
                        selected = last_label
                    else:
                        available = {_retry_class(group[0]) for group in ready.values()}
                        desired = choose_retry_reason(previous_failures, attempted_reasons, available)
                        selected = min(
                            (label for label, group in ready.items() if _retry_class(group[0]) == desired),
                            key=lambda label: (_retry_index(ready[label][0]), label),
                            default="",
                        )
                else:
                    available = {_retry_class(group[0]) for group in ready.values()}
                    desired = choose_retry_reason(previous_failures, attempted_reasons, available)
                    selected = min(
                        (label for label, group in ready.items() if _retry_class(group[0]) == desired),
                        key=lambda label: (_retry_index(ready[label][0]), label),
                        default="",
                    )
            else:
                selected = ""
            if not selected:
                return {
                    "candidate": candidate,
                    "passed": False,
                    "reason": "finite_attempts_exhausted",
                    "failures": previous_failures,
                }

        for sid in structures:
            directory, record = _attempt_records(root, sid, candidate)[selected]
            status = str(record.get("status"))
            if status in {"READY"} or (
                status == "FAIL"
                and recorded_or_classified_failure(record, directory) == "INFRASTRUCTURE_FAILURE"
                and not list(record.get("execution_history", []))
            ):
                result = run_attempt(root, sid, candidate, selected, execute=True, mpi_processes=mpi_processes)
                failure_class = result["attempt"].get("failure_class") if not result.get("passed") else None
                if failure_class == "INTERFACE_FAILURE":
                    raise WorkflowBlocked(
                        f"{sid}/{candidate}/{selected} 接口/工具链失败：VASP-Wannier90 "
                        "不兼容或进程崩溃（.amn/.eig 未生成），需更换 site 组合；不继续消耗候选。"
                    )
                if failure_class == "NUMERIC_FAILURE":
                    raise WorkflowBlocked(f"{sid}/{candidate}/{selected} 数值失败。")


def _candidate_layout(config: dict[str, Any], spec: dict[str, Any]) -> list[tuple[list[str], str]]:
    structures = [str(item["id"]) for item in config["structures"]]
    if spec["ensemble_mode"] == "common_basis":
        return [(structures, str(candidate["candidate"])) for candidate in spec["shared_candidates"]]
    return [
        ([sid], str(candidate["candidate"]))
        for sid in structures
        for candidate in spec["independent_candidates"][sid]
    ]


def _accepted_coverage(root: Path, config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    structures = [str(item["id"]) for item in config["structures"]]
    accepted: dict[str, list[str]] = {sid: [] for sid in structures}
    for sid in structures:
        for quality_path in sorted((root / "structures" / sid / "04_wannier_attempts").glob("*/attempt-*/quality.json")):
            quality = read_json(quality_path)
            if quality.get("passed"):
                accepted[sid].append(f"{quality['candidate']}/{quality['attempt']}")
        baseline = root / "structures" / sid / "05_quality/quality.json"
        if baseline.is_file() and read_json(baseline).get("passed"):
            accepted[sid].append("baseline")
    if spec["ensemble_mode"] == "common_basis":
        common = set(accepted[structures[0]]) if structures else set()
        for sid in structures[1:]:
            common &= set(accepted[sid])
        passed = bool(common)
    else:
        common = set()
        passed = all(accepted[sid] for sid in structures)
    return {"passed": passed, "accepted": accepted, "common": sorted(common)}


def _run_candidate_search(
    root: Path,
    config: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run independent candidate directories serially or with bounded workers."""
    layout = _candidate_layout(config, spec)
    settings = config.get("execution", {}).get("candidate_search", {})
    plan = candidate_search_plan(
        available=detect_resources(),
        candidate_count=len(layout),
        mpi_processes_per_candidate=int(settings.get("mpi_processes_per_candidate", config["backend"]["mpi_processes"])),
        policy=str(settings.get("policy", "serial")),
        max_workers=settings.get("max_workers", "auto"),
    )
    mpi = int(plan["mpi_processes_per_candidate"])
    if int(plan["workers"]) == 1:
        return ([{"structures": structures, **_process_candidate(root, structures, candidate, mpi_processes=mpi)}
                 for structures, candidate in layout], plan)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(plan["workers"]), thread_name_prefix="w90-candidate") as pool:
        futures = {
            pool.submit(_process_candidate, root, structures, candidate, mpi_processes=mpi): (structures, candidate)
            for structures, candidate in layout
        }
        for future in as_completed(futures):
            structures, candidate = futures[future]
            results.append({"structures": structures, **future.result()})
    order = {(tuple(structures), candidate): index for index, (structures, candidate) in enumerate(layout)}
    results.sort(key=lambda item: order[(tuple(item["structures"]), str(item["candidate"]))])
    return results, plan


def _ensure_vasp_stage(root: Path, stage: str, events: list[dict[str, Any]]) -> None:
    if _stage_status(root, stage) == "PASS":
        events.append({"stage": stage, "action": "skip_passed"})
        return
    if _stage_status(root, stage) == "RUNNING":
        stage_name = STAGE_NAME[stage]
        state_path = root / stage_name / "stage.json"
        state = read_json(state_path)
        active = state.get("active_process") or {}
        active_pid = int(active.get("pid") or 0)
        if _pid_alive(active_pid):
            raise WorkflowBlocked(f"{stage} worker PID {active_pid} 仍在运行，拒绝重复提交。")
        reconciled = reconcile_stage(root, stage)
        if reconciled.get("passed"):
            events.append({"stage": stage, "action": "reconciled"})
            return
        recovery_history = list(state.get("recovery_history", []))
        if recovery_history:
            raise WorkflowBlocked(f"{stage} 的一次中断恢复预算已用完，需人工检查环境。")
        archived_logs: list[str] = []
        for structure_dir in sorted((root / "structures").iterdir()):
            source = structure_dir / stage_name / "runner.log"
            target = structure_dir / stage_name / "runner.infrastructure-failure-001.log"
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
                archived_logs.append(str(target))
        recovery_history.append({
            "detected_utc": _utc_now(),
            "previous_active_process": active or None,
            "failure_class": "INFRASTRUCTURE_FAILURE",
            "archived_logs": archived_logs,
        })
        update_stage(
            root,
            stage_name,
            status="FAIL",
            active_process=None,
            failure_class="INFRASTRUCTURE_FAILURE",
            recovery_history=recovery_history,
        )
        events.append({"stage": stage, "action": "recover_orphan_once"})
    result = run_vasp_stage(root, stage, execute=True)
    events.append({"stage": stage, "action": "execute", "passed": result.get("passed")})
    if not result.get("passed"):
        raise WorkflowBlocked(f"{stage} 未通过完整性/收敛检查。")


def run_workflow(run_dir: str | Path, *, execute: bool = False, until: str = "export") -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    if until not in WORKFLOW_STOPS:
        raise ValueError(f"until 必须是：{', '.join(WORKFLOW_STOPS)}")
    if not execute:
        return {"mode": "DRY_RUN", **inspect_workflow(root), "until": until}
    config = _load_snapshot(root)
    events: list[dict[str, Any]] = []
    try:
        with RunLock(root):
            if _stage_status(root, "validate") != "PASS":
                raise WorkflowBlocked("输入尚未通过 prepare/哈希校验，不能启动科学计算。")
            _ensure_vasp_stage(root, "scf", events)
            if until == "scf":
                raise StopIteration
            _ensure_vasp_stage(root, "band", events)
            if until == "band":
                raise StopIteration

            if _stage_status(root, "candidates") != "PASS":
                generate_candidates(root)
                events.append({"stage": "candidates", "action": "generated"})
            else:
                events.append({"stage": "candidates", "action": "skip_passed"})
            if until == "candidates":
                raise StopIteration

            attempt_plan = root / "03_candidates/attempt-plan.json"
            if not attempt_plan.is_file():
                create_attempts(root)
                events.append({"stage": "wannier", "action": "planned_attempts"})
            spec = read_json(root / "03_candidates/candidate-spec.json")
            candidate_results, resource_plan = _run_candidate_search(root, config, spec)
            coverage = _accepted_coverage(root, config, spec)
            if not coverage["passed"]:
                update_stage(root, "04_wannier", status="FAIL", finished_utc=_utc_now(), outputs=candidate_results)
                raise WorkflowBlocked("有限候选/能窗尝试结束后仍没有覆盖全部结构的合格 Wannier 表示。")
            update_stage(root, "04_wannier", status="PASS", finished_utc=_utc_now(), outputs=candidate_results)
            update_stage(root, "05_quality", status="PASS", finished_utc=_utc_now(), outputs=[coverage])
            events.append({"stage": "wannier", "action": "finite_search_complete", "coverage": coverage, "resource_plan": resource_plan})
            if until in {"wannier", "quality"}:
                raise StopIteration

            if _stage_status(root, "rank") != "PASS":
                ranking = rank_attempts(root)
                update_stage(root, "06_rank", status="PASS", finished_utc=_utc_now(), outputs=[ranking["ranking"]])
                events.append({"stage": "rank", "action": "ranked", "ranking": ranking["ranking"]})
            else:
                events.append({"stage": "rank", "action": "skip_passed"})
            if until == "rank":
                raise StopIteration

            if _stage_status(root, "export") != "PASS":
                exported = export_ranked_attempts(root)
                update_stage(root, "07_export", status="PASS", finished_utc=_utc_now(), outputs=[exported["export"]])
                events.append({"stage": "export", "action": "exported", "path": exported["export"]})
            else:
                events.append({"stage": "export", "action": "skip_passed"})
    except StopIteration:
        pass
    except (WorkflowBlocked, OSError, PermissionError, ValueError, json.JSONDecodeError) as exc:
        summary = {
            "schema_version": 1,
            "updated_utc": _utc_now(),
            "status": "BLOCKED",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "events": events,
            "workflow": inspect_workflow(root),
        }
        atomic_json(root / "workflow-summary.json", summary)
        return {"passed": False, **summary}

    passed = all(
        _stage_status(root, name) == "PASS"
        for name in WORKFLOW_STOPS[: WORKFLOW_STOPS.index(until) + 1]
    )
    summary = {
        "schema_version": 1,
        "updated_utc": _utc_now(),
        "status": "PASS" if passed else "INCOMPLETE",
        "until": until,
        "events": events,
        "workflow": inspect_workflow(root),
    }
    atomic_json(root / "workflow-summary.json", summary)
    return {"passed": passed, **summary}
