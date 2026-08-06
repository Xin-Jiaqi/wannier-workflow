from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .config import load_project
from .orchestrator import fit_from_preprocess, inspect_workflow
from .parsers import parse_poscar
from .preprocess import (
    StageRecoveryRequired,
    inspect_preprocess_run,
    normalize_preprocess_config,
    patch_incar_contract,
    prepare_preprocess_run,
    reconcile_preprocess_stage,
    recover_interrupted_stage,
    run_preprocess_workflow,
    stage_incar_contract,
    validate_incar_structure_compatibility,
    validate_potcar_recipe,
)
from .state import atomic_json, read_json, sha256_file
from .structure import resolve_periodicity
from .supervisor import supervise_workflow


STATE_FILE = "stage1-coldstart-state.json"
LOCK_FILE = ".stage1-coldstart.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ColdStartLock:
    """One owner for POSCAR -> preprocess -> exact Wannier run handoff."""

    def __init__(self, campaign_dir: Path):
        self.path = campaign_dir / LOCK_FILE
        self.acquired = False

    def __enter__(self) -> "ColdStartLock":
        if self.path.is_file():
            previous = read_json(self.path)
            old_pid = int(previous.get("pid") or 0)
            if previous.get("host") == socket.gethostname() and _pid_alive(old_pid):
                raise RuntimeError(f"Stage-1 cold start 已由 PID {old_pid} 管理。")
            archived = self.path.with_name(
                f"{LOCK_FILE}.stale-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{old_pid}"
            )
            os.replace(self.path, archived)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, (json.dumps({
                "schema_version": 1,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_utc": _utc_now(),
            }, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired or not self.path.is_file():
            return
        try:
            owner = read_json(self.path)
        except (OSError, json.JSONDecodeError):
            owner = {}
        if int(owner.get("pid") or 0) == os.getpid():
            self.path.unlink()


def _resolve_preprocess_inputs(config_path: Path, poscar: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError("Stage-1 配置根节点必须是对象。")
    analysis = resolve_periodicity(parse_poscar(poscar), {"dimensionality": "auto"})
    protocol = normalize_preprocess_config(document, int(analysis["dimensionality"]))
    source = document.get("preprocess_inputs")
    if not isinstance(source, dict):
        raise ValueError("Stage-1 配置缺少 preprocess_inputs。")
    inputs = json.loads(json.dumps(source))
    missing = sorted({"templates", "potcar", "commands"} - set(inputs))
    if missing:
        raise ValueError(f"preprocess_inputs 缺少：{', '.join(missing)}")

    def resolved(value: str) -> str:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (config_path.parent / path).resolve())

    for stage in inputs["templates"].values():
        for role in ("incar", "kpoints"):
            if isinstance(stage.get(role), str):
                stage[role] = resolved(stage[role])
    potcar = inputs["potcar"]
    if isinstance(potcar.get("path"), str):
        potcar["path"] = resolved(potcar["path"])
    for component in potcar.get("components", []):
        if isinstance(component, dict) and isinstance(component.get("path"), str):
            component["path"] = resolved(component["path"])
    return protocol, inputs


def _positive_mpi_processes(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} 必须是正整数。")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 必须是正整数。") from exc
    if number <= 0 or str(number) != str(value).strip():
        raise ValueError(f"{path} 必须是正整数。")
    return number


def _validate_execution_contract(
    config: dict[str, Any],
    protocol: dict[str, Any],
    commands: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check the scientific SOC switch against every local launcher."""
    requested_soc = config.get("physics", {}).get("soc")
    if not isinstance(requested_soc, bool):
        raise ValueError("physics.soc 必须显式为 true/false。")
    for stage in ("scf", "band"):
        if protocol["soc"][stage] is not requested_soc:
            raise ValueError(f"preprocess.soc.{stage} 与 physics.soc 不一致。")

    expected_backend = "ncl" if requested_soc else "std"
    backend = config.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("backend 必须是对象。")
    backend_vasp = str(backend.get("vasp", "")).strip().lower()
    if backend_vasp != expected_backend:
        raise ValueError(
            f"physics.soc={str(requested_soc).lower()} 要求 backend.vasp={expected_backend}；"
            f"当前为 {backend_vasp or '<missing>'}。"
        )
    mpi_processes = _positive_mpi_processes(backend.get("mpi_processes"), "backend.mpi_processes")

    if not isinstance(commands, dict):
        raise ValueError("preprocess_inputs.commands 必须是对象。")
    active = ["02_scf", "03_band"]
    if protocol["relaxation"]["enabled"]:
        active.insert(0, "01_relax")
    soc_keys = {"01_relax": "relaxation", "02_scf": "scf", "03_band": "band"}
    checked: dict[str, Any] = {}
    for stage in active:
        command = commands.get(stage)
        if not isinstance(command, list) or len(command) < 3 or any(not isinstance(item, str) for item in command):
            raise ValueError(
                f"preprocess_inputs.commands.{stage} 必须是参数列表，末两项依次为 std/ncl 与 MPI 数。"
            )
        command_vasp = command[-2].strip().lower()
        expected_stage_vasp = "ncl" if protocol["soc"][soc_keys[stage]] else "std"
        if command_vasp != expected_stage_vasp:
            raise ValueError(
                f"preprocess_inputs.commands.{stage} 应使用 {expected_stage_vasp}；当前为 {command_vasp}。"
            )
        command_mpi = _positive_mpi_processes(command[-1], f"preprocess_inputs.commands.{stage} MPI 数")
        if command_mpi != mpi_processes:
            raise ValueError(
                f"preprocess_inputs.commands.{stage} MPI 数 {command_mpi} "
                f"与 backend.mpi_processes={mpi_processes} 不一致。"
            )
        checked[stage] = {"vasp": command_vasp, "mpi_processes": command_mpi}
    return {
        "physics_soc": requested_soc,
        "backend_vasp": backend_vasp,
        "mpi_processes": mpi_processes,
        "preprocess_commands": checked,
    }


def _validate_locked_potcar(
    config: dict[str, Any],
    structure: dict[str, Any],
    elements: list[str],
    recipe_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require the read-only component recipe to equal the named project lock set."""
    pseudo = config.get("pseudopotentials", {})
    set_name = structure.get("pseudopotential_set", pseudo.get("set"))
    if not isinstance(set_name, str) or not set_name:
        raise ValueError("Stage-1 完整预检要求 pseudopotentials.set。")
    lock_path = pseudo.get("lock_resolved")
    if not isinstance(lock_path, str):
        raise ValueError("Stage-1 完整预检要求 pseudopotentials.lock。")
    lock = read_json(lock_path)
    specification = (lock.get("sets") or {}).get(set_name)
    if not isinstance(specification, dict):
        raise ValueError(f"POTCAR lock 中没有 set：{set_name}")
    order = specification.get("order")
    if order != elements:
        raise ValueError(f"POTCAR lock 顺序 {order} 与 POSCAR 元素顺序 {elements} 不一致。")
    if recipe_manifest.get("mode") != "ordered_component_recipe":
        raise ValueError("Stage-1 完整预检要求 ordered components recipe，不能只校验拼接后的 POTCAR。")
    source_records = recipe_manifest.get("sources") or []
    actual_components = {
        item.get("element"): item.get("sha256")
        for item in source_records
        if isinstance(item, dict)
    }
    expected_components = specification.get("component_sha256")
    if not isinstance(expected_components, dict) or actual_components != expected_components:
        raise ValueError(f"preprocess POTCAR 组件哈希与 lock set {set_name} 不一致。")
    expected_combined = specification.get("combined_sha256")
    if recipe_manifest.get("combined_sha256") != expected_combined:
        raise ValueError(f"preprocess POTCAR combined hash 与 lock set {set_name} 不一致。")
    return {
        "set": set_name,
        "order": list(elements),
        "component_sha256": actual_components,
        "combined_sha256": expected_combined,
    }


def _validate_structure_preflight(
    config: dict[str, Any],
    structure: dict[str, Any],
    poscar: Path,
    protocol: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Perform the complete read-only gate used by dry-run and execute alike."""
    execution = _validate_execution_contract(config, protocol, inputs["commands"])
    parsed = parse_poscar(poscar)
    atom_count = sum(parsed["counts"])
    active = ["02_scf", "03_band"]
    if protocol["relaxation"]["enabled"]:
        active.insert(0, "01_relax")
    soc_keys = {"01_relax": "relaxation", "02_scf": "scf", "03_band": "band"}
    incar_checks: dict[str, Any] = {}
    for stage in active:
        specification = inputs["templates"].get(stage)
        if not isinstance(specification, dict) or not isinstance(specification.get("incar"), str):
            raise ValueError(f"preprocess_inputs.templates.{stage}.incar 缺失。")
        source = Path(specification["incar"])
        if not source.is_file():
            raise FileNotFoundError(source)
        patched = patch_incar_contract(
            source.read_text(encoding="utf-8", errors="strict"),
            stage_incar_contract(protocol, stage),
        )
        stage_soc = bool(protocol["soc"][soc_keys[stage]])
        validate_incar_structure_compatibility(
            patched,
            atom_count,
            require_soc_magmom=stage_soc,
        )
        incar_checks[stage] = {
            "soc": stage_soc,
            "magmom_components": 3 * atom_count if stage_soc else None,
        }

    recipe = validate_potcar_recipe(inputs["potcar"], parsed["elements"])
    potcar = _validate_locked_potcar(config, structure, parsed["elements"], recipe)
    return {
        "passed": True,
        "execution": execution,
        "incar": incar_checks,
        "potcar": potcar,
    }


def plan_stage1_coldstart(config_path: str | Path, campaign_dir: str | Path, runs_root: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    campaign = Path(campaign_dir).expanduser().resolve()
    runs = Path(runs_root).expanduser().resolve()
    config = load_project(config_file)
    stage1_templates = {
        f"{stage}:{role}": sha256_file(value)
        for stage, specification in config.get("templates", {}).items()
        for role, value in specification.items()
        if role.endswith("_resolved")
    }
    preprocess_templates: dict[str, str] = {}
    structures: list[dict[str, Any]] = []
    for structure in config["structures"]:
        poscar = Path(structure["poscar_resolved"])
        protocol, preprocess_inputs = _resolve_preprocess_inputs(config_file, poscar)
        preflight = _validate_structure_preflight(
            config,
            structure,
            poscar,
            protocol,
            preprocess_inputs,
        )
        for stage, specification in preprocess_inputs["templates"].items():
            for role in ("incar", "kpoints"):
                preprocess_templates[f"{stage}:{role}"] = sha256_file(specification[role])
        structures.append({
            "id": str(structure["id"]),
            "poscar": str(poscar),
            "poscar_sha256": sha256_file(poscar),
            "dimensionality": structure["structure_analysis"]["dimensionality"],
            "preprocess_protocol_sha256": protocol["protocol_sha256"],
            "preflight": preflight,
        })
    return {
        "mode": "DRY_RUN",
        "project": config["project"],
        "config": str(config_file),
        "config_sha256": sha256_file(config_file),
        "campaign_dir": str(campaign),
        "runs_root": str(runs),
        "structures": structures,
        "immutable_input_sha256": {
            "stage1_templates": stage1_templates,
            "preprocess_templates": preprocess_templates,
            "potcar_lock": sha256_file(config["pseudopotentials"]["lock_resolved"]),
        },
        "preflight": {"passed": True, "execution": "NOT_STARTED"},
        "sequence": [
            "prepare immutable preprocess inputs",
            "SCF",
            "Band + PROCAR",
            "bootstrap exactly one Wannier run",
            "test every finite projection candidate",
            "hard quality gates",
            "rank and export Top-K",
        ],
        "stage2": "not executed",
    }


def _save(path: Path, state: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    state.update({"schema_version": 1, "updated_utc": _utc_now(), "status": status, **extra})
    atomic_json(path, state)
    return state


def _advance_preprocess(
    run: Path,
    state_path: Path,
    state: dict[str, Any],
    structure_id: str,
    poll_seconds: float,
) -> None:
    recoveries = state.setdefault("preprocess_recoveries", {}).setdefault(structure_id, {})
    while True:
        inspection = inspect_preprocess_run(run)
        action = inspection["next_action"]
        name = str(action["action"])
        _save(state_path, state, "PREPROCESS_RUNNING", current_structure=structure_id, next_action=action)
        if name == "COMPLETE":
            return
        if name == "WAIT":
            time.sleep(poll_seconds)
            continue
        if name == "RECOVER_REQUIRED":
            stage = str(action["stage"])
            used = int(recoveries.get(stage, 0))
            if used >= 1:
                raise RuntimeError(f"{structure_id}/{stage} 的一次基础设施恢复预算已用完。")
            recover_interrupted_stage(run, stage)
            recoveries[stage] = used + 1
            _save(state_path, state, "PREPROCESS_RECOVERED", current_structure=structure_id, recovered_stage=stage)
            reconciled = reconcile_preprocess_stage(run, stage)
            if reconciled.get("passed"):
                continue
            # Incomplete outputs are not adopted; the same immutable inputs may
            # be submitted once more by the normal state-machine action.
            continue
        if name == "SCIENTIFIC_REVIEW_REQUIRED":
            raise RuntimeError(f"{structure_id}/{action['stage']} 是科学/数值失败，拒绝自动改参数。")
        if name != "RUN_SAME_FROZEN_PROTOCOL":
            raise RuntimeError(f"{structure_id} 未知 preprocess 动作：{name}")
        result = run_preprocess_workflow(run, execute=True)
        if result.get("passed"):
            continue
        post = inspect_preprocess_run(run)["next_action"]
        if post["action"] in {"WAIT", "RECOVER_REQUIRED"}:
            continue
        raise RuntimeError(f"{structure_id} preprocess 未通过：{result}")


def run_stage1_coldstart(
    config_path: str | Path,
    campaign_dir: str | Path,
    runs_root: str | Path,
    *,
    execute: bool = False,
    poll_seconds: float = 30.0,
) -> dict[str, Any]:
    plan = plan_stage1_coldstart(config_path, campaign_dir, runs_root)
    if not execute:
        return plan
    if poll_seconds <= 0:
        raise ValueError("poll_seconds 必须大于 0。")
    config_file = Path(config_path).expanduser().resolve()
    campaign = Path(campaign_dir).expanduser().resolve()
    runs = Path(runs_root).expanduser().resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    state_path = campaign / STATE_FILE
    config = load_project(config_file)
    signature = {
        "config": str(config_file),
        "config_sha256": sha256_file(config_file),
        "structures": {item["id"]: item["poscar_sha256"] for item in plan["structures"]},
        "immutable_input_sha256": plan["immutable_input_sha256"],
    }
    signature_sha256 = hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with ColdStartLock(campaign):
        if state_path.is_file():
            state = read_json(state_path)
            if state.get("input_signature") != signature:
                raise RuntimeError("已有 cold-start 状态与当前配置/POSCAR 哈希不同，拒绝覆盖。")
        else:
            preprocess_runs = {
                item["id"]: str(
                    runs / "preprocess" /
                    f"{config['project']}-{item['id']}-{signature_sha256[:8]}-{item['poscar_sha256'][:8]}"
                )
                for item in plan["structures"]
            }
            state = {
                "created_utc": _utc_now(),
                "input_signature": signature,
                "input_signature_sha256": signature_sha256,
                "campaign_dir": str(campaign),
                "runs_root": str(runs),
                "preprocess_runs": preprocess_runs,
                "preprocess_recoveries": {},
                "wannier_run": None,
            }
            _save(state_path, state, "PLANNED")

        sources: dict[str, str] = {}
        for structure in config["structures"]:
            sid = str(structure["id"])
            poscar = Path(structure["poscar_resolved"])
            protocol, inputs = _resolve_preprocess_inputs(config_file, poscar)
            preprocess_run = Path(state["preprocess_runs"][sid])
            if not preprocess_run.exists():
                _save(state_path, state, "PREPROCESS_PREPARING", current_structure=sid)
                prepare_preprocess_run(
                    preprocess_run,
                    poscar,
                    protocol,
                    templates=inputs["templates"],
                    potcar=inputs["potcar"],
                    commands=inputs["commands"],
                )
            else:
                stored_protocol = read_json(preprocess_run / "protocol.json")
                input_lock = read_json(preprocess_run / "input-lock.json")
                locked_poscar = (input_lock.get("files") or {}).get("POSCAR") or {}
                if (
                    stored_protocol.get("protocol_sha256") != protocol["protocol_sha256"]
                    or locked_poscar.get("sha256") != sha256_file(poscar)
                ):
                    raise RuntimeError(f"{sid} 已有 preprocess run 的协议/POSCAR 哈希不同。")
            _advance_preprocess(preprocess_run, state_path, state, sid, poll_seconds)
            sources[sid] = str(preprocess_run)

        wannier_run_value = state.get("wannier_run")
        if wannier_run_value:
            wannier_run = Path(str(wannier_run_value))
            if not wannier_run.is_dir():
                raise RuntimeError(f"记录的 Wannier run 不存在：{wannier_run}")
            validation = wannier_run / "00_validate/stage.json"
            if not validation.is_file() or read_json(validation).get("status") != "PASS":
                raise RuntimeError(
                    f"记录的 Wannier run 未完成不可变导入；拒绝创建重复 run：{wannier_run}"
                )
        else:
            wannier_run = runs / (
                f"{config['project']}-coldstart-"
                f"{signature_sha256[:16]}"
            )
            state["wannier_run"] = str(wannier_run)
            _save(state_path, state, "WANNIER_BOOTSTRAPPING", preprocess_sources=sources)
            bootstrapped = fit_from_preprocess(
                config,
                runs,
                sources,
                bootstrap_only=True,
                run_dir=wannier_run,
            )
            if Path(bootstrapped["run_dir"]) != wannier_run:
                raise RuntimeError("bootstrap 返回了非预期的 Wannier run_dir。")
            _save(state_path, state, "WANNIER_BOOTSTRAPPED", preprocess_sources=sources)

        _save(state_path, state, "WANNIER_RUNNING", wannier_run=str(wannier_run))
        supervised = supervise_workflow(
            wannier_run,
            execute=True,
            until="export",
            poll_seconds=poll_seconds,
            heartbeat_seconds=max(60.0, poll_seconds),
            no_progress_seconds=3600.0,
            stage_timeout_seconds=172800.0,
        )
        workflow = inspect_workflow(wannier_run)
        if workflow["next_action"]["action"] == "complete":
            return _save(
                state_path,
                state,
                "COMPLETE",
                current_structure=None,
                wannier_run=str(wannier_run),
                workflow=workflow,
                supervisor=supervised,
                stage2="not executed",
            )
        status = "PAUSED" if str(supervised.get("status", "")).startswith("PAUSED") else "BLOCKED"
        return _save(
            state_path,
            state,
            status,
            wannier_run=str(wannier_run),
            workflow=workflow,
            supervisor=supervised,
            stage2="not executed",
        )
