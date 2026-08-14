from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from .state import atomic_json, read_json, sha256_file, update_stage
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

    wannier_vasp = commands.get("04_wannier")
    if (
        not isinstance(wannier_vasp, list)
        or len(wannier_vasp) < 3
        or any(not isinstance(item, str) for item in wannier_vasp)
    ):
        raise ValueError(
            "preprocess_inputs.commands.04_wannier 必须是参数列表，末两项依次为 std/ncl 与 MPI 数。"
        )
    wannier_vasp_kind = wannier_vasp[-2].strip().lower()
    if wannier_vasp_kind != expected_backend:
        raise ValueError(
            f"preprocess_inputs.commands.04_wannier 应使用 {expected_backend}；当前为 {wannier_vasp_kind}。"
        )
    wannier_mpi = _positive_mpi_processes(
        wannier_vasp[-1], "preprocess_inputs.commands.04_wannier MPI 数"
    )
    if wannier_mpi != mpi_processes:
        raise ValueError(
            f"preprocess_inputs.commands.04_wannier MPI 数 {wannier_mpi} "
            f"与 backend.mpi_processes={mpi_processes} 不一致。"
        )
    wannier90 = commands.get("04_wannier90")
    if (
        not isinstance(wannier90, list)
        or len(wannier90) < 3
        or any(not isinstance(item, str) for item in wannier90)
    ):
        raise ValueError(
            "preprocess_inputs.commands.04_wannier90 必须是至少 3 项参数列表，末项固定为 wannier90。"
        )
    if wannier90[-1] != "wannier90":
        raise ValueError(
            "preprocess_inputs.commands.04_wannier90 末项必须是固定逻辑 seed wannier90；"
            f"当前为 {wannier90[-1]!r}。"
        )
    checked["04_wannier"] = {"vasp": wannier_vasp_kind, "mpi_processes": wannier_mpi}
    checked["04_wannier90"] = {"seed": wannier90[-1]}
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


def coldstart_signature(config_file: str | Path, plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Canonical input signature shared by coldstart and the run-material entry."""
    signature = {
        "config": str(Path(config_file).expanduser().resolve()),
        "config_sha256": sha256_file(config_file),
        "structures": {item["id"]: item["poscar_sha256"] for item in plan["structures"]},
        "immutable_input_sha256": plan["immutable_input_sha256"],
    }
    signature_sha256 = hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return signature, signature_sha256


def derived_preprocess_run(
    runs_root: str | Path,
    project: str,
    structure_id: str,
    poscar_sha256: str,
    signature_sha256: str,
) -> Path:
    """Deterministic preprocess run path for one structure (single source of truth)."""
    return (
        Path(runs_root).expanduser().resolve()
        / "preprocess"
        / f"{project}-{structure_id}-{signature_sha256[:8]}-{poscar_sha256[:8]}"
    )


def derived_wannier_run(runs_root: str | Path, project: str, signature_sha256: str) -> Path:
    """Deterministic Wannier run path for one coldstart campaign (single source of truth)."""
    return Path(runs_root).expanduser().resolve() / f"{project}-coldstart-{signature_sha256[:16]}"


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


def _verify_preprocess_reuse_source(
    preprocess_run: str | Path,
    structure: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Read-only verification of one prior preprocess/SCF/Band source.

    Scientific fingerprint = protocol_sha256 + POSCAR + POTCAR combined +
    INCAR/KPOINTS template hashes + PASS stage status + complete band handoff.
    Launcher/commands and Python paths are deliberately NOT part of the
    fingerprint, so a pure environment fix keeps reuse valid.
    """
    root = Path(preprocess_run).expanduser().resolve()
    sid = str(structure["id"])
    protocol_path = root / "protocol.json"
    lock_path = root / "input-lock.json"
    if not protocol_path.is_file() or not lock_path.is_file():
        raise ValueError(f"{sid} reuse 源不是完整 preprocess run（缺 protocol.json/input-lock.json）：{root}")
    expected_protocol = str(structure["preprocess_protocol_sha256"])
    stored_protocol = read_json(protocol_path)
    if str(stored_protocol.get("protocol_sha256") or "") != expected_protocol:
        raise ValueError(
            f"{sid} 科学协议哈希不一致（protocol 变化，拒绝复用）："
            f"{stored_protocol.get('protocol_sha256', '?')[:8]}… != {expected_protocol[:8]}…"
        )
    input_lock = read_json(lock_path)
    locked_poscar = (input_lock.get("files") or {}).get("POSCAR") or {}
    if locked_poscar.get("sha256") != str(structure["poscar_sha256"]):
        raise ValueError(f"{sid} POSCAR 哈希不一致，拒绝复用")
    materialized = input_lock.get("materialized_files") or {}
    potcar_record = materialized.get("POTCAR") or {}
    expected_potcar = (structure.get("preflight") or {}).get("potcar", {}).get("combined_sha256")
    if not expected_potcar or potcar_record.get("sha256") != expected_potcar:
        raise ValueError(f"{sid} POTCAR combined 哈希不一致，拒绝复用")
    templates = (plan.get("immutable_input_sha256") or {}).get("preprocess_templates") or {}
    active = sorted((structure.get("preflight") or {}).get("incar") or {})
    if not active:
        raise ValueError(f"{sid} 当前计划无激活 preprocess 阶段，拒绝复用")
    template_hashes: dict[str, str] = {}
    for stage in active:
        for role in ("incar", "kpoints"):
            record = materialized.get(f"{stage}:{role.upper()}") or {}
            expected = templates.get(f"{stage}:{role}")
            if not record.get("sha256") or record["sha256"] != expected:
                raise ValueError(f"{sid} {stage} {role} 模板哈希不一致（kmesh/INCAR 变化，拒绝复用）")
            template_hashes[f"{stage}:{role}"] = record["sha256"]
        state_path = root / stage / "stage.json"
        if not state_path.is_file():
            raise ValueError(f"{sid} 缺少 {stage}/stage.json，拒绝复用")
        status = str(read_json(state_path).get("status"))
        if status != "PASS":
            raise ValueError(f"{sid}/{stage} 状态是 {status}（非 PASS），拒绝复用")
    handoff_path = root / "03_band/band-handoff.json"
    if not handoff_path.is_file():
        raise ValueError(f"{sid} 缺少 03_band/band-handoff.json（Band 未交接），拒绝复用")
    handoff_files = read_json(handoff_path).get("files")
    if not isinstance(handoff_files, dict) or not handoff_files:
        raise ValueError(f"{sid} band-handoff 无 files 记录，拒绝复用")
    for name, record in handoff_files.items():
        path = root / "03_band" / name
        if not isinstance(record, dict) or not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"{sid} Band 产物 {name} 缺失或哈希改变，拒绝复用")
    return {
        "source_run": str(root),
        "source_stage": "preprocess",
        "stage_status": "PASS",
        "hashes": {
            "protocol_sha256": expected_protocol,
            "poscar_sha256": str(structure["poscar_sha256"]),
            "potcar_sha256": str(expected_potcar),
            "templates": template_hashes,
        },
    }


def _as_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected bool, got {type(value).__name__}")
    return value


_CANDIDATE_POLICY_METADATA_KEYS = frozenset({"role", "description", "comment", "note", "reason"})
_CANDIDATE_POLICY_SCHEMA: dict[str, Any] = {
    "max_candidates": int,
    "max_num_wann_fraction_of_bands": float,
    "shell_family_min_weight": float,
    "shell_combination_min_weight": float,
    "chemical_valence_used": _as_bool,
    "pyw90_dos": {
        "enabled": _as_bool,
        "required": _as_bool,
        "lower_bound": float,
    },
}


def _canonicalize_candidate_policy(
    policy: dict[str, Any],
    *,
    source: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize a candidate policy to the scientific fields that matter.

    Only keys in the explicit science schema are compared.  Known metadata
    keys (role/description/comment/note/reason) are dropped.  Any other
    unknown key is treated conservatively as possibly scientific and rejects
    reuse, unless it can be proven to be metadata by name.
    """
    schema = _CANDIDATE_POLICY_SCHEMA if schema is None else schema
    if not isinstance(policy, dict):
        raise ValueError(f"候选策略无效（{source}），拒绝复用 candidates")
    canonical: dict[str, Any] = {}
    for key, converter in schema.items():
        if key not in policy:
            raise ValueError(f"候选策略缺少科学字段 {key}（{source}），拒绝复用 candidates")
        if isinstance(converter, dict):
            canonical[key] = _canonicalize_candidate_policy(
                policy[key], source=f"{source}.{key}", schema=converter
            )
        else:
            try:
                canonical[key] = converter(policy[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"候选策略字段 {key} 无法转换（{source}）：{policy[key]!r}") from exc
    for key in policy:
        if key not in schema and key not in _CANDIDATE_POLICY_METADATA_KEYS:
            raise ValueError(
                f"候选策略含未知字段 {key}（{source}），无法证明是 metadata，拒绝复用 candidates"
            )
    return canonical


def _verify_candidates_reuse_source(
    wannier_prior: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Read-only verification of a prior PASS 03_candidates stage."""
    root = Path(wannier_prior).expanduser().resolve()
    stage_path = root / "03_candidates/stage.json"
    spec_path = root / "03_candidates/candidate-spec.json"
    if not stage_path.is_file() or str(read_json(stage_path).get("status")) != "PASS":
        raise ValueError(f"候选源 03_candidates 不是 PASS，拒绝复用：{root}")
    if not spec_path.is_file():
        raise ValueError(f"候选源缺少 candidate-spec.json，拒绝复用：{root}")
    spec = read_json(spec_path)
    physics = config.get("physics") or {}
    candidates = config.get("candidates") or {}
    pyw90 = candidates.get("pyw90_dos") or {}
    expected_ids = {str(item["id"]) for item in config["structures"]}
    spec_structures = spec.get("structures")
    if not isinstance(spec_structures, dict) or set(spec_structures) != expected_ids:
        raise ValueError("候选结构覆盖不一致，拒绝复用 candidates")
    policy = {
        "max_candidates": int(candidates.get("max_candidates", 8)),
        "max_num_wann_fraction_of_bands": float(candidates.get("max_num_wann_fraction_of_bands", 1.0)),
        "shell_family_min_weight": float(candidates.get("shell_family_min_weight", 0.08)),
        "shell_combination_min_weight": float(candidates.get("shell_combination_min_weight", 0.45)),
        "chemical_valence_used": _as_bool(bool(candidates.get("chemical_valence"))),
        "pyw90_dos": {
            "enabled": _as_bool(bool(pyw90.get("enabled", False))),
            "required": _as_bool(bool(pyw90.get("required", False))),
            "lower_bound": float(pyw90.get("lower_bound", 0.1)),
        },
    }
    if spec.get("ensemble_mode") != str(config.get("ensemble", {}).get("mode", "independent")):
        raise ValueError("候选 ensemble 模式不一致，拒绝复用 candidates")
    if bool(spec.get("soc")) != bool(physics.get("soc")):
        raise ValueError("候选 SOC 设定不一致，拒绝复用 candidates")
    if spec.get("energy_reference") != physics.get("energy_reference"):
        raise ValueError("候选能量参考不一致，拒绝复用 candidates")
    target = [float(physics["target_energy_eV"][0]), float(physics["target_energy_eV"][1])]
    if list(spec.get("target_energy_eV") or []) != target:
        raise ValueError("候选目标能窗不一致，拒绝复用 candidates")
    spec_policy = _canonicalize_candidate_policy(
        spec.get("candidate_policy") or {}, source="candidate-spec"
    )
    expected_policy = _canonicalize_candidate_policy(policy, source="config")
    if spec_policy != expected_policy:
        raise ValueError("候选策略不一致，拒绝复用 candidates")
    files = [spec_path]
    for dos in sorted(root.glob("03_candidates/pyw90-dos-*.json")):
        files.append(dos)
    return {
        "source_run": str(root),
        "source_stage": "03_candidates",
        "stage_status": "PASS",
        "files": [{"path": str(path), "sha256": sha256_file(path)} for path in files],
        "candidate_spec_sha256": sha256_file(spec_path),
    }


def _resolve_legacy_preprocess_sources(
    prior: Path,
    structure_ids: list[str],
) -> tuple[dict[str, str], tuple[str, str]]:
    """Resolve preprocess parents from legacy root-level import manifests.

    Stage1 Wannier runs recorded explicit parent references in
    ``01_scf/import-manifest.json`` and ``02_band/import-manifest.json`` at the
    run root (``results[].source`` points into the preprocess run).  Some
    fixtures/layouts use ``02_scf``/``03_band`` for the same pair, so both
    stage-name pairs are probed; the first pair with complete manifests wins.
    Directory paths are never guessed: when neither pair has complete evidence,
    the caller must pass the preprocess run directly as ``--reuse-from``.
    """
    stage_pairs = (("01_scf", "02_band"), ("02_scf", "03_band"))
    manifests: dict[str, dict[str, dict[str, Any]]] | None = None
    used_pairs: tuple[str, str] | None = None
    for scf_stage, band_stage in stage_pairs:
        scf_manifest = prior / scf_stage / "import-manifest.json"
        band_manifest = prior / band_stage / "import-manifest.json"
        if not scf_manifest.is_file() or not band_manifest.is_file():
            continue
        parsed: dict[str, dict[str, dict[str, Any]]] = {}
        for stage, manifest_path in ((scf_stage, scf_manifest), (band_stage, band_manifest)):
            document = read_json(manifest_path)
            results = document.get("results") if isinstance(document, dict) else None
            if not isinstance(results, list) or not results:
                raise ValueError(f"候选源 {stage}/import-manifest.json 无 results 记录，拒绝解析 parent")
            by_structure: dict[str, dict[str, Any]] = {}
            for record in results:
                if isinstance(record, dict) and isinstance(record.get("structure"), str):
                    by_structure[str(record["structure"])] = record
            parsed[stage] = by_structure
        manifests = parsed
        used_pairs = (scf_stage, band_stage)
        break
    if manifests is None or used_pairs is None:
        raise ValueError(
            "候选源无法解析 SCF parent run：run 根级既无 "
            "01_scf/02_band 也无 02_scf/03_band 的完整 import-manifest；"
            "拒绝猜测目录，请直接传 preprocess run 作为 --reuse-from"
        )
    parents: dict[str, str] = {}
    for sid in structure_ids:
        scf_record = manifests[used_pairs[0]].get(sid)
        band_record = manifests[used_pairs[1]].get(sid)
        if not scf_record or not band_record:
            raise ValueError(
                f"候选源（旧版布局）缺少 {sid} 的 SCF/Band parent 记录，拒绝猜测目录；"
                "请直接传 preprocess run 作为 --reuse-from"
            )
        scf_source = scf_record.get("source")
        band_source = band_record.get("source")
        if not isinstance(scf_source, str) or not isinstance(band_source, str):
            raise ValueError(f"候选源（旧版布局）{sid} 的 import source 无效，拒绝解析 parent")
        scf_parent = Path(scf_source).expanduser().resolve().parent
        band_parent = Path(band_source).expanduser().resolve().parent
        if scf_parent != band_parent:
            raise ValueError(f"候选源（旧版布局）{sid} 的 SCF/Band parent 不一致，拒绝复用")
        if not scf_parent.is_dir():
            raise ValueError(f"候选源（旧版布局）{sid} 的 SCF parent 不存在：{scf_parent}")
        parents[sid] = str(scf_parent)
    return parents, used_pairs


def verify_reuse_source(
    prior_run: str | Path,
    config_file: str | Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Read-only gate for ``--reuse-from`` (technical-recovery cross-run reuse).

    The prior run is never modified.  Only PASS preprocess/SCF/band (and,
    when the prior run is a Stage1 Wannier run, PASS 03_candidates) with a
    hash-identical scientific fingerprint are accepted.  Parent references are
    resolved through the prior run's own import manifests, never guessed.
    """
    prior = Path(prior_run).expanduser().resolve()
    config = load_project(Path(config_file).expanduser().resolve())
    if not prior.is_dir():
        raise ValueError(f"--reuse-from 目标不是目录：{prior}")
    if (prior / "protocol.json").is_file() and (prior / "input-lock.json").is_file():
        preprocess_sources = {str(item["id"]): str(prior) for item in plan["structures"]}
        candidates_info = None
        layout_version = "preprocess"
        scf_stage_label, band_stage_label = "02_scf", "03_band"
    elif (prior / "03_candidates/stage.json").is_file():
        structure_ids = [str(item["id"]) for item in plan["structures"]]
        per_structure_manifests = [
            prior / "structures" / sid / "02_scf/import-manifest.json"
            for sid in structure_ids
        ]
        present = [manifest.is_file() for manifest in per_structure_manifests]
        if all(present):
            preprocess_sources = {}
            for item in plan["structures"]:
                sid = str(item["id"])
                manifest = prior / "structures" / sid / "02_scf/import-manifest.json"
                results = read_json(manifest).get("results") or []
                if not results or not results[0].get("source"):
                    raise ValueError(f"SCF import manifest 无 source 记录：{manifest}")
                parent = Path(str(results[0]["source"])).expanduser().resolve().parent
                if not parent.is_dir():
                    raise ValueError(f"SCF parent run 不存在：{parent}")
                preprocess_sources[sid] = str(parent)
            layout_version = "current"
            scf_stage_label, band_stage_label = "02_scf", "03_band"
        elif not any(present):
            preprocess_sources, (scf_stage_label, band_stage_label) = _resolve_legacy_preprocess_sources(
                prior, structure_ids
            )
            layout_version = "legacy"
        else:
            raise ValueError(
                "候选源同时含新版与旧版 SCF import manifest，拒绝猜测目录；"
                "请直接传 preprocess run 作为 --reuse-from"
            )
        candidates_info = _verify_candidates_reuse_source(prior, config)
        candidates_info["source_layout_version"] = layout_version
        candidates_info["preprocess_parents"] = dict(preprocess_sources)
    else:
        raise ValueError(
            "--reuse-from 必须是 preprocess run（含 protocol.json/input-lock.json）"
            "或 Stage1 Wannier run（含 03_candidates）"
        )
    preprocess_verified: dict[str, Any] = {}
    for item in plan["structures"]:
        sid = str(item["id"])
        source = preprocess_sources[sid]
        preprocess_verified[sid] = _verify_preprocess_reuse_source(source, item, plan)
    stages: dict[str, Any] = {"preprocess": preprocess_verified}
    for sid, verified in preprocess_verified.items():
        stages.setdefault("scf", {})[sid] = {
            "source_run": preprocess_sources[sid],
            "source_stage": scf_stage_label,
            "stage_status": "PASS",
            "source_layout_version": layout_version,
            "hashes": verified["hashes"],
        }
        stages.setdefault("band", {})[sid] = {
            "source_run": preprocess_sources[sid],
            "source_stage": band_stage_label,
            "stage_status": "PASS",
            "source_layout_version": layout_version,
            "hashes": verified["hashes"],
        }
    if candidates_info is not None:
        stages["candidates"] = candidates_info
    return {
        "schema_version": 1,
        "mode": "technical_recovery_reuse",
        "prior_run": str(prior),
        "source_layout_version": layout_version,
        "preprocess_sources": preprocess_sources,
        "candidates": candidates_info,
        "stages": stages,
    }


def _apply_reuse_candidates(wannier_run: Path, candidates_info: dict[str, Any]) -> dict[str, Any]:
    """Copy hash-verified candidate artifacts into the new immutable run."""
    target = wannier_run / "03_candidates"
    target.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for record in candidates_info.get("files") or []:
        source = Path(str(record["path"])).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"候选源文件缺失：{source}")
        if sha256_file(source) != record.get("sha256"):
            raise ValueError(f"候选源文件哈希改变：{source}")
        destination = target / source.name
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        elif sha256_file(destination) != record["sha256"]:
            raise ValueError(f"候选目标已存在且哈希不同：{destination}")
        copied.append({"source": str(source), "target": str(destination), "sha256": record["sha256"]})
    update_stage(
        wannier_run,
        "03_candidates",
        status="PASS",
        reused_utc=_utc_now(),
        provenance={
            "source_run": str(candidates_info.get("source_run")),
            "source_stage": "03_candidates",
        },
    )
    return {"stage": "03_candidates", "status": "PASS", "copied": copied}


def run_stage1_coldstart(
    config_path: str | Path,
    campaign_dir: str | Path,
    runs_root: str | Path,
    *,
    execute: bool = False,
    poll_seconds: float = 30.0,
    python: str | None = None,
    reuse: dict[str, Any] | None = None,
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
    signature, signature_sha256 = coldstart_signature(config_file, plan)

    with ColdStartLock(campaign):
        if state_path.is_file():
            state = read_json(state_path)
            if state.get("input_signature") != signature:
                raise RuntimeError("已有 cold-start 状态与当前配置/POSCAR 哈希不同，拒绝覆盖。")
        else:
            preprocess_runs = {
                item["id"]: str(
                    derived_preprocess_run(
                        runs,
                        config["project"],
                        item["id"],
                        item["poscar_sha256"],
                        signature_sha256,
                    )
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
            plan_item = next(item for item in plan["structures"] if item["id"] == sid)
            if reuse and reuse.get("preprocess_sources", {}).get(sid):
                preprocess_run = Path(reuse["preprocess_sources"][sid])
                verified = _verify_preprocess_reuse_source(preprocess_run, plan_item, plan)
                state.setdefault("preprocess_reuse", {})[sid] = verified
                _save(state_path, state, "PREPROCESS_REUSED", current_structure=sid,
                      source_run=str(preprocess_run))
                sources[sid] = str(preprocess_run)
                continue
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
            wannier_run = derived_wannier_run(runs, config["project"], signature_sha256)
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
            if reuse:
                provenance = {
                    "schema_version": 1,
                    "mode": "technical_recovery_reuse",
                    "created_utc": _utc_now(),
                    "prior_run": reuse["prior_run"],
                    "stages": reuse["stages"],
                    "imported_via": "fit_from_preprocess/import_completed_stage (hash-verified parent references)",
                }
                if reuse.get("candidates"):
                    applied = _apply_reuse_candidates(wannier_run, reuse["candidates"])
                    provenance["candidates_files"] = applied["copied"]
                atomic_json(wannier_run / "00_reuse/provenance.json", provenance)
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
            python=python,
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
