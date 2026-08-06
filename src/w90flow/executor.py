from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from .candidates import num_wann_from_projection, suggest_projection_candidates
from .parsers import parse_eigenval, parse_fermi_outcar, parse_outcar_status, parse_poscar, parse_procar_orbital_scores, parse_wout
from .paths import launcher_root, pyw90_executable as pyw90_executable_env
from .pyw90_adapter import element_projection_envelope, run_dos_suggest
from .state import atomic_json, initialize_run, read_json, sha256_file, update_stage
from .structure import (
    generate_ase_band_path,
    patch_wannier_band_path,
    patch_wannier_uniform_grid,
    render_gamma_kpoints,
)


VASP_STAGES = {
    "scf": "01_scf",
    "band": "02_band",
    "wannier": "04_wannier",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise FileExistsError(f"拒绝覆盖不同内容：{target}")
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _copy_exact(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise FileExistsError(f"拒绝覆盖不同内容：{target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_exact_text(target: Path, payload: str) -> None:
    encoded = payload.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if target.exists():
        if sha256_file(target) != digest:
            raise FileExistsError(f"拒绝覆盖不同内容：{target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)


def _load_potcar_lock(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    pseudo = config.get("pseudopotentials", {})
    lock_path = Path(pseudo.get("lock_resolved", ""))
    if not lock_path.is_file():
        raise ValueError("prepare 需要 pseudopotentials.lock。")
    lock = read_json(lock_path)
    registry = Path(pseudo.get("registry", lock.get("registry", ""))).expanduser().resolve()
    if not registry.is_dir():
        raise ValueError(f"找不到私有 POTCAR registry：{registry}")
    return lock, registry


def _build_potcar(config: dict[str, Any], set_name: str, elements: list[str], output: Path) -> dict[str, Any]:
    lock, registry = _load_potcar_lock(config)
    try:
        specification = lock["sets"][set_name]
    except KeyError as exc:
        raise ValueError(f"POTCAR lock 中没有 set：{set_name}") from exc
    order = list(specification["order"])
    if order != elements:
        raise ValueError(f"POTCAR 顺序 {order} 与 POSCAR 元素顺序 {elements} 不一致。")
    labels = specification.get("labels", {})
    if not isinstance(labels, dict):
        raise ValueError(f"POTCAR set {set_name} 的 labels 必须是元素到 registry 标签的映射。")
    unknown_labels = sorted(set(labels) - set(order))
    if unknown_labels:
        raise ValueError(
            f"POTCAR set {set_name} 的 labels 含非 POSCAR 元素：{', '.join(unknown_labels)}"
        )

    chunks: list[bytes] = []
    components: list[dict[str, Any]] = []
    for element in order:
        registry_label = str(labels.get(element, element))
        source = registry / registry_label / "POTCAR"
        if not source.is_file():
            raise ValueError(f"registry 缺少 {registry_label}/POTCAR（POSCAR 元素 {element}）")
        actual = sha256_file(source)
        expected = specification["component_sha256"][element]
        if actual != expected:
            raise ValueError(f"{element}（registry 标签 {registry_label}）POTCAR SHA256 不符。")
        chunks.append(source.read_bytes())
        components.append({
            "element": element,
            "registry_label": registry_label,
            "source": str(source),
            "sha256": actual,
        })
    payload = b"".join(chunks)
    combined = hashlib.sha256(payload).hexdigest()
    if combined != specification["combined_sha256"]:
        raise ValueError(f"组合 POTCAR SHA256 不符：{combined}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256_file(output) != combined:
        raise FileExistsError(f"拒绝覆盖不同 POTCAR：{output}")
    if not output.exists():
        output.write_bytes(payload)
    return {"set": set_name, "order": order, "combined_sha256": combined, "components": components}


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "project.snapshot.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("project.snapshot.yaml 无效。")
    return data


def prepare_run(
    config: dict[str, Any],
    runs_root: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> Path:
    band_path_mode = config.get("band_path", {}).get("mode", "template")
    if band_path_mode in {"vaspkit", "both"}:
        raise ValueError(
            "VASPKIT provider 尚未冻结可审计的非交互任务接口；"
            "请先使用 band_path.mode=template 或 native。"
        )
    all_have_scf_mesh = all(item.get("meshes_resolved", {}).get("scf") for item in config["structures"])
    all_have_wannier_mesh = all(item.get("meshes_resolved", {}).get("wannier") for item in config["structures"])
    required_templates = {
        "scf": ("incar",) if all_have_scf_mesh else ("incar", "kpoints"),
        "band": ("incar",) if band_path_mode == "native" else ("incar", "kpoints"),
        "wannier": ("incar",) if all_have_wannier_mesh else ("incar", "kpoints"),
    }
    for stage, roles in required_templates.items():
        for role in roles:
            if not config.get("templates", {}).get(stage, {}).get(f"{role}_resolved"):
                raise ValueError(f"prepare 缺少 templates.{stage}.{role}。")
    pseudo_default = config.get("pseudopotentials", {}).get("set")
    if not pseudo_default and any(not item.get("pseudopotential_set") for item in config["structures"]):
        raise ValueError("每个结构或 pseudopotentials 必须指定 set。")

    run_dir = initialize_run(config, runs_root, run_dir=run_dir)
    manifests: list[dict[str, Any]] = []
    for structure in config["structures"]:
        sid = str(structure["id"])
        source_poscar = Path(structure["poscar_resolved"])
        poscar_info = parse_poscar(source_poscar)
        set_name = structure.get("pseudopotential_set", pseudo_default)
        private_potcar = run_dir / "private" / f"POTCAR.{set_name}"
        potcar = _build_potcar(config, set_name, poscar_info["elements"], private_potcar)
        base = run_dir / "structures" / sid
        _copy_exact(source_poscar, base / "00_input/POSCAR")
        links: dict[str, str] = {}
        generated_inputs: dict[str, Any] = {}
        band_path = None
        if band_path_mode == "native":
            band_path = generate_ase_band_path(
                poscar_info,
                list(structure["structure_analysis"]["periodic_axes"]),
                int(config.get("band_path", {}).get("points_per_segment", 40)),
            )
            generated_inputs["band_path"] = {
                key: value for key, value in band_path.items()
                if key not in {"kpoints_text", "wannier_kpoint_path"}
            }
        for short_stage, directory_name in VASP_STAGES.items():
            directory = base / directory_name
            _copy_exact(source_poscar, directory / "POSCAR")
            templates = config["templates"][short_stage]
            incar_text = Path(templates["incar_resolved"]).read_text(encoding="utf-8", errors="replace")
            if short_stage == "wannier":
                wannier_mesh = structure.get("meshes_resolved", {}).get("wannier")
                if wannier_mesh:
                    incar_text = patch_wannier_uniform_grid(incar_text, wannier_mesh)
                if band_path is not None:
                    incar_text = patch_wannier_band_path(incar_text, band_path["wannier_kpoint_path"])
            _write_exact_text(directory / "INCAR", incar_text)

            resolved_mesh = structure.get("meshes_resolved", {}).get(short_stage)
            preserve_precomputed = bool(structure.get("preserve_precomputed_inputs"))
            if preserve_precomputed and short_stage in {"scf", "band"}:
                _copy_exact(Path(templates["kpoints_resolved"]), directory / "KPOINTS")
                generated_inputs[f"{short_stage}_kpoints"] = "preserved_precomputed_handoff"
            elif short_stage == "band" and band_path is not None:
                _write_exact_text(directory / "KPOINTS", band_path["kpoints_text"])
                generated_inputs["band_kpoints"] = "native_ase_path"
            elif short_stage in {"scf", "wannier"} and resolved_mesh:
                _write_exact_text(directory / "KPOINTS", render_gamma_kpoints(resolved_mesh))
                generated_inputs[f"{short_stage}_mesh"] = resolved_mesh
            else:
                _copy_exact(Path(templates["kpoints_resolved"]), directory / "KPOINTS")
            links[directory_name] = _link_or_copy(private_potcar, directory / "POTCAR")
        manifests.append({
            "structure": sid,
            "poscar": poscar_info,
            "structure_analysis": structure.get("structure_analysis"),
            "meshes_resolved": structure.get("meshes_resolved", {}),
            "generated_inputs": generated_inputs,
            "poscar_sha256": sha256_file(source_poscar),
            "potcar": potcar,
            "potcar_materialization": links,
        })

    atomic_json(run_dir / "prepare-manifest.json", {
        "schema_version": 1,
        "prepared_utc": _utc_now(),
        "structures": manifests,
        "note": "POTCAR is private and excluded from public export.",
    })
    update_stage(run_dir, "00_validate", status="PASS", finished_utc=_utc_now(), outputs=["prepare-manifest.json"])
    return run_dir


def _launcher(run_dir: Path, config: dict[str, Any], directory: Path) -> list[str]:
    workspace = launcher_root()
    executable = workspace / "00_命令入口/计算/运行VASP.command"
    backend = config.get("backend", {})
    default_kind = "ncl" if config.get("physics", {}).get("soc") else "std"
    return [str(executable), str(directory), str(backend.get("vasp", default_kind)), str(backend.get("mpi_processes", 2))]


def _wannier_launcher(directory: Path) -> list[str]:
    workspace = launcher_root()
    executable = workspace / "00_命令入口/计算/运行Wannier90.command"
    return [str(executable), str(directory), "wannier90", "run"]


def _required_files(directory: Path, stage: str) -> list[Path]:
    base = [directory / name for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")]
    if stage in {"band", "wannier"}:
        base.append(directory / "CHGCAR")
    return base


def _handoff_chgcar(run_dir: Path, structure_id: str) -> dict[str, str]:
    base = run_dir / "structures" / structure_id
    source = base / "01_scf/CHGCAR"
    if not source.is_file():
        raise ValueError(f"SCF 没有生成 CHGCAR：{source}")
    return {
        "band": _link_or_copy(source, base / "02_band/CHGCAR"),
        "wannier": _link_or_copy(source, base / "04_wannier/CHGCAR"),
    }


def _postcheck(directory: Path, stage: str, returncode: int) -> dict[str, Any]:
    result: dict[str, Any] = {"returncode": returncode, "passed": returncode == 0, "issues": []}
    if returncode != 0:
        result["issues"].append(f"进程退出码 {returncode}")
    if stage == "scf":
        for name in ("OUTCAR", "OSZICAR", "CHGCAR"):
            if not (directory / name).is_file():
                result["issues"].append(f"缺少 {name}")
        if (directory / "OUTCAR").is_file():
            result["outcar"] = parse_outcar_status(directory / "OUTCAR")
            if not result["outcar"]["converged"]:
                result["issues"].append("SCF 未达到电子收敛")
    elif stage == "band":
        for name in ("OUTCAR", "EIGENVAL", "PROCAR", "vasprun.xml"):
            if not (directory / name).is_file():
                result["issues"].append(f"缺少 {name}")
        if (directory / "EIGENVAL").is_file():
            try:
                eig = parse_eigenval(directory / "EIGENVAL")
                result["eigenval"] = {key: eig[key] for key in ("nelect", "nkpts", "nbands")}
            except ValueError as exc:
                result["issues"].append(f"EIGENVAL 尚未完整：{exc}")
    elif stage == "wannier":
        for name in ("wannier90.amn", "wannier90.mmn", "wannier90.eig", "wannier90.wout", "wannier90_hr.dat", "wannier90_centres.xyz"):
            if not (directory / name).is_file():
                result["issues"].append(f"缺少 {name}")
        if (directory / "wannier90.wout").is_file():
            result["wout"] = parse_wout(directory / "wannier90.wout")
            if not result["wout"]["completed"]:
                result["issues"].append("Wannier90 未正常完成")
    result["passed"] = not result["issues"]
    return result


def run_vasp_stage(run_dir: str | Path, stage: str, execute: bool = False, structure_ids: list[str] | None = None) -> dict[str, Any]:
    if stage not in VASP_STAGES:
        raise ValueError(f"stage 只能是：{', '.join(VASP_STAGES)}")
    root = Path(run_dir).expanduser().resolve()
    config = _read_snapshot(root)
    if execute and not config.get("execution", {}).get("allow_execute", False):
        raise PermissionError("project 配置中的 execution.allow_execute 不是 true。")
    selected = structure_ids or [str(item["id"]) for item in config["structures"]]
    stage_dir_name = VASP_STAGES[stage]
    commands: list[dict[str, Any]] = []
    for sid in selected:
        directory = root / "structures" / sid / stage_dir_name
        missing = [str(path) for path in _required_files(directory, stage) if not path.is_file()]
        command_chain = [_launcher(root, config, directory)]
        if stage == "wannier":
            command_chain.append(_wannier_launcher(directory))
        commands.append({"structure": sid, "directory": str(directory), "command_chain": command_chain, "missing_inputs": missing})
    if not execute:
        return {"mode": "DRY_RUN", "stage": stage, "would_run": commands, "passed": not any(item["missing_inputs"] for item in commands)}
    if any(item["missing_inputs"] for item in commands):
        raise ValueError("存在缺失输入；先完成上一阶段或重新 prepare。")

    workflow_stage = stage_dir_name
    update_stage(
        root,
        workflow_stage,
        status="RUNNING",
        started_utc=_utc_now(),
        commands=commands,
        active_process=None,
    )
    results: list[dict[str, Any]] = []
    for item in commands:
        directory = Path(item["directory"])
        log_path = directory / "runner.log"
        with log_path.open("wb") as log:
            returncode = 0
            executed_commands: list[dict[str, Any]] = []
            for command_index, command in enumerate(item["command_chain"]):
                if stage == "wannier" and command_index > 0:
                    wout_path = directory / "wannier90.wout"
                    if wout_path.is_file() and (directory / "wannier90_hr.dat").is_file() and parse_wout(wout_path)["completed"]:
                        executed_commands.append({"command": command, "skipped": True, "reason": "VASP library mode already completed Wannier90"})
                        continue
                process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
                active_process = {
                    "pid": process.pid,
                    "structure": item["structure"],
                    "command": command,
                    "started_utc": _utc_now(),
                }
                update_stage(root, workflow_stage, active_process=active_process)
                returncode = process.wait()
                update_stage(root, workflow_stage, active_process=None)
                executed_commands.append({"command": command, "returncode": returncode})
                if returncode != 0:
                    break
        check = _postcheck(directory, stage, returncode)
        check["executed_commands"] = executed_commands
        if stage == "scf" and check["passed"]:
            check["handoff"] = _handoff_chgcar(root, item["structure"])
        check.update({"structure": item["structure"], "log": str(log_path)})
        results.append(check)
        if not check["passed"]:
            break
    passed = len(results) == len(commands) and all(item["passed"] for item in results)
    update_stage(
        root,
        workflow_stage,
        status="PASS" if passed else "FAIL",
        finished_utc=_utc_now(),
        outputs=results,
        active_process=None,
    )
    return {"mode": "EXECUTE", "stage": stage, "passed": passed, "results": results}


def reconcile_stage(run_dir: str | Path, stage: str, structure_ids: list[str] | None = None) -> dict[str, Any]:
    """Promote an externally/background-completed stage without resubmitting it.

    Incomplete output remains RUNNING/PENDING. Reconcile never guesses FAIL because
    the worker may still be active; a non-zero launcher exit is recorded by run-stage.
    """
    if stage not in VASP_STAGES:
        raise ValueError(f"stage 只能是：{', '.join(VASP_STAGES)}")
    root = Path(run_dir).expanduser().resolve()
    config = _read_snapshot(root)
    selected = structure_ids or [str(item["id"]) for item in config["structures"]]
    results: list[dict[str, Any]] = []
    for sid in selected:
        directory = root / "structures" / sid / VASP_STAGES[stage]
        check = _postcheck(directory, stage, 0)
        if stage == "scf" and check["passed"]:
            check["handoff"] = _handoff_chgcar(root, sid)
        check["structure"] = sid
        results.append(check)
    passed = all(item["passed"] for item in results)
    state_path = root / VASP_STAGES[stage] / "stage.json"
    current = read_json(state_path)
    if passed:
        update_stage(root, VASP_STAGES[stage], status="PASS", reconciled_utc=_utc_now(), outputs=results)
    return {
        "stage": stage,
        "passed": passed,
        "status_before": current.get("status"),
        "status_after": "PASS" if passed else current.get("status"),
        "results": results,
    }


def import_completed_stage(
    run_dir: str | Path,
    stage: str,
    sources: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Import a completed, input-identical SCF/Band stage without recomputation.

    Files are hard-linked when possible to save space. Exact hashes of POSCAR,
    POTCAR, INCAR, and KPOINTS must match the prepared immutable run; Band also
    verifies the SCF CHGCAR that fixes its charge-density input.
    """
    if stage not in {"scf", "band"}:
        raise ValueError("import-stage 当前只支持 scf 或 band。")
    root = Path(run_dir).expanduser().resolve()
    config = _read_snapshot(root)
    expected_ids = [str(item["id"]) for item in config["structures"]]
    if set(sources) != set(expected_ids):
        raise ValueError(f"source 映射必须恰好覆盖结构：{', '.join(expected_ids)}")
    stage_name = VASP_STAGES[stage]
    manifest_path = root / stage_name / "import-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 import manifest：{manifest_path}")
    output_names = {
        "scf": ("OUTCAR", "OSZICAR", "CHGCAR", "EIGENVAL"),
        "band": ("OUTCAR", "EIGENVAL", "PROCAR", "vasprun.xml"),
    }[stage]
    results: list[dict[str, Any]] = []
    for sid in expected_ids:
        source = Path(sources[sid]).expanduser().resolve()
        target = root / "structures" / sid / stage_name
        if not source.is_dir():
            raise ValueError(f"导入源不是目录：{source}")
        input_hashes: dict[str, str] = {}
        input_names = ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR") if stage == "band" else ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
        for name in input_names:
            old = source / name
            new = target / name
            if not old.is_file() or not new.is_file():
                raise ValueError(f"无法验证导入输入 {sid}/{stage}/{name}")
            old_hash, new_hash = sha256_file(old), sha256_file(new)
            if old_hash != new_hash:
                raise ValueError(f"导入源与新 run 的 {sid}/{stage}/{name} 不一致。")
            input_hashes[name] = old_hash
        materialization: dict[str, str] = {}
        output_hashes: dict[str, str] = {}
        for name in output_names:
            old = source / name
            if not old.is_file():
                raise ValueError(f"导入源缺少 {sid}/{stage}/{name}")
            materialization[name] = _link_or_copy(old, target / name)
            output_hashes[name] = sha256_file(old)
        check = _postcheck(target, stage, 0)
        if not check["passed"]:
            raise ValueError(f"导入后的 {sid}/{stage} 未通过完整性检查：{check['issues']}")
        if stage == "scf":
            check["handoff"] = _handoff_chgcar(root, sid)
        results.append({
            "structure": sid,
            "source": str(source),
            "input_sha256": input_hashes,
            "output_sha256": output_hashes,
            "materialization": materialization,
            "postcheck": check,
        })
    document = {
        "schema_version": 1,
        "stage": stage,
        "imported_utc": _utc_now(),
        "results": results,
        "note": (
            "Imported only after exact POSCAR/POTCAR/INCAR/KPOINTS/CHGCAR hash equivalence."
            if stage == "band" else
            "Imported only after exact POSCAR/POTCAR/INCAR/KPOINTS hash equivalence."
        ),
    }
    atomic_json(manifest_path, document)
    update_stage(root, stage_name, status="PASS", imported_utc=_utc_now(), outputs=results)
    return {"passed": True, "manifest": str(manifest_path), "results": results}


def generate_candidates(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    config = _read_snapshot(root)
    target_min, target_max = map(float, config["physics"]["target_energy_eV"])
    energy_reference = config["physics"].get("energy_reference", "absolute")
    soc = bool(config["physics"].get("soc"))
    mode = config.get("ensemble", {}).get("mode", "independent")
    chemical_valence = config.get("candidates", {}).get("chemical_valence")
    max_candidates = int(config.get("candidates", {}).get("max_candidates", 8))
    shell_family_min_weight = float(config.get("candidates", {}).get("shell_family_min_weight", 0.08))
    shell_combination_min_weight = float(config.get("candidates", {}).get("shell_combination_min_weight", 0.45))
    max_model_fraction = float(config.get("candidates", {}).get("max_num_wann_fraction_of_bands", 1.0))
    if not 0.0 < max_model_fraction <= 1.0:
        raise ValueError("candidates.max_num_wann_fraction_of_bands 必须在 (0, 1]。")
    pyw90_config = config.get("candidates", {}).get("pyw90_dos", {})
    if not isinstance(pyw90_config, dict):
        raise ValueError("candidates.pyw90_dos 必须是对象。")
    pyw90_enabled = bool(pyw90_config.get("enabled", False))
    pyw90_required = bool(pyw90_config.get("required", False))
    if pyw90_required and not pyw90_enabled:
        raise ValueError("candidates.pyw90_dos.required=true 时 enabled 也必须为 true。")
    pyw90_lower_bound = float(pyw90_config.get("lower_bound", 0.1))
    pyw90_executable = str(pyw90_config.get("executable") or pyw90_executable_env() or "")
    analyses: dict[str, Any] = {}
    element_counts_by_structure: dict[str, dict[str, int]] = {}
    for item in config["structures"]:
        sid = str(item["id"])
        base = root / "structures" / sid
        poscar = parse_poscar(base / "00_input/POSCAR")
        element_counts = dict(zip(poscar["elements"], poscar["counts"]))
        element_counts_by_structure[sid] = element_counts
        band_dir = base / "02_band"
        eig = parse_eigenval(band_dir / "EIGENVAL")
        fermi = parse_fermi_outcar(base / "01_scf/OUTCAR")
        absolute_target = [target_min, target_max]
        if energy_reference == "fermi_relative":
            absolute_target = [target_min + fermi, target_max + fermi]
        scores = parse_procar_orbital_scores(band_dir / "PROCAR", poscar, *absolute_target)
        pyw90_baseline: dict[str, Any] = {"enabled": pyw90_enabled, "status": "DISABLED"}
        if pyw90_enabled:
            output_json = root / "03_candidates" / f"pyw90-dos-{sid}.json"
            try:
                baseline = run_dos_suggest(
                    band_dir / "vasprun.xml",
                    target_energy_eV=(float(absolute_target[0]), float(absolute_target[1])),
                    degeneracy=2 if soc else 1,
                    lower_bound=pyw90_lower_bound,
                    energy_reference="absolute",
                    executable=pyw90_executable,
                    output_json=output_json,
                )
                result = baseline["result"]
                represented = element_projection_envelope(result, element_counts)
                projection = list(represented["projection"])
                expected_num_wann = num_wann_from_projection(projection, element_counts, soc)
                original_num_wann = int(result["num_wann_selected"])
                if represented["representation"] != "element_level_envelope" and original_num_wann != expected_num_wann:
                    raise ValueError(
                        "PyW90 num_wann 与 POSCAR 元素计数/SOC 简并不一致："
                        f"{original_num_wann} != {expected_num_wann}"
                    )
                pyw90_baseline = {
                    "enabled": True,
                    "status": "PASS",
                    "document": str(output_json),
                    "source_sha256": baseline["source_sha256"],
                    "tool_sha256": baseline["tool_sha256"],
                    "projection": projection,
                    "num_wann": expected_num_wann,
                    "original_num_wann": original_num_wann,
                    "representation": represented["representation"],
                    "site_uniform": represented["site_uniform"],
                    "site_projections": result.get("site_projections", []),
                    "extra_input": result.get("extra_input"),
                    "fermi_eV_reported": result.get("fermi_eV"),
                    "dos_gap_eV": result.get("dos_gap_eV"),
                    "lower_bound": pyw90_lower_bound,
                }
            except (OSError, RuntimeError, ValueError) as exc:
                pyw90_baseline = {
                    "enabled": True,
                    "status": "FAILED",
                    "error": str(exc),
                    "required": pyw90_required,
                }
                if pyw90_required:
                    raise ValueError(f"{sid} 的必需 PyW90 DOS baseline 失败：{exc}") from exc
        analyses[sid] = {
            "source": {"EIGENVAL": str(band_dir / "EIGENVAL"), "PROCAR": str(band_dir / "PROCAR")},
            "eigenval": {key: eig[key] for key in ("nelect", "nkpts", "nbands")},
            "fermi_eV": fermi,
            "target_energy_reference_eV": [target_min, target_max],
            "target_energy_absolute_eV": absolute_target,
            "projection_analysis": scores,
            "pyw90_dos_baseline": pyw90_baseline,
        }

    def pyw90_external(structure_ids: list[str]) -> list[dict[str, Any]]:
        unique: list[tuple[str, list[str]]] = []
        for sid in structure_ids:
            baseline = analyses[sid]["pyw90_dos_baseline"]
            projection = baseline.get("projection") if baseline.get("status") == "PASS" else None
            if projection and list(projection) not in [item[1] for item in unique]:
                unique.append((sid, list(projection)))
        external = []
        for sid, projection in unique:
            baseline = analyses[sid]["pyw90_dos_baseline"]
            envelope = baseline.get("representation") == "element_level_envelope"
            if len(unique) == 1:
                candidate = "pyw90-dos-envelope" if envelope else "pyw90-dos"
            else:
                candidate = f"pyw90-dos-envelope-{sid}" if envelope else f"pyw90-dos-{sid}"
            external.append({
                "candidate": candidate,
                "projection": projection,
                "candidate_family": "external_baseline",
            })
        if len(unique) > 1:
            union = list(dict.fromkeys(label for _, projection in unique for label in projection))
            if union not in [item[1] for item in unique]:
                external.append({
                    "candidate": "pyw90-dos-union",
                    "projection": union,
                    "candidate_family": "external_baseline",
                })
        return external

    if mode == "common_basis":
        counts_values = list(element_counts_by_structure.values())
        if any(value != counts_values[0] for value in counts_values[1:]):
            raise ValueError("common_basis 当前要求结构具有相同元素和原子数；不同化学组成请用 independent。")
        labels = sorted({label for data in analyses.values() for label in data["projection_analysis"]["orbital_scores"]})
        combined = {
            label: sum(data["projection_analysis"]["orbital_scores"].get(label, 0.0) for data in analyses.values()) / len(analyses)
            for label in labels
        }
        maximum_dimension = math.floor(min(data["eigenval"]["nbands"] for data in analyses.values()) * max_model_fraction)
        shared = suggest_projection_candidates(
            combined,
            counts_values[0],
            soc,
            chemical_valence,
            shell_family_min_weight=shell_family_min_weight,
            shell_combination_min_weight=shell_combination_min_weight,
            max_num_wann=maximum_dimension,
            max_candidates=max_candidates,
            external_candidates=pyw90_external(list(analyses)),
        )
        candidate_payload: dict[str, Any] = {"shared_candidates": shared, "combined_orbital_scores": combined}
    else:
        candidate_payload = {
            "independent_candidates": {
                sid: suggest_projection_candidates(
                    analyses[sid]["projection_analysis"]["orbital_scores"],
                    element_counts_by_structure[sid],
                    soc,
                    chemical_valence,
                    shell_family_min_weight=shell_family_min_weight,
                    shell_combination_min_weight=shell_combination_min_weight,
                    max_num_wann=math.floor(analyses[sid]["eigenval"]["nbands"] * max_model_fraction),
                    max_candidates=max_candidates,
                    external_candidates=pyw90_external([sid]),
                )
                for sid in analyses
            }
        }
    payload = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "ensemble_mode": mode,
        "soc": soc,
        "energy_reference": energy_reference,
        "target_energy_eV": [target_min, target_max],
        "candidate_policy": {
            "max_candidates": max_candidates,
            "max_num_wann_fraction_of_bands": max_model_fraction,
            "shell_family_min_weight": shell_family_min_weight,
            "shell_combination_min_weight": shell_combination_min_weight,
            "chemical_valence_used": bool(chemical_valence),
            "pyw90_dos": {
                "enabled": pyw90_enabled,
                "required": pyw90_required,
                "lower_bound": pyw90_lower_bound,
                "role": "optional_projection_baseline_not_authoritative_decision",
            },
        },
        "structures": analyses,
        **candidate_payload,
    }
    output = root / "03_candidates/candidate-spec.json"
    atomic_json(output, payload)
    update_stage(root, "03_candidates", status="PASS", finished_utc=_utc_now(), outputs=[str(output)])
    return payload


def finalize_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    config = _read_snapshot(root)
    mode = config.get("ensemble", {}).get("mode", "independent")
    evaluated: dict[str, dict[str, Any]] = {}
    for structure in config["structures"]:
        sid = str(structure["id"])
        quality_path = root / "structures" / sid / "05_quality/quality.json"
        if not quality_path.is_file():
            raise ValueError(f"{sid} 尚无 quality.json。")
        evaluated[sid] = read_json(quality_path)
        if not evaluated[sid].get("passed"):
            raise ValueError(f"{sid} 未通过质量门槛，不能导出。")

    if mode == "common_basis":
        candidates = {result["candidate"] for result in evaluated.values()}
        projections = {tuple(result["projection"]) for result in evaluated.values()}
        dimensions = {result["num_wann"] for result in evaluated.values()}
        if len(candidates) != 1 or len(projections) != 1 or len(dimensions) != 1:
            raise ValueError("common_basis 结构没有使用同一候选/投影/NUM_WANN。")
        ranking = [{
            "rank": 1,
            "candidate": next(iter(candidates)),
            "projection": list(next(iter(projections))),
            "num_wann": next(iter(dimensions)),
            "structures": {sid: result["metrics"] for sid, result in evaluated.items()},
            "worst_metrics": {
                key: max(abs(result["metrics"][key]) for result in evaluated.values())
                for key in ("mae_mev", "rmse_mev", "p95_mev", "max_mev", "gap_error_mev")
            },
        }]
        ranking_document: dict[str, Any] = {"ensemble_mode": mode, "ranking": ranking}
    else:
        ranking_document = {
            "ensemble_mode": mode,
            "rankings_by_structure": {
                sid: [{
                    "rank": 1,
                    "candidate": result["candidate"],
                    "projection": result["projection"],
                    "num_wann": result["num_wann"],
                    "nocc": result["nocc"],
                    "metrics": result["metrics"],
                }]
                for sid, result in evaluated.items()
            },
        }
    ranking_path = root / "06_rank/ranking.json"
    atomic_json(ranking_path, ranking_document)
    update_stage(root, "06_rank", status="PASS", finished_utc=_utc_now(), outputs=[str(ranking_path)])

    packages: list[dict[str, Any]] = []
    for sid, result in evaluated.items():
        source_base = root / "structures" / sid
        output = root / "07_export" / sid / "rank-1"
        output.mkdir(parents=True, exist_ok=False)
        sources = {
            "POSCAR": source_base / "00_input/POSCAR",
            "wannier90_hr.dat": source_base / "04_wannier/wannier90_hr.dat",
            "wannier90_centres.xyz": source_base / "04_wannier/wannier90_centres.xyz",
            "VASP_W90_cmp.pdf": source_base / "05_quality/VASP_W90_cmp.pdf",
            "quality.json": source_base / "05_quality/quality.json",
        }
        materialization: dict[str, str] = {}
        for name, source in sources.items():
            materialization[name] = _link_or_copy(source, output / name)
        centre_lines = [
            line for line in sources["wannier90_centres.xyz"].read_text(encoding="utf-8", errors="replace").splitlines()
            if line.split() and line.split()[0] == "X"
        ]
        (output / "wannier90-wcenter.dat").write_text("\n".join(centre_lines) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "structure": sid,
            "rank": 1,
            "candidate": result["candidate"],
            "projection": result["projection"],
            "num_wann": result["num_wann"],
            "nocc": result["nocc"],
            "contains_potcar": False,
            "materialization": materialization,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in output.iterdir() if path.is_file() and path.name != "manifest.json"
            },
        }
        atomic_json(output / "manifest.json", manifest)
        packages.append({"structure": sid, "path": str(output), "manifest": manifest})
    update_stage(root, "07_export", status="PASS", finished_utc=_utc_now(), outputs=[item["path"] for item in packages])
    return {"passed": True, "ranking": str(ranking_path), "packages": packages}
