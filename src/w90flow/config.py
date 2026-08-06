from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .parsers import parse_poscar
from .structure import resolve_periodicity, resolve_uniform_mesh


class ConfigError(ValueError):
    """Raised when a project file cannot define a reproducible run."""


def load_project(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("项目配置必须是 YAML/JSON 对象。")

    required = ("project", "structures", "physics", "backend", "ranking")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"项目配置缺少字段：{', '.join(missing)}")
    if not isinstance(data["structures"], list) or not data["structures"]:
        raise ConfigError("structures 至少需要一个结构。")

    seen: set[str] = set()
    structure_dimensions: set[int] = set()
    for structure in data["structures"]:
        if not isinstance(structure, dict) or not structure.get("id") or not structure.get("poscar"):
            raise ConfigError("每个 structure 都需要 id 和 poscar。")
        sid = str(structure["id"])
        if sid in seen:
            raise ConfigError(f"结构 id 重复：{sid}")
        seen.add(sid)
        poscar = (config_path.parent / str(structure["poscar"])).resolve()
        if not poscar.is_file():
            raise ConfigError(f"找不到 POSCAR：{poscar}")
        structure["poscar_resolved"] = str(poscar)
        try:
            parsed = parse_poscar(poscar)
            analysis = resolve_periodicity(parsed, data["physics"])
        except ValueError as exc:
            raise ConfigError(f"结构 {sid} 的维度/周期性无法确定：{exc}") from exc
        structure["structure_analysis"] = analysis
        structure_dimensions.add(int(analysis["dimensionality"]))

        meshes = data.get("meshes", {})
        resolved_meshes: dict[str, list[int]] = {}
        mesh_aliases = {
            "scf": meshes.get("scf"),
            "wannier": meshes.get("wannier", meshes.get("screening")),
            "production": meshes.get("production"),
        }
        for role, specification in mesh_aliases.items():
            if specification is None:
                continue
            try:
                resolved_meshes[role] = resolve_uniform_mesh(
                    specification,
                    parsed["lattice_angstrom"],
                    analysis["periodic_axes"],
                )
            except ValueError as exc:
                raise ConfigError(f"结构 {sid} 的 meshes.{role} 无效：{exc}") from exc
        structure["meshes_resolved"] = resolved_meshes

    lock = data.get("pseudopotentials", {}).get("lock")
    if lock:
        lock_path = (config_path.parent / str(lock)).resolve()
        if not lock_path.is_file():
            raise ConfigError(f"找不到 POTCAR lock：{lock_path}")
        data["pseudopotentials"]["lock_resolved"] = str(lock_path)

    # A portable project may keep its private POTCAR registry beside the
    # configuration file.  Resolve that path here, just as we do for the
    # lock and templates, so a copied project never falls back to the
    # absolute path recorded on the machine that created the lock.
    registry = data.get("pseudopotentials", {}).get("registry")
    if registry:
        registry_path = (config_path.parent / str(registry)).resolve()
        if not registry_path.is_dir():
            raise ConfigError(f"找不到 POTCAR registry：{registry_path}")
        data["pseudopotentials"]["registry"] = str(registry_path)

    for stage, templates in data.get("templates", {}).items():
        if not isinstance(templates, dict):
            raise ConfigError(f"templates.{stage} 必须是对象。")
        for role, source in list(templates.items()):
            if role.endswith("_resolved"):
                continue
            source_path = (config_path.parent / str(source)).resolve()
            if not source_path.is_file():
                raise ConfigError(f"找不到模板 templates.{stage}.{role}：{source_path}")
            templates[f"{role}_resolved"] = str(source_path)

    target = data["physics"].get("target_energy_eV")
    if not isinstance(target, list) or len(target) != 2 or float(target[0]) >= float(target[1]):
        raise ConfigError("physics.target_energy_eV 必须是递增的两个数。")
    energy_reference = data["physics"].get("energy_reference", "absolute")
    if energy_reference not in {"absolute", "fermi_relative"}:
        raise ConfigError("physics.energy_reference 只能是 absolute 或 fermi_relative。")
    data["physics"]["energy_reference"] = energy_reference
    ensemble_mode = data.get("ensemble", {}).get(
        "mode",
        "common_basis" if data.get("candidates", {}).get("same_basis_across_structures") else "independent",
    )
    if ensemble_mode not in {"independent", "common_basis"}:
        raise ConfigError("ensemble.mode 只能是 independent 或 common_basis。")
    data.setdefault("ensemble", {})["mode"] = ensemble_mode
    if ensemble_mode == "common_basis" and len(structure_dimensions) != 1:
        raise ConfigError("common_basis 的结构必须具有相同 dimensionality。")

    wannier_input_mode = data.get("backend", {}).get("wannier_input_mode", "embedded")
    if wannier_input_mode not in {"embedded", "standalone_win"}:
        raise ConfigError("backend.wannier_input_mode 只能是 embedded 或 standalone_win。")
    data["backend"]["wannier_input_mode"] = wannier_input_mode

    band_path = data.get("band_path", {})
    band_path_mode = band_path.get("mode", "template")
    if band_path_mode not in {"template", "native", "vaspkit", "both"}:
        raise ConfigError("band_path.mode 只能是 template、native、vaspkit 或 both。")
    if int(band_path.get("points_per_segment", 40)) < 2:
        raise ConfigError("band_path.points_per_segment 必须至少为 2。")
    data.setdefault("band_path", {})["mode"] = band_path_mode

    iteration = data.get("iteration", {})
    max_window_attempts = int(iteration.get("max_window_attempts_per_candidate", 2))
    max_localization_retries = int(iteration.get("max_localization_retries", 1))
    if not 1 <= max_window_attempts <= 4:
        raise ConfigError("iteration.max_window_attempts_per_candidate 必须在 1 到 4 之间。")
    if not 0 <= max_localization_retries <= 3:
        raise ConfigError("iteration.max_localization_retries 必须在 0 到 3 之间。")
    output_top_k = int(data.get("ranking", {}).get("output_top_k", 3))
    if output_top_k <= 0:
        raise ConfigError("ranking.output_top_k 必须是正整数。")
    max_candidates = int(data.get("candidates", {}).get("max_candidates", 8))
    if max_candidates <= 0:
        raise ConfigError("candidates.max_candidates 必须是正整数。")
    if output_top_k > max_candidates:
        raise ConfigError("ranking.output_top_k 不能大于 candidates.max_candidates。")
    shell_minimum = float(data.get("candidates", {}).get("shell_family_min_weight", 0.08))
    if not 0.0 <= shell_minimum <= 1.0:
        raise ConfigError("candidates.shell_family_min_weight 必须在 [0, 1]。")
    combination_minimum = float(data.get("candidates", {}).get("shell_combination_min_weight", 0.45))
    if not 0.0 <= combination_minimum <= 1.0:
        raise ConfigError("candidates.shell_combination_min_weight 必须在 [0, 1]。")
    pyw90_dos = data.get("candidates", {}).get("pyw90_dos", {})
    if not isinstance(pyw90_dos, dict):
        raise ConfigError("candidates.pyw90_dos 必须是对象。")
    if bool(pyw90_dos.get("required", False)) and not bool(pyw90_dos.get("enabled", False)):
        raise ConfigError("candidates.pyw90_dos.required=true 时 enabled 也必须为 true。")
    lower_bound = float(pyw90_dos.get("lower_bound", 0.1))
    if not 0.0 < lower_bound <= 1.0:
        raise ConfigError("candidates.pyw90_dos.lower_bound 必须在 (0, 1]。")

    # Runtime allocation is operational only.  It may change how independent
    # candidates share a machine, but never candidate generation, energy
    # windows, SOC, or quality thresholds.
    search = data.setdefault("execution", {}).setdefault("candidate_search", {})
    policy = search.setdefault("policy", "serial")
    if policy not in {"serial", "candidate_parallel"}:
        raise ConfigError("execution.candidate_search.policy 只能是 serial 或 candidate_parallel。")
    workers = search.setdefault("max_workers", "auto")
    if workers != "auto" and (isinstance(workers, bool) or int(workers) <= 0):
        raise ConfigError("execution.candidate_search.max_workers 必须是 auto 或正整数。")
    mpi = search.setdefault("mpi_processes_per_candidate", data["backend"].get("mpi_processes"))
    if isinstance(mpi, bool) or int(mpi) <= 0:
        raise ConfigError("execution.candidate_search.mpi_processes_per_candidate 必须是正整数。")
    search["max_workers"] = workers if workers == "auto" else int(workers)
    search["mpi_processes_per_candidate"] = int(mpi)
    # A one-member common-basis ensemble is valid and reduces to an ordinary
    # single-structure workflow. Do not encode ferroelectric L1/L2 assumptions.

    data["_config_path"] = str(config_path)
    return data


def public_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only keys before hashing or saving a snapshot."""
    return {key: value for key, value in config.items() if not key.startswith("_")}
