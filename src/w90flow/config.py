from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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

    lock = data.get("pseudopotentials", {}).get("lock")
    if lock:
        lock_path = (config_path.parent / str(lock)).resolve()
        if not lock_path.is_file():
            raise ConfigError(f"找不到 POTCAR lock：{lock_path}")
        data["pseudopotentials"]["lock_resolved"] = str(lock_path)

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
    # A one-member common-basis ensemble is valid and reduces to an ordinary
    # single-structure workflow. Do not encode ferroelectric L1/L2 assumptions.

    data["_config_path"] = str(config_path)
    return data


def public_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only keys before hashing or saving a snapshot."""
    return {key: value for key, value in config.items() if not key.startswith("_")}
