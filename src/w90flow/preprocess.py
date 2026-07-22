from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .parsers import parse_eigenval, parse_poscar
from .state import atomic_json, read_json, sha256_file


PREPROCESS_STAGES = ("01_relax", "02_scf", "03_band")
STRESS_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "zx")
_ELECTRONIC_RE = re.compile(r"^\s*(DAV|RMM|CG)\s*:\s*(\d+)", re.MULTILINE)
_IONIC_RE = re.compile(r"^\s*(\d+)\s+F=", re.MULTILINE)
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


class PreprocessConfigError(ValueError):
    """The requested physical protocol is incomplete or internally unsafe."""


class StageBusyError(RuntimeError):
    """A live process already owns the immutable stage input."""


class StageRecoveryRequired(RuntimeError):
    """A stale/foreign lock requires an explicit recovery decision."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_number(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PreprocessConfigError(f"{path} 必须是正数。") from exc
    if not math.isfinite(number) or number <= 0:
        raise PreprocessConfigError(f"{path} 必须是正数。")
    return number


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise PreprocessConfigError(f"{path} 必须是正整数。")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PreprocessConfigError(f"{path} 必须是正整数。") from exc
    if number <= 0 or number != float(value):
        raise PreprocessConfigError(f"{path} 必须是正整数。")
    return number


def _normalise_soc(value: Any, relaxation_enabled: bool) -> dict[str, bool]:
    """Require SOC to be a scientific input, never an element-name heuristic."""
    if isinstance(value, bool):
        return {"relaxation": value, "scf": value, "band": value}
    if not isinstance(value, Mapping):
        raise PreprocessConfigError(
            "preprocess.soc 必须显式为 true/false，或包含 relaxation/scf/band 的布尔对象。"
        )
    unknown = sorted(set(value) - {"relaxation", "scf", "band"})
    if unknown:
        raise PreprocessConfigError(f"preprocess.soc 含未知阶段：{', '.join(unknown)}")
    non_boolean = sorted(key for key, item in value.items() if not isinstance(item, bool))
    if non_boolean:
        raise PreprocessConfigError(f"preprocess.soc 阶段值必须是布尔值：{', '.join(non_boolean)}")
    required = {"scf", "band"} | ({"relaxation"} if relaxation_enabled else set())
    missing = sorted(required - set(value))
    if missing:
        raise PreprocessConfigError(f"preprocess.soc 缺少显式阶段：{', '.join(missing)}")
    result = {
        "relaxation": value.get("relaxation", False),
        "scf": value.get("scf"),
        "band": value.get("band"),
    }
    if any(not isinstance(item, bool) for item in result.values()):
        raise PreprocessConfigError("preprocess.soc 的每个阶段值都必须是布尔值。")
    return result


def standard_semilocal_preprocess(
    *,
    soc: bool,
    dimensionality: int,
    periodic_axes: list[bool],
    relaxation_enabled: bool = True,
    relaxation_soc: bool = False,
    band_path_provider: str = "ase",
) -> dict[str, Any]:
    """Build the named nonmagnetic semilocal default from SOC + geometry.

    This profile makes the common path concise without pretending that U,
    magnetism, hybrid functionals, vdW choices, or POTCAR variants can be
    inferred from a POSCAR.  Those method choices remain frozen templates.
    """
    if not isinstance(soc, bool) or not isinstance(relaxation_soc, bool):
        raise PreprocessConfigError("soc 与 relaxation_soc 必须是布尔值。")
    if len(periodic_axes) != 3 or not all(isinstance(value, bool) for value in periodic_axes):
        raise PreprocessConfigError("periodic_axes 必须是三个布尔值。")
    if sum(periodic_axes) != dimensionality:
        raise PreprocessConfigError("periodic_axes 与 dimensionality 不一致。")
    vacuum_axes = [axis for axis, periodic in enumerate(periodic_axes) if not periodic]
    if dimensionality == 2 and len(vacuum_axes) != 1:
        raise PreprocessConfigError("标准 2D profile 要求恰好一个真空轴。")
    document = {
        "preprocess": {
            "profile": "standard_nonmagnetic_semilocal_v1",
            "soc": {
                "relaxation": relaxation_soc if relaxation_enabled else False,
                "scf": soc,
                "band": soc,
            },
            "electronic": {"ediff_eV": 1.0e-6, "max_steps": 120},
            "relaxation": {
                "enabled": relaxation_enabled,
                "cell_mode": "ions_only",
                "force_threshold_eV_A": 0.01,
                "max_steps": 200,
                **({"vacuum_axis": vacuum_axes[0]} if dimensionality == 2 else {}),
            },
            "band": {"path_provider": band_path_provider},
        }
    }
    normalized = normalize_preprocess_config(document, dimensionality)
    normalized["profile"] = "standard_nonmagnetic_semilocal_v1"
    normalized["profile_limitations"] = [
        "nonmagnetic_only",
        "no_automatic_hubbard_u",
        "no_automatic_hybrid_functional",
        "no_automatic_vdw_choice",
        "potcar_variant_must_be_locked_separately",
    ]
    normalized["protocol_sha256"] = _canonical_hash({
        key: value for key, value in normalized.items() if key != "protocol_sha256"
    })
    return normalized


def normalize_preprocess_config(document: Mapping[str, Any], dimensionality: int) -> dict[str, Any]:
    """Validate and freeze the deterministic POSCAR -> relaxation -> SCF -> Band policy.

    No convergence threshold has a hidden default.  A changed threshold therefore
    creates a different protocol hash rather than silently becoming a retry.
    """
    if dimensionality not in {2, 3}:
        raise PreprocessConfigError("前处理当前只支持显式的 2D 或 3D。")
    raw = document.get("preprocess", document)
    if not isinstance(raw, Mapping):
        raise PreprocessConfigError("preprocess 必须是对象。")
    relaxation = raw.get("relaxation")
    if not isinstance(relaxation, Mapping) or not isinstance(relaxation.get("enabled"), bool):
        raise PreprocessConfigError("preprocess.relaxation.enabled 必须显式为 true 或 false。")
    relaxation_enabled = bool(relaxation["enabled"])
    soc = _normalise_soc(raw.get("soc"), relaxation_enabled)

    electronic = raw.get("electronic")
    if not isinstance(electronic, Mapping):
        raise PreprocessConfigError("preprocess.electronic 必须显式给出。")
    electronic_frozen = {
        "ediff_eV": _positive_number(electronic.get("ediff_eV"), "preprocess.electronic.ediff_eV"),
        "max_steps": _positive_integer(electronic.get("max_steps"), "preprocess.electronic.max_steps"),
    }

    band = raw.get("band")
    if not isinstance(band, Mapping):
        raise PreprocessConfigError("preprocess.band 必须显式给出 band 路径来源。")
    path_provider = band.get("path_provider")
    if path_provider not in {"template", "seekpath", "ase", "vaspkit"}:
        raise PreprocessConfigError(
            "preprocess.band.path_provider 只能是 template、seekpath、ase 或 vaspkit。"
        )

    frozen_relaxation: dict[str, Any] = {"enabled": relaxation_enabled}
    if relaxation_enabled:
        cell_mode = relaxation.get("cell_mode")
        if cell_mode not in {"ions_only", "full_cell", "constrained_cell"}:
            raise PreprocessConfigError(
                "结构优化必须显式选择 cell_mode: ions_only/full_cell/constrained_cell。"
            )
        force_threshold = _positive_number(
            relaxation.get("force_threshold_eV_A"),
            "preprocess.relaxation.force_threshold_eV_A",
        )
        max_steps = _positive_integer(
            relaxation.get("max_steps"), "preprocess.relaxation.max_steps"
        )
        locked_axes = relaxation.get("locked_lattice_axes", [])
        if not isinstance(locked_axes, list) or any(axis not in {0, 1, 2} for axis in locked_axes):
            raise PreprocessConfigError("locked_lattice_axes 必须是 0/1/2 的列表。")
        locked_axes = sorted(set(int(axis) for axis in locked_axes))
        vacuum_axis = relaxation.get("vacuum_axis")
        if vacuum_axis is not None and vacuum_axis not in {0, 1, 2}:
            raise PreprocessConfigError("vacuum_axis 必须是 0、1 或 2。")

        if dimensionality == 2:
            if vacuum_axis is None:
                raise PreprocessConfigError("2D 优化必须显式给出 vacuum_axis，不能猜测真空方向。")
            if cell_mode == "full_cell" and relaxation.get("allow_vacuum_axis_relaxation") is not True:
                raise PreprocessConfigError(
                    "2D full_cell 会改变真空轴；必须显式设置 allow_vacuum_axis_relaxation: true。"
                )
            if cell_mode == "constrained_cell" and vacuum_axis not in locked_axes:
                raise PreprocessConfigError("2D constrained_cell 必须把 vacuum_axis 放入 locked_lattice_axes。")

        stress_threshold: float | None = None
        stress_components: list[str] = []
        if cell_mode != "ions_only":
            stress_threshold = _positive_number(
                relaxation.get("stress_threshold_kbar"),
                "preprocess.relaxation.stress_threshold_kbar",
            )
            components = relaxation.get("stress_components")
            if not isinstance(components, list) or not components:
                raise PreprocessConfigError("改变晶格时必须显式给出 stress_components。")
            unknown = sorted(set(components) - set(STRESS_COMPONENTS))
            if unknown:
                raise PreprocessConfigError(f"未知应力分量：{', '.join(unknown)}")
            stress_components = list(dict.fromkeys(str(item) for item in components))

        frozen_relaxation = {
            "enabled": True,
            "cell_mode": cell_mode,
            "force_threshold_eV_A": force_threshold,
            "stress_threshold_kbar": stress_threshold,
            "stress_components": stress_components,
            "max_steps": max_steps,
            "vacuum_axis": vacuum_axis,
            "locked_lattice_axes": locked_axes,
            "allow_vacuum_axis_relaxation": relaxation.get("allow_vacuum_axis_relaxation") is True,
            "custom_isif": relaxation.get("custom_isif"),
        }
        if cell_mode == "constrained_cell":
            isif = relaxation.get("custom_isif")
            if not isinstance(isif, int) or isinstance(isif, bool) or not 0 <= isif <= 8:
                raise PreprocessConfigError(
                    "constrained_cell 必须显式给出 custom_isif；约束的具体实现仍由输入模板负责。"
                )

    normalized = {
        "schema_version": 1,
        "dimensionality": dimensionality,
        "soc": soc,
        "relaxation": frozen_relaxation,
        "electronic": electronic_frozen,
        "band": {"path_provider": path_provider},
    }
    normalized["protocol_sha256"] = _canonical_hash(normalized)
    return normalized


def stage_incar_contract(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Return mandatory tags; callers may add method choices but may not weaken these."""
    if stage not in PREPROCESS_STAGES:
        raise ValueError(f"未知前处理阶段：{stage}")
    soc_key = {"01_relax": "relaxation", "02_scf": "scf", "03_band": "band"}[stage]
    base: dict[str, Any] = {
        "LSORBIT": bool(config["soc"][soc_key]),
        "EDIFF": float(config["electronic"]["ediff_eV"]),
        "NELM": int(config["electronic"]["max_steps"]),
    }
    if stage == "01_relax":
        relaxation = config["relaxation"]
        if not relaxation["enabled"]:
            raise ValueError("relaxation 已禁用。")
        cell_mode = relaxation["cell_mode"]
        if cell_mode == "ions_only":
            isif = 2
        elif cell_mode == "full_cell":
            isif = 3
        else:
            isif = int(relaxation["custom_isif"])
        base.update({
            "NSW": int(relaxation["max_steps"]),
            "EDIFFG": -float(relaxation["force_threshold_eV_A"]),
            "ISIF": isif,
        })
    elif stage == "02_scf":
        base.update({"NSW": 0, "IBRION": -1, "LCHARG": True})
    else:
        base.update({"NSW": 0, "IBRION": -1, "ICHARG": 11, "LCHARG": False})
    return base


def build_preprocess_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    if config["relaxation"]["enabled"]:
        stages.append({
            "stage": "01_relax",
            "poscar_source": "00_input/POSCAR",
            "handoff": {"CONTCAR": "02_scf/POSCAR"},
            "incar_contract": stage_incar_contract(config, "01_relax"),
        })
        scf_poscar = "01_relax/CONTCAR"
    else:
        scf_poscar = "00_input/POSCAR"
    stages.extend([
        {
            "stage": "02_scf",
            "poscar_source": scf_poscar,
            "handoff": {"POSCAR": "03_band/POSCAR", "CHGCAR": "03_band/CHGCAR"},
            "incar_contract": stage_incar_contract(config, "02_scf"),
        },
        {
            "stage": "03_band",
            "poscar_source": "02_scf/POSCAR",
            "charge_source": "02_scf/CHGCAR",
            "band_path_provider": config["band"]["path_provider"],
            "handoff": {
                "EIGENVAL": "wannier-input/EIGENVAL",
                "PROCAR": "wannier-input/PROCAR",
                "vasprun.xml": "wannier-input/vasprun.xml",
            },
            "incar_contract": stage_incar_contract(config, "03_band"),
        },
    ])
    return {
        "schema_version": 1,
        "protocol_sha256": config["protocol_sha256"],
        "stages": stages,
        "scientific_choices": {
            "soc": config["soc"],
            "relaxation_cell_mode": config["relaxation"].get("cell_mode"),
            "band_path_provider": config["band"]["path_provider"],
        },
    }


def parse_oszicar(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    electronic_cycles: list[int] = []
    pending_electronic = 0
    for line in text.splitlines():
        electronic = re.match(r"^\s*(?:DAV|RMM|CG)\s*:\s*(\d+)", line)
        if electronic:
            pending_electronic = max(pending_electronic, int(electronic.group(1)))
            continue
        if re.match(r"^\s*\d+\s+F=", line):
            electronic_cycles.append(pending_electronic)
            pending_electronic = 0
    if pending_electronic:
        electronic_cycles.append(pending_electronic)
    return {
        "ionic_steps": len(_IONIC_RE.findall(text)),
        "electronic_steps_by_cycle": electronic_cycles,
        "last_electronic_steps": electronic_cycles[-1] if electronic_cycles else None,
        "has_energy_record": bool(_IONIC_RE.search(text)),
    }


def parse_procar_dimensions(path: str | Path) -> dict[str, int]:
    """Read the PROCAR contract without loading a potentially huge file."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        header = "".join(handle.readline() for _ in range(12))
    match = re.search(
        r"# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)",
        header,
    )
    if not match:
        raise ValueError("PROCAR 缺少 k 点/能带/离子数头信息。")
    nkpts, nbands, nions = map(int, match.groups())
    if min(nkpts, nbands, nions) <= 0:
        raise ValueError("PROCAR 头信息含非正维数。")
    return {"nkpts": nkpts, "nbands": nbands, "nions": nions}


def vasprun_xml_complete(path: str | Path) -> bool:
    """Check the closing root tag from the tail instead of parsing the full XML."""
    source = Path(path)
    with source.open("rb") as handle:
        handle.seek(max(0, source.stat().st_size - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
    return bool(re.search(r"</modeling>\s*$|<modeling\s*/>\s*$", tail))


def _last_force_block(text: str) -> list[list[float]]:
    blocks = re.findall(
        r"POSITION\s+TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\s*\n(.*?)(?=\n\s*-+)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not blocks:
        return []
    forces: list[list[float]] = []
    for line in blocks[-1].splitlines():
        values = line.split()
        if len(values) < 6:
            continue
        try:
            forces.append([float(value.replace("D", "E")) for value in values[3:6]])
        except ValueError:
            continue
    return forces


def parse_outcar_convergence(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    fatal_markers = [
        marker for marker in ("VERY BAD NEWS", "internal error in subroutine", "ERROR:")
        if marker.lower() in text.lower()
    ]
    forces = _last_force_block(text)
    force_norms = [math.sqrt(sum(value * value for value in row)) for row in forces]
    stress_matches = re.findall(
        rf"^\s*in kB\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    stress = None
    if stress_matches:
        stress = {
            name: float(value.replace("D", "E"))
            for name, value in zip(STRESS_COMPONENTS, stress_matches[-1])
        }
    def final_number(name: str) -> float | None:
        matches = re.findall(rf"\b{name}\s*=\s*({_FLOAT})", text, flags=re.IGNORECASE)
        return float(matches[-1].replace("D", "E")) if matches else None

    def final_boolean(name: str) -> bool | None:
        matches = re.findall(rf"\b{name}\s*=\s*([TF])\b", text, flags=re.IGNORECASE)
        return matches[-1].upper() == "T" if matches else None

    return {
        "fatal_markers": fatal_markers,
        "normal_termination": "General timing and accounting informations" in text,
        "electronic_convergence_markers": len(re.findall(
            r"aborting loop because EDIFF is reached", text, flags=re.IGNORECASE
        )),
        "ionic_converged_marker": bool(re.search(
            r"reached required accuracy\s*-\s*stopping structural energy minimisation",
            text,
            flags=re.IGNORECASE,
        )),
        "max_force_eV_A": max(force_norms) if force_norms else None,
        "force_count": len(forces),
        "stress_kbar": stress,
        "incar_observed": {
            "EDIFF": final_number("EDIFF"),
            "EDIFFG": final_number("EDIFFG"),
            "NELM": final_number("NELM"),
            "NSW": final_number("NSW"),
            "ISIF": final_number("ISIF"),
            "ICHARG": final_number("ICHARG"),
            "LSORBIT": final_boolean("LSORBIT"),
        },
    }


def _same_float(actual: float | None, expected: float) -> bool:
    return actual is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def _lattice_axis_deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[float]:
    return [
        math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(old, new)))
        for old, new in zip(left["lattice_angstrom"], right["lattice_angstrom"])
    ]


def analyze_preprocess_outputs(
    directory: str | Path,
    stage: str,
    config: Mapping[str, Any],
    *,
    source_poscar: str | Path | None = None,
) -> dict[str, Any]:
    """Strict postcheck.  It reports evidence; it never changes a threshold."""
    if stage not in PREPROCESS_STAGES:
        raise ValueError(f"未知前处理阶段：{stage}")
    root = Path(directory)
    required = {
        "01_relax": ("OUTCAR", "OSZICAR", "CONTCAR"),
        "02_scf": ("OUTCAR", "OSZICAR", "CHGCAR"),
        "03_band": ("OUTCAR", "OSZICAR", "EIGENVAL", "PROCAR", "vasprun.xml"),
    }[stage]
    missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
    issues = [f"缺少或为空：{name}" for name in missing]
    evidence: dict[str, Any] = {"missing": missing}
    if (root / "OSZICAR").is_file():
        evidence["oszicar"] = parse_oszicar(root / "OSZICAR")
    if (root / "OUTCAR").is_file():
        evidence["outcar"] = parse_outcar_convergence(root / "OUTCAR")
    outcar = evidence.get("outcar", {})
    oszicar = evidence.get("oszicar", {})
    if outcar.get("fatal_markers"):
        issues.append("OUTCAR 含 fatal marker")
    if not outcar.get("normal_termination"):
        issues.append("OUTCAR 没有正常结束标记")

    contract = stage_incar_contract(config, stage)
    observed = outcar.get("incar_observed", {})
    if observed.get("LSORBIT") is not contract["LSORBIT"]:
        issues.append("OUTCAR 中 LSORBIT 与冻结 SOC 协议不一致")
    for key in ("EDIFF", "NELM"):
        if not _same_float(observed.get(key), float(contract[key])):
            issues.append(f"OUTCAR 中 {key} 与冻结协议不一致")

    if stage in {"01_relax", "02_scf"}:
        required_cycles = max(int(oszicar.get("ionic_steps", 0)), 1)
        if int(outcar.get("electronic_convergence_markers", 0)) < required_cycles:
            issues.append("至少一个电子循环未达到 EDIFF")
        last_steps = oszicar.get("last_electronic_steps")
        if last_steps is None or int(last_steps) > int(config["electronic"]["max_steps"]):
            issues.append("电子步数缺失或超过 NELM")

    if stage == "01_relax":
        relaxation = config["relaxation"]
        for key in ("EDIFFG", "NSW", "ISIF"):
            if not _same_float(observed.get(key), float(contract[key])):
                issues.append(f"OUTCAR 中 {key} 与冻结协议不一致")
        if int(oszicar.get("ionic_steps", 0)) > int(relaxation["max_steps"]):
            issues.append("离子步数超过冻结 NSW")
        if not outcar.get("ionic_converged_marker"):
            issues.append("没有离子优化收敛标记")
        maximum_force = outcar.get("max_force_eV_A")
        if maximum_force is None or float(maximum_force) > float(relaxation["force_threshold_eV_A"]):
            issues.append("最大原子力未达到门槛")
        if relaxation["cell_mode"] != "ions_only":
            stress = outcar.get("stress_kbar")
            if not stress:
                issues.append("改变晶格但没有可解析的最终应力")
            else:
                selected = {name: abs(float(stress[name])) for name in relaxation["stress_components"]}
                evidence["selected_stress_abs_kbar"] = selected
                if max(selected.values()) > float(relaxation["stress_threshold_kbar"]):
                    issues.append("最终应力未达到门槛")
        if (root / "CONTCAR").is_file():
            try:
                final_structure = parse_poscar(root / "CONTCAR")
                evidence["contcar_sha256"] = sha256_file(root / "CONTCAR")
                if source_poscar is not None:
                    initial = parse_poscar(source_poscar)
                    if (initial["elements"], initial["counts"]) != (
                        final_structure["elements"], final_structure["counts"]
                    ):
                        issues.append("CONTCAR 改变了元素或原子数")
                    deltas = _lattice_axis_deltas(initial, final_structure)
                    evidence["lattice_axis_delta_A"] = deltas
                    locked = set(relaxation["locked_lattice_axes"])
                    if relaxation["cell_mode"] == "ions_only":
                        locked = {0, 1, 2}
                    for axis in sorted(locked):
                        if deltas[axis] > 1e-8:
                            issues.append(f"锁定的晶格轴 {axis} 被改变")
            except ValueError as exc:
                issues.append(f"CONTCAR 无效：{exc}")
    elif stage == "02_scf":
        for key in ("NSW",):
            if not _same_float(observed.get(key), float(contract[key])):
                issues.append(f"OUTCAR 中 {key} 与冻结协议不一致")
    else:
        for key in ("NSW", "ICHARG"):
            if not _same_float(observed.get(key), float(contract[key])):
                issues.append(f"OUTCAR 中 {key} 与冻结协议不一致")
        if (root / "EIGENVAL").is_file():
            try:
                eig = parse_eigenval(root / "EIGENVAL")
                evidence["eigenval"] = {key: eig[key] for key in ("nelect", "nkpts", "nbands")}
            except ValueError as exc:
                issues.append(f"EIGENVAL 不完整：{exc}")
        if (root / "PROCAR").is_file():
            try:
                procar = parse_procar_dimensions(root / "PROCAR")
                evidence["procar"] = procar
                eig_summary = evidence.get("eigenval")
                if eig_summary and (
                    procar["nkpts"] != int(eig_summary["nkpts"])
                    or procar["nbands"] != int(eig_summary["nbands"])
                ):
                    issues.append("PROCAR 与 EIGENVAL 的 k 点数或能带数不一致")
                if source_poscar is not None:
                    atom_count = sum(parse_poscar(source_poscar)["counts"])
                    if procar["nions"] != atom_count:
                        issues.append("PROCAR 离子数与 POSCAR 不一致")
            except ValueError as exc:
                issues.append(f"PROCAR 不完整：{exc}")
        if (root / "vasprun.xml").is_file() and not vasprun_xml_complete(root / "vasprun.xml"):
            issues.append("vasprun.xml 缺少完整 modeling 根节点结束标记")

    output_hashes = {
        name: sha256_file(root / name)
        for name in required if (root / name).is_file() and (root / name).stat().st_size > 0
    }
    return {
        "schema_version": 1,
        "stage": stage,
        "passed": not issues,
        "issues": issues,
        "protocol_sha256": config["protocol_sha256"],
        "evidence": evidence,
        "output_sha256": output_hashes,
    }


def stage_input_hash(protocol_sha256: str, inputs: Mapping[str, str | Path]) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for role, path_value in sorted(inputs.items()):
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"阶段输入不存在：{role}={path}")
        hashes[role] = sha256_file(path)
    return _canonical_hash({"protocol_sha256": protocol_sha256, "inputs": hashes}), hashes


def _copy_exact(source: str | Path, target: str | Path) -> str:
    old = Path(source).expanduser().resolve()
    new = Path(target)
    if not old.is_file():
        raise FileNotFoundError(old)
    expected = sha256_file(old)
    if new.exists():
        if not new.is_file() or sha256_file(new) != expected:
            raise FileExistsError(f"拒绝覆盖不同内容：{new}")
        return "existing"
    new.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(old, new)
        return "hardlink"
    except OSError:
        shutil.copy2(old, new)
        return "copy"


def _write_exact_bytes(target: str | Path, payload: bytes) -> str:
    destination = Path(target)
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise FileExistsError(f"拒绝覆盖不同内容：{destination}")
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return "write"


def _format_incar_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def patch_incar_contract(template_text: str, contract: Mapping[str, Any]) -> str:
    """Materialize mandatory tags while rejecting ambiguous duplicate assignments."""
    lines = template_text.splitlines()
    for key, value in contract.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=", flags=re.IGNORECASE)
        matches = [index for index, line in enumerate(lines) if pattern.match(line)]
        if len(matches) > 1:
            raise PreprocessConfigError(f"INCAR 模板中 {key} 重复；拒绝猜测生效值。")
        rendered = f"{key} = {_format_incar_value(value)}"
        if matches:
            lines[matches[0]] = rendered
        else:
            lines.append(rendered)
    return "\n".join(lines).rstrip() + "\n"


def validate_incar_structure_compatibility(incar_text: str, atom_count: int) -> None:
    """Reject deterministic INCAR/POSCAR cardinality errors before VASP.

    VASP treats spin-orbit calculations as noncollinear.  If MAGMOM is
    explicitly supplied, it therefore needs three components per ion.  This is
    a syntax/shape check only; it never invents a magnetic configuration.
    """
    if atom_count <= 0:
        raise PreprocessConfigError("POSCAR 原子数必须大于零。")

    def assignment(key: str) -> str | None:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^!#]+)", re.IGNORECASE)
        values = [match.group(1).strip() for line in incar_text.splitlines() if (match := pattern.match(line))]
        if len(values) > 1:
            raise PreprocessConfigError(f"INCAR 中 {key} 重复；拒绝猜测生效值。")
        return values[0] if values else None

    def enabled(value: str | None) -> bool:
        return value is not None and value.strip().upper() in {"T", ".T.", "TRUE", ".TRUE."}

    if not (enabled(assignment("LSORBIT")) or enabled(assignment("LNONCOLLINEAR"))):
        return
    magmom = assignment("MAGMOM")
    if magmom is None:
        return
    count = 0
    for token in magmom.replace(",", " ").split():
        repeated = re.fullmatch(r"(\d+)\*[-+0-9.eEdD]+", token)
        scalar = re.fullmatch(r"[-+0-9.eEdD]+", token)
        if repeated:
            count += int(repeated.group(1))
        elif scalar:
            count += 1
        else:
            raise PreprocessConfigError(f"无法解析 MAGMOM token：{token}")
    expected = 3 * atom_count
    if count != expected:
        raise PreprocessConfigError(
            f"SOC/noncollinear MAGMOM 必须有 3*NIONS={expected} 个分量；当前为 {count}。"
        )


def _validate_command(command: Any, stage: str, stage_dir: Path) -> list[str]:
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
        raise PreprocessConfigError(f"commands.{stage} 必须是非空字符串列表；不接受 shell 字符串。")
    replacements = {"{stage_dir}": str(stage_dir), "{run_dir}": str(stage_dir.parent)}
    rendered: list[str] = []
    for item in command:
        value = item
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        rendered.append(value)
    return rendered


def materialize_potcar_recipe(
    specification: Mapping[str, Any],
    elements: list[str],
    destination: str | Path,
) -> dict[str, Any]:
    """Resolve an explicitly locked POTCAR path or ordered component recipe.

    The manifest stores hashes and paths but never embeds POTCAR bytes.  Every
    component needs an expected SHA256; there is no implicit local lookup.
    """
    if not isinstance(specification, Mapping):
        raise PreprocessConfigError("potcar 必须显式给出锁定 path 或 components recipe。")
    target = Path(destination)
    mode: str
    sources: list[dict[str, Any]] = []
    payload: bytes
    if "path" in specification:
        if "components" in specification:
            raise PreprocessConfigError("potcar.path 与 potcar.components 不能同时使用。")
        order = specification.get("element_order")
        if order != elements:
            raise PreprocessConfigError("potcar.element_order 必须与 POSCAR 元素顺序完全一致。")
        source = Path(str(specification["path"])).expanduser().resolve()
        expected = specification.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise PreprocessConfigError("potcar.path 模式必须显式给出 64 位 sha256。")
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256_file(source)
        if actual != expected:
            raise PreprocessConfigError("锁定 POTCAR 的 SHA256 不匹配。")
        payload = source.read_bytes()
        sources.append({"element_order": list(order), "path": str(source), "sha256": actual})
        mode = "locked_path"
    else:
        components = specification.get("components")
        if not isinstance(components, list) or not components:
            raise PreprocessConfigError("potcar 必须显式给出 path 或非空 components recipe。")
        order = [item.get("element") for item in components if isinstance(item, Mapping)]
        if len(order) != len(components) or order != elements:
            raise PreprocessConfigError("POTCAR recipe 组件顺序必须与 POSCAR 元素顺序完全一致。")
        chunks: list[bytes] = []
        for item in components:
            source = Path(str(item.get("path", ""))).expanduser().resolve()
            expected = item.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise PreprocessConfigError(f"POTCAR 组件 {item.get('element')} 缺少 64 位 sha256。")
            if not source.is_file():
                raise FileNotFoundError(source)
            actual = sha256_file(source)
            if actual != expected:
                raise PreprocessConfigError(f"POTCAR 组件 {item.get('element')} SHA256 不匹配。")
            chunks.append(source.read_bytes())
            sources.append({"element": item["element"], "path": str(source), "sha256": actual})
        payload = b"".join(chunks)
        expected_combined = specification.get("combined_sha256")
        actual_combined = hashlib.sha256(payload).hexdigest()
        if not isinstance(expected_combined, str) or actual_combined != expected_combined:
            raise PreprocessConfigError("组合 POTCAR 的 combined_sha256 缺失或不匹配。")
        mode = "ordered_component_recipe"
    combined_sha256 = hashlib.sha256(payload).hexdigest()
    _write_exact_bytes(target, payload)
    return {
        "mode": mode,
        "element_order": elements,
        "combined_sha256": combined_sha256,
        "destination": str(target),
        "sources": sources,
    }


def _stage_input_paths(run_dir: Path, stage: str) -> dict[str, Path]:
    root = run_dir / stage
    names = ["POSCAR", "POTCAR", "INCAR", "KPOINTS"]
    if stage == "03_band":
        names.append("CHGCAR")
    return {name: root / name for name in names}


def _missing_stage_inputs(run_dir: Path, stage: str) -> list[str]:
    return [name for name, path in _stage_input_paths(run_dir, stage).items() if not path.is_file()]


def verify_materialized_stage_inputs(run_dir: str | Path, stage: str) -> dict[str, str]:
    root = Path(run_dir)
    state = read_json(root / stage / "stage.json")
    expected = state.get("prepared_input_sha256")
    if not isinstance(expected, Mapping):
        raise StageRecoveryRequired(f"{stage} 没有冻结的 prepared_input_sha256。")
    paths = _stage_input_paths(root, stage)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise StageRecoveryRequired(f"{stage} 缺少已物化输入：{', '.join(missing)}")
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if set(expected) != set(actual):
        raise StageRecoveryRequired(f"{stage} 冻结输入角色与当前文件不一致。")
    changed = [name for name in actual if actual[name] != expected[name]]
    if changed:
        raise StageRecoveryRequired(f"{stage} 输入哈希已改变：{', '.join(changed)}")
    return actual


def materialize_preprocess_inputs(
    run_dir: str | Path,
    poscar: str | Path,
    config: Mapping[str, Any],
    *,
    templates: Mapping[str, Mapping[str, str | Path]],
    potcar: Mapping[str, Any],
    commands: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Prepare immutable standard inputs for all active preprocessing stages."""
    root = Path(run_dir).expanduser().resolve()
    protocol = read_json(root / "protocol.json")
    if protocol.get("protocol_sha256") != config.get("protocol_sha256"):
        raise StageRecoveryRequired("待物化配置与 run 内冻结协议不一致。")
    source_poscar = Path(poscar).expanduser().resolve()
    structure = parse_poscar(source_poscar)
    input_poscar = root / "00_input/POSCAR"
    _copy_exact(source_poscar, input_poscar)
    active = ["02_scf", "03_band"]
    if config["relaxation"]["enabled"]:
        active.insert(0, "01_relax")
    unknown = sorted(set(templates) - set(PREPROCESS_STAGES))
    if unknown:
        raise PreprocessConfigError(f"templates 含未知阶段：{', '.join(unknown)}")
    for stage in active:
        template = templates.get(stage)
        if not isinstance(template, Mapping) or set(template) != {"incar", "kpoints"}:
            raise PreprocessConfigError(f"templates.{stage} 必须恰好给出 incar 与 kpoints。")
        if stage not in commands:
            raise PreprocessConfigError(f"commands 缺少 {stage}。")

    private_potcar = root / "private/POTCAR.locked"
    potcar_manifest = materialize_potcar_recipe(potcar, structure["elements"], private_potcar)
    materialized: list[dict[str, Any]] = []
    for stage in active:
        directory = root / stage
        template = templates[stage]
        source_incar = Path(template["incar"]).expanduser().resolve()
        source_kpoints = Path(template["kpoints"]).expanduser().resolve()
        if not source_incar.is_file() or not source_kpoints.is_file():
            raise FileNotFoundError(f"{stage} 的 INCAR/KPOINTS 模板不存在。")
        patched_incar = patch_incar_contract(
            source_incar.read_text(encoding="utf-8", errors="strict"),
            stage_incar_contract(config, stage),
        )
        validate_incar_structure_compatibility(patched_incar, sum(structure["counts"]))
        _write_exact_bytes(directory / "INCAR", patched_incar.encode("utf-8"))
        kpoints_mode = _copy_exact(source_kpoints, directory / "KPOINTS")
        potcar_mode = _copy_exact(private_potcar, directory / "POTCAR")
        command = _validate_command(commands[stage], stage, directory)
        if stage == "01_relax" or (stage == "02_scf" and not config["relaxation"]["enabled"]):
            _copy_exact(input_poscar, directory / "POSCAR")
        missing = _missing_stage_inputs(root, stage)
        state_path = directory / "stage.json"
        state = read_json(state_path)
        if state["status"] not in {"PENDING", "READY", "WAITING_HANDOFF"}:
            raise StageRecoveryRequired(f"{stage} 已进入 {state['status']}，拒绝重新物化。")
        state.update({
            "status": "READY" if not missing else "WAITING_HANDOFF",
            "prepared_utc": _utc_now(),
            "command": command,
            "missing_inputs": missing,
            "template_sha256": {
                "INCAR": sha256_file(source_incar),
                "KPOINTS": sha256_file(source_kpoints),
            },
            "prepared_input_sha256": {
                name: sha256_file(path)
                for name, path in _stage_input_paths(root, stage).items() if path.is_file()
            },
        })
        atomic_json(state_path, state)
        materialized.append({
            "stage": stage,
            "directory": str(directory),
            "status": state["status"],
            "missing_inputs": missing,
            "command": command,
            "incar_sha256": sha256_file(directory / "INCAR"),
            "kpoints_materialization": kpoints_mode,
            "potcar_materialization": potcar_mode,
        })
    manifest = {
        "schema_version": 1,
        "prepared_utc": _utc_now(),
        "protocol_sha256": config["protocol_sha256"],
        "source_poscar": {"path": str(source_poscar), "sha256": sha256_file(source_poscar)},
        "potcar": potcar_manifest,
        "stages": materialized,
    }
    atomic_json(root / "prepare-manifest.json", manifest)
    input_lock_path = root / "input-lock.json"
    input_lock = read_json(input_lock_path)
    input_lock["materialized_files"] = {
        "POSCAR": {"path": str(input_poscar), "sha256": sha256_file(input_poscar)},
        "POTCAR": {
            "path": str(private_potcar),
            "sha256": sha256_file(private_potcar),
            "recipe_mode": potcar_manifest["mode"],
        },
        **{
            f"{item['stage']}:{role}": {
                "path": str(Path(templates[item["stage"]][role.lower()]).expanduser().resolve()),
                "sha256": item["template_sha256"][role],
            }
            for item in (read_json(root / stage / "stage.json") for stage in active)
            for role in ("INCAR", "KPOINTS")
        },
    }
    atomic_json(input_lock_path, input_lock)
    return manifest


def prepare_preprocess_run(
    run_dir: str | Path,
    poscar: str | Path,
    config: Mapping[str, Any],
    *,
    templates: Mapping[str, Mapping[str, str | Path]],
    potcar: Mapping[str, Any],
    commands: Mapping[str, list[str]],
) -> Path:
    """Initialize and materialize a run without starting any subprocess."""
    root = initialize_preprocess_run(run_dir, poscar, config)
    materialize_preprocess_inputs(
        root,
        poscar,
        config,
        templates=templates,
        potcar=potcar,
        commands=commands,
    )
    return root


def initialize_preprocess_run(
    run_dir: str | Path,
    poscar: str | Path,
    config: Mapping[str, Any],
    immutable_inputs: Mapping[str, str | Path] | None = None,
) -> Path:
    """Create state only.  This function never launches VASP or writes scientific inputs."""
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    poscar_path = Path(poscar).expanduser().resolve()
    if not poscar_path.is_file():
        raise FileNotFoundError(poscar_path)
    locked_paths = {"POSCAR": poscar_path, **(immutable_inputs or {})}
    locked = {
        role: {"path": str(Path(path).expanduser().resolve()), "sha256": sha256_file(path)}
        for role, path in sorted(locked_paths.items())
    }
    atomic_json(root / "protocol.json", dict(config))
    atomic_json(root / "input-lock.json", {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "protocol_sha256": config["protocol_sha256"],
        "files": locked,
    })
    atomic_json(root / "plan.json", build_preprocess_plan(config))
    for stage in PREPROCESS_STAGES:
        stage_root = root / stage
        stage_root.mkdir(parents=True, exist_ok=False)
        status = "SKIPPED" if stage == "01_relax" and not config["relaxation"]["enabled"] else "PENDING"
        atomic_json(stage_root / "stage.json", {
            "schema_version": 1,
            "stage": stage,
            "status": status,
            "protocol_sha256": config["protocol_sha256"],
            "attempts": 0,
            "interruptions": 0,
            "input_sha256": {},
            "input_digest": None,
            "active_process": None,
        })
    return root


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_stage_lock(stage_dir: str | Path) -> dict[str, Any]:
    path = Path(stage_dir) / ".stage.lock"
    if not path.is_file():
        return {"status": "ABSENT", "path": str(path)}
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {"status": "INVALID", "path": str(path)}
    same_host = payload.get("hostname") == socket.gethostname()
    alive = bool(same_host and isinstance(payload.get("pid"), int) and _pid_alive(payload["pid"]))
    return {
        "status": "LIVE" if alive else ("STALE" if same_host else "FOREIGN_HOST"),
        "path": str(path),
        "owner": payload,
    }


@dataclass
class StageLease:
    stage_dir: Path
    input_digest: str
    fd: int | None = None

    @property
    def path(self) -> Path:
        return self.stage_dir / ".stage.lock"

    def acquire(self) -> "StageLease":
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_utc": _utc_now(),
            "input_digest": self.input_digest,
        }
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            inspection = inspect_stage_lock(self.stage_dir)
            if inspection["status"] == "LIVE":
                raise StageBusyError(f"阶段已有活动 PID：{inspection['owner'].get('pid')}") from exc
            raise StageRecoveryRequired(
                f"阶段锁状态为 {inspection['status']}；必须显式 recover_interrupted_stage。"
            ) from exc
        os.write(self.fd, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        os.fsync(self.fd)
        return self

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.path.is_file():
            owner = read_json(self.path)
            if owner.get("pid") == os.getpid() and owner.get("input_digest") == self.input_digest:
                self.path.unlink()

    def __enter__(self) -> "StageLease":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def begin_stage(
    run_dir: str | Path,
    stage: str,
    inputs: Mapping[str, str | Path],
    command: Iterable[str],
    *,
    worker_pid: int | None = None,
) -> StageLease:
    root = Path(run_dir)
    state_path = root / stage / "stage.json"
    state = read_json(state_path)
    protocol = read_json(root / "protocol.json")
    if state.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise StageRecoveryRequired("state 与冻结 protocol 哈希不一致。")
    digest, hashes = stage_input_hash(protocol["protocol_sha256"], inputs)
    previous = state.get("input_digest")
    if previous is not None and previous != digest:
        raise StageRecoveryRequired("阶段输入已改变；不能作为同一断点恢复。")
    if state.get("status") == "PASS":
        raise StageBusyError("阶段已经 PASS，不允许重复提交。")
    if state.get("status") == "FAIL":
        raise StageRecoveryRequired("科学/数值失败不能自动重试或放宽门槛。")
    predecessor = {
        "01_relax": None,
        "02_scf": "01_relax" if protocol["relaxation"]["enabled"] else None,
        "03_band": "02_scf",
    }.get(stage)
    if stage not in PREPROCESS_STAGES:
        raise ValueError(f"未知前处理阶段：{stage}")
    if predecessor is not None:
        previous_state = read_json(root / predecessor / "stage.json")
        if previous_state.get("status") != "PASS":
            raise StageRecoveryRequired(f"{stage} 的前一阶段 {predecessor} 尚未 PASS。")
    verify_materialized_stage_inputs(root, stage)
    lease = StageLease(root / stage, digest).acquire()
    try:
        state.update({
            "status": "RUNNING",
            "started_utc": _utc_now(),
            "attempts": int(state.get("attempts", 0)) + 1,
            "input_sha256": hashes,
            "input_digest": digest,
            "command": list(command),
            "active_process": {"pid": worker_pid, "launcher_pid": os.getpid()},
        })
        atomic_json(state_path, state)
    except Exception:
        lease.release()
        raise
    return lease


def finish_stage(
    run_dir: str | Path,
    stage: str,
    result: Mapping[str, Any],
    lease: StageLease,
) -> dict[str, Any]:
    root = Path(run_dir)
    state_path = root / stage / "stage.json"
    state = read_json(state_path)
    if state.get("input_digest") != lease.input_digest:
        raise StageRecoveryRequired("完成阶段时 input digest 已改变。")
    state.update({
        "status": "PASS" if result.get("passed") else "FAIL",
        "finished_utc": _utc_now(),
        "active_process": None,
        "postcheck": dict(result),
    })
    atomic_json(state_path, state)
    lease.release()
    return state


def recover_interrupted_stage(run_dir: str | Path, stage: str) -> dict[str, Any]:
    """Explicitly clear only a same-host dead PID lock; never changes inputs or thresholds."""
    root = Path(run_dir)
    inspection = inspect_stage_lock(root / stage)
    if inspection["status"] != "STALE":
        raise StageRecoveryRequired(f"只有 STALE 同主机锁可恢复；当前是 {inspection['status']}。")
    state_path = root / stage / "stage.json"
    state = read_json(state_path)
    owner = inspection["owner"]
    if state.get("input_digest") != owner.get("input_digest"):
        raise StageRecoveryRequired("锁与 stage.json 的 input digest 不一致。")
    Path(inspection["path"]).unlink()
    state.update({
        "status": "INTERRUPTED",
        "active_process": None,
        "interruptions": int(state.get("interruptions", 0)) + 1,
        "recovered_utc": _utc_now(),
        "recovered_dead_pid": owner.get("pid"),
    })
    atomic_json(state_path, state)
    return state


def _handoff_file(source: Path, target: Path) -> dict[str, Any]:
    source_hash = sha256_file(source)
    mode = _copy_exact(source, target)
    target_hash = sha256_file(target)
    if target_hash != source_hash:
        raise StageRecoveryRequired(f"交接哈希不一致：{source} -> {target}")
    return {
        "source": str(source),
        "target": str(target),
        "sha256": source_hash,
        "materialization": mode,
    }


def apply_preprocess_handoff(run_dir: str | Path, stage: str) -> dict[str, Any]:
    """Materialize only the documented, hash-checked handoff after a PASS postcheck."""
    root = Path(run_dir)
    records: dict[str, Any] = {}
    next_stage: str | None = None
    if stage == "01_relax":
        next_stage = "02_scf"
        records["CONTCAR_to_POSCAR"] = _handoff_file(
            root / stage / "CONTCAR", root / next_stage / "POSCAR"
        )
    elif stage == "02_scf":
        next_stage = "03_band"
        records["POSCAR"] = _handoff_file(root / stage / "POSCAR", root / next_stage / "POSCAR")
        records["CHGCAR"] = _handoff_file(root / stage / "CHGCAR", root / next_stage / "CHGCAR")
    elif stage == "03_band":
        for name in ("POSCAR", "OUTCAR", "EIGENVAL", "PROCAR", "vasprun.xml"):
            source = root / stage / name
            records[name] = {"path": str(source), "sha256": sha256_file(source)}
    else:
        raise ValueError(f"未知前处理阶段：{stage}")
    document = {
        "schema_version": 1,
        "stage": stage,
        "created_utc": _utc_now(),
        "files": records,
    }
    name = "band-handoff.json" if stage == "03_band" else "handoff.json"
    atomic_json(root / stage / name, document)
    if next_stage is not None:
        state_path = root / next_stage / "stage.json"
        state = read_json(state_path)
        missing = _missing_stage_inputs(root, next_stage)
        state.update({
            "status": "READY" if not missing else "WAITING_HANDOFF",
            "missing_inputs": missing,
            "prepared_input_sha256": {
                role: sha256_file(path)
                for role, path in _stage_input_paths(root, next_stage).items() if path.is_file()
            },
            "handoff_from": stage,
        })
        atomic_json(state_path, state)
    return document


def _mark_stage_interrupted(run_dir: Path, stage: str, reason: str, lease: StageLease) -> dict[str, Any]:
    state_path = run_dir / stage / "stage.json"
    state = read_json(state_path)
    state.update({
        "status": "INTERRUPTED",
        "active_process": None,
        "interrupted_utc": _utc_now(),
        "interruptions": int(state.get("interruptions", 0)) + 1,
        "interruption_reason": reason,
    })
    atomic_json(state_path, state)
    lease.release()
    return state


def _transfer_stage_lock_to_worker(lease: StageLease, worker_pid: int) -> None:
    """Leave a live child uniquely owning the lock when the synchronous waiter is interrupted."""
    owner = read_json(lease.path)
    if owner.get("pid") != os.getpid() or owner.get("input_digest") != lease.input_digest:
        raise StageRecoveryRequired("无法把阶段锁安全转交给活动 worker。")
    owner.update({"pid": worker_pid, "transferred_from_launcher_pid": os.getpid(), "transferred_utc": _utc_now()})
    atomic_json(lease.path, owner)
    if lease.fd is not None:
        os.close(lease.fd)
        lease.fd = None


def run_preprocess_stage(
    run_dir: str | Path,
    stage: str,
    *,
    execute: bool = False,
) -> dict[str, Any]:
    """Run one stage synchronously; dry-run is the default and read-only."""
    root = Path(run_dir).expanduser().resolve()
    if stage not in PREPROCESS_STAGES:
        raise ValueError(f"未知前处理阶段：{stage}")
    state = read_json(root / stage / "stage.json")
    command = state.get("command")
    if not isinstance(command, list) or not command:
        raise StageRecoveryRequired(f"{stage} 尚未物化命令。")
    inputs = _stage_input_paths(root, stage)
    missing = [name for name, path in inputs.items() if not path.is_file()]
    preview = {
        "mode": "EXECUTE" if execute else "DRY_RUN",
        "stage": stage,
        "directory": str(root / stage),
        "command": command,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "missing_inputs": missing,
    }
    if not missing:
        verify_materialized_stage_inputs(root, stage)
    if not execute:
        return {**preview, "passed": not missing}
    lease = begin_stage(root, stage, inputs, command)
    attempt = int(read_json(root / stage / "stage.json")["attempts"])
    log_path = root / stage / f"runner-attempt-{attempt:03d}.log"
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=root / stage,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running = read_json(root / stage / "stage.json")
            running["active_process"] = {
                "pid": process.pid,
                "launcher_pid": os.getpid(),
                "started_utc": _utc_now(),
            }
            running["log"] = str(log_path)
            atomic_json(root / stage / "stage.json", running)
            returncode = process.wait()
    except KeyboardInterrupt as exc:
        if process is not None and process.poll() is None:
            _transfer_stage_lock_to_worker(lease, process.pid)
            return {
                **preview,
                "passed": False,
                "status": "RUNNING_BACKGROUND",
                "worker_pid": process.pid,
                "note": "waiter 被中断，但活动 worker 保留唯一阶段锁；请等待后显式 recover/reconcile。",
            }
        _mark_stage_interrupted(root, stage, f"{type(exc).__name__}: {exc}", lease)
        return {**preview, "passed": False, "status": "INTERRUPTED", "error": str(exc)}
    except OSError as exc:
        _mark_stage_interrupted(root, stage, f"{type(exc).__name__}: {exc}", lease)
        return {**preview, "passed": False, "status": "INTERRUPTED", "error": str(exc)}

    result = analyze_preprocess_outputs(
        root / stage,
        stage,
        read_json(root / "protocol.json"),
        source_poscar=root / stage / "POSCAR",
    )
    result["returncode"] = returncode
    result["log"] = str(log_path)
    if returncode != 0:
        result["issues"].append(f"子进程退出码 {returncode}")
        result["passed"] = False
    if result["passed"]:
        result["handoff"] = apply_preprocess_handoff(root, stage)
    final_state = finish_stage(root, stage, result, lease)
    return {
        **preview,
        "passed": bool(result["passed"]),
        "status": final_state["status"],
        "postcheck": result,
    }


def reconcile_preprocess_stage(run_dir: str | Path, stage: str) -> dict[str, Any]:
    """Adopt externally completed outputs without submitting another process."""
    root = Path(run_dir).expanduser().resolve()
    if stage not in PREPROCESS_STAGES:
        raise ValueError(f"未知前处理阶段：{stage}")
    lock = inspect_stage_lock(root / stage)
    if lock["status"] in {"LIVE", "STALE", "FOREIGN_HOST", "INVALID"}:
        raise StageRecoveryRequired(
            f"阶段锁是 {lock['status']}；活动任务需等待，死锁需先显式恢复。"
        )
    state_path = root / stage / "stage.json"
    state = read_json(state_path)
    if state["status"] == "PASS":
        return {"stage": stage, "passed": True, "status": "PASS", "already_complete": True}
    if state["status"] == "FAIL":
        raise StageRecoveryRequired("FAIL 阶段不能通过 reconcile 绕过科学失败。")
    input_hashes = verify_materialized_stage_inputs(root, stage)
    digest, _ = stage_input_hash(read_json(root / "protocol.json")["protocol_sha256"], _stage_input_paths(root, stage))
    previous_digest = state.get("input_digest")
    if previous_digest is not None and previous_digest != digest:
        raise StageRecoveryRequired("reconcile 的阶段输入与原始提交哈希不一致。")
    result = analyze_preprocess_outputs(
        root / stage,
        stage,
        read_json(root / "protocol.json"),
        source_poscar=root / stage / "POSCAR",
    )
    if not result["passed"]:
        return {"stage": stage, "passed": False, "status": state["status"], "postcheck": result}
    result["handoff"] = apply_preprocess_handoff(root, stage)
    state.update({
        "status": "PASS",
        "reconciled_utc": _utc_now(),
        "active_process": None,
        "postcheck": result,
        "input_digest": digest,
        "input_sha256": input_hashes,
    })
    atomic_json(state_path, state)
    return {"stage": stage, "passed": True, "status": "PASS", "postcheck": result}


def run_preprocess_workflow(run_dir: str | Path, *, execute: bool = False) -> dict[str, Any]:
    """Advance the finite workflow to Band; dry-run only reports the next action."""
    root = Path(run_dir).expanduser().resolve()
    history: list[dict[str, Any]] = []
    while True:
        inspection = inspect_preprocess_run(root)
        action = inspection["next_action"]
        if action["action"] == "COMPLETE":
            return {
                "mode": "EXECUTE" if execute else "DRY_RUN",
                "passed": True,
                "status": "COMPLETE",
                "band_handoff": str(root / "03_band/band-handoff.json"),
                "history": history,
            }
        if action["action"] != "RUN_SAME_FROZEN_PROTOCOL":
            return {
                "mode": "EXECUTE" if execute else "DRY_RUN",
                "passed": False,
                "status": action["action"],
                "next_action": action,
                "history": history,
            }
        result = run_preprocess_stage(root, action["stage"], execute=execute)
        history.append(result)
        if not execute or not result.get("passed"):
            return {
                "mode": "EXECUTE" if execute else "DRY_RUN",
                "passed": bool(result.get("passed")),
                "status": "PLANNED" if not execute else result.get("status"),
                "next_action": action,
                "history": history,
            }


def inspect_preprocess_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    protocol = read_json(root / "protocol.json")
    states = {stage: read_json(root / stage / "stage.json") for stage in PREPROCESS_STAGES}
    order = ["02_scf", "03_band"]
    if protocol["relaxation"]["enabled"]:
        order.insert(0, "01_relax")
    next_action: dict[str, Any] = {"action": "COMPLETE"}
    for stage in order:
        state = states[stage]
        status = state["status"]
        if status == "PASS":
            continue
        lock = inspect_stage_lock(root / stage)
        if status == "RUNNING":
            next_action = {
                "action": "WAIT" if lock["status"] == "LIVE" else "RECOVER_REQUIRED",
                "stage": stage,
                "lock": lock,
            }
        elif status == "FAIL":
            next_action = {"action": "SCIENTIFIC_REVIEW_REQUIRED", "stage": stage}
        else:
            next_action = {"action": "RUN_SAME_FROZEN_PROTOCOL", "stage": stage}
        break
    return {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "states": {stage: {key: states[stage].get(key) for key in ("status", "attempts", "interruptions")} for stage in PREPROCESS_STAGES},
        "next_action": next_action,
    }
