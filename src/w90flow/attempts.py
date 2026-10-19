from __future__ import annotations

import json
import itertools
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .parsers import convergence_gate, parse_eigenval, parse_fermi_outcar, parse_hr_header, parse_poscar, parse_procar_orbital_scores, parse_wout
from .state import atomic_json, read_json, sha256_file
from .windows import minimum_outer_max, suggest_occupation_complete_window, validate_frozen_window, validate_outer_window, validate_window_containment


FAILURE_TRANSITIONS = {
    "NOT_ENOUGH_STATES": "expand_outer_window_to_next_required_band_edge",
    "TOO_MANY_FROZEN_STATES": "shrink_frozen_window_without_dropping_protected_target_bands",
    "DISENTANGLEMENT_NOT_CONVERGED": "increase_dis_num_iter_then_reduce_dis_mix_ratio_once",
    "CONVERGENCE_UNVERIFIED": "stop_and_check_stage_convergence_evidence_and_log_version",
    "W90_NOT_COMPLETED": "inspect_incomplete_wannier_outputs_before_retry",
    "LOCALIZATION_NOT_CONVERGED": "increase_num_iter_then_enable_guiding_centres_once",
    "BAND_CONNECTIVITY_FAILED": "change_projection_or_outer_window_not_localization_iterations",
    "QUALITY_GATE_FAILED": "protect_target_window_then_add_next_missing_orbital_shell",
    "INVALID_WINDOW_CONTAINMENT": "regenerate_attempt_with_frozen_window_inside_outer_window",
    "INFRASTRUCTURE_FAILURE": "retry_same_attempt_once_after_environment_check",
    "INTERFACE_FAILURE": "stop_and_require_site_environment_fix",
    "KPOINT_INDEX_ORDER_MISMATCH": "rebuild_wannier_inputs_from_one_shared_explicit_mesh",
    "NEIGHBOUR_SHELL_FAILURE": "check_bounded_kmesh_tol_and_search_shells_with_wannier_preprocessing_before_new_attempt",
    "NUMERIC_FAILURE": "stop_and_require_manual_diagnosis",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_snapshot(root: Path) -> dict[str, Any]:
    with (root / "project.snapshot.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("project.snapshot.yaml 无效。")
    return config


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


def _target_band_indices(energies: list[list[float]], target_min: float, target_max: float) -> list[int]:
    if not energies:
        return []
    nbands = len(energies[0])
    touched = [
        band + 1 for band in range(nbands)
        if any(target_min <= row[band] <= target_max for row in energies)
    ]
    if not touched:
        return []
    # Wannier eigenvalues are compared as a continuous sorted manifold.  If an
    # intermediate DFT band never enters the numeric energy interval, skipping
    # it would nevertheless map non-contiguous DFT indices onto a contiguous
    # Wannier slice.  Protect the full band-index interval instead.
    return list(range(min(touched), max(touched) + 1))


def target_occupation_coverage(
    energies: list[list[float]], nelect: int, soc: bool,
    target_min: float, target_max: float,
) -> dict[str, Any]:
    """Apply the existing optical quality gate before expensive attempts.

    Uses the same integer-occupation convention and protected band interval
    as planning/quality. It diagnoses the declared window; it never widens it.
    """
    occupied = int(round(nelect if soc else nelect / 2.0))
    indices = _target_band_indices(energies, target_min, target_max)
    target_occupied = sum(index <= occupied for index in indices)
    passed = 0 < target_occupied < len(indices)
    integer_occupation = abs((nelect if soc else nelect / 2.0) - occupied) < 1e-8
    proposal = (
        suggest_occupation_complete_window(energies, occupied, target_min, target_max)
        if not passed and integer_occupation else None
    )
    return {
        "passed": passed,
        "failure_class": None if passed else "TARGET_WINDOW_MISSING_OCCUPATION_SECTOR",
        "target_energy_eV": [target_min, target_max],
        "target_bands_1_based": indices,
        "occupied_band_count": occupied,
        "target_occupied_band_count": target_occupied,
        "target_empty_band_count": len(indices) - target_occupied,
        "policy": "require occupied and empty bands without changing the declared window",
        "separate_window_proposal": proposal,
    }


def _boundary_gaps(energies: list[list[float]], start: int, stop: int) -> dict[str, float | None]:
    nbands = len(energies[0])
    lower = None if start == 1 else min(row[start - 1] - row[start - 2] for row in energies)
    upper = None if stop == nbands else min(row[stop] - row[stop - 1] for row in energies)
    finite = [gap for gap in (lower, upper) if gap is not None]
    return {"lower_gap_eV": lower, "upper_gap_eV": upper, "minimum_boundary_gap_eV": min(finite) if finite else math.inf}


def _candidate_character(band_features: list[dict[str, Any]], projection: list[str]) -> list[float]:
    component_shell = {
        "px": "p", "py": "p", "pz": "p",
        "dxy": "d", "dyz": "d", "dz2": "d", "dxz": "d", "dx2": "d", "dx2-y2": "d",
    }

    def score(item: dict[str, Any], label: str) -> float:
        scores = item["orbital_scores"]
        if label in scores:
            return float(scores[label])
        element, orbital = label.split(":", 1)
        shell = component_shell.get(orbital)
        if shell is None:
            return 0.0
        # PROCAR analysis currently records shell totals.  Split that total
        # evenly only for band-selection heuristics; the real Wannier attempt
        # and the hard band-quality gate remain authoritative.
        divisor = 3 if shell == "p" else 5
        return float(scores.get(f"{element}:{shell}", 0.0)) / divisor

    return [sum(score(item, label) for label in projection) for item in band_features]


def _exact_plan(
    energies: list[list[float]],
    band_features: list[dict[str, Any]],
    projection: list[str],
    num_wann: int,
    target_indices: list[int],
    isolation_threshold_eV: float,
) -> dict[str, Any] | None:
    nbands = len(energies[0])
    if not target_indices or num_wann > nbands:
        return None
    target_start, target_stop = min(target_indices), max(target_indices)
    if target_stop - target_start + 1 > num_wann:
        return None
    first_start = max(1, target_stop - num_wann + 1)
    last_start = min(target_start, nbands - num_wann + 1)
    character = _candidate_character(band_features, projection)
    blocks: list[dict[str, Any]] = []
    for start in range(first_start, last_start + 1):
        stop = start + num_wann - 1
        gaps = _boundary_gaps(energies, start, stop)
        score = sum(character[start - 1:stop])
        blocks.append({"start": start, "stop": stop, "projection_character_score": score, **gaps})
    if not blocks:
        return None
    blocks.sort(key=lambda item: (-item["projection_character_score"], -float(item["minimum_boundary_gap_eV"]), item["start"]))
    best = blocks[0]
    if float(best["minimum_boundary_gap_eV"]) < isolation_threshold_eV:
        return None
    excluded = [index for index in range(1, nbands + 1) if not best["start"] <= index <= best["stop"]]
    return {
        "mode": "exact",
        "retained_vasp_bands_1_based": [best["start"], best["stop"]],
        "exclude_bands_1_based": excluded,
        "target_bands_1_based": target_indices,
        "selection": best,
        "alternatives": blocks,
    }


def _disentanglement_plans(
    energies: list[list[float]],
    band_features: list[dict[str, Any]],
    projection: list[str],
    num_wann: int,
    target_indices: list[int],
    target_min: float,
    target_max: float,
    max_window_attempts: int,
    max_localization_retries: int,
    disentanglement_budget_iterations: int = 0,
) -> list[dict[str, Any]]:
    character = _candidate_character(band_features, projection)
    desired = set(target_indices)
    for index in sorted(range(1, len(character) + 1), key=lambda band: (-character[band - 1], band)):
        desired.add(index)
        if len(desired) >= num_wann:
            break
    desired_sorted = sorted(desired)
    raw_min = min(energies[k][band - 1] for k in range(len(energies)) for band in desired_sorted) - 0.05
    raw_max = max(energies[k][band - 1] for k in range(len(energies)) for band in desired_sorted) + 0.05
    try:
        required_max = minimum_outer_max(energies, raw_min, num_wann, 0.02)
    except ValueError:
        raw_min = min(min(row) for row in energies) - 0.05
        required_max = minimum_outer_max(energies, raw_min, num_wann, 0.02)
    outer_max = max(raw_max, required_max, target_max)
    raw_min = min(raw_min, target_min)
    frozen = validate_frozen_window(energies, target_min, target_max, num_wann)
    frozen_max = target_max
    if not frozen["valid"]:
        candidate_edges = sorted({energy for row in energies for energy in row if target_min < energy <= target_max}, reverse=True)
        for edge in candidate_edges:
            check = validate_frozen_window(energies, target_min, edge - 0.01, num_wann)
            if check["valid"]:
                frozen_max = edge - 0.01
                frozen = check
                break
    seed = {
        "mode": "disentanglement",
        "dis_win_min_eV": round(raw_min, 6),
        "dis_win_max_eV": round(outer_max, 6),
        "dis_froz_min_eV": target_min,
        "dis_froz_max_eV": round(frozen_max, 6),
        "outer_check": validate_outer_window(energies, raw_min, outer_max, num_wann),
        "frozen_check": frozen,
        "containment_check": validate_window_containment(raw_min, outer_max, target_min, frozen_max),
        "target_bands_1_based": target_indices,
        "desired_character_bands_1_based": desired_sorted,
        "dis_num_iter": 500,
        "dis_mix_ratio": 0.5,
    }
    plans = [seed]
    if disentanglement_budget_iterations:
        plans.append({
            **seed,
            "dis_num_iter": disentanglement_budget_iterations,
            "retry_class": "disentanglement_budget",
            "retry_index": 1,
            "retry_reason": "disentanglement_budget_only",
        })
    for retry_index in range(1, max_window_attempts):
        expansion = 0.5 * retry_index
        plans.append({
            **seed,
            "dis_win_min_eV": round(raw_min - expansion, 6),
            "dis_win_max_eV": round(outer_max + expansion, 6),
            "retry_class": "outer_window_expansion",
            "retry_index": retry_index,
            "retry_reason": f"outer_window_expansion_{retry_index:03d}",
        })
    for retry_index in range(1, max_localization_retries + 1):
        plans.append({
            **seed,
            "dis_num_iter": 500 * (retry_index + 1),
            "dis_mix_ratio": round(max(0.1, 0.5 - 0.1 * retry_index), 3),
            "num_iter": 5000 * retry_index,
            "retry_class": "convergence_retry",
            "retry_index": retry_index,
            "retry_reason": f"convergence_retry_{retry_index:03d}",
        })
    return plans


def plan_candidate(
    energies: list[list[float]],
    band_features: list[dict[str, Any]],
    candidate: dict[str, Any],
    target_min: float,
    target_max: float,
    isolation_threshold_eV: float = 0.02,
    max_window_attempts: int = 2,
    max_localization_retries: int = 1,
    disentanglement_budget_iterations: int = 0,
) -> list[dict[str, Any]]:
    if not 1 <= max_window_attempts <= 4:
        raise ValueError("有限状态机支持 1 到 4 个 window attempts。")
    if not 0 <= max_localization_retries <= 3:
        raise ValueError("有限状态机支持 0 到 3 个 localization retries。")
    if (type(disentanglement_budget_iterations) is not int or
            disentanglement_budget_iterations != 0 and not 1000 <= disentanglement_budget_iterations <= 10000):
        raise ValueError("disentanglement_budget_iterations must be 0 or an integer from 1000 to 10000")
    target_indices = _target_band_indices(energies, target_min, target_max)
    # A Wannier Hamiltonian with fewer eigenvalues than the protected DFT band
    # manifold can never reproduce that manifold.  Reject it before launching
    # VASP/Wannier instead of spending a numerical attempt on an impossible
    # quality comparison.
    if len(target_indices) > int(candidate["num_wann"]):
        return []
    exact = _exact_plan(
        energies,
        band_features,
        list(candidate["projection"]),
        int(candidate["num_wann"]),
        target_indices,
        isolation_threshold_eV,
    )
    if exact:
        return [exact]
    return _disentanglement_plans(
        energies,
        band_features,
        list(candidate["projection"]),
        int(candidate["num_wann"]),
        target_indices,
        target_min,
        target_max,
        max_window_attempts,
        max_localization_retries,
        disentanglement_budget_iterations,
    )


def _format_ranges(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _replace_projection_block(text: str, projection: list[str]) -> str:
    block = "begin projections\n" + "\n".join(projection) + "\nend projections"
    replaced, count = re.subn(r"(?ims)^\s*begin\s+projections\s*$.*?^\s*end\s+projections\s*$", block, text)
    if count != 1:
        raise ValueError("WANNIER90_WIN 必须恰好有一个 projections block。")
    return replaced


def _build_w90_text(template: str, candidate: dict[str, Any], plan: dict[str, Any]) -> str:
    text = _replace_projection_block(template, list(candidate["projection"]))
    text = re.sub(r"(?im)^\s*(exclude_bands|dis_win_min|dis_win_max|dis_froz_min|dis_froz_max|dis_num_iter|dis_mix_ratio|use_ws_distance)\s*[=:].*$\n?", "", text)
    header: list[str] = ["use_ws_distance = false"]
    if plan["mode"] == "exact":
        excluded = _format_ranges(list(plan["exclude_bands_1_based"]))
        if excluded:
            header.append(f"exclude_bands = {excluded}")
    else:
        containment = validate_window_containment(
            float(plan["dis_win_min_eV"]),
            float(plan["dis_win_max_eV"]),
            float(plan["dis_froz_min_eV"]),
            float(plan["dis_froz_max_eV"]),
        )
        if not containment["valid"]:
            raise ValueError("冻结窗必须完整包含在 disentanglement 外窗内。")
        header.extend([
            f"dis_win_min = {plan['dis_win_min_eV']:.6f}",
            f"dis_win_max = {plan['dis_win_max_eV']:.6f}",
            f"dis_froz_min = {plan['dis_froz_min_eV']:.6f}",
            f"dis_froz_max = {plan['dis_froz_max_eV']:.6f}",
            f"dis_num_iter = {plan.get('dis_num_iter', 500)}",
            f"dis_mix_ratio = {plan.get('dis_mix_ratio', 0.5):.3f}",
        ])
    if plan.get("num_iter"):
        text = re.sub(r"(?im)^\s*num_iter\s*[=:].*$", f"num_iter = {plan['num_iter']}", text)
    return "\n".join(header) + "\n\n" + text.strip() + "\n"


def _build_incar(template_path: Path, candidate: dict[str, Any], plan: dict[str, Any]) -> str:
    text = template_path.read_text(encoding="utf-8", errors="replace")
    inline = re.search(r"(?ims)(WANNIER90_WIN\s*=\s*\"\s*\n)(.*?)(\n\")", text)
    if not inline:
        raise ValueError("Wannier INCAR 缺少多行 WANNIER90_WIN。")
    w90 = _build_w90_text(inline.group(2), candidate, plan)
    text = text[:inline.start(2)] + w90.rstrip("\n") + text[inline.end(2):]
    text, count = re.subn(r"(?im)^\s*NUM_WANN\s*=.*$", f"NUM_WANN = {candidate['num_wann']}", text)
    if count != 1:
        raise ValueError("Wannier INCAR 必须恰好有一个 NUM_WANN。")
    return text


def build_standalone_w90_input(
    incar_text: str,
    num_wann: int,
    total_vasp_bands: int,
    plan: dict[str, Any],
    poscar_text: str,
) -> str:
    """Build the pre-existing .win required by patched VASP 5.4.4.

    VASP 6.x accepts the inline ``WANNIER90_WIN`` INCAR block.  The patched
    VASP 5.4.4 interface used on some clusters instead reads ``wannier90.win``
    before VASP starts; without that file it silently sets num_wann=NBANDS and
    discards the requested projections/windows.
    """
    inline = re.search(r"(?ims)(WANNIER90_WIN\s*=\s*\"\s*\n)(.*?)(\n\")", incar_text)
    if not inline:
        raise ValueError("Wannier INCAR 缺少多行 WANNIER90_WIN。")
    if num_wann <= 0 or total_vasp_bands <= 0 or num_wann > total_vasp_bands:
        raise ValueError("独立 wannier90.win 的 NUM_WANN/NUM_BANDS 非法。")

    text = inline.group(2).strip()
    text = re.sub(r"(?im)^\s*num_(wann|bands)\s*[=:].*$\n?", "", text).strip()
    # The VASP2WANNIER90 patch on the Paracloud VASP 5.4.4 build links
    # Wannier90 2.1.0.  That parser rejects ``precond`` before VASP can write
    # AMN/MMN/EIG, although newer project templates may contain the keyword.
    # Removing it changes only the localization optimiser, not the Hamiltonian,
    # projections, energy windows, or acceptance criteria.
    text = re.sub(r"(?im)^\s*precond\s*[=:].*$\n?", "", text).strip()
    excluded = [int(value) for value in plan.get("exclude_bands_1_based", [])]
    if any(value < 1 or value > total_vasp_bands for value in excluded):
        raise ValueError("exclude_bands 超出 VASP 能带范围。")
    num_bands = total_vasp_bands - len(set(excluded))
    if num_bands < num_wann:
        raise ValueError("排除能带后 NUM_BANDS 小于 NUM_WANN。")
    cell = _poscar_cell_blocks(poscar_text)
    return (
        f"num_wann = {num_wann}\n"
        f"num_bands = {num_bands}\n\n"
        f"{text}\n\n"
        f"{cell}"
    )


def _poscar_cell_blocks(poscar_text: str) -> str:
    """从 POSCAR 生成 wannier90.win 的 unit_cell_cart + atoms_cart 区块。

    2026-08-13 修复（ZrI2 complete-sd/attempt-003 失败证据）：
    build_standalone_w90_input 原先只复制 INCAR 内联 WANNIER90_WIN 块，而内联块不含
    晶胞几何；VASP 5.4.4 MLWF 接口（wannier90 2.1.0）在缺少 unit_cell_cart/atoms_cart
    时把 4 行投影展开为 288 个投影并与 num_wann=144 冲突，且报
    "Did not find the cell information in the input file"。A1（模板自带 cell 块）同
    参数可跑通，补齐 cell/atoms 块即为最小修复。
    """
    lines = [line for line in poscar_text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError("POSCAR 内容不完整，无法生成 unit_cell_cart/atoms_cart。")
    try:
        scale = float(lines[1].split()[0])
        lattice = [
            [float(value) for value in row.split()[:3]]
            for row in lines[2:5]
        ]
        species = lines[5].split()
        counts = [int(value) for value in lines[6].split()[: len(species)]]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"POSCAR 晶格/种类行非法：{exc}") from exc
    if sum(counts) <= 0:
        raise ValueError("POSCAR 原子数为零。")
    index = 7
    if lines[7].strip().lower().startswith("selective"):
        index = 8
    mode = lines[index].strip().lower()
    rows = lines[index + 1 : index + 1 + sum(counts)]
    if len(rows) != sum(counts):
        raise ValueError("POSCAR 坐标行数与声明原子数不一致。")

    cell_rows: list[str] = []
    atom_index = 0
    for species_index, count in enumerate(counts):
        for _ in range(count):
            fields = rows[atom_index].split()
            if len(fields) < 3:
                raise ValueError(f"POSCAR 坐标行非法：{rows[atom_index]}")
            try:
                coord = [float(value) for value in fields[:3]]
            except ValueError as exc:
                raise ValueError(f"POSCAR 坐标非数值：{rows[atom_index]}") from exc
            if mode.startswith("direct") or mode.startswith("frac"):
                cart = [
                    coord[0] * lattice[0][j]
                    + coord[1] * lattice[1][j]
                    + coord[2] * lattice[2][j]
                    for j in range(3)
                ]
            else:
                cart = [value * scale for value in coord]
            element = fields[3] if len(fields) >= 4 and fields[3] in species else species[species_index]
            cell_rows.append(
                f"{element:2s} {cart[0]:13.7f} {cart[1]:13.7f} {cart[2]:13.7f}"
            )
            atom_index += 1

    scaled_lattice = [[value * scale for value in row] for row in lattice]
    unit_cell = "\n".join(
        "  %13.7f %13.7f %13.7f" % tuple(row) for row in scaled_lattice
    )
    return (
        "begin unit_cell_cart\n"
        f"{unit_cell}\n"
        "end unit_cell_cart\n"
        "\n"
        "begin atoms_cart\n"
        + "\n".join(cell_rows)
        + "\n"
        "end atoms_cart\n"
    )


def _explicit_w90_kpoints(win_text: str) -> list[list[float]]:
    blocks = re.findall(r"(?ims)^\s*begin\s+kpoints\s*$\n(.*?)^\s*end\s+kpoints\s*$", win_text)
    if len(blocks) != 1:
        raise ValueError(f"wannier90.win 必须恰好包含一个 kpoints 区块，当前为 {len(blocks)} 个。")
    points: list[list[float]] = []
    for raw in blocks[0].splitlines():
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 3:
            raise ValueError(f"非法 Wannier90 k 点行：{raw}")
        points.append([float(value) for value in fields])
    return points


def _ibzkpt_points(path: str | Path) -> list[list[float]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError("IBZKPT 内容不完整。")
    count = int(lines[1].split()[0])
    rows = [line.split() for line in lines[3:3 + count]]
    if len(rows) != count or any(len(row) < 3 for row in rows):
        raise ValueError("IBZKPT k 点数量与声明不一致。")
    return [[float(value) for value in row[:3]] for row in rows]


def _wrapped_fractional_delta(left: float, right: float) -> float:
    return abs((left - right + 0.5) % 1.0 - 0.5)


def validate_standalone_w90_kpoint_order(
    win_path: str | Path,
    ibzkpt_path: str | Path,
    *,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Fail closed when VASP and Wannier90 index the same mesh differently."""
    win_points = _explicit_w90_kpoints(Path(win_path).read_text(encoding="utf-8", errors="replace"))
    vasp_points = _ibzkpt_points(ibzkpt_path)
    if len(win_points) != len(vasp_points):
        raise ValueError(
            "KPOINT_INDEX_ORDER_MISMATCH: VASP/Wannier k 点数不同："
            f"{len(vasp_points)} != {len(win_points)}"
        )
    maximum = 0.0
    for index, (vasp, wannier) in enumerate(zip(vasp_points, win_points), 1):
        delta = max(_wrapped_fractional_delta(a, b) for a, b in zip(vasp, wannier))
        maximum = max(maximum, delta)
        if delta > tolerance:
            raise ValueError(
                "KPOINT_INDEX_ORDER_MISMATCH: 第 "
                f"{index} 个 VASP/Wannier k 点不同（mod 1 最大差 {delta:.3e}）。"
            )
    return {
        "mode": "explicit_full_mesh_index_validation",
        "kpoint_count": len(win_points),
        "maximum_wrapped_fractional_delta": maximum,
        "passed": True,
    }


def canonicalize_standalone_w90_kpoints(win_text: str) -> tuple[str, int]:
    """Match an old VASP2WANNIER90 MMN file's signed k-point convention.

    Some patched VASP 5.4.4 builds calculate MMN neighbour cells from VASP's
    internal ``[-0.5, 0.5)`` coordinates, but append equivalent ``[0, 1)``
    coordinates to ``wannier90.win``.  External Wannier90 then reconstructs a
    different neighbour-cell labelling and rejects the otherwise valid MMN.
    Canonicalising only the explicit k-point block restores the representation
    used to create MMN; it does not change the physical mesh.
    """
    match = re.search(r"(?ims)(^\s*begin\s+kpoints\s*$)(.*?)(^\s*end\s+kpoints\s*$)", win_text)
    if not match:
        raise ValueError("VASP 完成后的 wannier90.win 缺少 kpoints 区块。")
    converted: list[str] = []
    moved = 0
    for raw in match.group(2).splitlines():
        if not raw.strip():
            converted.append(raw)
            continue
        fields = raw.split()
        if len(fields) != 3:
            raise ValueError(f"非法 Wannier90 k 点行：{raw}")
        values = [float(value) for value in fields]
        canonical: list[float] = []
        for value in values:
            shifted = value
            while shifted >= 0.5:
                shifted -= 1.0
            while shifted < -0.5:
                shifted += 1.0
            if abs(shifted - value) > 1e-10:
                moved += 1
            canonical.append(shifted)
        converted.append("  " + " ".join(f"{value:.12f}" for value in canonical))
    body = "\n" + "\n".join(converted) + "\n"
    return win_text[:match.start(2)] + body + win_text[match.end(2):], moved


def reconcile_mmn_neighbour_cells(
    mmn_path: str | Path, nnkp_path: str | Path, *,
    win_path: str | Path | None = None, ibzkpt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reconcile a VASP 5.4.4 MMN cell-axis convention with Wannier90.

    The Paracloud VASP2WANNIER90 patch writes the correct k-pairs and overlap
    matrices, but its reciprocal-cell components can be permuted relative to
    the ``.win``/``.nnkp`` basis.  Try every signed axis permutation and apply
    one only when it makes the complete MMN neighbour set exactly equal to the
    neighbour set independently generated by Wannier90. With WIN and IBZKPT
    evidence, also allow G_new = G_old + shift[j] - shift[i] when canonicalising
    equivalent k-point representatives. Require exact full-set agreement before
    writing; never change overlap values or infer shifts from headers alone.
    """
    mmn = Path(mmn_path)
    nnkp = Path(nnkp_path)
    with mmn.open("r", encoding="utf-8", errors="replace") as handle:
        comment = handle.readline()
        dimensions = handle.readline()
        num_bands, num_kpoints, num_neighbours = map(int, dimensions.split())
        headers: list[tuple[int, int, int, int, int]] = []
        for _ in range(num_kpoints * num_neighbours):
            header = tuple(map(int, handle.readline().split()))
            if len(header) != 5:
                raise ValueError("MMN 邻接块头必须包含五个整数。")
            headers.append(header)
            for _ in range(num_bands * num_bands):
                if not handle.readline():
                    raise ValueError("MMN overlap 数据提前结束。")

    nnkp_lines = nnkp.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        (index for index, line in enumerate(nnkp_lines) if line.strip().lower() == "begin nnkpts"),
        None,
    )
    if start is None:
        raise ValueError("NNKP 缺少 nnkpts 区块。")
    expected_neighbours = int(nnkp_lines[start + 1].split()[0])
    if expected_neighbours != num_neighbours:
        raise ValueError("MMN 与 NNKP 的每个 k 点邻接数不同。")
    expected = [
        tuple(map(int, line.split()))
        for line in nnkp_lines[start + 2:start + 2 + num_kpoints * expected_neighbours]
    ]
    if len(expected) != len(headers) or any(len(header) != 5 for header in expected):
        raise ValueError("NNKP 邻接块长度非法。")

    transforms: list[tuple[tuple[int, ...], tuple[int, ...], list[tuple[int, ...]]]] = []
    expected_counter = Counter(expected)
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((0, 1, 2), (1, 1, 1))]
    candidates.extend(
        (permutation, signs)
        for permutation in itertools.permutations(range(3))
        for signs in itertools.product((-1, 1), repeat=3)
        if (permutation, signs) != ((0, 1, 2), (1, 1, 1))
    )
    for permutation, signs in candidates:
        transformed = []
        for first, second, gx, gy, gz in headers:
            cell = (gx, gy, gz)
            mapped = tuple(signs[index] * cell[permutation[index]] for index in range(3))
            transformed.append((first, second, *mapped))
        if Counter(transformed) == expected_counter:
            transforms.append((permutation, signs, transformed))
            break
    mode = "signed_axis_permutation"
    translated_points = 0
    if not transforms and win_path is not None and ibzkpt_path is not None:
        wannier_points = _explicit_w90_kpoints(Path(win_path).read_text(encoding="utf-8"))
        vasp_points = _ibzkpt_points(ibzkpt_path)
        if len(wannier_points) != num_kpoints or len(vasp_points) != num_kpoints:
            raise ValueError("MMN k 点数量与 WIN/IBZKPT 不一致。")
        shifts = []
        for vasp, wannier in zip(vasp_points, wannier_points):
            delta = [left - right for left, right in zip(vasp, wannier)]
            if any(abs(value - round(value)) > 1.0e-8 for value in delta):
                raise ValueError("WIN/IBZKPT k 点不是相同索引的整倒格矢平移。")
            shifts.append(tuple(round(value) for value in delta))
        translated = []
        for first, second, *cell_components in headers:
            if not (1 <= first <= num_kpoints and 1 <= second <= num_kpoints):
                raise ValueError("MMN 邻居 k 点索引越界。")
            mapped = tuple(cell_components[axis] + shifts[second - 1][axis] - shifts[first - 1][axis]
                           for axis in range(3))
            translated.append((first, second, *mapped))
        if Counter(translated) == expected_counter:
            transforms.append(((0, 1, 2), (1, 1, 1), translated))
            mode = "ibzkpt_to_win_integer_translation"
            translated_points = sum(any(shift) for shift in shifts)
    if not transforms:
        raise ValueError("找不到能使 MMN 与 NNKP 完整一致的 reciprocal-cell 轴变换或有证据的 k 点平移。")
    permutation, signs, transformed = transforms[0]
    changed = sum(before != after for before, after in zip(headers, transformed))
    if changed:
        temporary = mmn.with_name(mmn.name + ".reconcile.tmp")
        with mmn.open("r", encoding="utf-8", errors="replace") as source, temporary.open(
            "w", encoding="utf-8"
        ) as target:
            target.write(source.readline())
            target.write(source.readline())
            for fixed in transformed:
                source.readline()
                target.write("".join(f"{value:5d}" for value in fixed) + "\n")
                for _ in range(num_bands * num_bands):
                    target.write(source.readline())
        os.replace(temporary, mmn)
    return {
        "permutation": list(permutation),
        "signs": list(signs),
        "changed_headers": changed,
        "mode": mode,
        "translated_kpoints": translated_points,
        "num_headers": len(headers),
        "matched_exactly": True,
        "sha256": sha256_file(mmn),
    }


def materialize_plan_energy_reference(
    plan: dict[str, Any],
    fermi_eV: float,
    energy_reference: str,
) -> dict[str, Any]:
    """Translate a reference-frame plan into the absolute energies Wannier90 reads."""
    materialized = json.loads(json.dumps(plan))
    if energy_reference == "absolute" or plan.get("mode") == "exact":
        return materialized
    if energy_reference != "fermi_relative":
        raise ValueError(f"不支持的 energy_reference：{energy_reference}")
    for key in ("dis_win_min_eV", "dis_win_max_eV", "dis_froz_min_eV", "dis_froz_max_eV"):
        materialized[key] = round(float(materialized[key]) + fermi_eV, 6)
    materialized["outer_check"] = {
        **materialized.get("outer_check", {}),
        "reference_note": "check was evaluated in the Fermi-relative frame",
    }
    materialized["frozen_check"] = {
        **materialized.get("frozen_check", {}),
        "reference_note": "check was evaluated in the Fermi-relative frame",
    }
    materialized["containment_check"] = validate_window_containment(
        materialized["dis_win_min_eV"],
        materialized["dis_win_max_eV"],
        materialized["dis_froz_min_eV"],
        materialized["dis_froz_max_eV"],
    )
    return materialized


def _existing_projection(root: Path, sid: str) -> list[str] | None:
    win = root / "structures" / sid / "04_wannier/wannier90.win"
    if not win.is_file():
        return None
    match = re.search(r"(?ims)^\s*begin\s+projections\s*$\n(.*?)^\s*end\s+projections\s*$", win.read_text(encoding="utf-8", errors="replace"))
    return [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else None


def create_attempts(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    config = _load_snapshot(root)
    spec = read_json(root / "03_candidates/candidate-spec.json")
    target_min, target_max = map(float, spec["target_energy_eV"])
    energy_reference = spec.get("energy_reference", "absolute")
    mode = spec["ensemble_mode"]
    isolation_threshold = float(config.get("candidates", {}).get("exact_min_boundary_gap_eV", 0.05))
    if isolation_threshold <= 0:
        raise ValueError("candidates.exact_min_boundary_gap_eV 必须大于零。")
    iteration = config.get("iteration", {})
    max_window_attempts = int(iteration.get("max_window_attempts_per_candidate", 2))
    max_localization_retries = int(iteration.get("max_localization_retries", 1))
    disentanglement_budget_iterations = iteration.get("disentanglement_budget_iterations", 0)
    if not 1 <= max_window_attempts <= 4:
        raise ValueError("iteration.max_window_attempts_per_candidate 必须在 1 到 4 之间。")
    if not 0 <= max_localization_retries <= 3:
        raise ValueError("iteration.max_localization_retries 必须在 0 到 3 之间。")
    contexts: dict[str, dict[str, Any]] = {}
    for structure in config["structures"]:
        sid = str(structure["id"])
        base = root / "structures" / sid
        poscar = parse_poscar(base / "00_input/POSCAR")
        eig = parse_eigenval(base / "02_band/EIGENVAL")
        fermi = parse_fermi_outcar(base / "01_scf/OUTCAR")
        energy_offset = fermi if energy_reference == "fermi_relative" else 0.0
        planning_energies = [
            [energy - energy_offset for energy in row]
            for row in eig["energies_eV"]
        ]
        uniform_eigenval = base / "01_scf/EIGENVAL"
        if uniform_eigenval.is_file():
            uniform = parse_eigenval(uniform_eigenval)
            if uniform["nbands"] != eig["nbands"]:
                raise ValueError(f"{sid} 的 SCF/Band NBANDS 不一致。")
            planning_energies.extend([
                [energy - energy_offset for energy in row]
                for row in uniform["energies_eV"]
            ])
        coverage = target_occupation_coverage(
            planning_energies, int(eig["nelect"]), bool(config["physics"].get("soc")),
            target_min, target_max,
        )
        if not coverage["passed"]:
            diagnostic = {
                **coverage, "structure": sid, "energy_reference": energy_reference,
                "fermi_eV": fermi,
                "band_eigenval_sha256": sha256_file(base / "02_band/EIGENVAL"),
            }
            atomic_json(base / "target-window-diagnostic.json", diagnostic)
            raise ValueError(
                f"{coverage['failure_class']}: {sid} 的目标能区必须同时包含占据带和空带；"
                "保持原目标窗口，停止创建 Wannier attempts。"
            )
        absolute_target = [target_min + energy_offset, target_max + energy_offset]
        contexts[sid] = {
            "base": base,
            "fermi_eV": fermi,
            "planning_energies": planning_energies,
            "features": parse_procar_orbital_scores(base / "02_band/PROCAR", poscar, *absolute_target)["band_orbital_scores"],
        }

    shared_plans: dict[str, list[dict[str, Any]]] = {}
    if mode == "common_basis":
        combined_energies = [row for context in contexts.values() for row in context["planning_energies"]]
        feature_sets = [context["features"] for context in contexts.values()]
        if len({len(features) for features in feature_sets}) != 1:
            raise ValueError("common_basis 结构的 NBANDS 不一致。")
        combined_features: list[dict[str, Any]] = []
        for band_index in range(len(feature_sets[0])):
            labels = {label for features in feature_sets for label in features[band_index]["orbital_scores"]}
            combined_features.append({
                "band_index_1_based": band_index + 1,
                "orbital_scores": {
                    label: sum(features[band_index]["orbital_scores"].get(label, 0.0) for features in feature_sets) / len(feature_sets)
                    for label in labels
                },
            })
        for candidate in spec["shared_candidates"]:
            shared_plans[candidate["candidate"]] = plan_candidate(
                combined_energies,
                combined_features,
                candidate,
                target_min,
                target_max,
                isolation_threshold,
                max_window_attempts,
                max_localization_retries,
                disentanglement_budget_iterations,
            )
    all_plans: dict[str, Any] = {}
    for structure in config["structures"]:
        sid = str(structure["id"])
        base = contexts[sid]["base"]
        planning_energies = contexts[sid]["planning_energies"]
        features = contexts[sid]["features"]
        fermi = float(contexts[sid]["fermi_eV"])
        candidates = spec["shared_candidates"] if mode == "common_basis" else spec["independent_candidates"][sid]
        existing_projection = _existing_projection(root, sid)
        structure_plans: list[dict[str, Any]] = []
        for candidate in candidates:
            plans = shared_plans[candidate["candidate"]] if mode == "common_basis" else plan_candidate(
                planning_energies,
                features,
                candidate,
                target_min,
                target_max,
                isolation_threshold,
                max_window_attempts,
                max_localization_retries,
                disentanglement_budget_iterations,
            )
            candidate_record = {"candidate": candidate, "attempts": []}
            if existing_projection == candidate["projection"] and (base / "05_quality/quality.json").is_file():
                candidate_record["existing_accepted"] = str(base / "05_quality/quality.json")
                structure_plans.append(candidate_record)
                continue
            candidate_root = base / "04_wannier_attempts" / candidate["candidate"]
            existing_records: list[tuple[Path, dict[str, Any]]] = []
            if candidate_root.is_dir():
                for record_path in sorted(candidate_root.glob("attempt-*/attempt.json")):
                    existing_records.append((record_path.parent, read_json(record_path)))
            for reference_plan in plans:
                plan = materialize_plan_energy_reference(reference_plan, fermi, energy_reference)
                reused = next((
                    directory for directory, record in existing_records
                    if record.get("plan_reference", record.get("plan")) == reference_plan
                ), None)
                if reused is not None:
                    candidate_record["attempts"].append({
                        "path": str(reused),
                        "plan_reference": reference_plan,
                        "plan": plan,
                        "reused": True,
                    })
                    continue
                next_index = max([int(path.name.split("-")[-1]) for path, _ in existing_records] or [0]) + 1
                attempt_name = f"attempt-{next_index:03d}"
                directory = candidate_root / attempt_name
                directory.mkdir(parents=True, exist_ok=False)
                materialization = {}
                for name in ("POSCAR", "POTCAR", "KPOINTS", "CHGCAR"):
                    materialization[name] = _link_or_copy(base / "04_wannier" / name, directory / name)
                incar = _build_incar(base / "04_wannier/INCAR", candidate, plan)
                (directory / "INCAR").write_text(incar, encoding="utf-8")
                record = {
                    "schema_version": 1,
                    "structure": sid,
                    "candidate": candidate["candidate"],
                    "projection": candidate["projection"],
                    "num_wann": candidate["num_wann"],
                    "attempt": attempt_name,
                    "status": "READY",
                    "energy_reference": energy_reference,
                    "fermi_eV": fermi,
                    "plan_reference": reference_plan,
                    "plan": plan,
                    "materialization": materialization,
                    "failure_transitions": FAILURE_TRANSITIONS,
                    "created_utc": _utc_now(),
                }
                atomic_json(directory / "attempt.json", record)
                existing_records.append((directory, record))
                candidate_record["attempts"].append({
                    "path": str(directory),
                    "plan_reference": reference_plan,
                    "plan": plan,
                })
            structure_plans.append(candidate_record)
        all_plans[sid] = structure_plans
    document = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "ensemble_mode": mode,
        "energy_reference": energy_reference,
        "exact_min_boundary_gap_eV": isolation_threshold,
        "iteration_budget": {
            "max_window_attempts_per_candidate": max_window_attempts,
            "max_localization_retries": max_localization_retries,
            "disentanglement_budget_iterations": disentanglement_budget_iterations,
        },
        "common_plan_across_structures": mode == "common_basis",
        "structures": all_plans,
        "failure_transitions": FAILURE_TRANSITIONS,
    }
    atomic_json(root / "03_candidates/attempt-plan.json", document)
    return document


def import_completed_attempt(
    run_dir: str | Path,
    structure: str,
    candidate: str,
    attempt: str,
    source: str | Path,
) -> dict[str, Any]:
    """Reuse a completed Wannier attempt only when every scientific input matches.

    The imported numerical outputs are evaluated again against the *new* run's
    declared target energy window and quality gates.  Historical quality or
    ranking files are deliberately not imported.
    """
    root = Path(run_dir).expanduser().resolve()
    target = root / "structures" / structure / "04_wannier_attempts" / candidate / attempt
    source_dir = Path(source).expanduser().resolve()
    record_path = target / "attempt.json"
    if not record_path.is_file():
        raise ValueError(f"找不到目标 attempt：{target}")
    if not source_dir.is_dir():
        raise ValueError(f"找不到来源 attempt：{source_dir}")
    record = read_json(record_path)
    if record.get("status") != "READY":
        raise ValueError(f"目标 attempt 状态必须是 READY：{record.get('status')}")

    input_names = ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")
    input_hashes: dict[str, dict[str, str]] = {}
    for name in input_names:
        target_file = target / name
        source_file = source_dir / name
        if not target_file.is_file() or not source_file.is_file():
            raise ValueError(f"复用输入缺少 {name}：target={target_file.is_file()} source={source_file.is_file()}")
        target_hash = sha256_file(target_file)
        source_hash = sha256_file(source_file)
        input_hashes[name] = {"target": target_hash, "source": source_hash}
        if target_hash != source_hash:
            raise ValueError(f"拒绝复用：{name} SHA256 不一致。")

    required_outputs = (
        "wannier90.win",
        "wannier90.wout",
        "wannier90_hr.dat",
        "wannier90_centres.xyz",
        "wannier90_band.dat",
    )
    for name in required_outputs:
        if not (source_dir / name).is_file():
            raise ValueError(f"来源 attempt 缺少完成输出：{name}")
    source_convergence = parse_wout(source_dir / "wannier90.wout")
    source_failure = convergence_gate(source_convergence,
        require_disentanglement=record.get("plan", {}).get("mode") != "exact")
    if source_failure:
        raise ValueError(f"来源 Wannier attempt 缺少严格收敛证据：{source_failure}")
    hr = parse_hr_header(source_dir / "wannier90_hr.dat")
    if int(hr["num_wann"]) != int(record["num_wann"]):
        raise ValueError("来源 HR 维数与目标 NUM_WANN 不一致。")

    materialization: dict[str, str] = {}
    optional_outputs = ("wannier90_band.gnu", "wannier90_band.kpt")
    for name in (*required_outputs, *optional_outputs):
        source_file = source_dir / name
        if source_file.is_file():
            materialization[name] = _link_or_copy(source_file, target / name)
    record.update({
        "status": "PASS",
        "finished_utc": _utc_now(),
        "returncode": 0,
        "failure_class": None,
        "next_action": None,
        "missing_outputs": [],
        "imported_completed_attempt": {
            "source": str(source_dir),
            "input_sha256": input_hashes,
            "materialization": materialization,
            "policy": "all_scientific_inputs_sha256_identical_re_evaluate_quality",
        },
    })
    atomic_json(record_path, record)
    return {
        "passed": True,
        "target": str(target),
        "source": str(source_dir),
        "input_sha256": input_hashes,
        "materialization": materialization,
    }


def classify_failure(directory: Path, returncode: int) -> str | None:
    text = ""
    names = ["runner.log", "wannier90.wout"]
    names.extend(path.name for path in sorted(directory.glob("wannier90*.werr")))
    for name in names:
        path = directory / name
        if path.is_file():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    if "kmesh_get_bvector: not enough bvectors found" in lower:
        return "NEIGHBOUR_SHELL_FAILURE"
    if (
        "no sockets were able to be opened" in lower
        or "bind() failed for port" in lower
        or "prte error" in lower
        or "all nodes which are allocated for this job are already filled" in lower
    ):
        return "INFRASTRUCTURE_FAILURE"
    if "more states in the frozen window" in lower or "frozen states" in lower and "num_wann" in lower:
        return "TOO_MANY_FROZEN_STATES"
    if "not enough states" in lower or "less states than wannier functions" in lower:
        return "NOT_ENOUGH_STATES"
    if "disentanglement convergence" in lower and ("not" in lower or "failed" in lower):
        return "DISENTANGLEMENT_NOT_CONVERGED"
    if "convergence not reached" in lower or "localisation convergence" in lower and "not" in lower:
        return "LOCALIZATION_NOT_CONVERGED"
    if "kpoint_index_order_mismatch" in lower:
        return "KPOINT_INDEX_ORDER_MISMATCH"
    if (
        "corrupted unsorted chunks" in lower
        or "corrupted top size" in lower
        or "double free or corruption" in lower
        or "malloc(): corrupted" in lower
        or "heap corruption" in lower
        or "killed by signal" in lower
        or "bad termination" in lower
        or "signal: 6 (aborted)" in lower
        or "signal: 11 (segmentation fault)" in lower
    ):
        return "INTERFACE_FAILURE"
    if "nan" in lower or returncode != 0:
        return "NUMERIC_FAILURE"
    return None


def recorded_or_classified_failure(record: dict[str, Any], directory: Path) -> str | None:
    """Prefer an explicit runner diagnosis over incomplete/truncated logs.

    A detached local job can be terminated by the app or operating system before
    ``run_attempt`` records a return code.  Recovery tooling marks that orphaned
    attempt explicitly as ``INFRASTRUCTURE_FAILURE``; reclassifying its truncated
    VASP log as a numeric failure would incorrectly forbid the one safe retry.
    """
    recorded = record.get("failure_class")
    if recorded:
        return str(recorded)
    return classify_failure(directory, int(record.get("returncode", 1)))


def _expand_command_template(template: list[str], directory: Path, run_dir: Path) -> list[str]:
    """Expand the site command placeholders; no shell-level string concatenation."""
    return [
        item.replace("{stage_dir}", str(directory)).replace("{run_dir}", str(run_dir))
        for item in template
    ]


def _commands(
    config: dict[str, Any],
    directory: Path,
    *,
    mpi_processes: int | None = None,
    run_dir: Path | None = None,
) -> list[list[str]]:
    """Resolve Wannier attempt commands exclusively from the explicit site config.

    ``preprocess_inputs.commands.04_wannier`` is the VASP launcher (same contract
    as SCF/Band) and ``04_wannier90`` is the Wannier90 launcher whose final
    logical seed is fixed to ``wannier90``; the running mode (``pre``/``run``) is
    appended by this function.  A missing field is a configuration error and is
    never silently resolved from ``W90FLOW_LAUNCHER_ROOT`` or the home directory.
    """
    root = Path(run_dir).expanduser() if run_dir is not None else directory.parents[4]
    inputs = config.get("preprocess_inputs", {})
    commands = inputs.get("commands")
    if not isinstance(commands, dict):
        raise ValueError(
            "Stage-1 配置缺少 preprocess_inputs.commands（Wannier 需要 04_wannier 与 04_wannier90）。"
        )
    backend = config.get("backend", {})
    default_kind = "ncl" if config.get("physics", {}).get("soc") else "std"
    kind = str(backend.get("vasp", default_kind))
    mpi = str(mpi_processes if mpi_processes is not None else backend.get("mpi_processes", 2))

    vasp_template = commands.get("04_wannier")
    if (
        not isinstance(vasp_template, list)
        or len(vasp_template) < 3
        or any(not isinstance(item, str) for item in vasp_template)
    ):
        raise ValueError(
            "preprocess_inputs.commands.04_wannier 必须是参数列表，末两项依次为 std/ncl 与 MPI 数。"
        )
    if vasp_template[-2].strip().lower() != kind:
        raise ValueError(
            f"preprocess_inputs.commands.04_wannier 应使用 {kind}；当前为 {vasp_template[-2]}。"
        )
    if str(vasp_template[-1]) != mpi:
        raise ValueError(
            f"preprocess_inputs.commands.04_wannier MPI 数 {vasp_template[-1]} "
            f"与 backend.mpi_processes={mpi} 不一致。"
        )
    vasp_command = _expand_command_template(vasp_template, directory, root)

    w90_template = commands.get("04_wannier90")
    if (
        not isinstance(w90_template, list)
        or len(w90_template) < 3
        or any(not isinstance(item, str) for item in w90_template)
    ):
        raise ValueError(
            "preprocess_inputs.commands.04_wannier90 必须是至少 3 项参数列表，末项固定为 wannier90。"
        )
    if w90_template[-1] != "wannier90":
        raise ValueError(
            "preprocess_inputs.commands.04_wannier90 末项必须是固定逻辑 seed wannier90；"
            f"当前为 {w90_template[-1]!r}。"
        )
    if config.get("physics", {}).get("soc"):
        from .doctor import capability_errors

        problems = capability_errors(config, commands)
        if problems:
            raise ValueError(
                "SOC/spinor Wannier 需要兼容 site capability（VASP>=6.4 + Wannier90>=3）："
                + "；".join(problems)
            )
    w90_base = _expand_command_template(w90_template, directory, root)

    result: list[list[str]] = [vasp_command]
    if backend.get("wannier_input_mode", "embedded") == "standalone_win":
        result.append(w90_base + ["pre"])
    result.append(w90_base + ["run"])
    return result


def run_attempt(
    run_dir: str | Path,
    structure: str,
    candidate: str,
    attempt: str,
    execute: bool = False,
    mpi_processes: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    config = _load_snapshot(root)
    directory = root / "structures" / structure / "04_wannier_attempts" / candidate / attempt
    record_path = directory / "attempt.json"
    if not record_path.is_file():
        raise ValueError(f"找不到 attempt：{directory}")
    record = read_json(record_path)
    required = [directory / name for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")]
    missing = [str(path) for path in required if not path.is_file()]
    if mpi_processes is not None and mpi_processes <= 0:
        raise ValueError("mpi_processes 必须是正整数。")
    commands = _commands(config, directory, mpi_processes=mpi_processes, run_dir=root)
    if not execute:
        return {"mode": "DRY_RUN", "path": str(directory), "commands": commands, "missing_inputs": missing, "passed": not missing}
    if not config.get("execution", {}).get("allow_execute", False):
        raise PermissionError("project 配置中的 execution.allow_execute 不是 true。")
    if missing:
        raise ValueError("attempt 输入不完整。")
    if record.get("status") in {"PASS", "ACCEPTED", "REJECTED_QUALITY"}:
        raise ValueError(f"attempt 状态为 {record['status']}，拒绝重复运行。")
    if record.get("status") == "FAIL":
        prior_failure = recorded_or_classified_failure(record, directory)
        history = list(record.get("execution_history", []))
        manual_retry = bool(record.pop("manual_infrastructure_retry_authorized", False))
        if prior_failure != "INFRASTRUCTURE_FAILURE":
            raise ValueError(f"attempt 已因 {prior_failure} 失败；必须创建下一 attempt，拒绝覆盖重跑。")
        if history and not manual_retry:
            raise ValueError("同一 attempt 的基础设施重试预算已用完。")
        old_log = directory / "runner.log"
        archived_log = directory / f"runner.infrastructure-failure-{len(history) + 1:03d}.log"
        if old_log.is_file() and not archived_log.exists():
            shutil.copy2(old_log, archived_log)
        history.append({
            "status": record.get("status"),
            "started_utc": record.get("started_utc"),
            "finished_utc": record.get("finished_utc"),
            "returncode": record.get("returncode"),
            "failure_class": prior_failure,
            "manual_retry_authorized": manual_retry,
            "executed_commands": record.get("executed_commands", []),
            "runner_log": str(archived_log),
        })
        record["execution_history"] = history
    if config.get("backend", {}).get("wannier_input_mode", "embedded") == "standalone_win":
        eigenval = parse_eigenval(root / "structures" / structure / "02_band/EIGENVAL")
        standalone = build_standalone_w90_input(
            (directory / "INCAR").read_text(encoding="utf-8", errors="replace"),
            int(record["num_wann"]),
            int(eigenval["nbands"]),
            dict(record["plan"]),
            (directory / "POSCAR").read_text(encoding="utf-8", errors="replace"),
        )
        (directory / "wannier90.win").write_text(standalone, encoding="utf-8")
        record["standalone_w90_input"] = {
            "mode": "vasp54_preexisting_win",
            "path": str(directory / "wannier90.win"),
            "sha256": sha256_file(directory / "wannier90.win"),
            "num_wann": int(record["num_wann"]),
            "num_bands": int(eigenval["nbands"]) - len(set(record["plan"].get("exclude_bands_1_based", []))),
        }
    record.update({
        "status": "RUNNING",
        "started_utc": _utc_now(),
        "commands": commands,
        "runtime_mpi_processes": mpi_processes if mpi_processes is not None else int(config.get("backend", {}).get("mpi_processes", 2)),
    })
    atomic_json(record_path, record)
    log_path = directory / "runner.log"
    executed: list[dict[str, Any]] = []
    returncode = 0
    with log_path.open("wb") as log:
        for index, command in enumerate(commands):
            if index > 0:
                wout = directory / "wannier90.wout"
                if wout.is_file() and (directory / "wannier90_hr.dat").is_file() and parse_wout(wout)["completed"]:
                    executed.append({"command": command, "skipped": True, "reason": "VASP library mode already completed Wannier90"})
                    continue
            process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            record["active_process"] = {
                "pid": process.pid,
                "command": command,
                "started_utc": _utc_now(),
            }
            atomic_json(record_path, record)
            returncode = process.wait()
            record["active_process"] = None
            atomic_json(record_path, record)
            executed.append({"command": command, "returncode": returncode})
            if returncode != 0:
                break
            if config.get("backend", {}).get("wannier_input_mode", "embedded") == "standalone_win":
                try:
                    if index == 0:
                        win_path = directory / "wannier90.win"
                        record["standalone_w90_kpoint_order"] = validate_standalone_w90_kpoint_order(
                            win_path,
                            directory / "IBZKPT",
                        )
                        canonical, moved = canonicalize_standalone_w90_kpoints(
                            win_path.read_text(encoding="utf-8", errors="replace")
                        )
                        win_path.write_text(canonical, encoding="utf-8")
                        record["standalone_w90_post_vasp"] = {
                            "mode": "vasp54_signed_kpoint_compatibility",
                            "path": str(win_path),
                            "sha256": sha256_file(win_path),
                            "shifted_components": moved,
                        }
                    elif index == 1:
                        record["standalone_w90_mmn_reconciliation"] = reconcile_mmn_neighbour_cells(
                            directory / "wannier90.mmn",
                            directory / "wannier90.nnkp",
                            win_path=directory / "wannier90.win",
                            ibzkpt_path=directory / "IBZKPT",
                        )
                    atomic_json(record_path, record)
                except Exception as exc:
                    message = f"W90FLOW_VASP54_COMPATIBILITY_ERROR: {exc}\n"
                    log.write(message.encode("utf-8", errors="replace"))
                    log.flush()
                    executed.append({
                        "compatibility_step": "post_vasp" if index == 0 else "post_preprocessor",
                        "returncode": 1,
                        "error": str(exc),
                    })
                    record["compatibility_error"] = str(exc)
                    returncode = 1
                    break
    required_outputs = [directory / name for name in ("wannier90.wout", "wannier90_hr.dat", "wannier90_centres.xyz", "wannier90_band.dat")]
    output_missing = [path.name for path in required_outputs if not path.is_file()]
    convergence = parse_wout(directory / "wannier90.wout") if (directory / "wannier90.wout").is_file() else {}
    convergence_failure = convergence_gate(convergence,
        require_disentanglement=record.get("plan", {}).get("mode") != "exact")
    # Preserve execution/missing-output failures before evaluating scientific convergence.
    failure = (classify_failure(directory, returncode) if returncode != 0 or output_missing
               else convergence_failure)
    passed = returncode == 0 and not output_missing and failure is None
    record["convergence_check"] = convergence
    record.update({
        "status": "PASS" if passed else "FAIL",
        "finished_utc": _utc_now(),
        "returncode": returncode,
        "executed_commands": executed,
        "missing_outputs": output_missing,
        "failure_class": failure,
        "next_action": FAILURE_TRANSITIONS.get(failure) if failure else None,
        "runner_log": str(log_path),
        "active_process": None,
    })
    atomic_json(record_path, record)
    return {"passed": passed, "attempt": record}
