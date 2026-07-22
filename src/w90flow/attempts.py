from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .parsers import parse_eigenval, parse_fermi_outcar, parse_hr_header, parse_poscar, parse_procar_orbital_scores, parse_wout
from .state import atomic_json, read_json, sha256_file
from .windows import minimum_outer_max, validate_frozen_window, validate_outer_window, validate_window_containment


FAILURE_TRANSITIONS = {
    "NOT_ENOUGH_STATES": "expand_outer_window_to_next_required_band_edge",
    "TOO_MANY_FROZEN_STATES": "shrink_frozen_window_without_dropping_protected_target_bands",
    "DISENTANGLEMENT_NOT_CONVERGED": "increase_dis_num_iter_then_reduce_dis_mix_ratio_once",
    "LOCALIZATION_NOT_CONVERGED": "increase_num_iter_then_enable_guiding_centres_once",
    "BAND_CONNECTIVITY_FAILED": "change_projection_or_outer_window_not_localization_iterations",
    "QUALITY_GATE_FAILED": "protect_target_window_then_add_next_missing_orbital_shell",
    "INVALID_WINDOW_CONTAINMENT": "regenerate_attempt_with_frozen_window_inside_outer_window",
    "INFRASTRUCTURE_FAILURE": "retry_same_attempt_once_after_environment_check",
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
) -> list[dict[str, Any]]:
    if not 1 <= max_window_attempts <= 4:
        raise ValueError("有限状态机支持 1 到 4 个 window attempts。")
    if not 0 <= max_localization_retries <= 3:
        raise ValueError("有限状态机支持 0 到 3 个 localization retries。")
    target_indices = _target_band_indices(energies, target_min, target_max)
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
    if not parse_wout(source_dir / "wannier90.wout")["completed"]:
        raise ValueError("来源 wannier90.wout 未正常完成。")
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
    for name in ("runner.log", "wannier90.wout"):
        path = directory / name
        if path.is_file():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
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
    if "nan" in lower or "segmentation fault" in lower or returncode != 0:
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


def _commands(config: dict[str, Any], directory: Path) -> list[list[str]]:
    workspace = Path(__file__).resolve().parents[4]
    backend = config.get("backend", {})
    default_kind = "ncl" if config.get("physics", {}).get("soc") else "std"
    return [
        [
            str(workspace / "00_命令入口/计算/运行VASP.command"),
            str(directory),
            str(backend.get("vasp", default_kind)),
            str(backend.get("mpi_processes", 2)),
        ],
        [
            str(workspace / "00_命令入口/计算/运行Wannier90.command"),
            str(directory),
            "wannier90",
            "run",
        ],
    ]


def run_attempt(
    run_dir: str | Path,
    structure: str,
    candidate: str,
    attempt: str,
    execute: bool = False,
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
    commands = _commands(config, directory)
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
    record.update({"status": "RUNNING", "started_utc": _utc_now(), "commands": commands})
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
    required_outputs = [directory / name for name in ("wannier90.wout", "wannier90_hr.dat", "wannier90_centres.xyz", "wannier90_band.dat")]
    output_missing = [path.name for path in required_outputs if not path.is_file()]
    completed = False
    if (directory / "wannier90.wout").is_file():
        completed = parse_wout(directory / "wannier90.wout")["completed"]
    passed = returncode == 0 and not output_missing and completed
    failure = None if passed else classify_failure(directory, returncode)
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
