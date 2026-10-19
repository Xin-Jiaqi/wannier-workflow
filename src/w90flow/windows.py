from __future__ import annotations

import math
from typing import Any, Iterable


def suggest_occupation_complete_window(
    energies: list[list[float]], occupied: int,
    target_min: float, target_max: float, margin_eV: float = 0.5,
) -> dict[str, Any] | None:
    """Propose a separately recorded gapped-band control; never apply it.

    Preserve the entire requested window and cover the top valence and first
    conduction band at every sampled k point. Metals and truncated DFT band
    sets have no safe integer-occupation proposal under this policy.
    """
    if not energies or occupied < 1 or any(len(row) <= occupied for row in energies):
        return None
    if not math.isfinite(margin_eV) or margin_eV <= 0:
        raise ValueError("margin_eV must be finite and positive")
    valence = [row[occupied - 1] for row in energies]
    conduction = [row[occupied] for row in energies]
    if not all(math.isfinite(v) for v in [target_min, target_max, *valence, *conduction]):
        return None
    if target_min > target_max or min(conduction) <= max(valence):
        return None
    return {
        "target_energy_eV": [min(target_min, min(valence) - margin_eV),
                             max(target_max, max(conduction) + margin_eV)],
        "input_target_energy_eV": [target_min, target_max],
        "energy_reference": "same_as_input_eigenvalues",
        "margin_eV": margin_eV,
        "requires_new_protocol": True,
        "status": "proposal_only; refit_and_quality_validation_required",
    }


def state_counts(energies: Iterable[Iterable[float]], window_min: float, window_max: float) -> list[int]:
    return [sum(window_min <= energy <= window_max for energy in row) for row in energies]


def validate_outer_window(energies: list[list[float]], window_min: float, window_max: float, num_wann: int) -> dict[str, Any]:
    counts = state_counts(energies, window_min, window_max)
    minimum = min(counts) if counts else 0
    return {
        "valid": minimum >= num_wann,
        "minimum_states_over_k": minimum,
        "maximum_states_over_k": max(counts) if counts else 0,
        "failing_kpoint_indices": [index for index, count in enumerate(counts) if count < num_wann],
    }


def validate_frozen_window(energies: list[list[float]], window_min: float, window_max: float, num_wann: int) -> dict[str, Any]:
    counts = state_counts(energies, window_min, window_max)
    maximum = max(counts) if counts else 0
    return {
        "valid": maximum <= num_wann,
        "minimum_states_over_k": min(counts) if counts else 0,
        "maximum_states_over_k": maximum,
        "failing_kpoint_indices": [index for index, count in enumerate(counts) if count > num_wann],
    }


def validate_window_containment(
    outer_min: float,
    outer_max: float,
    frozen_min: float,
    frozen_max: float,
) -> dict[str, Any]:
    valid = outer_min <= frozen_min <= frozen_max <= outer_max
    return {
        "valid": valid,
        "outer_eV": [outer_min, outer_max],
        "frozen_eV": [frozen_min, frozen_max],
    }


def minimum_outer_max(energies: list[list[float]], window_min: float, num_wann: int, margin_eV: float = 0.02) -> float:
    thresholds: list[float] = []
    for index, row in enumerate(energies):
        available = sorted(energy for energy in row if energy >= window_min)
        if len(available) < num_wann:
            raise ValueError(f"k 点 {index} 在外窗下界之上只有 {len(available)} 个状态，小于 NUM_WANN={num_wann}。")
        thresholds.append(available[num_wann - 1])
    return math.ceil((max(thresholds) + margin_eV) * 1000.0) / 1000.0


def suggest_windows(
    energies: list[list[float]],
    num_wann: int,
    target_min: float,
    target_max: float,
    margin_eV: float = 0.02,
) -> list[dict[str, Any]]:
    """Generate a finite, deterministic seed grid; this does not claim fit quality."""
    candidates: list[dict[str, Any]] = []
    for outer_min in (target_min - 2.0, target_min - 1.0, target_min):
        try:
            required_max = minimum_outer_max(energies, outer_min, num_wann, margin_eV)
        except ValueError as exc:
            candidates.append({"outer_min_eV": outer_min, "valid": False, "reason": str(exc)})
            continue
        outer_max = max(required_max, target_max)
        outer = validate_outer_window(energies, outer_min, outer_max, num_wann)
        frozen_options = []
        for frozen_max in (target_max, target_max - 0.5, target_max - 1.0):
            frozen_options.append({
                "frozen_min_eV": target_min,
                "frozen_max_eV": frozen_max,
                **validate_frozen_window(energies, target_min, frozen_max, num_wann),
            })
        candidates.append({
            "outer_min_eV": outer_min,
            "outer_max_eV": outer_max,
            "valid": outer["valid"],
            "outer_check": outer,
            "frozen_options": frozen_options,
        })
    return candidates


def suggest_fixed_deep_band_exclusion(
    energies: list[list[float]], occupied: int, num_wann: int,
    target_min: float, target_max: float, margin_eV: float = 0.5,
    isolation_eV: float = 0.05,
) -> dict[str, Any] | None:
    """Propose a separate subspace control with constant retained occupation.

    Only an index-stable block strictly below the target may be removed.
    Freeze all retained valence and the original target; preserve every quality
    gate. This function never edits files, slices matrices, or runs a fit.
    All energies and returned bounds use the same frame as the input.
    """
    if (not energies or occupied < 1 or num_wann < 1
            or not math.isfinite(margin_eV) or margin_eV <= 0
            or not math.isfinite(isolation_eV) or isolation_eV <= 0):
        return None
    nbands = len(energies[0])
    if (occupied >= nbands or num_wann > nbands or target_min > target_max
            or not all(math.isfinite(x) for x in [target_min, target_max])
            or any(len(row) != nbands or any(not math.isfinite(x) for x in row)
                   or any(a > b for a, b in zip(row, row[1:])) for row in energies)):
        return None
    if min(row[occupied] for row in energies) <= max(row[occupied - 1] for row in energies):
        return None
    choices = []
    for count in range(1, min(occupied, nbands - num_wann + 1)):
        removed_top = max(row[count - 1] for row in energies)
        retained_bottom = min(row[count] for row in energies)
        if (removed_top < target_min - margin_eV
                and retained_bottom - removed_top >= isolation_eV
                and occupied - count < num_wann):
            choices.append(count)
    if not choices:
        return None
    count = max(choices)
    lower = min(row[count] for row in energies) - margin_eV
    frozen_upper = max(target_max, max(row[occupied - 1] for row in energies))
    if max(sum(lower <= e <= frozen_upper for e in row[count:]) for row in energies) > num_wann:
        return None
    upper = max(frozen_upper, max(row[count + num_wann - 1] for row in energies)) + margin_eV
    return {
        "exclude_bands_1_based": list(range(1, count + 1)),
        "retained_occupied_bands": occupied - count,
        "remaining_dft_bands": nbands - count,
        "outer_window_eV": [lower, upper],
        "frozen_window_eV": [lower, frozen_upper],
        "input_target_energy_eV": [target_min, target_max],
        "removed_highest_energy_eV": max(row[count - 1] for row in energies),
        "energy_reference": "same_as_input_eigenvalues",
        "requires_new_protocol": True,
        "status": "proposal_only; independent_refit_and_original_quality_gates_required",
    }
