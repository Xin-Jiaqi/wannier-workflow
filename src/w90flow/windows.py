from __future__ import annotations

import math
from typing import Any, Iterable


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
