from __future__ import annotations

from itertools import combinations
import math
from typing import Any


DEGENERACY = {"s": 1, "p": 3, "d": 5, "f": 7}
COMPONENT_DEGENERACY = {
    "px": 1, "py": 1, "pz": 1,
    "dxy": 1, "dyz": 1, "dz2": 1, "dxz": 1, "dx2": 1, "dx2-y2": 1,
}
ORBITAL_DEGENERACY = {**DEGENERACY, **COMPONENT_DEGENERACY}


def candidate_dimension_budget(
    candidates: list[dict[str, Any]], nbands: int, fraction: float,
) -> dict[str, Any]:
    """Explain projection exclusions without changing the configured search.

    An exhausted narrow basis is different from an adequately explored model
    space. Keep the original cap; report the DFT budget needed for each excluded
    projection so a subsequent, separately recorded control can address it.
    """
    if isinstance(nbands, bool) or not isinstance(nbands, int) or nbands <= 0:
        raise ValueError("nbands must be a positive integer")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("candidate fraction must be in (0, 1]")
    limit = math.floor(nbands * fraction)
    excluded = []
    for candidate in candidates:
        dimension = int(candidate["num_wann"])
        if dimension > limit:
            excluded.append({
                "candidate": candidate["candidate"],
                "projection": candidate["projection"],
                "num_wann": dimension,
                "reason": "insufficient_dft_bands" if dimension > nbands else "fraction_cap_only",
                "minimum_nbands_at_current_fraction": math.ceil(dimension / fraction),
            })
    return {"nbands": nbands, "fraction": fraction, "maximum_num_wann": limit,
            "excluded_candidates": excluded,
            "policy": "diagnostic_only; no automatic cap or quality-gate change"}


def _projection_order(label: str, element_order: list[str]) -> tuple[int, int, str]:
    element, orbital = label.split(":", 1)
    try:
        element_index = element_order.index(element)
    except ValueError:
        element_index = len(element_order)
    orbital_order = list(ORBITAL_DEGENERACY)
    orbital_index = orbital_order.index(orbital) if orbital in ORBITAL_DEGENERACY else len(orbital_order)
    return element_index, orbital_index, label


def normalize_projection(projection: list[str]) -> list[str]:
    """Remove component labels already covered by a complete shell.

    This makes a union of independent external suggestions mathematically a
    union of subspaces, not a double count of e.g. ``In:p`` and ``In:px``.
    """
    labels = list(dict.fromkeys(projection))
    complete = set(labels)
    component_shell = {
        "px": "p", "py": "p", "pz": "p",
        "dxy": "d", "dyz": "d", "dz2": "d", "dxz": "d",
        "dx2": "d", "dx2-y2": "d",
    }
    return [
        label for label in labels
        if (
            (orbital := label.split(":", 1)[1]) not in component_shell
            or f"{label.split(':', 1)[0]}:{component_shell[orbital]}" not in complete
        )
    ]


def num_wann_from_projection(projection: list[str], element_counts: dict[str, int], soc: bool) -> int:
    spatial = 0
    for label in normalize_projection(projection):
        try:
            element, orbital = label.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"投影必须是 Element:orbital：{label}") from exc
        if element not in element_counts:
            raise ValueError(f"投影元素不在 POSCAR 中：{element}")
        if orbital not in ORBITAL_DEGENERACY:
            raise ValueError(f"暂不支持轨道：{orbital}")
        spatial += int(element_counts[element]) * ORBITAL_DEGENERACY[orbital]
    return spatial * (2 if soc else 1)


def suggest_projection_candidates(
    orbital_scores: dict[str, float],
    element_counts: dict[str, int],
    soc: bool,
    chemical_valence: list[str] | None = None,
    coverage_targets: tuple[float, ...] = (0.80, 0.90, 0.95),
    relevant_fraction: float = 0.01,
    shell_family_min_weight: float = 0.08,
    shell_combination_min_weight: float = 0.45,
    max_num_wann: int | None = None,
    max_candidates: int | None = None,
    external_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create a finite, chemically structured candidate family.

    Coverage candidates are nested with each other. Chemical-valence and complete
    shell candidates are separate branches: forcing every branch into one global
    chain can create an unnecessarily huge model. These are trial seeds, not
    accepted models; real Wannier quality gates determine the final ranking.
    """
    if not orbital_scores or any(float(value) < 0 for value in orbital_scores.values()):
        raise ValueError("orbital_scores 必须是非负且非空。")
    total = sum(float(value) for value in orbital_scores.values())
    if total <= 0:
        raise ValueError("orbital_scores 总和必须大于零。")
    if not 0.0 <= shell_family_min_weight <= 1.0:
        raise ValueError("shell_family_min_weight 必须在 [0, 1]。")
    if not 0.0 <= shell_combination_min_weight <= 1.0:
        raise ValueError("shell_combination_min_weight 必须在 [0, 1]。")
    element_order = list(element_counts)
    normalized = {label: float(value) / total for label, value in orbital_scores.items()}
    ordered = sorted(normalized, key=lambda label: (-normalized[label], _projection_order(label, element_order)))

    raw: list[tuple[str, list[str], float | None, str]] = []
    for target in coverage_targets:
        selected: list[str] = []
        cumulative = 0.0
        for label in ordered:
            selected.append(label)
            cumulative += normalized[label]
            if cumulative + 1e-12 >= target:
                break
        raw.append((f"weight-{int(round(target * 100))}", selected, cumulative, "weight_coverage"))

    for index, external in enumerate(external_candidates or (), 1):
        if not isinstance(external, dict) or not isinstance(external.get("projection"), list):
            raise ValueError("external_candidates 每项都必须含 projection 列表。")
        name = str(external.get("candidate") or f"external-{index:02d}")
        labels = [str(label) for label in external["projection"]]
        raw.append((
            name,
            labels,
            sum(normalized.get(label, 0.0) for label in labels),
            str(external.get("candidate_family") or "external_baseline"),
        ))

    # Enumerate angular-momentum families from the PROCAR labels themselves.
    # The old p-only/complete-sp branch is therefore a special case of a general
    # s/p/d/f shell-Pareto family, not a material-name rule.
    observed_by_shell = {
        shell: [label for label in ordered if label.split(":", 1)[1] == shell]
        for shell in DEGENERACY
    }
    shell_weights_all = {
        shell: sum(normalized.get(label, 0.0) for label in labels)
        for shell, labels in observed_by_shell.items()
        if labels
    }
    by_shell = {
        shell: labels for shell, labels in observed_by_shell.items()
        if labels and shell_weights_all[shell] >= shell_family_min_weight
    }
    shell_weights = {shell: shell_weights_all[shell] for shell in by_shell}
    shell_order = sorted(by_shell, key=lambda shell: (-shell_weights[shell], list(DEGENERACY).index(shell)))
    for shell in shell_order:
        name = "p-only" if shell == "p" else f"{shell}-only"
        raw.append((name, by_shell[shell], shell_weights[shell], "single_shell"))

    observed_shells = [shell for shell in DEGENERACY if observed_by_shell[shell]]
    for size in range(2, len(observed_shells) + 1):
        for shell_group in combinations(observed_shells, size):
            coverage = sum(shell_weights_all[shell] for shell in shell_group)
            if coverage < shell_combination_min_weight:
                continue
            labels = [label for label in ordered if label.split(":", 1)[1] in shell_group]
            name = "complete-" + "".join(shell_group)
            raw.append((
                name,
                labels,
                coverage,
                "complete_shell_combination",
            ))
    if chemical_valence:
        raw.append(("chemical-valence", list(chemical_valence), sum(normalized.get(label, 0.0) for label in chemical_valence), "explicit_optional_prior"))
        all_observed = [label for label in ordered if normalized[label] > 0]
        expanded = list(dict.fromkeys([*chemical_valence, *all_observed]))
        raw.append(("expanded-valence", expanded, sum(normalized.get(label, 0.0) for label in expanded), "explicit_optional_prior"))
    relevant = [label for label in ordered if normalized[label] >= relevant_fraction]
    raw.append(("all-relevant", relevant or ordered, sum(normalized[label] for label in (relevant or ordered)), "relevance_threshold"))
    raw.append(("complete-observed", ordered, 1.0, "all_observed_shells"))

    priority = {
        "weight-80": 0,
        "chemical-valence": 1,
        "weight-90": 3,
        "weight-95": 4,
        "expanded-valence": 5,
        "all-relevant": 6,
        "complete-observed": 7,
    }
    raw.sort(key=lambda item: (
        priority.get(
            item[0],
            2 if item[3] in {"single_shell", "complete_shell_combination", "external_baseline"} else 99,
        ),
        item[0],
    ))
    prepared: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for name, labels, _coverage, family in raw:
        projection = sorted(set(normalize_projection(labels)), key=lambda label: _projection_order(label, element_order))
        key = tuple(projection)
        if key in seen:
            continue
        num_wann = num_wann_from_projection(projection, element_counts, soc)
        if max_num_wann is not None and num_wann > max_num_wann:
            continue
        seen.add(key)
        component_shell = {
            "px": ("p", 3), "py": ("p", 3), "pz": ("p", 3),
            "dxy": ("d", 5), "dyz": ("d", 5), "dz2": ("d", 5),
            "dxz": ("d", 5), "dx2": ("d", 5), "dx2-y2": ("d", 5),
        }
        approximate = False
        actual_coverage = 0.0
        for label in projection:
            if label in normalized:
                actual_coverage += normalized[label]
                continue
            element, orbital = label.split(":", 1)
            fallback = component_shell.get(orbital)
            if fallback is not None:
                shell, divisor = fallback
                actual_coverage += normalized.get(f"{element}:{shell}", 0.0) / divisor
                approximate = True
        prepared.append({
            "candidate": name,
            "projection": projection,
            "integrated_weight_fraction": actual_coverage,
            "integrated_weight_estimate": "equal_component_split" if approximate else "direct_shell_total",
            "num_wann": num_wann,
            "soc": soc,
            "candidate_family": family,
        })

    def trial_order(item: dict[str, Any]) -> tuple[int, int, float, str]:
        name = str(item["candidate"])
        family = str(item["candidate_family"])
        if name == "weight-80":
            group = 0
        elif family in {"single_shell", "complete_shell_combination", "external_baseline"}:
            group = 1
        elif name in {"weight-90", "weight-95"}:
            group = 2
        else:
            group = 3
        return (group, int(item["num_wann"]), -float(item["integrated_weight_fraction"]), name)

    prepared.sort(key=trial_order)
    return prepared if max_candidates is None else prepared[:max_candidates]
