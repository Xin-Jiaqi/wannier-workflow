from __future__ import annotations

from typing import Any


DEGENERACY = {"s": 1, "p": 3, "d": 5, "f": 7}


def _projection_order(label: str, element_order: list[str]) -> tuple[int, int, str]:
    element, orbital = label.split(":", 1)
    try:
        element_index = element_order.index(element)
    except ValueError:
        element_index = len(element_order)
    orbital_index = list(DEGENERACY).index(orbital) if orbital in DEGENERACY else len(DEGENERACY)
    return element_index, orbital_index, label


def num_wann_from_projection(projection: list[str], element_counts: dict[str, int], soc: bool) -> int:
    spatial = 0
    for label in projection:
        try:
            element, orbital = label.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"投影必须是 Element:orbital：{label}") from exc
        if element not in element_counts:
            raise ValueError(f"投影元素不在 POSCAR 中：{element}")
        if orbital not in DEGENERACY:
            raise ValueError(f"暂不支持轨道：{orbital}")
        spatial += int(element_counts[element]) * DEGENERACY[orbital]
    return spatial * (2 if soc else 1)


def suggest_projection_candidates(
    orbital_scores: dict[str, float],
    element_counts: dict[str, int],
    soc: bool,
    chemical_valence: list[str] | None = None,
    coverage_targets: tuple[float, ...] = (0.80, 0.90, 0.95),
    relevant_fraction: float = 0.01,
    max_num_wann: int | None = None,
    max_candidates: int | None = None,
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
    element_order = list(element_counts)
    normalized = {label: float(value) / total for label, value in orbital_scores.items()}
    ordered = sorted(normalized, key=lambda label: (-normalized[label], _projection_order(label, element_order)))

    raw: list[tuple[str, list[str], float | None]] = []
    for target in coverage_targets:
        selected: list[str] = []
        cumulative = 0.0
        for label in ordered:
            selected.append(label)
            cumulative += normalized[label]
            if cumulative + 1e-12 >= target:
                break
        raw.append((f"weight-{int(round(target * 100))}", selected, cumulative))
    # Always retain a complete s/p shell candidate when p character is present.
    # This is important for optical-response completeness even if s weight in the
    # target window is small, and does not require a hand-written valence list.
    sp_shells = [label for label in ordered if label.split(":", 1)[1] in {"s", "p"}]
    if any(label.endswith(":p") for label in sp_shells):
        raw.append(("complete-sp", sp_shells, sum(normalized.get(label, 0.0) for label in sp_shells)))
    if chemical_valence:
        raw.append(("chemical-valence", list(chemical_valence), sum(normalized.get(label, 0.0) for label in chemical_valence)))
        expanded = list(dict.fromkeys([*chemical_valence, *sp_shells]))
        raw.append(("expanded-valence", expanded, sum(normalized.get(label, 0.0) for label in expanded)))
    relevant = [label for label in ordered if normalized[label] >= relevant_fraction]
    raw.append(("all-relevant", relevant or ordered, sum(normalized[label] for label in (relevant or ordered))))

    priority = {
        "weight-80": 0,
        "chemical-valence": 1,
        "complete-sp": 2,
        "weight-90": 3,
        "weight-95": 4,
        "expanded-valence": 5,
        "all-relevant": 6,
    }
    raw.sort(key=lambda item: (priority.get(item[0], 99), item[0]))
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for name, labels, _coverage in raw:
        projection = sorted(set(labels), key=lambda label: _projection_order(label, element_order))
        key = tuple(projection)
        if key in seen:
            continue
        num_wann = num_wann_from_projection(projection, element_counts, soc)
        if max_num_wann is not None and num_wann > max_num_wann:
            continue
        seen.add(key)
        actual_coverage = sum(normalized.get(label, 0.0) for label in projection)
        results.append({
            "candidate": name,
            "projection": projection,
            "integrated_weight_fraction": actual_coverage,
            "num_wann": num_wann,
            "soc": soc,
        })
        if max_candidates is not None and len(results) >= max_candidates:
            break
    return results
