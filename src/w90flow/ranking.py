from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    mae_mev: float = 5.0
    rmse_mev: float = 10.0
    p95_mev: float = 20.0
    max_mev: float = 50.0
    gap_error_mev: float = 10.0


def normalized_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    gap = metrics.get("abs_gap_error_mev", metrics.get("gap_error_mev", float("inf")))
    return {
        "mae_mev": float(metrics.get("mae_mev", float("inf"))),
        "rmse_mev": float(metrics.get("rmse_mev", float("inf"))),
        "p95_mev": float(metrics.get("p95_mev", float("inf"))),
        "max_mev": float(metrics.get("max_mev", float("inf"))),
        "abs_gap_error_mev": abs(float(gap)),
    }


def passes_hard_gates(metrics: dict[str, Any], thresholds: Thresholds) -> bool:
    values = normalized_metrics(metrics)
    return (
        values["mae_mev"] <= thresholds.mae_mev
        and values["rmse_mev"] <= thresholds.rmse_mev
        and values["p95_mev"] <= thresholds.p95_mev
        and values["max_mev"] <= thresholds.max_mev
        and values["abs_gap_error_mev"] <= thresholds.gap_error_mev
    )


def rank_common_candidates(
    candidates: dict[str, dict[str, Any]],
    structure_ids: list[str],
    thresholds: Thresholds,
) -> dict[str, Any]:
    """Rank candidates by the worst member; all structures must share one candidate."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate, payload in candidates.items():
        structures = payload.get("structures", {})
        missing = [sid for sid in structure_ids if sid not in structures]
        if missing:
            rejected.append({"candidate": candidate, "reason": "missing_structures", "missing": missing})
            continue
        per_structure = {sid: normalized_metrics(structures[sid].get("metrics", structures[sid])) for sid in structure_ids}
        failed = [sid for sid in structure_ids if not passes_hard_gates(per_structure[sid], thresholds)]
        if failed:
            rejected.append({"candidate": candidate, "reason": "hard_gate", "structures": failed})
            continue
        worst = {key: max(values[key] for values in per_structure.values()) for key in next(iter(per_structure.values()))}
        accepted.append({
            "candidate": candidate,
            "projection": payload.get("projection", []),
            "num_wann": payload.get("num_wann"),
            "nocc": payload.get("nocc"),
            "pair_worst_metrics": worst,
            "structures": per_structure,
        })
    accepted.sort(key=lambda item: (
        item["pair_worst_metrics"]["p95_mev"],
        item["pair_worst_metrics"]["max_mev"],
        item["pair_worst_metrics"]["rmse_mev"],
        item["pair_worst_metrics"]["mae_mev"],
    ))
    for rank, item in enumerate(accepted, 1):
        item["rank"] = rank
    return {"ranking": accepted, "rejected": rejected}


def audit_ranking_document(document: dict[str, Any]) -> dict[str, Any]:
    joint = document.get("joint_common_basis")
    is_joint_v04 = isinstance(joint, dict)
    if is_joint_v04:
        entries = joint.get("ranking")
    else:
        entries = document.get("rankings", document.get("ranking"))
    if not isinstance(entries, list) or not entries:
        label = "joint_common_basis/ranking" if is_joint_v04 else "ranking/rankings"
        return {"passed": False, "issues": [f"{label} 列表缺失"]}
    issues: list[str] = []
    ranks = [entry.get("rank") for entry in entries]
    if ranks != list(range(1, len(entries) + 1)):
        issues.append("rank 不是从 1 开始的连续整数")
    seen: set[str] = set()
    for entry in entries:
        name = entry.get("candidate")
        if not name:
            issues.append("候选缺少 candidate 名称")
        elif name in seen:
            issues.append(f"候选重复：{name}")
        seen.add(name)
        if not isinstance(entry.get("projection"), list) or not entry.get("projection"):
            issues.append(f"{name} 缺少 projection")
        if not isinstance(entry.get("num_wann"), int) or entry.get("num_wann", 0) <= 0:
            issues.append(f"{name} 的 num_wann 非法")
        if is_joint_v04:
            if not isinstance(entry.get("attempt"), str) or not entry.get("attempt"):
                issues.append(f"{name} 缺少共同 attempt")
            if not isinstance(entry.get("nocc_model"), int) or entry.get("nocc_model", 0) <= 0:
                issues.append(f"{name} 的 nocc_model 非法")
            if entry.get("passed") is not True:
                issues.append(f"{name} 在共同通过排名中却未标记 PASS")
            if not isinstance(entry.get("pair_worst_metrics"), dict):
                issues.append(f"{name} 缺少 pair_worst_metrics")
            members = entry.get("structures")
            if not isinstance(members, dict):
                issues.append(f"{name} 缺少 structures")
            else:
                expected = set(document.get("structures", {}))
                missing = sorted(expected - set(members))
                if missing:
                    issues.append(f"{name} 缺少共同结构：{','.join(missing)}")

    if is_joint_v04:
        top_k = joint.get("top_k")
        if not isinstance(top_k, list) or not top_k:
            issues.append("joint_common_basis/top_k 列表缺失")
        else:
            ranking_keys = [(entry.get("candidate"), entry.get("attempt")) for entry in entries[:len(top_k)]]
            top_keys = [(entry.get("candidate"), entry.get("attempt")) for entry in top_k]
            if top_keys != ranking_keys:
                issues.append("top_k 不是共同排名的连续前缀")
        recommendation = joint.get("optical_basis_recommendation")
        if recommendation is not None:
            if not isinstance(recommendation, dict) or recommendation.get("candidate") not in seen:
                issues.append("optical_basis_recommendation 不在共同通过排名中")

    return {
        "passed": not issues,
        "format": "joint-v04" if is_joint_v04 else "legacy",
        "candidate_count": len(entries),
        "rank1": entries[0].get("candidate"),
        "issues": issues,
    }
