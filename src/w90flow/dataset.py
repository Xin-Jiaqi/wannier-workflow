from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REQUIRED_ATTEMPT_FIELDS = (
    "material_id",
    "structure_hash",
    "candidate",
    "projection",
    "num_wann",
    "soc",
    "status",
)


def validate_attempt_record(record: dict[str, Any]) -> list[str]:
    issues = [f"缺少 {field}" for field in REQUIRED_ATTEMPT_FIELDS if field not in record]
    if "projection" in record and not isinstance(record["projection"], list):
        issues.append("projection 必须是列表")
    if "status" in record and record["status"] not in {"PASS", "FAIL", "CANCELLED"}:
        issues.append("status 必须是 PASS/FAIL/CANCELLED")
    if record.get("status") == "FAIL" and not record.get("failure_class"):
        issues.append("失败样本必须记录 failure_class")
    return issues


def make_attempt_record(**fields: Any) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    issues = validate_attempt_record(record)
    if issues:
        raise ValueError("; ".join(issues))
    return record

