"""Single-material Stage-1 production entry (AI-free, deterministic).

``run-material`` is a thin orchestrator on top of the existing coldstart and
state-machine APIs.  It adds:

* one command per material: POSCAR + material-id + profile config + work-root;
* a machine-readable manifest with input fingerprints and per-stage reuse;
* a terminal taxonomy: TOP3_READY / PARTIAL_TOPK / NO_MODEL / TECHNICAL_FAILED;
* exit codes decoupled from scientific success (exit 0 != scientific success).

It never reimplements SCF/band/candidate/Wannier/quality/rank logic and never
submits PBS jobs itself: execution is delegated to
``coldstart.run_stage1_coldstart`` and the existing supervisor state machine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .coldstart import (
    coldstart_signature,
    derived_preprocess_run,
    derived_wannier_run,
    plan_stage1_coldstart,
    run_stage1_coldstart,
    verify_reuse_source,
)
from .config import load_project
from .orchestrator import CONVERGENCE_FAILURES, STAGE_NAME, WINDOW_FAILURES
from .parsers import parse_poscar
from .state import atomic_json, read_json, read_status

TERMINAL_STATUSES = ("TOP3_READY", "PARTIAL_TOPK", "NO_MODEL", "TECHNICAL_FAILED")
SCIENTIFIC_TERMINAL_STATUSES = ("TOP3_READY", "PARTIAL_TOPK", "NO_MODEL")
MANIFEST_FILE = "run-material-manifest.json"
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_TECHNICAL = 2

SCIENTIFIC_FAILURES = frozenset(
    WINDOW_FAILURES | CONVERGENCE_FAILURES | {"QUALITY_GATE_FAILED"}
)
_STAGE_SHORT = {value: key for key, value in STAGE_NAME.items()}
# "数值失败" 不再是候选耗尽标记：NUMERIC_FAILURE/INTERFACE_FAILURE 阻断代表
# 需要人工/环境诊断，属于 TECHNICAL_FAILED，绝不能误报 NO_MODEL/CANDIDATES_EXHAUSTED。
_COVERAGE_EXHAUSTED_MARKERS = ("仍没有覆盖", "有限候选")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reuse_report(
    config: dict[str, Any],
    plan: dict[str, Any],
    runs: Path,
    signature_sha256: str,
) -> dict[str, Any]:
    """Read-only reuse conditions: which existing runs/stages will be reused."""
    report: dict[str, Any] = {"preprocess": [], "wannier": {"run_dir": None, "stages": {}}, "export": None}
    for item in plan["structures"]:
        sid = str(item["id"])
        run_dir = derived_preprocess_run(
            runs,
            config["project"],
            sid,
            str(item["poscar_sha256"]),
            signature_sha256,
        )
        entry: dict[str, Any] = {"structure": sid, "run_dir": str(run_dir), "status": "CREATE"}
        if run_dir.is_dir():
            entry["status"] = (
                "REUSE"
                if (run_dir / "protocol.json").is_file() and (run_dir / "input-lock.json").is_file()
                else "PARTIAL_VERIFY"
            )
        report["preprocess"].append(entry)

    wannier = derived_wannier_run(runs, config["project"], signature_sha256)
    if wannier.is_dir():
        report["wannier"]["run_dir"] = str(wannier)
        for stage in read_status(wannier)["stages"]:
            name = _STAGE_SHORT.get(str(stage["stage"]), str(stage["stage"]))
            status = str(stage["status"])
            if status == "PASS":
                report["wannier"]["stages"][name] = "REUSE"
            elif status == "FAIL":
                report["wannier"]["stages"][name] = "BLOCKED_EXISTING_FAILURE"
            elif status == "RUNNING":
                report["wannier"]["stages"][name] = "RECONCILE_OR_RECOVER_ONCE"
            else:
                report["wannier"]["stages"][name] = "EXECUTE"
        export = wannier / "07_export"
        if export.is_dir():
            summaries = sorted(export.glob("*/export-summary.json"))
            packages = None
            if summaries:
                payload = read_json(summaries[0])
                packages = len(payload.get("packages", [])) if isinstance(payload.get("packages"), list) else None
            report["export"] = {"path": str(export), "packages": packages}
    return report


def material_plan(
    poscar: str | Path,
    material_id: str,
    config_path: str | Path,
    work_root: str | Path,
) -> dict[str, Any]:
    """Read-only validation plus stage/reuse plan. Never creates state."""
    poscar_path = Path(poscar).expanduser().resolve()
    if not poscar_path.is_file():
        raise ValueError(f"POSCAR 不存在：{poscar_path}")
    parse_poscar(poscar_path)
    config_file = Path(config_path).expanduser().resolve()
    config = load_project(config_file)
    structures = config["structures"]
    if len(structures) != 1:
        raise ValueError("run-material 需要恰好一个 structure；profile 配置只能包含目标材料。")
    structure = structures[0]
    if str(structure["id"]) != material_id:
        raise ValueError(f"material-id {material_id!r} 与配置 structure id {structure['id']!r} 不一致。")
    if Path(str(structure["poscar_resolved"])).resolve() != poscar_path:
        raise ValueError("配置中的 structure.poscar 必须与命令行 POSCAR 是同一文件。")

    work = Path(work_root).expanduser().resolve()
    campaign = work / "campaigns" / material_id
    runs = work / "runs"
    plan = plan_stage1_coldstart(config_file, campaign, runs)
    _signature, signature_sha256 = coldstart_signature(config_file, plan)
    steps = [
        {"stage": "preprocess", "action": "REUSE_OR_CREATE", "detail": "relax/SCF/Band frozen protocol per profile"},
        {"stage": "scf", "action": "EXECUTE", "detail": "preprocess 02_scf 或已导入 SCF"},
        {"stage": "band", "action": "EXECUTE", "detail": "preprocess 03_band + band-handoff"},
        {"stage": "candidates", "action": "GENERATE", "detail": "从 Band EIGENVAL+PROCAR 生成投影候选"},
        {"stage": "wannier", "action": "PLAN_AND_RUN_ATTEMPTS", "detail": "finite candidate/attempt orchestration"},
        {"stage": "quality", "action": "EVALUATE", "detail": "统一目标能区硬门槛"},
        {"stage": "rank", "action": "RANK", "detail": f"top-{int(config['ranking'].get('output_top_k', 3))} 排序"},
        {"stage": "export", "action": "EXPORT", "detail": "不含 POTCAR 的最小包"},
    ]
    return {
        "schema_version": 1,
        "mode": "DRY_RUN",
        "material_id": material_id,
        "project": config["project"],
        "config": str(config_file),
        "config_sha256": plan["config_sha256"],
        "poscar": str(poscar_path),
        "poscar_sha256": str(plan["structures"][0]["poscar_sha256"]),
        "input_signature_sha256": signature_sha256,
        "campaign_dir": str(campaign),
        "runs_root": str(runs),
        "requested_top_k": int(config["ranking"].get("output_top_k", 3)),
        "terminal": False,
        "status": "DRY_RUN",
        "exit_code": EXIT_OK,
        "stage_plan": steps,
        "reuse": _reuse_report(config, plan, runs, signature_sha256),
        "structures": plan["structures"],
        "immutable_input_sha256": plan["immutable_input_sha256"],
        "preflight": plan["preflight"],
        "sequence": plan["sequence"],
    }


def count_accepted(run_dir: str | Path) -> tuple[int, list[str]]:
    """Count accepted ranked models: exported packages are authoritative."""
    root = Path(run_dir).expanduser().resolve()
    export = root / "07_export"
    if export.is_dir():
        summaries = sorted(export.glob("*/export-summary.json"))
        if summaries:
            payload = read_json(summaries[0])
            packages = payload.get("packages")
            if isinstance(packages, list):
                return len(packages), [
                    f"{str(pkg.get('candidate'))}/{str(pkg.get('attempt'))}"
                    for pkg in packages
                    if isinstance(pkg, dict)
                ]
    accepted: dict[str, str] = {}
    structures = root / "structures"
    if structures.is_dir():
        for sid_dir in sorted(structures.iterdir()):
            attempts = sid_dir / "04_wannier_attempts"
            if attempts.is_dir():
                for quality_path in sorted(attempts.glob("*/attempt-*/quality.json")):
                    quality = read_json(quality_path)
                    if quality.get("passed"):
                        accepted.setdefault(str(quality.get("candidate")), str(quality.get("attempt")))
            baseline = sid_dir / "05_quality/quality.json"
            if baseline.is_file() and read_json(baseline).get("passed"):
                accepted["baseline"] = "baseline"
    return len(accepted), [f"{candidate}/{attempt}" for candidate, attempt in sorted(accepted.items())]


def _wannier_stage_failure(root: Path) -> dict[str, Any]:
    stage = root / "04_wannier/stage.json"
    if not stage.is_file():
        return {"failed": False}
    state = read_json(stage)
    if state.get("status") != "FAIL":
        return {"failed": False}
    return {"failed": True, "failure_class": str(state.get("failure_class") or "")}


def _scientific_exhaustion(root: Path) -> bool:
    """True when the pipeline reached its scientific end (no remaining attempts)."""
    failure = _wannier_stage_failure(root)
    if failure["failed"]:
        failure_class = failure["failure_class"]
        if failure_class in SCIENTIFIC_FAILURES:
            return True
    summary = root / "workflow-summary.json"
    if summary.is_file():
        reason = str(read_json(summary).get("reason") or "")
        if any(marker in reason for marker in _COVERAGE_EXHAUSTED_MARKERS):
            return True
    return False


def classify_terminal(
    coldstart_result: dict[str, Any],
    wannier_run: Path | None,
    requested_top_k: int,
) -> dict[str, Any]:
    """Map coldstart/supervisor outcome to the production terminal taxonomy."""
    accepted_count, accepted = (
        count_accepted(wannier_run) if wannier_run is not None and wannier_run.is_dir() else (0, [])
    )
    status = str(coldstart_result.get("status", ""))
    if status == "COMPLETE":
        if accepted_count >= requested_top_k:
            terminal, failure_class, reason = "TOP3_READY", None, None
        elif accepted_count >= 1:
            terminal, failure_class, reason = "PARTIAL_TOPK", None, "导出项少于 requested_top_k"
        else:
            terminal, failure_class, reason = "NO_MODEL", None, "全部候选未通过质量门槛"
    elif wannier_run is not None and wannier_run.is_dir() and _scientific_exhaustion(wannier_run):
        if accepted_count >= 1:
            terminal, failure_class, reason = "PARTIAL_TOPK", None, "候选耗尽，仅部分候选通过质量门槛"
        else:
            terminal, failure_class, reason = "NO_MODEL", "CANDIDATES_EXHAUSTED", "候选耗尽且无合格 Wannier 模型"
    else:
        terminal, failure_class, reason = "TECHNICAL_FAILED", "INFRASTRUCTURE_OR_PROGRAM", (
            f"未到达科学终态（coldstart={status}）；请检查 supervisor 状态与日志"
        )
    return {
        "coldstart_status": status,
        "accepted_count": accepted_count,
        "accepted": accepted,
        "requested_top_k": requested_top_k,
        "terminal": terminal,
        "failure_class": failure_class,
        "reason": reason,
    }


def _terminal_report(manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    status = str(manifest.get("status"))
    return {
        "schema_version": 1,
        "mode": "EXECUTE",
        "material_id": plan["material_id"],
        "status": status,
        "terminal": bool(manifest.get("terminal")),
        "accepted_count": manifest.get("accepted_count"),
        "accepted": manifest.get("accepted"),
        "requested_top_k": plan["requested_top_k"],
        "failure_class": manifest.get("failure_class"),
        "reason": manifest.get("reason"),
        "coldstart_status": manifest.get("coldstart_status"),
        "wannier_run": manifest.get("wannier_run"),
        "campaign_dir": plan["campaign_dir"],
        "runs_root": plan["runs_root"],
        "manifest": str(Path(plan["campaign_dir"]) / MANIFEST_FILE),
        "exit_code": EXIT_TECHNICAL if status == "TECHNICAL_FAILED" else EXIT_OK,
    }


def run_material(
    poscar: str | Path,
    material_id: str,
    config_path: str | Path,
    work_root: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    poll_seconds: float = 30.0,
    python: str | None = None,
    reuse_from: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministic single-material production entry.

    ``--dry-run`` is read-only: it returns the stage plan, input fingerprints
    and reuse conditions and creates no state.  ``--resume`` continues an
    existing non-terminal campaign; PASS stages are never rerun (delegated to
    the coldstart state machine).  ``--reuse-from`` is an explicit
    technical-recovery switch (default off): it reuses hash-identical PASS
    preprocess/SCF/band (and PASS 03_candidates when the prior run is a Stage1
    Wannier run) from a prior run, read-only, with provenance.  Failed
    04_wannier and later stages are never reused.  Exit code 0 does not mean
    scientific success: the machine-readable ``status`` is authoritative.
    """
    plan = material_plan(poscar, material_id, config_path, work_root)
    reuse = None
    prior_run_path: str | None = None
    if reuse_from is not None:
        config_file_for_reuse = Path(config_path).expanduser().resolve()
        prior = Path(reuse_from).expanduser().resolve()
        prior_run_path = str(prior)
        reuse = verify_reuse_source(prior, config_file_for_reuse, plan)
    if dry_run:
        if reuse is not None:
            plan = {
                **plan,
                "reuse_from": prior_run_path,
                "reuse_verification": reuse,
            }
        return plan
    if poll_seconds <= 0:
        raise ValueError("poll_seconds 必须大于 0。")
    config_file = Path(config_path).expanduser().resolve()
    work = Path(work_root).expanduser().resolve()
    campaign = work / "campaigns" / material_id
    runs = work / "runs"
    manifest_path = campaign / MANIFEST_FILE
    signature = plan["input_signature_sha256"]

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("input_signature_sha256") != signature:
            raise RuntimeError("已有 run-material manifest 与当前 POSCAR/配置指纹不同，拒绝覆盖。")
        if str(manifest.get("status")) in SCIENTIFIC_TERMINAL_STATUSES:
            return _terminal_report(manifest, plan)
        if not resume:
            raise RuntimeError(
                f"campaign 尚未完成（status={manifest.get('status')}）；确认后使用 --resume 继续。"
            )

    manifest.update({
        "schema_version": 1,
        "material_id": material_id,
        "project": plan["project"],
        "config": plan["config"],
        "config_sha256": plan["config_sha256"],
        "poscar": plan["poscar"],
        "poscar_sha256": plan["poscar_sha256"],
        "input_signature_sha256": signature,
        "campaign_dir": str(campaign),
        "runs_root": str(runs),
        "requested_top_k": plan["requested_top_k"],
        "status": "STARTED",
        "terminal": False,
        "created_utc": manifest.get("created_utc", _utc_now()),
        "updated_utc": _utc_now(),
        "stage_plan": plan["stage_plan"],
        "reuse": plan["reuse"],
        "reuse_from": prior_run_path,
        "reuse_stages": reuse["stages"] if reuse else None,
    })
    atomic_json(manifest_path, manifest)

    try:
        result = run_stage1_coldstart(
            config_file,
            campaign,
            runs,
            execute=True,
            poll_seconds=poll_seconds,
            python=python,
            reuse=reuse,
        )
    except Exception as exc:  # noqa: BLE001 - production entry records any failure
        manifest.update({
            "status": "TECHNICAL_FAILED",
            "terminal": True,
            "failure_class": type(exc).__name__,
            "reason": str(exc),
            "updated_utc": _utc_now(),
        })
        atomic_json(manifest_path, manifest)
        return _terminal_report(manifest, plan)

    wannier_run_value = result.get("wannier_run") or manifest.get("wannier_run")
    wannier_run = Path(str(wannier_run_value)).expanduser().resolve() if wannier_run_value else None
    if wannier_run is not None and wannier_run.is_dir():
        manifest["wannier_run"] = str(wannier_run)
    terminal = classify_terminal(result, wannier_run, plan["requested_top_k"])
    manifest.update({
        "status": terminal["terminal"],
        "terminal": True,
        "accepted_count": terminal["accepted_count"],
        "accepted": terminal["accepted"],
        "failure_class": terminal["failure_class"],
        "reason": terminal["reason"],
        "coldstart_status": terminal["coldstart_status"],
        "updated_utc": _utc_now(),
    })
    atomic_json(manifest_path, manifest)
    return _terminal_report(manifest, plan)
