from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ConfigError, load_project
from .coldstart import run_stage1_coldstart
from .paths import runs_root
from .attempts import create_attempts, import_completed_attempt, run_attempt
from .candidates import suggest_projection_candidates
from .executor import finalize_run, generate_candidates, import_completed_stage, prepare_run, reconcile_stage, run_vasp_stage
from .parsers import parse_eigenval, parse_poscar
from .pipeline import audit_reference, plan_project
from .pyw90_adapter import run_eig_suggest
from .quality import evaluate_attempt, evaluate_run, export_ranked_attempts, rank_attempts
from .ranking import audit_ranking_document
from .material_pool import screen_c2db, write_material_pool
from .orchestrator import WORKFLOW_STOPS, fit_from_precomputed, fit_from_preprocess, inspect_workflow, run_workflow
from .run_material import run_material
from .state import initialize_run, read_status, sha256_file
from .preprocess import (
    PREPROCESS_STAGES,
    build_preprocess_plan,
    inspect_preprocess_run,
    normalize_preprocess_config,
    prepare_preprocess_run,
    reconcile_preprocess_stage,
    recover_interrupted_stage,
    run_preprocess_workflow,
    standard_semilocal_preprocess,
)
from .structure import resolve_periodicity
from .supervisor import inspect_supervisor, supervise_workflow
from .windows import suggest_windows
from .resources import candidate_search_plan, detect_resources


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _source_mappings(values: list[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("source 必须是 STRUCTURE=/absolute/directory")
        structure, source = value.split("=", 1)
        if not structure or not source or structure in mappings:
            raise ValueError("source 结构名/目录为空或结构重复。")
        mappings[structure] = source
    return mappings


def _preprocess_project(config_path: str, poscar_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one portable YAML file and resolve its file inputs against that file."""
    import yaml  # type: ignore[import-untyped]

    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError("preprocess 配置根节点必须是对象。")
    poscar = Path(poscar_path).expanduser().resolve()
    analysis = resolve_periodicity(parse_poscar(poscar), {"dimensionality": "auto"})
    protocol = normalize_preprocess_config(document, int(analysis["dimensionality"]))
    inputs = document.get("preprocess_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("preprocess 配置缺少 preprocess_inputs。")
    resolved = json.loads(json.dumps(inputs))
    base = config_file.parent

    def resolve_file(value: str) -> str:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (base / path).resolve())

    templates = resolved.get("templates")
    if isinstance(templates, dict):
        for stage in templates.values():
            if isinstance(stage, dict):
                for role in ("incar", "kpoints"):
                    if isinstance(stage.get(role), str):
                        stage[role] = resolve_file(stage[role])
    potcar = resolved.get("potcar")
    if isinstance(potcar, dict):
        if isinstance(potcar.get("path"), str):
            potcar["path"] = resolve_file(potcar["path"])
        for item in potcar.get("components", []):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                item["path"] = resolve_file(item["path"])
    return protocol, resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="w90flow", description="AI-free VASP→Wannier workflow core")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="校验配置并显示完整阶段；不启动计算")
    plan.add_argument("config")

    init = sub.add_parser("init", help="创建不可覆盖的 run 状态目录")
    init.add_argument("config")
    init.add_argument("--runs-root", default="runs")

    prepare = sub.add_parser("prepare", help="校验私有 POTCAR 并准备 SCF/Band/Wannier 目录；不启动计算")
    prepare.add_argument("config")
    prepare.add_argument("--runs-root", default="runs")

    status = sub.add_parser("status", help="读取 run 的各阶段状态")
    status.add_argument("run_dir")

    run = sub.add_parser("run", help="从当前断点确定性推进完整流程；默认只预览下一动作")
    run.add_argument("run_dir")
    run.add_argument("--until", choices=WORKFLOW_STOPS, default="export")
    run.add_argument("--execute", action="store_true")

    resume = sub.add_parser("resume", help="安全恢复同一 run；与 run 共用状态机且拒绝重复重作业")
    resume.add_argument("run_dir")
    resume.add_argument("--until", choices=WORKFLOW_STOPS, default="export")
    resume.add_argument("--execute", action="store_true")

    fit_bands = sub.add_parser(
        "fit-from-bands",
        help="从哈希等价的既有 SCF/Band 交接文件开始，自动完成 Wannier top-K",
    )
    fit_bands.add_argument("config")
    fit_bands.add_argument("--runs-root", default="runs")
    fit_bands.add_argument("--scf-source", action="append", required=True, help="STRUCTURE=/absolute/scf-directory")
    fit_bands.add_argument("--band-source", action="append", required=True, help="STRUCTURE=/absolute/band-directory")
    fit_bands.add_argument("--until", choices=WORKFLOW_STOPS, default="export")
    fit_bands.add_argument("--execute", action="store_true")

    fit_preprocess = sub.add_parser(
        "fit-from-preprocess",
        help="核验已完成的前处理交接，并用优化后结构自动进入 Wannier top-K",
    )
    fit_preprocess.add_argument("config")
    fit_preprocess.add_argument("--runs-root", default="runs")
    fit_preprocess.add_argument("--run-dir", help="可选的确定性唯一 run 路径；必须位于 runs-root 内")
    fit_preprocess.add_argument(
        "--preprocess-source",
        action="append",
        required=True,
        help="STRUCTURE=/absolute/preprocess-run",
    )
    fit_preprocess.add_argument("--until", choices=WORKFLOW_STOPS, default="export")
    fit_preprocess_mode = fit_preprocess.add_mutually_exclusive_group()
    fit_preprocess_mode.add_argument("--execute", action="store_true")
    fit_preprocess_mode.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="只创建一次 Wannier run 并导入 SCF/Band；随后必须 resume 同一 run_dir",
    )

    coldstart = sub.add_parser(
        "stage1-coldstart",
        help="从优化好的 POSCAR 无人值守推进 SCF/Band/有限 Wannier 候选/Top-K",
    )
    coldstart.add_argument("config")
    coldstart.add_argument("--campaign-dir", required=True)
    coldstart.add_argument("--runs-root", default=None)
    coldstart.add_argument("--poll-seconds", type=float, default=30.0)
    coldstart.add_argument("--execute", action="store_true")

    run_material_parser = sub.add_parser(
        "run-material",
        help="单材料生产入口：POSCAR→SCF→Band→候选→Wannier→质量→排名→导出；"
             "统一终态 TOP3_READY/PARTIAL_TOPK/NO_MODEL/TECHNICAL_FAILED",
    )
    run_material_parser.add_argument("poscar", help="结构优化好的 POSCAR")
    run_material_parser.add_argument("material_id", help="材料 ID（必须等于配置 structure id）")
    run_material_parser.add_argument("--config", required=True, help="单结构 Stage-1 profile 配置")
    run_material_parser.add_argument("--work-root", required=True, help="runs/campaigns 根目录")
    run_material_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读：输出 stage plan、输入指纹与复用条件；不创建状态、不提交计算",
    )
    run_material_parser.add_argument(
        "--resume",
        action="store_true",
        help="继续已有非终态 campaign（成功 stage 不重跑，遵循现有 retry 语义）",
    )
    run_material_parser.add_argument("--poll-seconds", type=float, default=30.0)

    inspect = sub.add_parser("inspect", help="只读显示自动状态机的下一动作")
    inspect.add_argument("run_dir")

    supervise = sub.add_parser("supervise", help="长时间运行：提交、轮询、心跳、阶段衔接和安全断点")
    supervise.add_argument("run_dir")
    supervise.add_argument("--until", choices=WORKFLOW_STOPS, default="export")
    supervise.add_argument("--execute", action="store_true")
    supervise.add_argument("--poll-seconds", type=float, default=30.0)
    supervise.add_argument("--heartbeat-seconds", type=float, default=300.0)
    supervise.add_argument("--no-progress-seconds", type=float, default=1800.0)
    supervise.add_argument("--stage-timeout-seconds", type=float, default=86400.0)
    supervise.add_argument("--max-hours", type=float)

    supervisor_status = sub.add_parser("supervisor-status", help="只读检查监督器、child 和实际 worker")
    supervisor_status.add_argument("run_dir")

    audit = sub.add_parser("audit-reference", help="审计已有最小实例的关键交接文件")
    audit.add_argument("example_dir")

    ranking = sub.add_parser("audit-ranking", help="审计已有共同投影排名文件")
    ranking.add_argument("ranking_json")

    windows = sub.add_parser("window-suggest", help="按每个 k 点状态数生成有限能窗种子")
    windows.add_argument("eigenval")
    windows.add_argument("--num-wann", type=int, required=True)
    windows.add_argument("--target-min", type=float, required=True)
    windows.add_argument("--target-max", type=float, required=True)
    windows.add_argument("--margin", type=float, default=0.02)

    candidates = sub.add_parser("candidate-suggest", help="由目标能区积分轨道权重产生有限嵌套候选")
    candidates.add_argument("scores_json", help="含 orbital_scores、element_counts、chemical_valence 的 JSON")
    candidates.add_argument("--soc", action="store_true")

    resources = sub.add_parser("resource-plan", help="探测本机/Slurm 核数和候选并行方案；不启动计算")
    resources.add_argument("--candidates", type=int, required=True)
    resources.add_argument("--mpi-per-candidate", type=int, required=True)
    resources.add_argument("--policy", choices=("serial", "candidate_parallel"), default="serial")
    resources.add_argument("--max-workers", default="auto")

    pyw90 = sub.add_parser("pyw90-baseline", help="将 PyW90 eig suggest 转为可审计 JSON 对照")
    pyw90.add_argument("eigenval")
    pyw90.add_argument("--num-wann", type=int, required=True)
    pyw90.add_argument("--target-min", type=float, required=True)
    pyw90.add_argument("--target-max", type=float, required=True)
    pyw90.add_argument("--fermi", type=float, required=True)
    pyw90.add_argument("--energy-reference", choices=("absolute", "fermi_relative"), default="fermi_relative")
    pyw90.add_argument("--executable", default=None, help="PyW90 可执行文件；默认取 W90FLOW_PYW90_EXECUTABLE 环境变量")
    pyw90.add_argument("--output")

    run_stage = sub.add_parser("run-stage", help="默认 dry-run；加 --execute 才实际运行 VASP/Wannier90")
    run_stage.add_argument("run_dir")
    run_stage.add_argument("stage", choices=("scf", "band", "wannier"))
    run_stage.add_argument("--structure", action="append", dest="structures")
    run_stage.add_argument("--execute", action="store_true")

    reconcile = sub.add_parser("reconcile", help="根据已完成输出回收后台 RUNNING 状态；不会重复提交")
    reconcile.add_argument("run_dir")
    reconcile.add_argument("stage", choices=("scf", "band", "wannier"))
    reconcile.add_argument("--structure", action="append", dest="structures")

    import_stage = sub.add_parser("import-stage", help="仅在四个输入哈希完全一致时复用已完成 SCF/Band")
    import_stage.add_argument("run_dir")
    import_stage.add_argument("stage", choices=("scf", "band"))
    import_stage.add_argument("--source", action="append", required=True, help="STRUCTURE=/absolute/source/directory")

    generate = sub.add_parser("generate-candidates", help="从已完成 Band 的 EIGENVAL+PROCAR 生成投影候选")
    generate.add_argument("run_dir")

    evaluate = sub.add_parser("evaluate", help="评估 DFT/Wannier 能带、HR/中心、spread 并绘制比较图")
    evaluate.add_argument("run_dir")

    finalize = sub.add_parser("finalize", help="形成单结构或共同排名，并导出不含 POTCAR 的最小包")
    finalize.add_argument("run_dir")

    attempts = sub.add_parser("plan-attempts", help="为每个投影生成 exact/disentanglement 有限 attempt 和完整输入")
    attempts.add_argument("run_dir")

    run_attempt_parser = sub.add_parser("run-attempt", help="默认 dry-run；运行指定候选 attempt")
    run_attempt_parser.add_argument("run_dir")
    run_attempt_parser.add_argument("structure")
    run_attempt_parser.add_argument("candidate")
    run_attempt_parser.add_argument("attempt")
    run_attempt_parser.add_argument("--execute", action="store_true")

    import_attempt_parser = sub.add_parser(
        "import-attempt",
        help="仅在五个科学输入 SHA256 完全一致时复用已完成 Wannier 数值输出",
    )
    import_attempt_parser.add_argument("run_dir")
    import_attempt_parser.add_argument("structure")
    import_attempt_parser.add_argument("candidate")
    import_attempt_parser.add_argument("attempt")
    import_attempt_parser.add_argument("--source", required=True)

    evaluate_attempt_parser = sub.add_parser("evaluate-attempt", help="在统一目标能区评估指定候选 attempt")
    evaluate_attempt_parser.add_argument("run_dir")
    evaluate_attempt_parser.add_argument("structure")
    evaluate_attempt_parser.add_argument("candidate")
    evaluate_attempt_parser.add_argument("attempt")

    rank_attempt_parser = sub.add_parser("rank-attempts", help="分别生成试算优先级、真实拟合排名和光学基组建议")
    rank_attempt_parser.add_argument("run_dir")

    export_attempts = sub.add_parser("export-ranked-attempts", help="导出不含 POTCAR 的共同 top-K 光学最小包")
    export_attempts.add_argument("run_dir")

    material_pool = sub.add_parser("material-pool", help="以只读方式筛选 C2DB 非中心对称半导体候选")
    material_pool.add_argument("database")
    material_pool.add_argument("config", help="材料池筛选 YAML")
    material_pool.add_argument("--output-dir", required=True)

    preprocess_profile = sub.add_parser(
        "preprocess-profile",
        help="从 POSCAR+SOC 生成非磁性半局域 relax→SCF→Band 冻结协议；不启动 VASP",
    )
    preprocess_profile.add_argument("poscar")
    soc_group = preprocess_profile.add_mutually_exclusive_group(required=True)
    soc_group.add_argument("--soc", dest="soc", action="store_true")
    soc_group.add_argument("--no-soc", dest="soc", action="store_false")
    preprocess_profile.add_argument("--no-relax", action="store_true")
    preprocess_profile.add_argument("--relax-with-soc", action="store_true")
    preprocess_profile.add_argument(
        "--band-path-provider",
        choices=("template", "seekpath", "ase", "vaspkit"),
        default="ase",
    )

    preprocess_prepare = sub.add_parser(
        "preprocess-prepare",
        help="物化不可覆盖的 Relax→SCF→Band 输入；不启动 VASP",
    )
    preprocess_prepare.add_argument("poscar")
    preprocess_prepare.add_argument("config", help="含 preprocess 与 preprocess_inputs 的 YAML")
    preprocess_prepare.add_argument("--run-dir", required=True)

    preprocess_inspect = sub.add_parser("preprocess-inspect", help="只读查看前处理下一动作")
    preprocess_inspect.add_argument("run_dir")

    preprocess_run = sub.add_parser(
        "preprocess-run",
        help="从断点推进 Relax→SCF→Band；默认只预览，--execute 才启动",
    )
    preprocess_run.add_argument("run_dir")
    preprocess_run.add_argument("--execute", action="store_true")

    preprocess_recover = sub.add_parser(
        "preprocess-recover",
        help="仅清理同主机已死亡 PID 的阶段锁；不提交作业",
    )
    preprocess_recover.add_argument("run_dir")
    preprocess_recover.add_argument("stage", choices=PREPROCESS_STAGES)

    preprocess_reconcile = sub.add_parser(
        "preprocess-reconcile",
        help="验收外部/中断后已完整输出并应用哈希交接；不提交作业",
    )
    preprocess_reconcile.add_argument("run_dir")
    preprocess_reconcile.add_argument("stage", choices=PREPROCESS_STAGES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            _emit(plan_project(load_project(args.config)))
        elif args.command == "init":
            run = initialize_run(load_project(args.config), args.runs_root)
            _emit({"created": str(run), "status": read_status(run)})
        elif args.command == "prepare":
            run = prepare_run(load_project(args.config), args.runs_root)
            _emit({"prepared": str(run), "status": read_status(run)})
        elif args.command == "status":
            _emit(read_status(args.run_dir))
        elif args.command in {"run", "resume"}:
            result = run_workflow(args.run_dir, execute=args.execute, until=args.until)
            _emit(result)
            return 0 if result.get("passed") or not args.execute else 1
        elif args.command == "fit-from-bands":
            result = fit_from_precomputed(
                load_project(args.config),
                args.runs_root,
                _source_mappings(args.scf_source),
                _source_mappings(args.band_source),
                execute=args.execute,
                until=args.until,
            )
            _emit(result)
            return 0 if result.get("passed") or not args.execute else 1
        elif args.command == "inspect":
            _emit(inspect_workflow(args.run_dir))
        elif args.command == "supervise":
            result = supervise_workflow(
                args.run_dir,
                execute=args.execute,
                until=args.until,
                poll_seconds=args.poll_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
                no_progress_seconds=args.no_progress_seconds,
                stage_timeout_seconds=args.stage_timeout_seconds,
                max_hours=args.max_hours,
            )
            _emit(result)
            return 0 if result.get("passed") or not args.execute else 1
        elif args.command == "supervisor-status":
            _emit(inspect_supervisor(args.run_dir))
        elif args.command == "audit-reference":
            result = audit_reference(args.example_dir)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "fit-from-preprocess":
            result = fit_from_preprocess(
                load_project(args.config),
                args.runs_root,
                _source_mappings(args.preprocess_source),
                execute=args.execute,
                bootstrap_only=args.bootstrap_only,
                run_dir=args.run_dir,
                until=args.until,
            )
            _emit(result)
            return 0 if result.get("passed") or not args.execute else 1
        elif args.command == "stage1-coldstart":
            result = run_stage1_coldstart(
                args.config,
                args.campaign_dir,
                args.runs_root or runs_root(),
                execute=args.execute,
                poll_seconds=args.poll_seconds,
            )
            _emit(result)
            return 0 if (not args.execute or result.get("status") == "COMPLETE") else 1
        elif args.command == "run-material":
            try:
                result = run_material(
                    args.poscar,
                    args.material_id,
                    args.config,
                    args.work_root,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    poll_seconds=args.poll_seconds,
                )
            except (ValueError, RuntimeError) as exc:
                _emit({
                    "mode": "ERROR",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "exit_code": 1,
                })
                return 1
            _emit(result)
            return int(result["exit_code"])
        elif args.command == "audit-ranking":
            with Path(args.ranking_json).open("r", encoding="utf-8") as handle:
                result = audit_ranking_document(json.load(handle))
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "window-suggest":
            eig = parse_eigenval(args.eigenval)
            _emit({
                "source": {key: eig[key] for key in ("nelect", "nkpts", "nbands")},
                "num_wann": args.num_wann,
                "target_energy_eV": [args.target_min, args.target_max],
                "candidates": suggest_windows(eig["energies_eV"], args.num_wann, args.target_min, args.target_max, args.margin),
            })
        elif args.command == "candidate-suggest":
            with Path(args.scores_json).open("r", encoding="utf-8") as handle:
                scores = json.load(handle)
            _emit({"candidates": suggest_projection_candidates(
                scores["orbital_scores"],
                scores["element_counts"],
                args.soc,
                scores.get("chemical_valence"),
            )})
        elif args.command == "resource-plan":
            raw_workers: int | str = args.max_workers
            if raw_workers != "auto":
                raw_workers = int(raw_workers)
            _emit(candidate_search_plan(
                available=detect_resources(),
                candidate_count=args.candidates,
                mpi_processes_per_candidate=args.mpi_per_candidate,
                policy=args.policy,
                max_workers=raw_workers,
            ))
        elif args.command == "pyw90-baseline":
            _emit(run_eig_suggest(
                args.eigenval,
                num_wann=args.num_wann,
                target_energy_eV=(args.target_min, args.target_max),
                fermi_eV=args.fermi,
                energy_reference=args.energy_reference,
                executable=args.executable,
                output_json=args.output,
            ))
        elif args.command == "run-stage":
            result = run_vasp_stage(args.run_dir, args.stage, execute=args.execute, structure_ids=args.structures)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "reconcile":
            result = reconcile_stage(args.run_dir, args.stage, structure_ids=args.structures)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "import-stage":
            mappings = _source_mappings(args.source)
            result = import_completed_stage(args.run_dir, args.stage, mappings)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "generate-candidates":
            _emit(generate_candidates(args.run_dir))
        elif args.command == "evaluate":
            result = evaluate_run(args.run_dir)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "finalize":
            _emit(finalize_run(args.run_dir))
        elif args.command == "plan-attempts":
            _emit(create_attempts(args.run_dir))
        elif args.command == "run-attempt":
            result = run_attempt(args.run_dir, args.structure, args.candidate, args.attempt, execute=args.execute)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "import-attempt":
            result = import_completed_attempt(
                args.run_dir,
                args.structure,
                args.candidate,
                args.attempt,
                args.source,
            )
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "evaluate-attempt":
            result = evaluate_attempt(args.run_dir, args.structure, args.candidate, args.attempt)
            _emit(result)
            return 0 if result.get("passed") else 1
        elif args.command == "rank-attempts":
            _emit(rank_attempts(args.run_dir))
        elif args.command == "export-ranked-attempts":
            _emit(export_ranked_attempts(args.run_dir))
        elif args.command == "material-pool":
            import yaml
            with Path(args.config).open("r", encoding="utf-8") as handle:
                pool_config = yaml.safe_load(handle) or {}
            document = screen_c2db(args.database, pool_config)
            _emit({**write_material_pool(document, args.output_dir), "selected_count": document["selected_count"]})
        elif args.command == "preprocess-profile":
            poscar_path = Path(args.poscar).expanduser().resolve()
            parsed = parse_poscar(poscar_path)
            analysis = resolve_periodicity(parsed, {"dimensionality": "auto"})
            protocol = standard_semilocal_preprocess(
                soc=args.soc,
                dimensionality=int(analysis["dimensionality"]),
                periodic_axes=list(analysis["periodic_axes"]),
                relaxation_enabled=not args.no_relax,
                relaxation_soc=args.relax_with_soc,
                band_path_provider=args.band_path_provider,
            )
            _emit({
                "source": {"POSCAR": str(poscar_path), "sha256": sha256_file(poscar_path)},
                "structure_analysis": analysis,
                "protocol": protocol,
                "plan": build_preprocess_plan(protocol),
                "execution": "DRY_RUN_ONLY",
            })
        elif args.command == "preprocess-prepare":
            protocol, inputs = _preprocess_project(args.config, args.poscar)
            required = {"templates", "potcar", "commands"}
            missing = sorted(required - set(inputs))
            if missing:
                raise ValueError(f"preprocess_inputs 缺少：{', '.join(missing)}")
            run = prepare_preprocess_run(
                args.run_dir,
                args.poscar,
                protocol,
                templates=inputs["templates"],
                potcar=inputs["potcar"],
                commands=inputs["commands"],
            )
            _emit({"prepared": str(run), "status": inspect_preprocess_run(run), "execution": "NOT_STARTED"})
        elif args.command == "preprocess-inspect":
            _emit(inspect_preprocess_run(args.run_dir))
        elif args.command == "preprocess-run":
            result = run_preprocess_workflow(args.run_dir, execute=args.execute)
            _emit(result)
            return 0 if result.get("passed") or not args.execute else 1
        elif args.command == "preprocess-recover":
            _emit(recover_interrupted_stage(args.run_dir, args.stage))
        elif args.command == "preprocess-reconcile":
            result = reconcile_preprocess_stage(args.run_dir, args.stage)
            _emit(result)
            return 0 if result.get("passed") else 1
        return 0
    except (ConfigError, OSError, PermissionError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        _emit({"error": str(exc), "type": type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
