from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ConfigError, load_project
from .attempts import create_attempts, import_completed_attempt, run_attempt
from .candidates import suggest_projection_candidates
from .executor import finalize_run, generate_candidates, import_completed_stage, prepare_run, reconcile_stage, run_vasp_stage
from .parsers import parse_eigenval
from .pipeline import audit_reference, plan_project
from .pyw90_adapter import run_eig_suggest
from .quality import evaluate_attempt, evaluate_run, export_ranked_attempts, rank_attempts
from .ranking import audit_ranking_document
from .material_pool import screen_c2db, write_material_pool
from .orchestrator import WORKFLOW_STOPS, inspect_workflow, run_workflow
from .state import initialize_run, read_status
from .supervisor import inspect_supervisor, supervise_workflow
from .windows import suggest_windows


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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

    pyw90 = sub.add_parser("pyw90-baseline", help="将 PyW90 eig suggest 转为可审计 JSON 对照")
    pyw90.add_argument("eigenval")
    pyw90.add_argument("--num-wann", type=int, required=True)
    pyw90.add_argument("--target-min", type=float, required=True)
    pyw90.add_argument("--target-max", type=float, required=True)
    pyw90.add_argument("--fermi", type=float, required=True)
    pyw90.add_argument("--energy-reference", choices=("absolute", "fermi_relative"), default="fermi_relative")
    pyw90.add_argument("--executable", default="/private/tmp/LocalVASP-mount/bin/pyw90")
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
            mappings: dict[str, str] = {}
            for value in args.source:
                if "=" not in value:
                    raise ValueError("--source 必须是 STRUCTURE=/absolute/directory")
                structure, source = value.split("=", 1)
                if not structure or not source or structure in mappings:
                    raise ValueError("--source 结构名/目录为空或结构重复。")
                mappings[structure] = source
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
        return 0
    except (ConfigError, OSError, PermissionError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        _emit({"error": str(exc), "type": type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
