from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parsers import (
    count_wannier_centres,
    parse_eigenval,
    parse_hr_header,
    parse_outcar_status,
    parse_poscar,
    parse_wout,
)
from .ranking import audit_ranking_document


STAGE_CONTRACTS = [
    ("00_validate", "POSCAR + project.yaml + POTCAR lock", "normalized input + input-lock.json"),
    ("01_scf", "POSCAR/POTCAR/INCAR/KPOINTS", "CHGCAR + OUTCAR + OSZICAR + vasprun.xml"),
    ("02_band", "SCF CHGCAR + ICHARG=11 band input", "EIGENVAL + PROCAR + vasprun.xml + bnd.dat"),
    ("03_candidates", "projected bands + target energy range", "finite candidate/window seed list"),
    ("04_wannier", "candidate + uniform mesh + VASP interface", "win/eig/mmn/amn/wout/hr/centres"),
    ("05_quality", "DFT/W90 bands + wout + hr + centres", "quality.json + comparison plot"),
    ("06_rank", "all candidates for every paired structure", "hard-gated common ranking"),
    ("07_export", "production top-K", "minimal packages + manifest + SHA256"),
]


def plan_project(config: dict[str, Any]) -> dict[str, Any]:
    backend = config.get("backend", {})
    kind = "ncl" if config.get("physics", {}).get("soc") else "std"
    workspace_root = Path(__file__).resolve().parents[4]
    launcher = str((workspace_root / "00_命令入口/计算/运行VASP.command").resolve())
    structures = []
    for item in config["structures"]:
        structures.append({
            "id": item["id"],
            "poscar": item["poscar_resolved"],
            "vasp_kind": backend.get("vasp", kind),
            "mpi_processes": int(backend.get("mpi_processes", 2)),
            "structure_analysis": item.get("structure_analysis"),
            "meshes_resolved": item.get("meshes_resolved", {}),
        })
    return {
        "project": config["project"],
        "mode": "PLAN_ONLY",
        "note": "plan 不会启动 VASP；后续 execute 必须显式开启。",
        "structures": structures,
        "band_path_provider": config.get("band_path", {}).get("mode", "template"),
        "stages": [{"stage": stage, "input": inputs, "output": outputs} for stage, inputs, outputs in STAGE_CONTRACTS],
        "launcher_template": [launcher, "<stage-directory>", f"{backend.get('vasp', kind)}", str(backend.get("mpi_processes", 2))],
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _audit_single(example: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required = {
        "POSCAR": example / "00_input/POSCAR",
        "SCF OUTCAR": example / "01_scf/OUTCAR",
        "SCF CHGCAR": example / "01_scf/CHGCAR",
        "Band EIGENVAL": example / "02_band/EIGENVAL",
        "Wannier hr": example / "04_wannier/wannier90_hr.dat",
        "Wannier centres": example / "04_wannier/wannier90_centres.xyz",
        "Wannier wout": example / "04_wannier/wannier90.wout",
        "quality": example / "04_wannier/quality-summary.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"passed": False, "kind": "single", "issues": [f"缺少：{name}" for name in missing]}
    poscar = parse_poscar(required["POSCAR"])
    outcar = parse_outcar_status(required["SCF OUTCAR"])
    eig = parse_eigenval(required["Band EIGENVAL"])
    hr = parse_hr_header(required["Wannier hr"])
    centres = count_wannier_centres(required["Wannier centres"])
    wout = parse_wout(required["Wannier wout"])
    quality = _load_json(required["quality"])
    quality_num_wann = quality.get("num_wann", quality.get("wannier_model", {}).get("num_wann"))
    checks.extend([
        {"name": "SCF convergence", "passed": outcar["converged"], "detail": outcar},
        {"name": "NBANDS handoff", "passed": outcar["nbands"] == eig["nbands"], "detail": {"scf": outcar["nbands"], "band": eig["nbands"]}},
        {"name": "HR/centres dimension", "passed": hr["num_wann"] == centres, "detail": {"hr": hr["num_wann"], "centres": centres}},
        {"name": "quality/HR dimension", "passed": quality_num_wann == hr["num_wann"], "detail": {"quality": quality_num_wann, "hr": hr["num_wann"]}},
        {"name": "Wannier90 completion", "passed": wout["completed"], "detail": wout},
    ])
    return {"passed": all(check["passed"] for check in checks), "kind": "single", "structure": poscar, "eigenval": {key: eig[key] for key in ("nelect", "nkpts", "nbands")}, "checks": checks}


def _quality_path(structure: Path) -> Path | None:
    for name in ("quality-source.json", "quality-summary.json"):
        path = structure / name
        if path.is_file():
            return path
    return None


def _audit_pair(example: Path) -> dict[str, Any]:
    ranking_path = example / "projection-ranking.json"
    reference = example / "02_reference"
    issues: list[str] = []
    if not ranking_path.is_file():
        issues.append("缺少 projection-ranking.json")
        ranking = {"passed": False, "issues": ["missing"]}
    else:
        ranking = audit_ranking_document(_load_json(ranking_path))
        issues.extend(ranking["issues"])
    structures: dict[str, Any] = {}
    for directory in sorted(path for path in reference.iterdir() if path.is_dir() and not path.name.startswith(".")):
        hr_path = directory / "wannier90_hr.dat"
        centre_path = directory / "wannier90-wcenter.dat"
        quality_path = _quality_path(directory)
        if not hr_path.is_file() or not centre_path.is_file() or quality_path is None:
            issues.append(f"{directory.name} 缺少 hr/centres/quality")
            continue
        hr = parse_hr_header(hr_path)
        centres = count_wannier_centres(centre_path)
        quality = _load_json(quality_path)
        candidate = quality.get("candidate")
        passed = hr["num_wann"] == centres == quality.get("num_wann") and candidate == ranking.get("rank1")
        if not passed:
            issues.append(f"{directory.name} 的 HR/centres/quality/rank-1 不一致")
        structures[directory.name] = {
            "passed": passed,
            "candidate": candidate,
            "num_wann": hr["num_wann"],
            "centres": centres,
            "nocc": quality.get("nocc"),
        }
    if len(structures) < 2:
        issues.append("成对实例少于两个 reference 结构")
    common_candidates = {value["candidate"] for value in structures.values()}
    common_dimensions = {value["num_wann"] for value in structures.values()}
    if len(common_candidates) > 1 or len(common_dimensions) > 1:
        issues.append("成对 reference 没有使用共同候选/NUM_WANN")
    return {"passed": not issues, "kind": "pair", "ranking": ranking, "structures": structures, "issues": issues}


def audit_reference(example_dir: str | Path) -> dict[str, Any]:
    example = Path(example_dir).expanduser().resolve()
    if (example / "02_reference").is_dir():
        return _audit_pair(example)
    return _audit_single(example)
