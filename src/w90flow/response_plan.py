"""Build deterministic MATLAB response jobs from a top-K export summary.

This module deliberately separates scientific choices (the campaign spec and
the existing MATLAB kernels) from orchestration.  Building a plan is entirely
offline and never launches MATLAB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parsers import count_wannier_centres, parse_hr_header


PLAN_SCHEMA = "w90flow-response-plan-v1"
STATUS_SCHEMA = "w90flow-response-status-v1"
REQUIRED_INPUTS = ("POSCAR", "wannier90_hr.dat", "wannier90-wcenter.dat")


class ResponsePlanError(ValueError):
    """The response campaign cannot be materialized without guessing."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponsePlanError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResponsePlanError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(base: Path, value: str, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ResponsePlanError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResponsePlanError(f"{field} must be a positive integer")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ResponsePlanError(f"{field} must be positive")
    return float(value)


def _validate_grid(config: dict[str, Any], name: str) -> dict[str, Any]:
    mesh = config.get("kmesh")
    if not isinstance(mesh, list) or len(mesh) != 3:
        raise ResponsePlanError(f"responses.{name}.kmesh must contain three integers")
    mesh = [_positive_int(value, f"responses.{name}.kmesh") for value in mesh]
    if mesh[2] != 1:
        raise ResponsePlanError(f"responses.{name}.kmesh[2] must be 1 for the current 2D MATLAB kernels")

    energy = config.get("energy_grid_eV")
    if not isinstance(energy, dict):
        raise ResponsePlanError(f"responses.{name}.energy_grid_eV must be an object")
    minimum = float(energy.get("min"))
    maximum = float(energy.get("max"))
    points = _positive_int(energy.get("points"), f"responses.{name}.energy_grid_eV.points")
    if points < 2 or maximum <= minimum:
        raise ResponsePlanError(f"responses.{name}.energy_grid_eV requires max > min and points >= 2")

    components = config.get("components")
    if not isinstance(components, list) or not components or not all(isinstance(x, str) and x for x in components):
        raise ResponsePlanError(f"responses.{name}.components must be a non-empty string list")
    return {
        "kmesh": mesh,
        "energy_grid_eV": {"min": minimum, "max": maximum, "points": points},
        "components": components,
    }


def _resolve_package(summary_dir: Path, package: dict[str, Any]) -> Path:
    declared = package.get("path")
    candidates: list[Path] = []
    if isinstance(declared, str) and declared:
        declared_path = Path(declared).expanduser()
        # Prefer the package beside export-summary.  Embedded paths commonly
        # point at an older /private/tmp mount which may still happen to exist.
        candidates.append((summary_dir / declared_path.name).resolve())
        if declared_path.is_dir():
            candidates.append(declared_path.resolve())
    rank = _positive_int(package.get("rank"), "packages[].rank")
    candidate = package.get("candidate")
    if isinstance(candidate, str) and candidate:
        candidates.append((summary_dir / f"rank-{rank:02d}_{candidate}").resolve())
    for path in candidates:
        if (path / "package-manifest.json").is_file():
            return path
    raise ResponsePlanError(
        f"Cannot resolve local package directory for rank={rank}, candidate={candidate!r}; "
        "the path embedded in export-summary may refer to an old mounted run"
    )


def _validate_member(
    rank_dir: Path,
    manifest: dict[str, Any],
    structure: str,
    member: dict[str, Any],
    verify_hashes: bool,
) -> dict[str, Any]:
    model_dir = rank_dir / structure
    files = member.get("files")
    if not isinstance(files, dict):
        raise ResponsePlanError(f"manifest structure {structure} has no files map")
    resolved: dict[str, str] = {}
    for filename in REQUIRED_INPUTS:
        path = (model_dir / filename).resolve()
        if not path.is_file():
            raise ResponsePlanError(f"Required MATLAB input is missing: {path}")
        expected = files.get(filename)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ResponsePlanError(f"manifest lacks SHA256 for {structure}/{filename}")
        if verify_hashes and _sha256(path) != expected:
            raise ResponsePlanError(f"SHA256 mismatch for {path}")
        resolved[filename] = str(path)

    quality_path = model_dir / "quality.json"
    if not quality_path.is_file():
        raise ResponsePlanError(f"Missing quality.json: {quality_path}")
    expected_quality = files.get("quality.json")
    if not isinstance(expected_quality, str) or len(expected_quality) != 64:
        raise ResponsePlanError(f"manifest lacks SHA256 for {structure}/quality.json")
    if verify_hashes and _sha256(quality_path) != expected_quality:
        raise ResponsePlanError(f"SHA256 mismatch for {quality_path}")
    quality = _read_json(quality_path)
    for field in ("num_wann", "nocc_target", "nocc_model", "passed"):
        if field not in quality:
            raise ResponsePlanError(f"quality.json lacks {field}: {quality_path}")
    if quality["passed"] is not True:
        raise ResponsePlanError(f"Rejected Wannier model cannot enter response plan: {quality_path}")

    num_wann = _positive_int(manifest.get("num_wann"), "manifest.num_wann")
    nocc_model = _positive_int(manifest.get("nocc_model"), "manifest.nocc_model")
    nocc_target = _positive_int(quality.get("nocc_target"), "quality.nocc_target")
    if int(quality["num_wann"]) != num_wann or int(quality["nocc_model"]) != nocc_model:
        raise ResponsePlanError(f"manifest/quality NUM_WANN or nocc_model mismatch: {quality_path}")
    if not (0 < nocc_model < num_wann):
        raise ResponsePlanError(f"nocc_model={nocc_model} must satisfy 0 < NOCC < NUM_WANN={num_wann}")
    if not (0 < nocc_target < num_wann):
        raise ResponsePlanError(f"nocc_target={nocc_target} is invalid in {quality_path}")

    hr = parse_hr_header(resolved["wannier90_hr.dat"])
    centres = count_wannier_centres(resolved["wannier90-wcenter.dat"])
    if hr["num_wann"] != num_wann or centres != num_wann:
        raise ResponsePlanError(
            f"HR/centres/manifest dimension mismatch for {structure}: "
            f"hr={hr['num_wann']}, centres={centres}, manifest={num_wann}"
        )
    checks = quality.get("checks")
    if not isinstance(checks, dict) or checks.get("hr_centres_match") is not True:
        raise ResponsePlanError(f"quality hr_centres_match gate is not true: {quality_path}")
    return {
        "model_dir": str(model_dir.resolve()),
        "files": resolved,
        "quality_json": str(quality_path.resolve()),
        "num_wann": num_wann,
        "nocc_model": nocc_model,
        "nocc_target": nocc_target,
        "hr_sha256": files["wannier90_hr.dat"],
    }


def _job_config(
    response_type: str,
    response: dict[str, Any],
    model: dict[str, Any],
    output_dir: Path,
    label: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    grid = response["energy_grid_eV"]
    points = int(grid["points"])
    step = (float(grid["max"]) - float(grid["min"])) / (points - 1)
    energies = [float(grid["min"]) + index * step for index in range(points)]
    energies[-1] = float(grid["max"])
    if response_type == "shift_current":
        return {
            "structure_name": label,
            "model_dir": model["model_dir"],
            "output_dir": str(output_dir),
            "nocc": model["nocc_model"],
            "nocc_source": provenance["manifest_file"],
            "components": response["components"],
            "nkx": response["kmesh"][0],
            "nky": response["kmesh"][1],
            "nkz": response["kmesh"][2],
            "ksi_eV": _positive_number(response.get("broadening_eV"), "shift_current.broadening_eV"),
            "photon_energy_eV": energies,
            "thickness_A": _positive_number(response.get("thickness_A"), "shift_current.thickness_A"),
            "parallel_workers": _positive_int(response.get("parallel_workers", 1), "shift_current.parallel_workers"),
            "save_png": bool(response.get("save_png", False)),
            "save_pdf": bool(response.get("save_pdf", False)),
            "provenance": provenance,
        }
    return {
        "input_dir": model["model_dir"],
        "output_root": str(output_dir),
        "seedname": "wannier90",
        "poscar": "POSCAR",
        "tensor_components": response["components"],
        "nk": response["kmesh"],
        "photon_energy_eV": energies,
        "nocc": model["nocc_model"],
        "spectral_broadening_eV": _positive_number(
            response.get("spectral_broadening_eV"), "shg.spectral_broadening_eV"
        ),
        "effective_thickness_A": _positive_number(
            response.get("effective_thickness_A"), "shg.effective_thickness_A"
        ),
        "save_diagnostics": bool(response.get("save_diagnostics", False)),
        "save_figures": bool(response.get("save_figures", False)),
        "continue_on_error": False,
        "k_resolved_mode": "none",
    }


def build_response_plan(
    spec_path: str | Path,
    *,
    plan_dir: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a campaign spec and materialize its immutable ordered plan."""

    spec_file = Path(spec_path).expanduser().resolve()
    spec = _read_json(spec_file)
    base = spec_file.parent
    if spec.get("schema_version") != 1:
        raise ResponsePlanError("response campaign schema_version must be 1")
    system = spec.get("system")
    if not isinstance(system, str) or not system:
        raise ResponsePlanError("system must be non-empty text")
    summary_path = _resolve(base, spec.get("export_summary"), "export_summary")
    summary = _read_json(summary_path)
    packages = summary.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ResponsePlanError(f"export summary contains no packages: {summary_path}")
    output_root = _resolve(base, spec.get("output_root"), "output_root")
    materialized_dir = (
        Path(plan_dir).expanduser().resolve()
        if plan_dir is not None
        else _resolve(base, spec.get("plan_dir"), "plan_dir")
    )
    matlab_executable = str(spec.get("matlab_executable", "matlab"))

    responses = spec.get("responses")
    if not isinstance(responses, dict):
        raise ResponsePlanError("responses must be an object")
    response_order = spec.get("response_order", ["shift_current", "shg"])
    if not isinstance(response_order, list) or not response_order:
        raise ResponsePlanError("response_order must be a non-empty list")
    normalized: dict[str, dict[str, Any]] = {}
    for name in response_order:
        if name not in {"shift_current", "shg"}:
            raise ResponsePlanError(f"Unsupported response type: {name}")
        value = responses.get(name)
        if not isinstance(value, dict) or not value.get("enabled", True):
            continue
        grid = _validate_grid(value, name)
        code_dir = _resolve(base, value.get("code_dir"), f"responses.{name}.code_dir")
        entry_default = "compute_shift_current_model" if name == "shift_current" else "SHG_unified_optimized"
        entry_point = str(value.get("entry_point", entry_default))
        entry_file = code_dir / f"{entry_point}.m"
        if not entry_file.is_file():
            raise ResponsePlanError(f"MATLAB entry point is missing: {entry_file}")
        normalized[name] = {
            **value,
            **grid,
            "code_dir": str(code_dir),
            "entry_point": entry_point,
            "profile": str(value.get("profile", "custom")),
        }
    if not normalized:
        raise ResponsePlanError("No enabled response was selected")

    package_rows: list[tuple[int, str, Path, dict[str, Any]]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ResponsePlanError("Every packages[] entry must be an object")
        rank_dir = _resolve_package(summary_path.parent, package)
        manifest_file = rank_dir / "package-manifest.json"
        manifest = _read_json(manifest_file)
        rank = _positive_int(manifest.get("rank"), "manifest.rank")
        candidate = manifest.get("candidate")
        if rank != package.get("rank") or candidate != package.get("candidate"):
            raise ResponsePlanError(f"export-summary/package manifest rank mismatch: {manifest_file}")
        if not isinstance(candidate, str) or not candidate:
            raise ResponsePlanError(f"manifest candidate is invalid: {manifest_file}")
        package_rows.append((rank, candidate, rank_dir, manifest))
    package_rows.sort(key=lambda row: (row[0], row[1]))

    jobs: list[dict[str, Any]] = []
    sequence = 0
    for response_name in response_order:
        if response_name not in normalized:
            continue
        response = normalized[response_name]
        for rank, candidate, rank_dir, manifest in package_rows:
            structures = manifest.get("structures")
            if not isinstance(structures, dict) or not structures:
                raise ResponsePlanError(f"manifest has no structures: {rank_dir}")
            for structure, member in structures.items():
                if not isinstance(member, dict):
                    raise ResponsePlanError(f"manifest member {structure} must be an object")
                model = _validate_member(rank_dir, manifest, structure, member, verify_hashes)
                sequence += 1
                safe_candidate = candidate.replace("/", "-").replace(" ", "-")
                safe_structure = structure.replace("/", "-").replace(" ", "-")
                job_id = f"{sequence:04d}-{response_name}-rank-{rank:02d}-{safe_candidate}-{safe_structure}"
                output_dir = (
                    output_root
                    / response_name
                    / response["profile"]
                    / f"rank-{rank:02d}_{candidate}"
                    / structure
                ).resolve()
                provenance = {
                    "system": system,
                    "rank": rank,
                    "candidate": candidate,
                    "attempt": manifest.get("attempt"),
                    "projection": manifest.get("projection"),
                    "num_wann": model["num_wann"],
                    "nocc_model": model["nocc_model"],
                    "nocc_target_fit_only": model["nocc_target"],
                    "hr_sha256": model["hr_sha256"],
                    "manifest_file": str((rank_dir / "package-manifest.json").resolve()),
                    "quality_file": model["quality_json"],
                    "export_summary": str(summary_path),
                }
                config = _job_config(
                    response_name,
                    response,
                    model,
                    output_dir,
                    f"{system}_{structure}_rank-{rank:02d}_{candidate}",
                    provenance,
                )
                config_file = materialized_dir / "jobs" / job_id / "matlab-config.json"
                interface = {
                    "matlab_executable": matlab_executable,
                    "code_dir": response["code_dir"],
                    "entry_point": response["entry_point"],
                    "config_file": str(config_file.resolve()),
                }
                job = {
                    "sequence": sequence,
                    "job_id": job_id,
                    "response_type": response_name,
                    "profile": response["profile"],
                    "system": system,
                    "rank": rank,
                    "candidate": candidate,
                    "attempt": manifest.get("attempt"),
                    "projection": manifest.get("projection"),
                    "structure": structure,
                    "num_wann": model["num_wann"],
                    "nocc_model": model["nocc_model"],
                    "nocc_target_fit_only": model["nocc_target"],
                    "inputs": model["files"],
                    "quality_json": model["quality_json"],
                    "output_dir": str(output_dir),
                    "components": response["components"],
                    "kmesh": response["kmesh"],
                    "energy_grid_eV": response["energy_grid_eV"],
                    "matlab": interface,
                    "matlab_config": config,
                }
                job["fingerprint"] = _object_sha256(job)
                jobs.append(job)

    plan_core = {
        "schema": PLAN_SCHEMA,
        "system": system,
        "source_spec": str(spec_file),
        "export_summary": str(summary_path),
        "output_root": str(output_root),
        "execution_policy": {
            "default_mode": "dry-run",
            "order": "response, rank, manifest structure order",
            "concurrency": 1,
            "resume": "skip only matching backend success markers",
            "nocc_rule": "MATLAB NOCC is nocc_model; nocc_target is fit-only metadata",
        },
        "jobs": jobs,
    }
    plan = {**plan_core, "plan_sha256": _object_sha256(plan_core), "created_at": _utc_now()}
    _atomic_json(materialized_dir / "response-plan.json", plan)
    for job in jobs:
        _atomic_json(Path(job["matlab"]["config_file"]), job["matlab_config"])
    status = {
        "schema": STATUS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "jobs": {
            job["job_id"]: {
                "state": "PENDING",
                "attempts": 0,
                "fingerprint": job["fingerprint"],
                "log": str((materialized_dir / "logs" / f"{job['job_id']}.log").resolve()),
            }
            for job in jobs
        },
        "summary": {"PENDING": len(jobs), "RUNNING": 0, "SUCCEEDED": 0, "FAILED": 0},
    }
    _atomic_json(materialized_dir / "response-status.json", status)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline top-K response planning and serial MATLAB backend")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="validate inputs and create plan/status; never runs MATLAB")
    build.add_argument("spec")
    build.add_argument("--plan-dir")
    build.add_argument("--skip-hash-check", action="store_true")
    preview = sub.add_parser("preview", help="show ordered matlab -batch commands")
    preview.add_argument("plan_dir")
    run = sub.add_parser("run", help="dry-run by default; --execute runs one MATLAB job at a time")
    run.add_argument("plan_dir")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    status = sub.add_parser("status", help="show the persistent response status JSON")
    status.add_argument("plan_dir")
    args = parser.parse_args(argv)

    if args.command == "build":
        payload = build_response_plan(
            args.spec, plan_dir=args.plan_dir, verify_hashes=not args.skip_hash_check
        )
    else:
        from .matlab_backend import preview_plan, read_response_status, run_response_plan

        if args.command == "preview":
            payload = preview_plan(args.plan_dir)
        elif args.command == "status":
            payload = read_response_status(args.plan_dir)
        else:
            payload = run_response_plan(
                args.plan_dir,
                execute=args.execute,
                continue_on_error=args.continue_on_error,
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
