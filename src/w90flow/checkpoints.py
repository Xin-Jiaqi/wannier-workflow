"""Preserve accepted candidates before the finite search finishes.

Checkpoints are immutable recovery packages, not ranked exports or workflow
completion. A later timeout must not erase an already verified model.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .acceptance import strict_quality_passed
from .state import atomic_json, read_json, sha256_file


def checkpoint_accepted_candidate(
    run_dir: str | Path, structures: list[str], candidate: str, attempt: str,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    for label in [*structures, candidate, attempt]:
        if not label or label in {".", ".."} or "/" in label or "\\" in label:
            raise ValueError("checkpoint identifiers must be single path components")
    files: dict[str, Path] = {}
    for sid in structures:
        source = root / "structures" / sid / "04_wannier_attempts" / candidate / attempt
        source = source.resolve()
        if not source.is_relative_to(root / "structures"):
            raise ValueError("checkpoint source escapes the run")
        quality = read_json(source / "quality.json")
        record = read_json(source / "attempt.json")
        if quality.get("passed") is not True or record.get("status") != "ACCEPTED":
            raise ValueError(f"{sid}/{candidate}/{attempt} is not quality-accepted")
        if (quality.get("structure"), quality.get("candidate"), quality.get("attempt")) != (sid, candidate, attempt):
            raise ValueError("checkpoint quality identity mismatch")
        if not strict_quality_passed(quality):
            raise ValueError("checkpoint requires current convergence/quality audit")
        checks = quality.get("checks")
        if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
            raise ValueError("checkpoint requires all recorded quality checks to pass")
        reference = root / "structures" / sid / "02_band" / "EIGENVAL"
        audited = quality.get("source_sha256", {})
        for name, path in [("EIGENVAL", reference), ("wannier90_hr.dat", source / "wannier90_hr.dat"),
                           ("wannier90.wout", source / "wannier90.wout")]:
            if not audited.get(name) or sha256_file(path) != audited[name]:
                raise ValueError(f"checkpoint audited source changed: {sid}/{name}")
        for name in ["POSCAR", "wannier90.win", "wannier90_hr.dat", "wannier90_centres.xyz",
                     "wannier90.wout", "quality.json", "attempt.json"]:
            files[f"{sid}/{name}"] = source / name
        files[f"{sid}/DFT_EIGENVAL"] = reference
        for name in ["VASP_W90_cmp.pdf", "wannier90-wcenter.dat", "wannier90_tb.dat"]:
            if (source / name).is_file():
                files[f"{sid}/{name}"] = source / name
    if not files:
        raise ValueError("checkpoint requires at least one accepted structure")
    hashes = {name: sha256_file(path) for name, path in sorted(files.items())}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    parent = root / "accepted-checkpoints"
    parent.mkdir(exist_ok=True)
    target = parent / digest
    manifest = {"schema_version": 1, "source_run": str(root), "structures": structures,
                "candidate": candidate, "attempt": attempt, "files_sha256": hashes,
                "package_sha256": digest, "workflow_complete": False, "ranked_export": False,
                "nlo_accepted": False, "scope": "quality-accepted recovery checkpoint; finite search continues"}
    if target.exists():
        if read_json(target / "checkpoint.json") != manifest:
            raise ValueError("existing checkpoint manifest changed")
        for name, expected in hashes.items():
            if sha256_file(target / name) != expected:
                raise ValueError(f"existing checkpoint content changed: {name}")
        return {**manifest, "path": str(target)}
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        for name, source in files.items():
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != hashes[name]:
                raise ValueError(f"checkpoint source changed while copying: {name}")
        atomic_json(temporary / "checkpoint.json", manifest)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {**manifest, "path": str(target)}
