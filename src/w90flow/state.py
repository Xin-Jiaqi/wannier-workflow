from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import public_snapshot


STAGES = (
    "00_validate",
    "01_scf",
    "02_band",
    "03_candidates",
    "04_wannier",
    "05_quality",
    "06_rank",
    "07_export",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(target)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_stage(run_dir: str | Path, stage: str, **changes: Any) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"未知阶段：{stage}")
    state_path = Path(run_dir) / stage / "stage.json"
    state = read_json(state_path)
    state.update(changes)
    atomic_json(state_path, state)
    return state


def initialize_run(config: dict[str, Any], runs_root: str | Path) -> Path:
    snapshot = public_snapshot(config)
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(canonical).hexdigest()
    suffix = config_hash[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_root).expanduser().resolve() / f"{config['project']}-{stamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "project.snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)

    locked_files: list[dict[str, Any]] = []
    for structure in config["structures"]:
        path = Path(structure["poscar_resolved"])
        locked_files.append({"role": f"POSCAR:{structure['id']}", "path": str(path), "sha256": sha256_file(path)})
    lock_path = config.get("pseudopotentials", {}).get("lock_resolved")
    if lock_path:
        locked_files.append({"role": "POTCAR_LOCK", "path": lock_path, "sha256": sha256_file(lock_path)})

    created = datetime.now(timezone.utc).isoformat()
    atomic_json(run_dir / "input-lock.json", {
        "schema_version": 1,
        "created_utc": created,
        "config_sha256": config_hash,
        "files": locked_files,
    })
    for stage in STAGES:
        atomic_json(run_dir / stage / "stage.json", {
            "schema_version": 1,
            "stage": stage,
            "status": "PENDING",
            "created_utc": created,
            "inputs": [],
            "outputs": [],
            "commands": [],
        })
    for structure in config["structures"]:
        sid = str(structure["id"])
        for stage in ("00_input", "01_scf", "02_band", "03_candidates", "04_wannier", "05_quality", "06_export"):
            (run_dir / "structures" / sid / stage).mkdir(parents=True, exist_ok=True)
    return run_dir


def read_status(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    stages: list[dict[str, Any]] = []
    for stage in STAGES:
        state_path = root / stage / "stage.json"
        if not state_path.is_file():
            stages.append({"stage": stage, "status": "MISSING"})
            continue
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        stages.append({"stage": stage, "status": state.get("status", "UNKNOWN")})
    return {"run_dir": str(root), "stages": stages}
