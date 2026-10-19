"""Generic Japan Stage-1 PBS material runner (manifest-driven, no whitelist).

The historical ``stage1_case.pbs`` hard-coded an allow-list of material ids and
invoked the lower-level ``stage1-coldstart`` entry.  This module replaces both
with a controlled-material contract:

* a material is submittable iff it is a key in the release's controlled cases
  manifest (``CONTROL/input-lock.json``) and its config/POSCAR SHA-256 hashes
  match the manifest row;
* every path is derived from the release/pool roots - callers never pass
  arbitrary POSCAR/config paths into the PBS job;
* the PBS template only contains validated, regex-constrained fields and runs
  ``w90flow run-material`` (real mode) under the fixed 1-node/32-core/128GB
  site allocation;
* duplicate submission is prevented by the wrapper (active-job query) plus
  run-material's own immutable campaign manifest semantics.

No material-specific branches, no site absolute paths, no secrets.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-untyped]

MATERIAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_POOL_REL = "02_本地算力工作流/06_材料池/test-pool-v01"
DEFAULT_MANIFEST_REL = "CONTROL/input-lock.json"
MANIFEST_FILE = "run-material-manifest.json"
TERMINAL_STATUSES = ("TOP3_READY", "PARTIAL_TOPK", "NO_MODEL", "TECHNICAL_FAILED")
QSTAT = "/opt/pbs/bin/qstat"
QSUB = "/opt/pbs/bin/qsub"

STAGE1_MATERIAL_PBS = r"""#!/bin/bash
#PBS -N xjq-stage1-%%MATERIAL_ID%%
#PBS -q CP_001
#PBS -l select=1:ncpus=32:mpiprocs=32:ompthreads=1:mem=128gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

MATERIAL_ID='%%MATERIAL_ID%%'
RELEASE='%%RELEASE%%'
POOL='%%POOL%%'
WORK_ROOT='%%WORK_ROOT%%'
PYTHON='%%PYTHON%%'
POSCAR='%%POSCAR%%'
CONFIG='%%CONFIG%%'
DRY_RUN=%%DRY_RUN%%
RESUME=%%RESUME%%
POLL_SECONDS=%%POLL_SECONDS%%
REUSE_FROM='%%REUSE_FROM%%'

if [[ ! -x "$PYTHON" ]]; then
  echo "python not executable: $PYTHON" >&2
  exit 2
fi

if [[ ! "$MATERIAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "invalid material id: $MATERIAL_ID" >&2
  exit 2
fi
test -f "$CONFIG" || { echo "config missing: $CONFIG" >&2; exit 2; }
test -f "$POSCAR" || { echo "poscar missing: $POSCAR" >&2; exit 2; }

"$PYTHON" - "$MATERIAL_ID" "$RELEASE" <<'PY'
import json
import sys

material_id, release = sys.argv[1], sys.argv[2]
lock_path = release + "/CONTROL/input-lock.json"
with open(lock_path, encoding="utf-8") as handle:
    lock = json.load(handle)
if material_id not in lock.get("cases", {}):
    raise SystemExit("material %s is not in the controlled cases manifest" % material_id)
PY

source "$RELEASE/site/japan/load_oneapi.sh"
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RELEASE/vendor:$RELEASE/02_本地算力工作流/05_Python自动化/src"
mkdir -p "$WORK_ROOT"
export MPLCONFIGDIR="$WORK_ROOT/.mplconfig"
test "$(wc -l < "$PBS_NODEFILE")" -ge 32

ARGS=(run-material "$POSCAR" "$MATERIAL_ID" --config "$CONFIG" --work-root "$WORK_ROOT" --poll-seconds "$POLL_SECONDS")
if [ "$DRY_RUN" = "1" ]; then
  ARGS+=(--dry-run)
fi
if [ "$RESUME" = "1" ]; then
  ARGS+=(--resume)
fi
ARGS+=(--python "$PYTHON")
if [ -n "$REUSE_FROM" ]; then
  ARGS+=(--reuse-from "$REUSE_FROM")
fi
exec "$PYTHON" -m w90flow "${ARGS[@]}"
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_material_id(material_id: str) -> str:
    """Reject shell metacharacters, path separators and empty ids."""
    if not isinstance(material_id, str) or not MATERIAL_ID_RE.fullmatch(material_id):
        raise ValueError(f"非法 material_id：{material_id!r}（只允许字母数字开头及 ._- ）")
    return material_id


def load_cases_manifest(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the controlled cases manifest (``CONTROL/input-lock.json``)."""
    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("cases"), dict):
        raise ValueError(f"cases manifest 无效：{path}")
    cases: dict[str, dict[str, Any]] = {}
    for key, row in document["cases"].items():
        if not isinstance(row, dict):
            raise ValueError(f"cases manifest 中 {key} 不是对象。")
        missing = {"config_sha256", "poscar_sha256"} - set(row)
        if missing:
            raise ValueError(f"cases manifest 中 {key} 缺少字段：{sorted(missing)}")
        cases[str(key)] = row
    return cases


def resolve_under_root(root: str | Path, path: str | Path) -> Path:
    """Resolve ``path`` and require it to stay inside ``root`` (fail closed)."""
    base = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"路径越界（不允许逃逸 release/pool 根）：{candidate}")
    return candidate


def release_commit(release_root: str | Path) -> str | None:
    """Read the recorded source commit from ``release/source.sha256``."""
    marker = Path(release_root).expanduser().resolve() / "source.sha256"
    if not marker.is_file():
        return None
    for line in marker.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("commit "):
            return line.split()[1].strip()
    return None


def derive_material_inputs(
    pool_root: str | Path,
    material_id: str,
    manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Derive and hash-validate the controlled config/POSCAR for a material.

    The config path is ``pool/coldstart/<material>/stage1.yaml``; the POSCAR
    path is read from the config's ``structures[0].poscar`` (config-relative)
    and must stay inside the pool root.  Both files must match the manifest
    row hashes.
    """
    material_id = validate_material_id(material_id)
    row = manifest.get(material_id)
    if row is None:
        raise ValueError(f"material_id 不在受控 cases manifest 中：{material_id}")
    pool = Path(pool_root).expanduser().resolve()
    config_path = resolve_under_root(pool, pool / "coldstart" / material_id / "stage1.yaml")
    if not config_path.is_file():
        raise ValueError(f"受控配置不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("structures"), list) or not config["structures"]:
        raise ValueError(f"配置缺少 structures：{config_path}")
    structure = config["structures"][0]
    if not isinstance(structure, dict) or not isinstance(structure.get("poscar"), str):
        raise ValueError(f"配置 structures[0] 缺少 poscar：{config_path}")

    config_sha = _sha256(config_path)
    if config_sha != str(row["config_sha256"]):
        raise ValueError(
            f"{material_id} 配置 SHA256 与 manifest 不一致：{config_sha} != {row['config_sha256']}"
        )

    poscar_abs = (config_path.parent / str(structure["poscar"])).resolve()
    poscar_path = resolve_under_root(pool, poscar_abs)
    if not poscar_path.is_file():
        raise ValueError(f"受控 POSCAR 不存在：{poscar_path}")
    poscar_sha = _sha256(poscar_path)
    if poscar_sha != str(row["poscar_sha256"]):
        raise ValueError(
            f"{material_id} POSCAR SHA256 与 manifest 不一致：{poscar_sha} != {row['poscar_sha256']}"
        )
    return {
        "material_id": material_id,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "poscar_path": str(poscar_path),
        "poscar_sha256": poscar_sha,
        "potcar_combined_sha256": row.get("potcar_combined_sha256"),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def site_qstat_active(material_id: str) -> list[str]:
    """Return active PBS job ids whose name matches ``xjq-stage1-<material>``."""
    validate_material_id(material_id)
    user = os.environ.get("USER", "")
    command = [QSTAT, "-u", user, "-w"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"无法查询 qstat：{exc}") from exc
    marker = f"xjq-stage1-{material_id}"
    jobs: list[str] = []
    for line in completed.stdout.splitlines():
        if marker in line:
            fields = line.split()
            if fields:
                jobs.append(fields[0])
    return jobs


def site_qsub(pbs_path: str | Path) -> dict[str, Any]:
    """Submit one PBS script and return the job id (single submission)."""
    path = Path(pbs_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"PBS 脚本不存在：{path}")
    try:
        completed = subprocess.run(
            [QSUB, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"qsub 失败：{exc}") from exc
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or not output:
        raise RuntimeError(f"qsub 返回 {completed.returncode}：{output}")
    job_id = output.splitlines()[-1].strip()
    return {"job_id": job_id, "output": output}


def campaign_manifest(work_root: str | Path, material_id: str) -> dict[str, Any] | None:
    path = Path(work_root).expanduser().resolve() / "campaigns" / material_id / MANIFEST_FILE
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def material_preflight(
    *,
    material_id: str,
    manifest_path: str | Path,
    pool_root: str | Path,
    work_root: str | Path,
    release_root: str | Path,
    expected_release_commit: str | None = None,
    resume: bool = False,
    active_jobs: list[str] | None = None,
) -> dict[str, Any]:
    """Read-only preflight: manifest membership, hashes, immutability, dupes."""
    errors: list[str] = []
    report: dict[str, Any] = {
        "material_id": material_id,
        "passed": False,
        "errors": errors,
        "checked_utc": _utc_now(),
    }
    try:
        validate_material_id(material_id)
    except ValueError as exc:
        errors.append(str(exc))
        return report
    try:
        manifest = load_cases_manifest(manifest_path)
    except ValueError as exc:
        errors.append(str(exc))
        return report
    if material_id not in manifest:
        errors.append(f"material_id 不在受控 cases manifest 中：{material_id}")
        return report

    try:
        inputs = derive_material_inputs(pool_root, material_id, manifest)
        report.update(inputs)
    except ValueError as exc:
        errors.append(str(exc))
        return report

    commit = release_commit(release_root)
    report["release_commit"] = commit
    if expected_release_commit is not None:
        report["expected_release_commit"] = expected_release_commit
        if commit != expected_release_commit:
            errors.append(
                f"release 记录 commit {commit} 与期望 {expected_release_commit} 不一致"
            )

    existing = campaign_manifest(work_root, material_id)
    if existing is not None:
        report["existing_campaign"] = {
            "status": existing.get("status"),
            "terminal": bool(existing.get("terminal", False)),
            "input_signature_sha256": existing.get("input_signature_sha256"),
        }
        expected_signature = _campaign_signature(pool_root, material_id, inputs)
        if expected_signature is not None and existing.get("input_signature_sha256") not in (None, expected_signature):
            errors.append("已有 campaign manifest 的输入指纹与当前输入不同，拒绝覆盖。")
        elif not resume:
            errors.append(
                "work-root 下已存在该材料 campaign；确认后需显式 --resume 才能继续。"
            )
    else:
        report["existing_campaign"] = None

    jobs = [str(job) for job in (active_jobs or [])]
    report["active_jobs"] = jobs
    if jobs:
        errors.append(f"检测到 active duplicate 作业：{jobs}")

    report["passed"] = not errors
    return report


def _campaign_signature(
    pool_root: str | Path,
    material_id: str,
    inputs: dict[str, Any],
) -> str | None:
    """Best-effort fingerprint of the controlled inputs (config+POSCAR)."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(material_id).encode("utf-8"))
    digest.update(str(inputs["config_sha256"]).encode("utf-8"))
    digest.update(str(inputs["poscar_sha256"]).encode("utf-8"))
    return digest.hexdigest()


def render_stage1_pbs(
    *,
    material_id: str,
    release_root: str | Path,
    pool_root: str | Path,
    work_root: str | Path,
    python: str,
    poscar_path: str | Path,
    config_path: str | Path,
    dry_run: bool = False,
    resume: bool = False,
    poll_seconds: float = 60.0,
    reuse_from: str | Path | None = None,
) -> str:
    """Render the generic PBS script from validated, root-derived fields only."""
    material_id = validate_material_id(material_id)
    if poll_seconds <= 0:
        raise ValueError("poll_seconds 必须大于 0。")
    values = {
        "MATERIAL_ID": material_id,
        "RELEASE": str(Path(release_root).expanduser().resolve()),
        "POOL": str(Path(pool_root).expanduser().resolve()),
        "WORK_ROOT": str(Path(work_root).expanduser().resolve()),
        "PYTHON": str(python),
        "POSCAR": str(Path(poscar_path).expanduser().resolve()),
        "CONFIG": str(Path(config_path).expanduser().resolve()),
        "DRY_RUN": "1" if dry_run else "0",
        "RESUME": "1" if resume else "0",
        "POLL_SECONDS": str(float(poll_seconds)),
        "REUSE_FROM": str(Path(reuse_from).expanduser().resolve()) if reuse_from else "",
    }
    script = STAGE1_MATERIAL_PBS
    for token, value in values.items():
        script = script.replace(f"%%{token}%%", value)
    for token in re.findall(r"%%([A-Z_]+)%%", script):
        raise ValueError(f"PBS 模板存在未替换 token：%%{token}%%")
    return script


def submit_material(
    *,
    material_id: str,
    release_root: str | Path,
    work_root: str | Path,
    manifest_path: str | Path | None = None,
    pool_root: str | Path | None = None,
    python: str | None = None,
    expected_release_commit: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
    poll_seconds: float = 60.0,
    reuse_from: str | Path | None = None,
    active_check: Callable[[str], list[str]] | None = None,
    qsub: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Preflight then render + single qsub for one controlled material.

    ``dry_run=True`` performs the read-only preflight and returns the rendered
    script without writing state or calling qsub.  In real mode both
    ``active_check`` and ``qsub`` must be provided by the caller (CLI wires
    them to the site's qstat/qsub); this keeps submission side effects
    explicit and testable.
    """
    release = Path(release_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve() if manifest_path else release / DEFAULT_MANIFEST_REL
    pool = Path(pool_root).expanduser().resolve() if pool_root else release / DEFAULT_POOL_REL
    py = str(python) if python else sys.executable

    preflight = material_preflight(
        material_id=material_id,
        manifest_path=manifest,
        pool_root=pool,
        work_root=work_root,
        release_root=release,
        expected_release_commit=expected_release_commit,
        resume=resume,
        active_jobs=[] if dry_run else None,
    )
    if not preflight["passed"]:
        if dry_run and any(
            ("不在受控" in error or "非法" in error or "越界" in error) for error in preflight["errors"]
        ):
            raise ValueError("；".join(preflight["errors"]))
        return {**preflight, "mode": "DRY_RUN" if dry_run else "PREFLIGHT_FAILED", "script": None}
    if dry_run:
        script = render_stage1_pbs(
            material_id=material_id,
            release_root=release,
            pool_root=pool,
            work_root=work_root,
            python=py,
            poscar_path=preflight["poscar_path"],
            config_path=preflight["config_path"],
            dry_run=True,
            resume=resume,
            poll_seconds=poll_seconds,
            reuse_from=reuse_from,
        )
        return {
            **preflight,
            "mode": "DRY_RUN",
            "script": script,
            "manifest_path": str(manifest),
            "pool_root": str(pool),
        }

    if active_check is None or qsub is None:
        raise ValueError("真实提交需要 active_check 与 qsub 调用（--dry-run 之外不得绕过）。")
    jobs = [str(job) for job in active_check(material_id)]
    if jobs:
        preflight = material_preflight(
            material_id=material_id,
            manifest_path=manifest,
            pool_root=pool,
            work_root=work_root,
            release_root=release,
            expected_release_commit=expected_release_commit,
            resume=resume,
            active_jobs=jobs,
        )
        if not preflight["passed"]:
            return {**preflight, "mode": "PREFLIGHT_FAILED", "script": None}

    script = render_stage1_pbs(
        material_id=material_id,
        release_root=release,
        pool_root=pool,
        work_root=work_root,
        python=py,
        poscar_path=preflight["poscar_path"],
        config_path=preflight["config_path"],
        dry_run=False,
        resume=resume,
        poll_seconds=poll_seconds,
        reuse_from=reuse_from,
    )
    submissions = Path(work_root).expanduser().resolve() / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pbs_path = submissions / f"stage1-{material_id}-{stamp}.pbs"
    pbs_path.write_text(script, encoding="utf-8")
    spec_path = pbs_path.with_suffix(".json")
    spec_path.write_text(
        json.dumps({
            "schema_version": 1,
            "material_id": material_id,
            "pbs_path": str(pbs_path),
            "poscar_path": preflight["poscar_path"],
            "config_path": preflight["config_path"],
            "poscar_sha256": preflight["poscar_sha256"],
            "config_sha256": preflight["config_sha256"],
            "release_root": str(release),
            "pool_root": str(pool),
            "work_root": str(Path(work_root).expanduser().resolve()),
            "dry_run": False,
            "resume": resume,
            "reuse_from": str(Path(reuse_from).expanduser().resolve()) if reuse_from else None,
            "submitted_utc": stamp,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = qsub(str(pbs_path))
    return {
        **preflight,
        "mode": "SUBMITTED",
        "pbs_path": str(pbs_path),
        "spec_path": str(spec_path),
        "job_id": result.get("job_id"),
        "qsub_output": result.get("output"),
        "submitted_utc": stamp,
    }
