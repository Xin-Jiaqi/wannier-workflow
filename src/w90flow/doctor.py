"""Stage-1 environment doctor (read-only; never launches materials or qsub).

Checks the explicit site contract the workflow already consumes:
``preprocess_inputs.commands.{02_scf,03_band,04_wannier,04_wannier90}``, the
POTCAR recipe/lock, Python dependencies, MPI/PBS availability, work-root
writability and cross-platform path risks.  The doctor introduces no second
site truth: it only reads the same config/site commands the stages use, and it
never falls back to ``Path.home()``/``W90FLOW_LAUNCHER_ROOT`` guessing.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-untyped]

SCHEMA_VERSION = 1
REQUIRED_IMPORTS = ("yaml", "numpy", "matplotlib")
OPTIONAL_IMPORTS = ("ase", "scipy")
VASP_STAGE_KEYS = ("02_scf", "03_band", "04_wannier")
W90_STAGE_KEY = "04_wannier90"
ALL_STAGE_KEYS = VASP_STAGE_KEYS + (W90_STAGE_KEY,)
MAC_ABSOLUTE_PREFIXES = ("/Users/", "/Volumes/")
CLUSTER_PATH_MARKERS = ("/work/", "/home/", "/scratch/", "/opt/pbs", "/share/")
PLACEHOLDER_RE = re.compile(r"\{stage_dir\}|\{run_dir\}")
UNKNOWN_PLACEHOLDER_RE = re.compile(r"\{[^}]*\}")
TARGET_PATH_RE = re.compile(r"(?<![A-Za-z0-9])(/[^\s'\"]*(?:vasp|wannier90)[^\s'\"]*)")
VASP_VERSION_RE = re.compile(r"vasp[._ /-]*v?[0-9]+(?:\.[0-9]+)+", re.IGNORECASE)
W90_VERSION_RE = re.compile(r"wannier90[._ /-]*[0-9]+(?:\.[0-9]+)+", re.IGNORECASE)

ImportProbe = Callable[[str], tuple[str, str]]

# SOC/spinor + VASP-Wannier90 interface capability contract.  VASP < 6.4 has a
# known-broken noncollinear Wannier90 interface (spinor projections crash with
# heap corruption after writing wannier90.mmn, so .amn/.eig are never produced).
SPINOR_WANNIER_CAPABILITY = "spinor_wannier_interface"
SPINOR_WANNIER_MIN_VASP = (6, 4)
SPINOR_WANNIER_MIN_W90 = (3, 0)


def parse_version(value: Any) -> tuple[int, ...] | None:
    """Parse a version declaration such as ``6.4.3`` into an integer tuple."""
    if not isinstance(value, str):
        return None
    parts = tuple(int(part) for part in re.findall(r"\d+", value))
    return parts or None


def capability_errors(config: dict[str, Any], commands: dict[str, Any] | None = None) -> list[str]:
    """Capability-contract violations for SOC/spinor Wannier runs.

    Non-spinor runs are free to follow the profile's default VASP selection
    (e.g. VASP 5.4.4).  Spinor Wannier runs must declare an explicit compatible
    site capability and the Wannier-stage launchers must come from it: no
    ``ncl``-default wrapper guessing and no material-id special cases.
    """
    physics = config.get("physics") or {}
    if not physics.get("soc"):
        return []
    backend = config.get("backend") or {}
    capability = backend.get(SPINOR_WANNIER_CAPABILITY)
    if not isinstance(capability, dict):
        return [
            f"backend.{SPINOR_WANNIER_CAPABILITY} 缺失：SOC/spinor Wannier 需要显式声明 "
            "VASP>=6.4 + Wannier90>=3 兼容 site 能力"
        ]
    problems: list[str] = []
    launcher = capability.get("vasp_launcher")
    if not isinstance(launcher, str) or not launcher:
        problems.append(f"backend.{SPINOR_WANNIER_CAPABILITY}.vasp_launcher 缺失")
    version = parse_version(capability.get("vasp_version"))
    if version is None:
        problems.append(f"backend.{SPINOR_WANNIER_CAPABILITY}.vasp_version 缺失/无法解析")
    elif version < SPINOR_WANNIER_MIN_VASP:
        problems.append(
            f"backend.{SPINOR_WANNIER_CAPABILITY}.vasp_version="
            f"{'.'.join(map(str, version))} < 6.4：VASP5.x NCL Wannier90 接口已知崩溃"
        )
    w90_launcher = capability.get("wannier90_launcher")
    if not isinstance(w90_launcher, str) or not w90_launcher:
        problems.append(f"backend.{SPINOR_WANNIER_CAPABILITY}.wannier90_launcher 缺失")
    w90_version = parse_version(capability.get("wannier90_version"))
    if w90_version is not None and w90_version < SPINOR_WANNIER_MIN_W90:
        problems.append(
            f"backend.{SPINOR_WANNIER_CAPABILITY}.wannier90_version="
            f"{'.'.join(map(str, w90_version))} < 3"
        )
    if isinstance(commands, dict):
        vasp_template = commands.get("04_wannier")
        if (
            isinstance(vasp_template, list)
            and vasp_template
            and isinstance(vasp_template[0], str)
            and launcher
            and str(vasp_template[0]) != str(launcher)
        ):
            problems.append(
                "preprocess_inputs.commands.04_wannier 的 launcher 与 "
                f"backend.{SPINOR_WANNIER_CAPABILITY}.vasp_launcher 不一致"
            )
        w90_template = commands.get("04_wannier90")
        if (
            isinstance(w90_template, list)
            and w90_template
            and isinstance(w90_template[0], str)
            and w90_launcher
            and str(w90_template[0]) != str(w90_launcher)
        ):
            problems.append(
                "preprocess_inputs.commands.04_wannier90 的 launcher 与 "
                f"backend.{SPINOR_WANNIER_CAPABILITY}.wannier90_launcher 不一致"
            )
    return problems


def _default_import_probe(name: str) -> tuple[str, str]:
    try:
        module = importlib.import_module(name)
        return "OK", str(getattr(module, "__version__", "ok"))
    except Exception as exc:  # pragma: no cover - exact exception is platform dependent
        return "MISSING", type(exc).__name__


def _item(item_id: str, name: str, status: str, detail: str = "") -> dict[str, str]:
    return {"id": item_id, "name": name, "status": status, "detail": detail}


def _platform_kind(platform: str) -> str:
    if platform == "darwin":
        return "mac"
    if platform.startswith("linux") or platform in {"", "other", "unknown"}:
        return "cluster"
    return "cluster"


def check_python(probe: ImportProbe) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    version = sys.version_info
    items.append(_item(
        "python.child", "child python (supervisor)", "PASS",
        f"{sys.executable}（supervisor 子进程继承 sys.executable / 显式 --python）",
    ))
    if version >= (3, 10):
        items.append(_item("python.version", "python version", "PASS",
                           f"{sys.version.split()[0]} (>=3.10)"))
    else:
        items.append(_item("python.version", "python version", "FAIL",
                           f"{sys.version.split()[0]} (<3.10)"))
    for name in REQUIRED_IMPORTS:
        status, detail = probe(name)
        items.append(_item(f"import.{name}", f"import {name}",
                           "PASS" if status == "OK" else "FAIL", detail))
    for name in OPTIONAL_IMPORTS:
        status, detail = probe(name)
        if status == "OK":
            items.append(_item(f"import.{name}", f"import {name} (optional)", "PASS", detail))
        else:
            items.append(_item(f"import.{name}", f"import {name} (optional)", "WARN",
                               f"{detail}: optional feature degraded"))
    return items


def _expand_placeholders(token: str, work_root: str) -> str:
    return token.replace("{stage_dir}", os.path.join(work_root, "doctor-stage")) \
                .replace("{run_dir}", work_root)


def _resolve_wrapper(template: list[str], work_root: str) -> tuple[str | None, str]:
    first = str(template[0])
    expanded = _expand_placeholders(first, work_root)
    if os.path.isabs(expanded):
        return expanded, "absolute"
    found = shutil.which(expanded)
    if found:
        return found, "which"
    if os.path.sep in expanded or expanded.endswith(".command"):
        return expanded, "relative"
    return expanded, "which"


def check_commands(config: dict[str, Any], work_root: str, base: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    inputs = config.get("preprocess_inputs")
    commands = inputs.get("commands") if isinstance(inputs, dict) else None
    if not isinstance(commands, dict):
        return [_item("commands.present", "preprocess_inputs.commands", "FAIL",
                      "配置缺失 commands（禁止回退到 launcher_root/home）")]
    missing = [key for key in ALL_STAGE_KEYS if key not in commands]
    if missing:
        return [_item("commands.present", "preprocess_inputs.commands", "FAIL",
                      "缺少 keys: " + ", ".join(missing))]
    items.append(_item("commands.present", "preprocess_inputs.commands", "PASS",
                       ", ".join(ALL_STAGE_KEYS)))

    physics = config.get("physics") or {}
    backend = config.get("backend") or {}
    kind = str(backend.get("vasp") or ("ncl" if physics.get("soc") else "std"))
    mpi = str(backend.get("mpi_processes") or 2)

    for key in VASP_STAGE_KEYS:
        template = commands[key]
        if not isinstance(template, list) or len(template) < 3 or \
                any(not isinstance(item, str) for item in template):
            items.append(_item(f"cmd.{key}", f"command {key}", "FAIL",
                               "必须是不短于 3 项的字符串列表"))
            continue
        ok_kind = str(template[-2]).strip().lower() == kind
        ok_mpi = str(template[-1]) == mpi
        items.append(_item(f"cmd.{key}", f"command {key}",
                           "PASS" if (ok_kind and ok_mpi) else "FAIL",
                           f"kind={template[-2]} mpi={template[-1]} vs backend {kind}/{mpi}"))
    template = commands[W90_STAGE_KEY]
    if not isinstance(template, list) or len(template) < 3 or \
            any(not isinstance(item, str) for item in template) or template[-1] != "wannier90":
        items.append(_item(f"cmd.{W90_STAGE_KEY}", f"command {W90_STAGE_KEY}", "FAIL",
                           "必须是不短于 3 项的字符串列表，末项固定为 wannier90"))
    else:
        items.append(_item(f"cmd.{W90_STAGE_KEY}", f"command {W90_STAGE_KEY}", "PASS",
                           f"seed=wannier90 len={len(template)}"))

    for key in ALL_STAGE_KEYS:
        template = commands[key]
        if not isinstance(template, list) or not template:
            continue
        for token in template:
            for match in UNKNOWN_PLACEHOLDER_RE.findall(str(token)):
                if not PLACEHOLDER_RE.fullmatch(match):
                    items.append(_item(f"cmd.{key}.placeholder", f"command {key} placeholder",
                                       "WARN", f"未知占位符 {match}"))
        bare = [token for token in template if token in {"python3", "python"}]
        if bare:
            items.append(_item(
                f"cmd.{key}.python", f"command {key} python", "FAIL",
                "命令含裸 python3/python；Stage1 子进程必须使用受控解释器（sys.executable/--python）",
            ))
        resolved, how = _resolve_wrapper(template, work_root)
        if not resolved:
            items.append(_item(f"wrap.{key}", f"wrapper {key}", "FAIL", "无法解析"))
            continue
        exists = os.path.exists(resolved)
        executable = os.access(resolved, os.X_OK)
        if exists and executable:
            items.append(_item(f"wrap.{key}", f"wrapper {key} ({how})", "PASS", resolved))
            items.extend(_scan_wrapper(key, resolved))
        elif exists:
            items.append(_item(f"wrap.{key}", f"wrapper {key} ({how})", "FAIL",
                               f"存在但不可执行: {resolved}"))
        else:
            items.append(_item(f"wrap.{key}", f"wrapper {key} ({how})", "FAIL",
                               f"不存在: {resolved}"))
    items.extend(check_interface_contract(config, commands, work_root))
    return items


def _scan_wrapper(stage_key: str, resolved: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    try:
        head = Path(resolved).read_bytes()[:8192]
    except OSError:
        return items
    if b"\x00" in head[:1024]:
        return items
    text = head.decode(errors="replace")
    version = VASP_VERSION_RE.search(text) or W90_VERSION_RE.search(text)
    for target in dict.fromkeys(TARGET_PATH_RE.findall(text)):
        label = "VASP" if stage_key != W90_STAGE_KEY else "W90"
        if os.path.exists(target):
            items.append(_item(f"target.{label}.{target}", f"{label} target", "PASS",
                               target + (f" (declared {version.group(0)})" if version else "")))
        else:
            items.append(_item(f"target.{label}.{target}", f"{label} target", "FAIL",
                               f"wrapper 声明但不存在: {target}"))
    return items


def _wrapper_declared_versions(resolved: str) -> list[tuple[str, tuple[int, ...]]]:
    """Static capability scan of a site launcher script (no execution)."""
    declared: list[tuple[str, tuple[int, ...]]] = []
    try:
        head = Path(resolved).read_bytes()[:32768]
    except OSError:
        return declared
    if b"\x00" in head[:1024]:
        return declared
    body = head.decode(errors="replace")
    for target in dict.fromkeys(TARGET_PATH_RE.findall(body)):
        vmatch = VASP_VERSION_RE.search(target)
        wmatch = W90_VERSION_RE.search(target)
        if vmatch:
            declared.append(("vasp", tuple(int(p) for p in re.findall(r"\d+", vmatch.group(0)))))
        elif wmatch:
            declared.append(("w90", tuple(int(p) for p in re.findall(r"\d+", wmatch.group(0)))))
    return declared


def check_interface_contract(
    config: dict[str, Any],
    commands: dict[str, Any],
    work_root: str,
) -> list[dict[str, str]]:
    """SOC/spinor Wannier requires a provably compatible VASP>=6.4 + W90>=3 site pair.

    The capability declaration is cross-checked against the resolved
    ``04_wannier`` wrapper: VASP 5.4.4 (or any wrapper that cannot statically
    prove VASP>=6.4) must FAIL, never PASS, for spinor Wannier runs.
    """
    if not config.get("physics", {}).get("soc"):
        return []
    problems = list(capability_errors(config, commands))
    if not problems:
        template = commands.get("04_wannier")
        if isinstance(template, list) and template:
            resolved, _ = _resolve_wrapper(template, work_root)
            if not resolved or not os.path.isfile(resolved):
                problems.append("04_wannier wrapper 无法静态解析：无法证明 VASP>=6.4 兼容")
            else:
                declared = _wrapper_declared_versions(resolved)
                vasp_versions = [version for label, version in declared if label == "vasp"]
                if not vasp_versions:
                    problems.append("04_wannier wrapper 未静态声明 VASP 版本（无法证明 spinor Wannier 兼容）")
                elif max(vasp_versions) < SPINOR_WANNIER_MIN_VASP:
                    problems.append(
                        "04_wannier wrapper 静态声明 VASP "
                        + ".".join(map(str, max(vasp_versions)))
                        + " < 6.4（VASP5.x NCL Wannier90 接口已知崩溃）"
                    )
    if problems:
        return [_item(
            "interface.vasp_wannier", "VASP-Wannier90 interface contract", "FAIL",
            "; ".join(problems),
        )]
    return [_item(
        "interface.vasp_wannier", "VASP-Wannier90 interface contract", "PASS",
        f"capability={SPINOR_WANNIER_CAPABILITY}: VASP>=6.4 + Wannier90>=3 已静态确认",
    )]


def _command_tokens(config: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    inputs = config.get("preprocess_inputs")
    if isinstance(inputs, dict):
        commands = inputs.get("commands")
        if isinstance(commands, dict):
            for template in commands.values():
                if isinstance(template, list):
                    tokens.extend(str(token) for token in template)
    return tokens


def check_platform_paths(config: dict[str, Any], config_text: str, platform: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    kind = _platform_kind(platform)
    tokens = _command_tokens(config)

    if kind == "cluster":
        mac_hits = [token for token in tokens
                    if token.startswith(MAC_ABSOLUTE_PREFIXES) or "Desktop" in token]
        text_hits = [line.strip()[:120] for line in config_text.splitlines()
                     if any(marker in line for marker in MAC_ABSOLUTE_PREFIXES) or "Desktop" in line]
        for hit in mac_hits:
            items.append(_item("platform.mac_path", "绝对 Mac 路径", "FAIL", hit))
        for line in text_hits[:5]:
            items.append(_item("platform.mac_path.config", "配置内 Mac 路径", "FAIL", line))
        if not mac_hits and not text_hits:
            items.append(_item("platform.mac_path", "绝对 Mac 路径", "PASS", "未发现 Mac 用户、外盘或桌面绝对路径"))
    else:
        cluster_hits = []
        for token in tokens:
            for marker in CLUSTER_PATH_MARKERS:
                if marker in token:
                    cluster_hits.append(token)
                    break
        for hit in dict.fromkeys(cluster_hits):
            exists = os.path.exists(hit)
            items.append(_item("platform.cluster_path", "集群绝对路径", "WARN" if exists else "FAIL",
                               hit + ("（本机存在，可能为挂载）" if exists else "（本机不存在）")))
        if not cluster_hits:
            items.append(_item("platform.cluster_path", "集群绝对路径", "PASS", "未发现 /work/ /home/ /scratch/ /opt/pbs /share/"))

    home_hits = [token for token in tokens if token.startswith("~") or token.startswith("$HOME")]
    for hit in dict.fromkeys(home_hits):
        items.append(_item("platform.home_path", "隐式 HOME 路径", "WARN", hit))
    if not home_hits:
        items.append(_item("platform.home_path", "隐式 HOME 路径", "PASS", "命令中无 ~/$HOME"))
    return items


def check_potcar(config: dict[str, Any], base: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    inputs = config.get("preprocess_inputs")
    potcar = inputs.get("potcar") if isinstance(inputs, dict) else None
    if not isinstance(potcar, dict) or not potcar:
        return [_item("potcar.spec", "POTCAR 配方", "FAIL", "preprocess_inputs.potcar 缺失")]
    paths: list[str] = []
    if isinstance(potcar.get("path"), str):
        paths.append(str(potcar["path"]))
    for component in potcar.get("components", []):
        if isinstance(component, dict) and isinstance(component.get("path"), str):
            paths.append(str(component["path"]))
    if not paths:
        items.append(_item("potcar.spec", "POTCAR 配方", "WARN", "path/components 均未给出显式文件路径"))
    else:
        items.append(_item("potcar.spec", "POTCAR 配方", "PASS", f"{len(paths)} 个文件"))
    for path in paths:
        resolved = Path(path) if os.path.isabs(path) else base / path
        items.append(_item(f"potcar.path.{path}", "POTCAR 文件", "PASS" if resolved.is_file() else "FAIL",
                           str(resolved)))
    pseudo = config.get("pseudopotentials")
    lock = pseudo.get("lock") if isinstance(pseudo, dict) else None
    if isinstance(lock, str):
        resolved = Path(lock) if os.path.isabs(lock) else base / lock
        items.append(_item("potcar.lock", "POTCAR lock", "PASS" if resolved.is_file() else "FAIL",
                           f"{resolved}（完整预检要求）"))
    elif not pseudo:
        items.append(_item("potcar.lock", "POTCAR lock", "WARN", "未配置 pseudopotentials（完整预检会要求 set+lock）"))
    return items


def check_work_root(work_root: str) -> list[dict[str, str]]:
    path = Path(work_root)
    if path.exists():
        status = "PASS" if os.access(path, os.W_OK) else "FAIL"
        return [_item("work.root", "work-root 可写", status, str(path))]
    parent = path.parent
    if parent.exists() and os.access(parent, os.W_OK):
        return [_item("work.root", "work-root 可写", "WARN", f"{path} 不存在（父目录可写，可由运行创建）")]
    return [_item("work.root", "work-root 可写", "FAIL", f"{path} 不存在且父目录不可写")]


def check_mpi_pbs(platform: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    kind = _platform_kind(platform)
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun:
        items.append(_item("mpi", "mpirun/mpiexec", "PASS", mpirun))
    else:
        items.append(_item("mpi", "mpirun/mpiexec", "FAIL" if kind == "cluster" else "WARN",
                           "未找到（本机规划模式可接受）" if kind == "mac" else "未找到"))
    qsub = shutil.which("qsub") or ("/opt/pbs/bin/qsub" if Path("/opt/pbs/bin/qsub").is_file() else None)
    qstat = shutil.which("qstat") or ("/opt/pbs/bin/qstat" if Path("/opt/pbs/bin/qstat").is_file() else None)
    if qsub and qstat:
        items.append(_item("pbs", "qsub/qstat", "PASS", f"{qsub} / {qstat}"))
    else:
        items.append(_item("pbs", "qsub/qstat", "FAIL" if kind == "cluster" else "WARN",
                           f"qsub={qsub} qstat={qstat}" + ("（本机无 PBS，设计上集群执行）" if kind == "mac" else "")))
    return items


def run_doctor(
    config_path: str,
    work_root: str,
    *,
    platform: str | None = None,
    import_probe: ImportProbe | None = None,
) -> dict[str, Any]:
    """Run all read-only Stage-1 environment checks and return the report."""
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "w90flow-doctor-mpl"))
    platform = platform or sys.platform
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config_text = handle.read()
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError(f"doctor：配置根节点必须是对象：{config_file}")
    base = config_file.parent

    items: list[dict[str, str]] = []
    items.append(_item("platform", "platform", "PASS", platform))
    items.append(_item("config", "config", "PASS", str(config_file.resolve())))
    items.extend(check_python(import_probe or _default_import_probe))
    items.extend(check_commands(config, work_root, base))
    items.extend(check_platform_paths(config, config_text, platform))
    items.extend(check_potcar(config, base))
    items.extend(check_work_root(work_root))
    items.extend(check_mpi_pbs(platform))

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    overall = "FAIL" if counts["FAIL"] else ("WARN" if counts["WARN"] else "PASS")
    return {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_file),
        "work_root": work_root,
        "platform": platform,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary": counts,
        "overall": overall,
        "items": items,
    }


def doctor_cli(args: Any) -> int:
    """CLI entry: prints human table or JSON; exit 1 on any FAIL."""
    report = run_doctor(args.config, args.work_root)
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["items"]:
            print("%-5s %-32s %s — %s" % (
                item["status"], item["id"], item["name"], item["detail"]))
        summary = report["summary"]
        print("SUMMARY pass=%d warn=%d fail=%d" % (
            summary.get("PASS", 0), summary.get("WARN", 0), summary.get("FAIL", 0)))
        print("OVERALL %s" % report["overall"])
    return 1 if report["overall"] == "FAIL" else 0
