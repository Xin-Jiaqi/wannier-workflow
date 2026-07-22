from __future__ import annotations

import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import atomic_json, sha256_file


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_eig_suggest_output(text: str) -> dict[str, Any]:
    """Turn PyW90's human-readable ``eig suggest`` tables into JSON data."""
    clean = ANSI_ESCAPE.sub("", text).replace("\r", "")
    calculated = re.search(
        rf"Calculated Energy Range:\s*({NUMBER}),\s*({NUMBER})\s+with Fermi level\s+({NUMBER})",
        clean,
    )
    counts = re.search(
        rf"There are at most\s+(\d+)\s+states and at least\s+(\d+)\s+states in\s*\[({NUMBER}),\s*({NUMBER})\]",
        clean,
    )
    outer_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    table: str | None = None
    for line in clean.splitlines():
        if "`dis_win_min` and `dis_win_max` Table" in line:
            table = "outer"
            continue
        if "`dis_froz_min` and `dis_froz_max` Table" in line:
            table = "frozen"
            continue
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        try:
            if table == "outer" and len(fields) == 7:
                outer_rows.append({
                    "index": int(fields[0]),
                    "dis_win_min_eV": float(fields[1]),
                    "dis_win_max_eV": float(fields[2]),
                    "next_band_min_eV": float(fields[3]),
                    "current_band_max_eV": float(fields[4]),
                    "nleast": int(fields[5]),
                    "nmost": int(fields[6]),
                })
            elif table == "frozen" and len(fields) == 4:
                frozen_rows.append({
                    "index": int(fields[0]),
                    "dis_froz_min_eV": float(fields[1]),
                    "dis_froz_max_eV": float(fields[2]),
                    "states": int(fields[3]),
                })
        except ValueError:
            continue
    return {
        "calculated_energy_range_eV": None if calculated is None else [
            float(calculated.group(1)),
            float(calculated.group(2)),
        ],
        "fermi_eV": None if calculated is None else float(calculated.group(3)),
        "state_count": None if counts is None else {
            "maximum": int(counts.group(1)),
            "minimum": int(counts.group(2)),
            "reported_range_eV": [float(counts.group(3)), float(counts.group(4))],
        },
        "outer_window_suggestions": outer_rows,
        "frozen_window_suggestions": frozen_rows,
        "raw_text": clean.strip(),
    }


def run_eig_suggest(
    eigenval: str | Path,
    *,
    num_wann: int,
    target_energy_eV: tuple[float, float],
    fermi_eV: float,
    energy_reference: str = "fermi_relative",
    executable: str | Path = "/private/tmp/LocalVASP-mount/bin/pyw90",
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run PyW90 as an optional, auditable baseline without changing decisions."""
    source = Path(eigenval).expanduser().resolve()
    tool = Path(executable).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"找不到 EIGENVAL：{source}")
    if not tool.is_file():
        raise ValueError(f"找不到 PyW90 可执行文件：{tool}")
    if num_wann <= 0:
        raise ValueError("num_wann 必须大于零。")
    low, high = map(float, target_energy_eV)
    if low >= high:
        raise ValueError("目标能区下限必须小于上限。")
    if energy_reference not in {"absolute", "fermi_relative"}:
        raise ValueError("energy_reference 只能是 absolute 或 fermi_relative。")
    command = [
        str(tool),
        "eig",
        "suggest",
        "--path",
        str(source.parent),
        "-i",
        source.name,
        "-w",
        str(num_wann),
        "-e",
        str(low),
        str(high),
        "--efermi",
        str(float(fermi_eV)),
    ]
    if energy_reference == "fermi_relative":
        command.append("--rm-fermi")
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="w90flow-pyw90-") as cache:
        environment["MPLCONFIGDIR"] = cache
        process = subprocess.run(
            command,
            cwd=source.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    parsed = parse_eig_suggest_output(process.stdout)
    document = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "adapter_role": "optional_baseline_not_authoritative_decision",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "tool": str(tool),
        "tool_sha256": sha256_file(tool),
        "command": command,
        "returncode": process.returncode,
        "input": {
            "num_wann": num_wann,
            "target_energy_eV": [low, high],
            "fermi_eV": float(fermi_eV),
            "energy_reference": energy_reference,
        },
        "result": parsed,
    }
    if process.returncode != 0:
        raise RuntimeError(f"PyW90 eig suggest 失败，退出码 {process.returncode}\n{parsed['raw_text']}")
    if output_json is not None:
        atomic_json(output_json, document)
    return document
