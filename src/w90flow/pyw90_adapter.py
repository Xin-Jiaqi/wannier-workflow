from __future__ import annotations

import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import pyw90_executable
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


def parse_pre_dos_output(text: str) -> dict[str, Any]:
    """Parse PyW90 ``pre dos`` into element shell/component labels.

    PyW90 can return a whole angular shell (``l=1``) or individual real
    harmonics (``px;py``).  Keeping the latter explicit is essential: silently
    widening ``px;py`` to a complete p shell changes the Hilbert-space
    dimension and invalidates a common-basis comparison.
    """
    clean = ANSI_ESCAPE.sub("", text).replace("\r", "")
    fermi = re.search(rf"Fermi level\s*:\s*({NUMBER})\s*eV", clean)
    gap = re.search(rf"DOS Gap\s*:\s*({NUMBER})\s*eV", clean)
    count = re.search(
        r"Number of WFs selected:\s*(\d+)\s*\(with degeneracy\s*(\d+)\)",
        clean,
    )
    block = re.search(
        r"Wannier90 Projection:\s*\n(.*?)(?:\n\s*\n|\npyw90 --extra input:)",
        clean,
        flags=re.DOTALL,
    )
    angular = {"0": "s", "1": "p", "2": "d", "3": "f"}
    components = {"s", "px", "py", "pz", "dxy", "dyz", "dz2", "dxz", "dx2", "dx2-y2"}
    projection: list[str] = []
    native_lines: list[str] = []
    if block:
        for raw in block.group(1).splitlines():
            line = raw.strip()
            if not line or ":" not in line:
                continue
            element, specification = line.split(":", 1)
            if not re.fullmatch(r"[A-Z][a-z]?", element):
                continue
            native_lines.append(line)
            for token in (part.strip() for part in specification.split(";")):
                angular_match = re.fullmatch(r"l=(\d+)", token)
                if angular_match and angular_match.group(1) in angular:
                    projection.append(f"{element}:{angular[angular_match.group(1)]}")
                elif token in components:
                    projection.append(f"{element}:{token}")
    extra = re.search(r"pyw90 --extra input:\s*\n([^\n]+)", clean)
    orbital_ids = {
        0: "s", 1: "py", 2: "pz", 3: "px",
        4: "dxy", 5: "dyz", 6: "dz2", 7: "dxz", 8: "dx2",
    }

    def expanded_indices(specification: str) -> list[int]:
        values: list[int] = []
        for token in specification.split("|"):
            token = token.strip()
            if re.fullmatch(r"\d+", token):
                values.append(int(token))
            elif match := re.fullmatch(r"(\d+)-(\d+)", token):
                start, stop = map(int, match.groups())
                if start > stop:
                    raise ValueError("PyW90 extra input 的索引范围必须递增。")
                values.extend(range(start, stop + 1))
            else:
                raise ValueError(f"无法解析 PyW90 extra input 索引：{token}")
        return list(dict.fromkeys(values))

    site_projections: list[dict[str, Any]] = []
    if extra is not None:
        for entry in extra.group(1).strip().split(";"):
            fields = [field.strip() for field in entry.split(",")]
            if len(fields) != 3 or not re.fullmatch(r"[A-Z][a-z]?", fields[0]):
                continue
            sites = expanded_indices(fields[1])
            indices = expanded_indices(fields[2])
            if any(index not in orbital_ids for index in indices):
                raise ValueError(f"PyW90 extra input 含不支持的轨道 id：{indices}")
            site_projections.append({
                "element": fields[0],
                "sites": sites,
                "orbitals": [orbital_ids[index] for index in indices],
            })
    return {
        "fermi_eV": None if fermi is None else float(fermi.group(1)),
        "dos_gap_eV": None if gap is None else float(gap.group(1)),
        "num_wann_selected": None if count is None else int(count.group(1)),
        "degeneracy": None if count is None else int(count.group(2)),
        "projection": list(dict.fromkeys(projection)),
        "wannier90_projection_lines": native_lines,
        "site_projections": site_projections,
        "extra_input": None if extra is None else extra.group(1).strip(),
        "raw_text": clean.strip(),
    }


def element_projection_envelope(
    result: dict[str, Any],
    element_counts: dict[str, int],
) -> dict[str, Any]:
    """Map site-specific PyW90 advice to the smallest element-level superset.

    W90Flow currently applies one projection specification to every atom of an
    element.  If PyW90 selects different components on symmetry-inequivalent
    sites, silently dropping those sites is wrong.  The auditable fallback is
    the union over all sites, with complete component sets collapsed to their
    angular shell.  The original site advice remains in the adapter document.
    """
    records = result.get("site_projections")
    if not isinstance(records, list) or not records:
        return {
            "projection": list(result.get("projection") or []),
            "representation": "native_element_lines",
            "site_uniform": None,
        }
    by_element: dict[str, dict[int, set[str]]] = {element: {} for element in element_counts}
    for record in records:
        element = str(record.get("element"))
        if element not in by_element:
            raise ValueError(f"PyW90 位点建议元素不在 POSCAR 中：{element}")
        orbitals = {str(value) for value in record.get("orbitals", [])}
        for site in record.get("sites", []):
            by_element[element].setdefault(int(site), set()).update(orbitals)

    shell_components = {
        "p": {"px", "py", "pz"},
        "d": {"dxy", "dyz", "dz2", "dxz", "dx2"},
    }
    orbital_order = ("s", "p", "px", "py", "pz", "d", "dxy", "dyz", "dz2", "dxz", "dx2")
    projection: list[str] = []
    uniform = True
    for element, count in element_counts.items():
        site_sets = list(by_element[element].values())
        if not site_sets:
            continue
        uniform = uniform and len(site_sets) == int(count) and all(values == site_sets[0] for values in site_sets[1:])
        envelope = set().union(*site_sets)
        for shell, components in shell_components.items():
            if components <= envelope:
                envelope.difference_update(components)
                envelope.add(shell)
        projection.extend(
            f"{element}:{orbital}" for orbital in orbital_order if orbital in envelope
        )
    return {
        "projection": projection,
        "representation": "exact_element_level" if uniform else "element_level_envelope",
        "site_uniform": uniform,
    }


def run_eig_suggest(
    eigenval: str | Path,
    *,
    num_wann: int,
    target_energy_eV: tuple[float, float],
    fermi_eV: float,
    energy_reference: str = "fermi_relative",
    executable: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run PyW90 as an optional, auditable baseline without changing decisions."""
    source = Path(eigenval).expanduser().resolve()
    tool = Path(executable or pyw90_executable() or "").expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"找不到 EIGENVAL：{source}")
    if not executable and not pyw90_executable():
        raise ValueError("未提供 PyW90 可执行文件：请通过 --executable 或 W90FLOW_PYW90_EXECUTABLE 环境变量指定。")
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


def run_dos_suggest(
    vasprun_xml: str | Path,
    *,
    target_energy_eV: tuple[float, float],
    degeneracy: int,
    lower_bound: float = 0.1,
    energy_reference: str = "fermi_relative",
    executable: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run PyW90 DOS projection analysis in an isolated scratch directory."""
    source = Path(vasprun_xml).expanduser().resolve()
    tool = Path(executable or pyw90_executable() or "").expanduser().resolve()
    if not source.is_file() or source.name != "vasprun.xml":
        raise ValueError("vasprun_xml 必须指向名为 vasprun.xml 的文件。")
    if not executable and not pyw90_executable():
        raise ValueError("未提供 PyW90 可执行文件：请通过 --executable 或 W90FLOW_PYW90_EXECUTABLE 环境变量指定。")
    if not tool.is_file():
        raise ValueError(f"找不到 PyW90 可执行文件：{tool}")
    low, high = map(float, target_energy_eV)
    if low >= high:
        raise ValueError("目标能区下限必须小于上限。")
    if degeneracy not in {1, 2}:
        raise ValueError("PyW90 DOS degeneracy 只能是 1 或 2。")
    if not 0.0 < float(lower_bound) <= 1.0:
        raise ValueError("PyW90 DOS lower_bound 必须在 (0, 1]。")
    if energy_reference not in {"absolute", "fermi_relative"}:
        raise ValueError("energy_reference 只能是 absolute 或 fermi_relative。")
    with tempfile.TemporaryDirectory(prefix="w90flow-pyw90-dos-") as scratch:
        scratch_path = Path(scratch)
        (scratch_path / "vasprun.xml").symlink_to(source)
        command = [
            str(tool), "pre", "dos", "--path", str(scratch_path),
            "-e", str(low), str(high), "--lb", str(float(lower_bound)),
            "--deg", str(degeneracy),
        ]
        if energy_reference == "fermi_relative":
            command.append("--rm-fermi")
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(scratch_path / "matplotlib")
        process = subprocess.run(
            command,
            cwd=scratch_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    parsed = parse_pre_dos_output(process.stdout)
    document = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "adapter_role": "optional_projection_baseline_not_authoritative_decision",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "tool": str(tool),
        "tool_sha256": sha256_file(tool),
        "command": command,
        "returncode": process.returncode,
        "input": {
            "target_energy_eV": [low, high],
            "energy_reference": energy_reference,
            "degeneracy": degeneracy,
            "lower_bound": float(lower_bound),
        },
        "result": parsed,
    }
    if process.returncode != 0:
        raise RuntimeError(f"PyW90 pre dos 失败，退出码 {process.returncode}\n{parsed['raw_text']}")
    if not parsed["projection"] or parsed["num_wann_selected"] is None:
        raise RuntimeError("PyW90 pre dos 成功退出但没有可解析的投影建议。")
    if output_json is not None:
        atomic_json(output_json, document)
    return document
