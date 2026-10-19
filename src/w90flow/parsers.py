from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _inverse_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(det) < 1e-12:
        raise ValueError("POSCAR 晶格矩阵不可逆。")
    cofactors = [
        [b[1] * c[2] - b[2] * c[1], -(b[0] * c[2] - b[2] * c[0]), b[0] * c[1] - b[1] * c[0]],
        [-(a[1] * c[2] - a[2] * c[1]), a[0] * c[2] - a[2] * c[0], -(a[0] * c[1] - a[1] * c[0])],
        [a[1] * b[2] - a[2] * b[1], -(a[0] * b[2] - a[2] * b[0]), a[0] * b[1] - a[1] * b[0]],
    ]
    return [[cofactors[j][i] / det for j in range(3)] for i in range(3)]


def _determinant_3x3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _row_times_matrix(row: list[float], matrix: list[list[float]]) -> list[float]:
    return [sum(row[index] * matrix[index][column] for index in range(3)) for column in range(3)]


def parse_poscar(path: str | Path) -> dict[str, Any]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR 太短：{path}")
    scale_token = float(lines[1].split()[0])
    if scale_token == 0:
        raise ValueError("POSCAR scale 不能为零。")
    raw_lattice = [[float(value) for value in lines[index].split()[:3]] for index in (2, 3, 4)]
    if any(len(vector) != 3 for vector in raw_lattice):
        raise ValueError(f"POSCAR 晶格行格式错误：{path}")
    if scale_token > 0:
        scale_factor = scale_token
    else:
        raw_volume = abs(_determinant_3x3(raw_lattice))
        if raw_volume < 1e-12:
            raise ValueError("POSCAR 负 scale 体积写法的原始晶格体积为零。")
        scale_factor = (abs(scale_token) / raw_volume) ** (1.0 / 3.0)
    lattice = [[scale_factor * value for value in vector] for vector in raw_lattice]
    element_tokens = lines[5].split()
    if element_tokens and all(token.lstrip("+-").isdigit() for token in element_tokens):
        raise ValueError("VASP 4 风格 POSCAR 没有元素名，无法自动核对 POTCAR。")
    counts = [int(token) for token in lines[6].split()]
    if len(element_tokens) != len(counts) or any(count <= 0 for count in counts):
        raise ValueError(f"POSCAR 元素/原子数行不一致：{path}")
    natoms = sum(counts)
    cursor = 7
    selective_dynamics = False
    if cursor < len(lines) and lines[cursor].lower().startswith("s"):
        selective_dynamics = True
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"POSCAR 缺少坐标模式：{path}")
    mode_token = lines[cursor].lower()
    cursor += 1
    if mode_token.startswith("d"):
        coordinate_mode = "direct"
    elif mode_token.startswith("c") or mode_token.startswith("k"):
        coordinate_mode = "cartesian"
    else:
        raise ValueError(f"POSCAR 坐标模式必须是 Direct/Cartesian：{path}")
    if cursor + natoms > len(lines):
        raise ValueError(f"POSCAR 原子坐标不足：{path}")
    coordinates: list[list[float]] = []
    for line in lines[cursor:cursor + natoms]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"POSCAR 原子坐标行格式错误：{path}")
        coordinates.append([float(value) for value in parts[:3]])
    if coordinate_mode == "direct":
        fractional = [[value % 1.0 for value in row] for row in coordinates]
    else:
        inverse = _inverse_3x3(lattice)
        fractional = [
            [value % 1.0 for value in _row_times_matrix([scale_factor * value for value in row], inverse)]
            for row in coordinates
        ]
    return {
        "comment": lines[0],
        "elements": element_tokens,
        "counts": counts,
        "natoms": natoms,
        "lattice_angstrom": lattice,
        "coordinate_mode": coordinate_mode,
        "selective_dynamics": selective_dynamics,
        "fractional_positions": fractional,
    }


def parse_eigenval(path: str | Path) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"EIGENVAL 太短：{path}")
    header = lines[5].split()
    if len(header) < 3:
        raise ValueError(f"无法读取 EIGENVAL 维数：{path}")
    nelect, nkpts, nbands = map(int, header[:3])
    cursor = 6
    energies: list[list[float]] = []
    kpoints: list[list[float]] = []
    for _ in range(nkpts):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise ValueError("EIGENVAL 在 k 点列表中提前结束。")
        kparts = lines[cursor].split()
        cursor += 1
        if len(kparts) < 4:
            raise ValueError("EIGENVAL k 点行格式错误。")
        kpoints.append([float(value) for value in kparts[:4]])
        row: list[float] = []
        for expected_band in range(1, nbands + 1):
            if cursor >= len(lines):
                raise ValueError("EIGENVAL 在能带列表中提前结束。")
            parts = lines[cursor].split()
            cursor += 1
            if len(parts) < 2 or int(parts[0]) != expected_band:
                raise ValueError("EIGENVAL band index 不连续。")
            row.append(float(parts[1]))
        energies.append(row)
    return {"nelect": nelect, "nkpts": nkpts, "nbands": nbands, "kpoints": kpoints, "energies_eV": energies}


def parse_hr_header(path: str | Path) -> dict[str, int]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"hr.dat 太短：{path}")
    return {"num_wann": int(lines[1].strip()), "num_R": int(lines[2].strip())}


def count_wannier_centres(path: str | Path) -> int:
    count = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "X" and len(parts) >= 4:
            count += 1
    return count


def parse_wout(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    spreads = [float(match.group(1).replace("D", "E")) for match in re.finditer(r"Omega Total\s*=\s*([-+0-9.EeDd]+)", text)]
    def stage_status(stage: str) -> bool | None:
        matches = re.findall(stage + r"\s+convergence criteria\s+(not\s+)?satisfied", text, re.I)
        return not bool(matches[-1]) if matches else None
    disentanglement = stage_status("disentanglement")
    localization = stage_status("wannieri[sz]ation")
    if localization is None and "CONVERGENCE ACHIEVED" in text:
        localization = True
    failure = ("DISENTANGLEMENT_NOT_CONVERGED" if disentanglement is False else
               "LOCALIZATION_NOT_CONVERGED" if localization is False else None)
    return {
        "completed": "All done" in text,
        "converged": localization is True and disentanglement is not False,
        "disentanglement_converged": disentanglement,
        "localization_converged": localization,
        "convergence_failure": failure,
        "final_total_spread_ang2": spreads[-1] if spreads else None,
    }


def parse_outcar_status(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    fatal_markers = ("VERY BAD NEWS", "ERROR:", "internal error in subroutine")
    fatal = [marker for marker in fatal_markers if marker.lower() in text.lower()]
    converged = (
        "aborting loop because EDIFF is reached" in text
        or "reached required accuracy" in text
    ) and not fatal
    match = re.search(r"NBANDS\s*=\s*(\d+)", text)
    return {"converged": converged, "fatal_markers": fatal, "nbands": int(match.group(1)) if match else None}


def parse_fermi_outcar(path: str | Path) -> float:
    """Return the final VASP Fermi energy in eV from an OUTCAR."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"E-fermi\s*:\s*([-+0-9.Ee]+)", text)
    if not matches:
        raise ValueError(f"OUTCAR 中没有 E-fermi：{path}")
    return float(matches[-1])


def parse_procar_orbital_scores(
    path: str | Path,
    poscar: dict[str, Any],
    energy_min_eV: float,
    energy_max_eV: float,
) -> dict[str, Any]:
    """Integrate the first (charge) projection table for states in an energy range.

    For non-collinear PROCAR files the later magnetisation tables are ignored here;
    they can become separate features in a later SOC-aware scorer.
    """
    ion_elements: list[str] = []
    for element, count in zip(poscar["elements"], poscar["counts"]):
        ion_elements.extend([element] * int(count))
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_match = re.search(r"# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)", text)
    if not header_match:
        raise ValueError(f"无法读取 PROCAR 维数：{path}")
    nkpts, nbands, nions = map(int, header_match.groups())
    if nions != len(ion_elements):
        raise ValueError(f"PROCAR 离子数 {nions} 与 POSCAR 原子数 {len(ion_elements)} 不同。")

    k_weight = 1.0 / max(nkpts, 1)
    band_energy: float | None = None
    first_table_consumed = False
    orbital_columns: list[str] = []
    state_scores: list[dict[str, float]] = []
    current_state: dict[str, float] | None = None
    accumulated: dict[str, float] = {}
    current_band_index: int | None = None
    band_accumulated: list[dict[str, float]] = [{} for _ in range(nbands)]

    k_re = re.compile(r"^\s*k-point\s+\d+.*weight\s*=\s*([-+0-9.EeDd]+)")
    band_re = re.compile(r"^\s*band\s+(\d+)\s+#\s+energy\s+([-+0-9.EeDd]+)")
    for line in lines:
        k_match = k_re.match(line)
        if k_match:
            k_weight = float(k_match.group(1).replace("D", "E"))
            continue
        band_match = band_re.match(line)
        if band_match:
            if current_state is not None:
                state_scores.append(current_state)
            current_band_index = int(band_match.group(1)) - 1
            band_energy = float(band_match.group(2).replace("D", "E"))
            first_table_consumed = False
            current_state = {} if energy_min_eV <= band_energy <= energy_max_eV else None
            continue
        parts = line.split()
        if parts and parts[0] == "ion":
            if first_table_consumed:
                orbital_columns = []
            else:
                orbital_columns = parts[1:]
                first_table_consumed = True
            continue
        if not orbital_columns or current_band_index is None or not parts or not parts[0].isdigit():
            continue
        ion_index = int(parts[0]) - 1
        if not 0 <= ion_index < len(ion_elements):
            raise ValueError("PROCAR ion index 越界。")
        values = [float(value) for value in parts[1:1 + len(orbital_columns)]]
        element = ion_elements[ion_index]
        by_l = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}
        for orbital, value in zip(orbital_columns, values):
            if orbital == "tot":
                continue
            if orbital == "s":
                by_l["s"] += value
            elif orbital.startswith("p"):
                by_l["p"] += value
            elif orbital.startswith("d") or orbital in {"x2-y2", "dx2"}:
                by_l["d"] += value
            elif orbital.startswith("f"):
                by_l["f"] += value
        for angular, value in by_l.items():
            if value:
                label = f"{element}:{angular}"
                weighted = k_weight * value
                band_accumulated[current_band_index][label] = band_accumulated[current_band_index].get(label, 0.0) + weighted
                if current_state is not None:
                    current_state[label] = current_state.get(label, 0.0) + value
                    accumulated[label] = accumulated.get(label, 0.0) + weighted
    if current_state is not None:
        state_scores.append(current_state)
    if not state_scores or not accumulated:
        raise ValueError("目标能区内没有可用的 PROCAR 投影状态。")
    total = sum(accumulated.values())
    normalized = {label: value / total for label, value in sorted(accumulated.items())}
    mean_projected_norm = sum(sum(state.values()) for state in state_scores) / len(state_scores)
    band_orbital_scores = []
    for index, raw_scores in enumerate(band_accumulated, 1):
        projected_norm = sum(raw_scores.values())
        normalized_band = {
            label: value / projected_norm for label, value in sorted(raw_scores.items())
        } if projected_norm > 0 else {}
        band_orbital_scores.append({
            "band_index_1_based": index,
            "projected_norm": projected_norm,
            "orbital_scores": normalized_band,
        })
    return {
        "nkpts": nkpts,
        "nbands": nbands,
        "nions": nions,
        "energy_range_eV": [energy_min_eV, energy_max_eV],
        "selected_state_count": len(state_scores),
        "mean_projected_norm": mean_projected_norm,
        "orbital_scores": normalized,
        "band_orbital_scores": band_orbital_scores,
    }


def convergence_gate(status: dict[str, Any], *, require_disentanglement: bool) -> str | None:
    """Require positive stage evidence, not merely successful termination."""
    if not status.get("completed"):
        return "W90_NOT_COMPLETED"
    if status.get("disentanglement_converged") is False:
        return "DISENTANGLEMENT_NOT_CONVERGED"
    if status.get("localization_converged") is False:
        return "LOCALIZATION_NOT_CONVERGED"
    if status.get("localization_converged") is not True:
        return "CONVERGENCE_UNVERIFIED"
    if require_disentanglement and status.get("disentanglement_converged") is not True:
        return "CONVERGENCE_UNVERIFIED"
    return None
