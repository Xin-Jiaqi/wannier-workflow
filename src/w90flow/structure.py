from __future__ import annotations

import math
import re
from typing import Any


SUPPORTED_DIMENSIONS = {2, 3}


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _largest_periodic_gap(values: list[float]) -> float:
    ordered = sorted(value % 1.0 for value in values)
    if not ordered:
        raise ValueError("结构没有原子坐标。")
    if len(ordered) == 1:
        return 1.0
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    gaps.append(ordered[0] + 1.0 - ordered[-1])
    return max(gaps)


def infer_periodicity(poscar: dict[str, Any], vacuum_gap_threshold_A: float = 8.0) -> dict[str, Any]:
    """Infer periodic axes from the largest empty fractional gap on each axis.

    This is intentionally conservative and auditable.  An explicit override is
    required for unusual low-density 3D cells or axes close to the threshold.
    """
    if vacuum_gap_threshold_A <= 0:
        raise ValueError("vacuum_gap_threshold_A 必须大于零。")
    lattice = poscar["lattice_angstrom"]
    fractional = poscar.get("fractional_positions")
    if not fractional:
        raise ValueError("POSCAR 解析结果缺少 fractional_positions。")
    lengths = [_norm(vector) for vector in lattice]
    gap_fractions = [
        _largest_periodic_gap([row[axis] for row in fractional])
        for axis in range(3)
    ]
    gaps = [fraction * length for fraction, length in zip(gap_fractions, lengths)]
    periodic_axes = [gap < vacuum_gap_threshold_A for gap in gaps]
    dimension = sum(periodic_axes)
    return {
        "method": "largest_empty_fractional_gap_v1",
        "vacuum_gap_threshold_A": vacuum_gap_threshold_A,
        "lattice_lengths_A": lengths,
        "largest_empty_gap_fraction": gap_fractions,
        "largest_empty_gap_A": gaps,
        "periodic_axes": periodic_axes,
        "dimensionality": dimension,
        "minimum_threshold_margin_A": min(abs(gap - vacuum_gap_threshold_A) for gap in gaps),
    }


def resolve_periodicity(poscar: dict[str, Any], physics: dict[str, Any]) -> dict[str, Any]:
    threshold = float(physics.get("vacuum_gap_threshold_A", 8.0))
    inferred = infer_periodicity(poscar, threshold)
    declared = physics.get("dimensionality", "auto")
    axes_override = physics.get("periodic_axes")
    if axes_override is not None:
        if not isinstance(axes_override, list) or len(axes_override) != 3 or not all(isinstance(value, bool) for value in axes_override):
            raise ValueError("physics.periodic_axes 必须是三个布尔值。")
        periodic_axes = list(axes_override)
        dimension = sum(periodic_axes)
        source = "explicit_periodic_axes"
    elif declared == "auto" or declared is None:
        periodic_axes = list(inferred["periodic_axes"])
        dimension = int(inferred["dimensionality"])
        source = "inferred"
    else:
        try:
            dimension = int(declared)
        except (TypeError, ValueError) as exc:
            raise ValueError("physics.dimensionality 必须是 auto、2 或 3。") from exc
        if int(inferred["dimensionality"]) != dimension:
            raise ValueError(
                "显式 dimensionality 与 POSCAR 真空层推断不一致；"
                "请核对结构或同时声明 physics.periodic_axes。"
            )
        periodic_axes = list(inferred["periodic_axes"])
        source = "declared_and_geometry_confirmed"
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"当前生产流程只验收 2D/3D；该结构解析为 {dimension}D。")
    if declared not in {None, "auto"} and int(declared) != dimension:
        raise ValueError("physics.dimensionality 与 periodic_axes 不一致。")
    return {
        **inferred,
        "periodic_axes": periodic_axes,
        "dimensionality": dimension,
        "resolution_source": source,
    }


def reciprocal_lengths_Ainv(lattice: list[list[float]]) -> list[float]:
    a, b, c = lattice
    volume = _dot(a, _cross(b, c))
    if abs(volume) < 1e-12:
        raise ValueError("晶格体积为零。")
    factor = 2.0 * math.pi / abs(volume)
    return [
        factor * _norm(_cross(b, c)),
        factor * _norm(_cross(c, a)),
        factor * _norm(_cross(a, b)),
    ]


def resolve_uniform_mesh(
    mesh_spec: list[int] | dict[str, Any],
    lattice: list[list[float]],
    periodic_axes: list[bool],
) -> list[int]:
    if isinstance(mesh_spec, list):
        if len(mesh_spec) != 3:
            raise ValueError("uniform mesh 必须有三个整数。")
        mesh = [int(value) for value in mesh_spec]
    elif isinstance(mesh_spec, dict):
        spacing = mesh_spec.get("spacing_Ainv", mesh_spec.get("kspacing_Ainv"))
        if spacing is None or float(spacing) <= 0:
            raise ValueError("自动 mesh 需要正的 spacing_Ainv。")
        minimum = int(mesh_spec.get("minimum", 1))
        maximum = int(mesh_spec.get("maximum", 99))
        if minimum <= 0 or maximum < minimum:
            raise ValueError("mesh minimum/maximum 无效。")
        reciprocal = reciprocal_lengths_Ainv(lattice)
        mesh = [
            min(maximum, max(minimum, int(math.ceil(length / float(spacing))))) if periodic else 1
            for length, periodic in zip(reciprocal, periodic_axes)
        ]
    else:
        raise ValueError("mesh 必须是 [nx,ny,nz] 或包含 spacing_Ainv 的对象。")
    if any(value <= 0 for value in mesh):
        raise ValueError("mesh 所有分量必须为正整数。")
    for axis, periodic in enumerate(periodic_axes):
        if not periodic and mesh[axis] != 1:
            raise ValueError(f"非周期轴 {axis} 的 k 网格必须为 1。")
    return mesh


def render_gamma_kpoints(mesh: list[int], comment: str = "W90Flow automatic Gamma mesh") -> str:
    return f"{comment}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"


def uniform_fractional_kpoints(mesh: list[int]) -> list[list[float]]:
    """Use VASP automatic-mesh indexing: x varies fastest, then y, then z.

    Matching point sets is insufficient: eigenvalues and overlap matrices are
    indexed by this order. The post-VASP IBZKPT comparison remains mandatory.
    """
    nx, ny, nz = mesh
    return [
        [ix / nx, iy / ny, iz / nz]
        for iz in range(nz)
        for iy in range(ny)
        for ix in range(nx)
    ]


def patch_wannier_uniform_grid(incar_text: str, mesh: list[int]) -> str:
    inline = re.search(r'(?ims)(WANNIER90_WIN\s*=\s*"\s*\n)(.*?)(\n")', incar_text)
    if not inline:
        raise ValueError("Wannier INCAR 缺少多行 WANNIER90_WIN。")
    win = inline.group(2)
    replacement = f"mp_grid = {mesh[0]} {mesh[1]} {mesh[2]}"
    win, count = re.subn(r"(?im)^\s*mp_grid\s*[=:].*$", replacement, win)
    if count == 0:
        win = replacement + "\n" + win
    elif count != 1:
        raise ValueError("WANNIER90_WIN 中 mp_grid 必须至多出现一次。")
    block = "begin kpoints\n" + "\n".join(
        "  {:.12f} {:.12f} {:.12f}".format(*point)
        for point in uniform_fractional_kpoints(mesh)
    ) + "\nend kpoints"
    win, count = re.subn(r"(?ims)^\s*begin\s+kpoints\s*$.*?^\s*end\s+kpoints\s*$", block, win)
    if count == 0:
        win = win.rstrip() + "\n\n" + block
    elif count != 1:
        raise ValueError("WANNIER90_WIN 中 kpoints block 必须至多出现一次。")
    return incar_text[:inline.start(2)] + win + incar_text[inline.end(2):]


def generate_ase_band_path(
    poscar: dict[str, Any],
    periodic_axes: list[bool],
    points_per_segment: int = 40,
) -> dict[str, Any]:
    """Generate a deterministic lattice path with ASE; no material name rules."""
    if points_per_segment < 2:
        raise ValueError("points_per_segment 必须至少为 2。")
    try:
        from ase.cell import Cell
        from ase.dft.kpoints import parse_path_string
    except ImportError as exc:
        raise RuntimeError("自动 band path 需要可选依赖 ASE；或改用 band KPOINTS 模板。") from exc
    path = Cell(poscar["lattice_angstrom"]).bandpath(npoints=points_per_segment, pbc=periodic_axes)
    parsed = parse_path_string(path.path)
    pairs: list[tuple[str, str]] = []
    for branch in parsed:
        pairs.extend(zip(branch[:-1], branch[1:]))
    if not pairs:
        raise ValueError("ASE 未生成可用的高对称路径。")
    lines = ["W90Flow ASE lattice band path", str(points_per_segment), "Line-mode", "Reciprocal"]
    w90_lines: list[str] = []
    for left, right in pairs:
        start = [float(value) for value in path.special_points[left]]
        stop = [float(value) for value in path.special_points[right]]
        lines.append(f"{start[0]:.12f} {start[1]:.12f} {start[2]:.12f} ! {left}")
        lines.append(f"{stop[0]:.12f} {stop[1]:.12f} {stop[2]:.12f} ! {right}")
        lines.append("")
        w90_lines.append(
            f"{left} {start[0]:.12f} {start[1]:.12f} {start[2]:.12f} "
            f"{right} {stop[0]:.12f} {stop[1]:.12f} {stop[2]:.12f}"
        )
    return {
        "provider": "ase_cell_bandpath",
        "path": path.path,
        "points_per_segment": points_per_segment,
        "segments": [[left, right] for left, right in pairs],
        "kpoints_text": "\n".join(lines).rstrip() + "\n",
        "wannier_kpoint_path": w90_lines,
    }


def patch_wannier_band_path(incar_text: str, path_lines: list[str]) -> str:
    inline = re.search(r'(?ims)(WANNIER90_WIN\s*=\s*"\s*\n)(.*?)(\n")', incar_text)
    if not inline:
        raise ValueError("Wannier INCAR 缺少多行 WANNIER90_WIN。")
    win = inline.group(2)
    block = "begin kpoint_path\n" + "\n".join(path_lines) + "\nend kpoint_path"
    win, count = re.subn(r"(?ims)^\s*begin\s+kpoint_path\s*$.*?^\s*end\s+kpoint_path\s*$", block, win)
    if count == 0:
        win = win.rstrip() + "\n\n" + block
    elif count != 1:
        raise ValueError("WANNIER90_WIN 中 kpoint_path block 必须至多出现一次。")
    return incar_text[:inline.start(2)] + win + incar_text[inline.end(2):]
