from __future__ import annotations

import bisect
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parsers import count_wannier_centres, parse_eigenval, parse_fermi_outcar, parse_hr_header, parse_poscar, parse_wout
from .state import atomic_json, read_json, sha256_file, update_stage


class IncompleteTargetSubspace(ValueError):
    """The Wannier model is too small to represent the protected DFT manifold."""

    def __init__(self, target_band_count: int, model_dimension: int):
        self.target_band_count = target_band_count
        self.model_dimension = model_dimension
        super().__init__(
            "目标 DFT 能带数大于 Wannier 模型维数："
            f"{target_band_count} > {model_dimension}。"
        )


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def reciprocal_lattice(lattice: list[list[float]]) -> list[list[float]]:
    volume = _dot(lattice[0], _cross(lattice[1], lattice[2]))
    if abs(volume) < 1.0e-12:
        raise ValueError("POSCAR 晶格体积为零。")
    factor = 2.0 * math.pi / volume
    return [
        [factor * value for value in _cross(lattice[1], lattice[2])],
        [factor * value for value in _cross(lattice[2], lattice[0])],
        [factor * value for value in _cross(lattice[0], lattice[1])],
    ]


def dft_band_blocks(eigenval: dict[str, Any], lattice: list[list[float]]) -> list[list[tuple[float, float]]]:
    reciprocal = reciprocal_lattice(lattice)
    path = [0.0]
    kpoints = [item[:3] for item in eigenval["kpoints"]]
    for previous, current in zip(kpoints, kpoints[1:]):
        delta_frac = [current[index] - previous[index] for index in range(3)]
        delta_cart = [sum(delta_frac[i] * reciprocal[i][j] for i in range(3)) for j in range(3)]
        path.append(path[-1] + _norm(delta_cart))
    return [
        [(path[k], eigenval["energies_eV"][k][band]) for k in range(eigenval["nkpts"])]
        for band in range(eigenval["nbands"])
    ]


def read_w90_band_blocks(path: str | Path) -> list[list[tuple[float, float]]]:
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if not parts:
            if current:
                blocks.append(current)
                current = []
            continue
        if len(parts) >= 2:
            current.append((float(parts[0]), float(parts[1])))
    if current:
        blocks.append(current)
    if not blocks or len({len(block) for block in blocks}) != 1:
        raise ValueError(f"Wannier band blocks 为空或长度不一致：{path}")
    return blocks


def _interpolate(block: list[tuple[float, float]], targets: list[float]) -> list[float]:
    xs = [point[0] for point in block]
    ys = [point[1] for point in block]
    tolerance = 1.0e-6 * max(1.0, abs(xs[0]), abs(xs[-1]))
    if targets[0] < xs[0] - tolerance or targets[-1] > xs[-1] + tolerance:
        raise ValueError("DFT path 超出 Wannier path，无法比较。")
    values: list[float] = []
    for target in targets:
        index = bisect.bisect_left(xs, target)
        if index <= 0:
            values.append(ys[0])
        elif index >= len(xs):
            values.append(ys[-1])
        else:
            x0, x1 = xs[index - 1], xs[index]
            y0, y1 = ys[index - 1], ys[index]
            fraction = 0.0 if abs(x1 - x0) < 1.0e-15 else (target - x0) / (x1 - x0)
            values.append(y0 + fraction * (y1 - y0))
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def band_metrics(
    dft_blocks: list[list[tuple[float, float]]],
    w90_blocks: list[list[tuple[float, float]]],
    occupied_count: int,
    *,
    error_energy_window_eV: tuple[float, float] | None = None,
) -> tuple[dict[str, Any], list[list[float]]]:
    if len(dft_blocks) != len(w90_blocks):
        raise ValueError("DFT/Wannier 能带数不同。")
    if not 0 < occupied_count < len(dft_blocks):
        raise ValueError("占据带数不在比较能带范围内。")
    if (
        error_energy_window_eV is not None
        and error_energy_window_eV[0] >= error_energy_window_eV[1]
    ):
        raise ValueError("误差统计能量窗口必须递增。")
    errors_mev: list[float] = []
    interpolated: list[list[float]] = []
    excluded_point_count = 0
    for dft, w90 in zip(dft_blocks, w90_blocks):
        targets = [point[0] for point in dft]
        values = _interpolate(w90, targets)
        interpolated.append(values)
        for w90_energy, dft_point in zip(values, dft):
            if (
                error_energy_window_eV is not None
                and not error_energy_window_eV[0] <= dft_point[1] <= error_energy_window_eV[1]
            ):
                excluded_point_count += 1
                continue
            errors_mev.append(1000.0 * (w90_energy - dft_point[1]))
    if not errors_mev:
        raise ValueError("声明的目标能量窗口内没有可比较的 DFT 能带点。")
    absolute = [abs(value) for value in errors_mev]
    dft_valence = max(point[1] for band in dft_blocks[:occupied_count] for point in band)
    dft_conduction = min(point[1] for band in dft_blocks[occupied_count:] for point in band)
    w90_valence = max(value for band in interpolated[:occupied_count] for value in band)
    w90_conduction = min(value for band in interpolated[occupied_count:] for value in band)
    metrics = {
        "mae_mev": sum(absolute) / len(absolute),
        "rmse_mev": math.sqrt(sum(value * value for value in errors_mev) / len(errors_mev)),
        "p95_mev": _percentile(absolute, 95.0),
        "max_mev": max(absolute),
        "dft_indirect_gap_ev": dft_conduction - dft_valence,
        "w90_indirect_gap_ev": w90_conduction - w90_valence,
        "gap_error_mev": 1000.0 * ((w90_conduction - w90_valence) - (dft_conduction - dft_valence)),
        "compared_point_count": len(errors_mev),
        "excluded_point_count": excluded_point_count,
        "error_energy_window_eV": (
            list(error_energy_window_eV)
            if error_energy_window_eV is not None
            else None
        ),
    }
    return metrics, interpolated


def _parse_excluded(win: Path, nbands: int) -> list[int]:
    text = win.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?im)^\s*exclude_bands\s*[=:]\s*([^!\n]+)", text)
    if not match:
        return []
    excluded: set[int] = set()
    for token in match.group(1).replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start, stop = map(int, token.split("-", 1))
            excluded.update(range(start, stop + 1))
        else:
            excluded.add(int(token))
    if any(index < 1 or index > nbands for index in excluded):
        raise ValueError("exclude_bands 超出 DFT 能带范围。")
    return sorted(excluded)


def _parse_projections(win: Path) -> list[str]:
    text = win.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?ims)^\s*begin\s+projections\s*$\n(.*?)^\s*end\s+projections\s*$", text)
    if not match:
        raise ValueError("wannier90.win 缺少 projections block。")
    return [line.split("!", 1)[0].strip() for line in match.group(1).splitlines() if line.split("!", 1)[0].strip()]


def _candidate_name(root: Path, sid: str, projection: list[str]) -> str:
    candidate_path = root / "03_candidates/candidate-spec.json"
    if not candidate_path.is_file():
        return "baseline-template"
    import json
    with candidate_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("ensemble_mode") == "common_basis":
        entries = document.get("shared_candidates", [])
    else:
        entries = document.get("independent_candidates", {}).get(sid, [])
    for entry in entries:
        if entry.get("projection") == projection:
            return str(entry.get("candidate"))
    return "baseline-template"


def _plot_comparison(
    output: Path,
    dft: list[list[tuple[float, float]]],
    w90: list[list[tuple[float, float]]],
    fermi: float,
    title: str,
    metrics: dict[str, Any],
    display_energy_window_eV: tuple[float, float] | None = None,
) -> None:
    cache = Path("/private/tmp/LocalVASP-mount/cache/w90flow-matplotlib")
    if not cache.parent.is_dir():
        cache = output.parents[4] / "cache/w90flow-matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for index, band in enumerate(dft):
        ax.plot([point[0] for point in band], [point[1] - fermi for point in band], color="black", lw=1.0, label="VASP" if index == 0 else None)
    for index, band in enumerate(w90):
        ax.plot([point[0] for point in band], [point[1] - fermi for point in band], color="#d62728", lw=1.0, ls="--", label="Wannier90" if index == 0 else None)
    ax.axhline(0.0, color="#777777", lw=0.8, ls=":")
    ax.set_xlabel("k path")
    ax.set_ylabel(r"Energy $-E_F$ (eV)")
    if display_energy_window_eV is not None:
        ax.set_ylim(*display_energy_window_eV)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.text(0.98, 0.02, f"MAE {metrics['mae_mev']:.3f} meV\nP95 {metrics['p95_mev']:.3f} meV\nmax {metrics['max_mev']:.3f} meV", transform=ax.transAxes, ha="right", va="bottom")
    fig.savefig(output)
    plt.close(fig)


def evaluate_run(run_dir: str | Path) -> dict[str, Any]:
    import yaml

    root = Path(run_dir).expanduser().resolve()
    with (root / "project.snapshot.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    soc = bool(config["physics"].get("soc"))
    gate_cfg = config.get("ranking", {}).get("hard_gates", {})
    thresholds = {
        "rmse_mev": float(gate_cfg.get("rmse_meV", 10.0)),
        "p95_mev": float(gate_cfg.get("p95_meV", 20.0)),
        "max_mev": float(gate_cfg.get("max_meV", 50.0)),
        "abs_gap_error_mev": float(gate_cfg.get("gap_error_meV", 10.0)),
    }
    results: dict[str, Any] = {}
    for structure in config["structures"]:
        sid = str(structure["id"])
        base = root / "structures" / sid
        band_dir = base / "02_band"
        wannier_dir = base / "04_wannier"
        quality_dir = base / "05_quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        poscar = parse_poscar(base / "00_input/POSCAR")
        eig = parse_eigenval(band_dir / "EIGENVAL")
        dft_all = dft_band_blocks(eig, poscar["lattice_angstrom"])
        w90 = read_w90_band_blocks(wannier_dir / "wannier90_band.dat")
        projection = _parse_projections(wannier_dir / "wannier90.win")
        candidate_name = _candidate_name(root, sid, projection)
        excluded = set(_parse_excluded(wannier_dir / "wannier90.win", eig["nbands"]))
        retained = [index for index in range(1, eig["nbands"] + 1) if index not in excluded]
        if len(retained) != len(w90):
            raise ValueError("解纠缠模型需要显式 quality band mapping；自动 exact mapping 不适用。")
        dft = [dft_all[index - 1] for index in retained]
        original_occupied = int(round(eig["nelect"] if soc else eig["nelect"] / 2.0))
        occupied = sum(index <= original_occupied for index in retained)
        metrics, _ = band_metrics(dft, w90, occupied)
        hr = parse_hr_header(wannier_dir / "wannier90_hr.dat")
        centres = count_wannier_centres(wannier_dir / "wannier90_centres.xyz")
        wout = parse_wout(wannier_dir / "wannier90.wout")
        checks = {
            "w90_completed": wout["completed"],
            "hr_centres_match": hr["num_wann"] == centres == len(w90),
            "rmse": metrics["rmse_mev"] <= thresholds["rmse_mev"],
            "p95": metrics["p95_mev"] <= thresholds["p95_mev"],
            "max": metrics["max_mev"] <= thresholds["max_mev"],
            "gap": abs(metrics["gap_error_mev"]) <= thresholds["abs_gap_error_mev"],
        }
        passed = all(checks.values())
        fermi = parse_fermi_outcar(base / "01_scf/OUTCAR")
        plot = quality_dir / "VASP_W90_cmp.pdf"
        _plot_comparison(plot, dft, w90, fermi, f"{sid} / baseline", metrics)
        result = {
            "schema_version": 1,
            "structure": sid,
            "candidate": candidate_name,
            "projection": projection,
            "soc": soc,
            "retained_vasp_bands_1_based": retained,
            "num_wann": hr["num_wann"],
            "nocc": occupied,
            "fermi_eV": fermi,
            "metrics": metrics,
            "thresholds": thresholds,
            "final_total_spread_ang2": wout["final_total_spread_ang2"],
            "centres_count": centres,
            "checks": checks,
            "passed": passed,
            "comparison_pdf": str(plot),
            "source_sha256": {
                "EIGENVAL": sha256_file(band_dir / "EIGENVAL"),
                "wannier90_hr.dat": sha256_file(wannier_dir / "wannier90_hr.dat"),
                "wannier90.wout": sha256_file(wannier_dir / "wannier90.wout"),
            },
        }
        atomic_json(quality_dir / "quality.json", result)
        results[sid] = result
    passed = all(result["passed"] for result in results.values())
    update_stage(root, "05_quality", status="PASS" if passed else "FAIL", outputs=results)
    return {"passed": passed, "structures": results}


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    gate_cfg = config.get("ranking", {}).get("hard_gates", {})
    return {
        "rmse_mev": float(gate_cfg.get("rmse_meV", 10.0)),
        "p95_mev": float(gate_cfg.get("p95_meV", 20.0)),
        "max_mev": float(gate_cfg.get("max_meV", 50.0)),
        "abs_gap_error_mev": float(gate_cfg.get("gap_error_meV", 10.0)),
    }


def _metric_checks(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, bool]:
    return {
        "rmse": metrics["rmse_mev"] <= thresholds["rmse_mev"],
        "p95": metrics["p95_mev"] <= thresholds["p95_mev"],
        "max": metrics["max_mev"] <= thresholds["max_mev"],
        "gap": abs(metrics["gap_error_mev"]) <= thresholds["abs_gap_error_mev"],
    }


def _best_contiguous_w90_slice(
    dft: list[list[tuple[float, float]]],
    w90: list[list[tuple[float, float]]],
    occupied_count: int,
    *,
    error_energy_window_eV: tuple[float, float] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Map a disentangled model to the target bands without manual band labels."""
    if len(dft) > len(w90):
        raise IncompleteTargetSubspace(len(dft), len(w90))
    ranked: list[tuple[tuple[float, float, float], int, dict[str, Any]]] = []
    for start in range(len(w90) - len(dft) + 1):
        metrics, _ = band_metrics(
            dft,
            w90[start:start + len(dft)],
            occupied_count,
            error_energy_window_eV=error_energy_window_eV,
        )
        ranked.append(((metrics["rmse_mev"], metrics["p95_mev"], metrics["max_mev"]), start, metrics))
    _, start, metrics = min(ranked, key=lambda item: item[0])
    return start, metrics


def evaluate_attempt(
    run_dir: str | Path,
    structure: str,
    candidate: str,
    attempt: str,
) -> dict[str, Any]:
    """Evaluate one immutable projection/window attempt on the declared target bands."""
    import yaml

    root = Path(run_dir).expanduser().resolve()
    with (root / "project.snapshot.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    directory = root / "structures" / structure / "04_wannier_attempts" / candidate / attempt
    record_path = directory / "attempt.json"
    record = read_json(record_path)
    if record.get("status") not in {"PASS", "ACCEPTED", "REJECTED_QUALITY"}:
        raise ValueError(f"attempt 尚未成功完成数值计算：{record.get('status')}")

    base = root / "structures" / structure
    band_dir = base / "02_band"
    eig = parse_eigenval(band_dir / "EIGENVAL")
    poscar = parse_poscar(base / "00_input/POSCAR")
    dft_all = dft_band_blocks(eig, poscar["lattice_angstrom"])
    w90_all = read_w90_band_blocks(directory / "wannier90_band.dat")
    fermi = parse_fermi_outcar(base / "01_scf/OUTCAR")
    target_window_declared = tuple(float(value) for value in config["physics"]["target_energy_eV"])
    energy_reference = str(config["physics"].get("energy_reference", "absolute"))
    if energy_reference == "fermi_relative":
        target_window_absolute = (
            fermi + target_window_declared[0],
            fermi + target_window_declared[1],
        )
        display_energy_window = target_window_declared
    else:
        target_window_absolute = target_window_declared
        display_energy_window = (
            target_window_declared[0] - fermi,
            target_window_declared[1] - fermi,
        )
    target_indices = list(record["plan"]["target_bands_1_based"])
    dft_target = [dft_all[index - 1] for index in target_indices]
    soc = bool(config["physics"].get("soc"))
    original_occupied = int(round(eig["nelect"] if soc else eig["nelect"] / 2.0))
    target_occupied = sum(index <= original_occupied for index in target_indices)
    if not 0 < target_occupied < len(target_indices):
        raise ValueError("目标能区必须同时包含占据带和空带，才能评估带隙与光学目标。")

    plan = record["plan"]
    native_metrics = None
    if plan["mode"] == "exact":
        retained_start, retained_stop = plan["retained_vasp_bands_1_based"]
        retained = list(range(retained_start, retained_stop + 1))
        if len(retained) != len(w90_all):
            raise ValueError("exact attempt 的 DFT/Wannier 模型维数不一致。")
        model_indices = [index - retained_start for index in target_indices]
        w90_target = [w90_all[index] for index in model_indices]
        model_occupied = sum(index <= original_occupied for index in retained)
        native_metrics, _ = band_metrics([dft_all[index - 1] for index in retained], w90_all, model_occupied)
        mapping = {
            "method": "exact_exclude_bands",
            "retained_vasp_bands_1_based": retained,
            "target_w90_bands_1_based": [index + 1 for index in model_indices],
        }
        target_metrics, _ = band_metrics(
            dft_target,
            w90_target,
            target_occupied,
            error_energy_window_eV=target_window_absolute,
        )
    else:
        start, target_metrics = _best_contiguous_w90_slice(
            dft_target,
            w90_all,
            target_occupied,
            error_energy_window_eV=target_window_absolute,
        )
        w90_target = w90_all[start:start + len(dft_target)]
        model_offset = target_indices[0] - (start + 1)
        model_occupied = original_occupied - model_offset
        if not 0 < model_occupied < len(w90_all):
            raise ValueError("由 DFT/Wannier 目标带映射得到的完整模型 NOCC 非法。")
        mapping = {
            "method": "minimum_rmse_contiguous_slice",
            "target_w90_bands_1_based": list(range(start + 1, start + len(dft_target) + 1)),
        }

    thresholds = _thresholds(config)
    metric_checks = _metric_checks(target_metrics, thresholds)
    hr = parse_hr_header(directory / "wannier90_hr.dat")
    centres = count_wannier_centres(directory / "wannier90_centres.xyz")
    wout = parse_wout(directory / "wannier90.wout")
    checks = {
        "w90_completed": bool(wout["completed"]),
        "hr_centres_match": hr["num_wann"] == centres == len(w90_all),
        **metric_checks,
    }
    passed = all(checks.values())
    failure_class = None
    if not passed:
        failure_class = "BAND_CONNECTIVITY_FAILED" if metric_checks["p95"] and not metric_checks["max"] else "QUALITY_GATE_FAILED"

    plot = directory / "VASP_W90_cmp.pdf"
    _plot_comparison(
        plot,
        dft_target,
        w90_target,
        fermi,
        f"{structure} / {candidate}",
        target_metrics,
        display_energy_window,
    )
    projection = _parse_projections(directory / "wannier90.win")
    spread = wout["final_total_spread_ang2"]
    result = {
        "schema_version": 1,
        "structure": structure,
        "candidate": candidate,
        "attempt": attempt,
        "projection": projection,
        "soc": soc,
        "plan_mode": plan["mode"],
        "target_vasp_bands_1_based": target_indices,
        "band_mapping": mapping,
        "num_wann": hr["num_wann"],
        "nocc_target": target_occupied,
        "nocc_model": model_occupied,
        "fermi_eV": fermi,
        "comparison_scope": {
            "energy_reference": energy_reference,
            "declared_target_energy_eV": list(target_window_declared),
            "absolute_target_energy_eV": list(target_window_absolute),
            "error_mask": "include points whose DFT energy lies inside the declared target window",
            "gap_check": "indirect gap evaluated independently on the mapped target bands",
        },
        "target_metrics": target_metrics,
        "native_model_metrics": native_metrics,
        "thresholds": thresholds,
        "final_total_spread_ang2": spread,
        "mean_spread_per_wf_ang2": None if spread is None else spread / hr["num_wann"],
        "centres_count": centres,
        "checks": checks,
        "passed": passed,
        "failure_class": failure_class,
        "comparison_pdf": str(plot),
        "source_sha256": {
            "EIGENVAL": sha256_file(band_dir / "EIGENVAL"),
            "wannier90_hr.dat": sha256_file(directory / "wannier90_hr.dat"),
            "wannier90.wout": sha256_file(directory / "wannier90.wout"),
        },
    }
    atomic_json(directory / "quality.json", result)
    record.update({
        "status": "ACCEPTED" if passed else "REJECTED_QUALITY",
        "quality_evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "quality": str(directory / "quality.json"),
        "failure_class": failure_class,
        "next_action": None if passed else (
            "change_projection_or_outer_window_not_localization_iterations"
            if failure_class == "BAND_CONNECTIVITY_FAILED"
            else "protect_target_window_then_add_next_missing_orbital_shell"
        ),
    })
    atomic_json(record_path, record)
    return result


def _is_complete_sp(projection: list[str]) -> bool:
    shells: dict[str, set[str]] = {}
    for item in projection:
        element, shell = item.split(":", 1)
        shells.setdefault(element, set()).add(shell)
    return bool(shells) and all({"s", "p"}.issubset(values) for values in shells.values())


def _is_complete_shell_candidate(item: dict[str, Any]) -> bool:
    return item.get("candidate_family") in {
        "complete_shell_combination",
        "all_observed_shells",
        "explicit_optional_prior",
    }


def _fit_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = item["metrics"]
    return (
        metrics["p95_mev"],
        metrics["max_mev"],
        metrics["rmse_mev"],
        metrics["mae_mev"],
        item["num_wann"],
    )


def _best_attempt_per_candidate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one best passing window/localization attempt per projection candidate."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        if item.get("passed"):
            grouped.setdefault(str(item["candidate"]), []).append(item)
    selected = []
    for candidate in sorted(grouped):
        attempts = sorted(grouped[candidate], key=_fit_sort_key)
        selected.append(attempts[0])
    return selected


def _common_attempt_ranking(
    results_by_structure: dict[str, list[dict[str, Any]]],
    structure_ids: list[str],
) -> dict[str, Any]:
    """Rank one shared candidate/attempt by its worst member.

    A common-basis ensemble is not allowed to silently combine different retry
    windows from different structures.  Each ranked entry therefore represents
    the same candidate *and* attempt label for every ensemble member.  When more
    than one common attempt passes, only the best shared attempt is retained for
    that candidate.
    """
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    all_candidates: set[str] = set()
    for sid in structure_ids:
        indexed[sid] = {}
        for result in results_by_structure.get(sid, []):
            candidate = str(result["candidate"])
            attempt = str(result["attempt"])
            indexed[sid][(candidate, attempt)] = result
            all_candidates.add(candidate)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in sorted(all_candidates):
        common_attempts = set.intersection(*(
            {attempt for name, attempt in indexed[sid] if name == candidate}
            for sid in structure_ids
        )) if structure_ids else set()
        passing: list[dict[str, Any]] = []
        failed_attempts: list[dict[str, Any]] = []
        for attempt in sorted(common_attempts):
            members = {sid: indexed[sid][(candidate, attempt)] for sid in structure_ids}
            projections = {tuple(member["projection"]) for member in members.values()}
            dimensions = {member["num_wann"] for member in members.values()}
            occupations = {member.get("nocc_model") for member in members.values()}
            if len(projections) != 1 or len(dimensions) != 1 or len(occupations) != 1 or None in occupations:
                failed_attempts.append({"attempt": attempt, "reason": "basis_mismatch"})
                continue
            failed = [sid for sid, member in members.items() if not member["passed"]]
            if failed:
                failed_attempts.append({"attempt": attempt, "reason": "quality_gate", "structures": failed})
                continue
            metric_names = ("mae_mev", "rmse_mev", "p95_mev", "max_mev", "gap_error_mev")
            worst = {
                name: max(abs(float(member["metrics"].get(name, 0.0))) for member in members.values())
                for name in metric_names
            }
            passing.append({
                "candidate": candidate,
                "attempt": attempt,
                "projection": next(iter(members.values()))["projection"],
                "num_wann": next(iter(dimensions)),
                "nocc_model": next(iter(occupations)),
                "metrics": worst,
                "pair_worst_metrics": worst,
                "structures": {
                    sid: {
                        "metrics": member["metrics"],
                        "quality": member["quality"],
                        "final_total_spread_ang2": member.get("final_total_spread_ang2"),
                        "mean_spread_per_wf_ang2": member.get("mean_spread_per_wf_ang2"),
                    }
                    for sid, member in members.items()
                },
                "passed": True,
                "candidate_family": next(iter(members.values())).get("candidate_family"),
                "integrated_weight_fraction": next(iter(members.values())).get("integrated_weight_fraction"),
            })
        if not common_attempts:
            rejected.append({"candidate": candidate, "reason": "no_common_attempt_across_structures"})
        elif not passing:
            rejected.append({"candidate": candidate, "reason": "no_common_passing_attempt", "attempts": failed_attempts})
        else:
            passing.sort(key=_fit_sort_key)
            accepted.append(passing[0])

    accepted.sort(key=_fit_sort_key)
    for rank, item in enumerate(accepted, 1):
        item["rank"] = rank
        item["complete_sp_shells"] = _is_complete_sp(item["projection"])
        item["complete_shell_candidate"] = _is_complete_shell_candidate(item)
    return {"ranking": accepted, "rejected": rejected}


def rank_attempts(run_dir: str | Path) -> dict[str, Any]:
    """Create separate trial-priority, fit-quality, and optical-basis recommendations."""
    import yaml

    root = Path(run_dir).expanduser().resolve()
    with (root / "project.snapshot.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    spec = read_json(root / "03_candidates/candidate-spec.json")
    output_top_k = int(config.get("ranking", {}).get("output_top_k", 3))
    structures: dict[str, Any] = {}
    results_by_structure: dict[str, list[dict[str, Any]]] = {}
    for structure in config["structures"]:
        sid = str(structure["id"])
        base = root / "structures" / sid
        candidates = spec["shared_candidates"] if spec["ensemble_mode"] == "common_basis" else spec["independent_candidates"][sid]
        candidate_metadata = {str(item["candidate"]): item for item in candidates}
        priority = [{
            "priority": index,
            "candidate": item["candidate"],
            "projection": item["projection"],
            "num_wann": item["num_wann"],
            "integrated_weight_fraction": item.get("integrated_weight_fraction"),
        } for index, item in enumerate(candidates, 1)]

        results: list[dict[str, Any]] = []
        baseline = base / "05_quality/quality.json"
        if baseline.is_file():
            raw = read_json(baseline)
            metadata = candidate_metadata.get(str(raw["candidate"]), {})
            results.append({
                "candidate": raw["candidate"],
                "attempt": "baseline",
                "projection": raw["projection"],
                "num_wann": raw["num_wann"],
                "nocc_model": raw.get("nocc_model"),
                "metrics": raw["metrics"],
                "native_model_metrics": raw["metrics"],
                "final_total_spread_ang2": raw.get("final_total_spread_ang2"),
                "mean_spread_per_wf_ang2": None if raw.get("final_total_spread_ang2") is None else raw["final_total_spread_ang2"] / raw["num_wann"],
                "passed": raw["passed"],
                "quality": str(baseline),
                "candidate_family": metadata.get("candidate_family", "legacy_baseline"),
                "integrated_weight_fraction": metadata.get("integrated_weight_fraction"),
            })
        for quality_path in sorted((base / "04_wannier_attempts").glob("*/attempt-*/quality.json")):
            raw = read_json(quality_path)
            metadata = candidate_metadata.get(str(raw["candidate"]), {})
            results.append({
                "candidate": raw["candidate"],
                "attempt": raw["attempt"],
                "projection": raw["projection"],
                "num_wann": raw["num_wann"],
                "nocc_model": raw.get("nocc_model"),
                "metrics": raw["target_metrics"],
                "native_model_metrics": raw.get("native_model_metrics"),
                "final_total_spread_ang2": raw.get("final_total_spread_ang2"),
                "mean_spread_per_wf_ang2": raw.get("mean_spread_per_wf_ang2"),
                "passed": raw["passed"],
                "quality": str(quality_path),
                "candidate_family": metadata.get("candidate_family"),
                "integrated_weight_fraction": metadata.get("integrated_weight_fraction"),
            })
        accepted = _best_attempt_per_candidate(results)
        accepted.sort(key=_fit_sort_key)
        for rank, item in enumerate(accepted, 1):
            item["rank"] = rank
            item["complete_sp_shells"] = _is_complete_sp(item["projection"])
            item["complete_shell_candidate"] = _is_complete_shell_candidate(item)

        optical = None
        if accepted:
            best = accepted[0]
            practical = [item for item in accepted if item["complete_shell_candidate"] and
                         item["metrics"]["p95_mev"] <= best["metrics"]["p95_mev"] + 2.0 and
                         item["metrics"]["max_mev"] <= best["metrics"]["max_mev"] + 10.0]
            selected = min(
                practical,
                key=lambda item: (
                    -float(item.get("integrated_weight_fraction") or 0.0),
                    item["num_wann"],
                    item["rank"],
                ),
            ) if practical else best
            optical = {
                "candidate": selected["candidate"],
                "fit_rank": selected["rank"],
                "reason": (
                    "complete_observed_shell_family_within_practical_fit_tolerance"
                    if selected["complete_shell_candidate"] else
                    "no_complete_shell_family_within_practical_fit_tolerance_use_best_fit_provisionally"
                ),
                "requires_observable_convergence_check": True,
            }
        structures[sid] = {
            "trial_priority": priority,
            "fit_ranking": accepted,
            "rejected": [item for item in results if not item["passed"]],
            "top_k": accepted[:output_top_k],
            "top_k_status": {
                "requested": output_top_k,
                "available": len(accepted),
                "complete": len(accepted) >= output_top_k,
            },
            "optical_basis_recommendation": optical,
        }
        results_by_structure[sid] = results

    joint: dict[str, Any] | None = None
    if spec["ensemble_mode"] == "common_basis":
        structure_ids = [str(item["id"]) for item in config["structures"]]
        joint = _common_attempt_ranking(results_by_structure, structure_ids)
        best = joint["ranking"][0] if joint["ranking"] else None
        practical = [] if best is None else [
            item for item in joint["ranking"]
            if item["complete_shell_candidate"]
            and item["pair_worst_metrics"]["p95_mev"] <= best["pair_worst_metrics"]["p95_mev"] + 2.0
            and item["pair_worst_metrics"]["max_mev"] <= best["pair_worst_metrics"]["max_mev"] + 10.0
        ]
        selected = None if best is None else (
            min(
                practical,
                key=lambda item: (
                    -float(item.get("integrated_weight_fraction") or 0.0),
                    item["num_wann"],
                    item["rank"],
                ),
            ) if practical else best
        )
        joint["optical_basis_recommendation"] = None if selected is None else {
            "candidate": selected["candidate"],
            "attempt": selected["attempt"],
            "fit_rank": selected["rank"],
            "reason": (
                "complete_observed_shell_family_within_practical_pair_worst_fit_tolerance"
                if selected["complete_shell_candidate"] else
                "no_complete_shell_family_within_practical_pair_worst_fit_tolerance_use_best_fit_provisionally"
            ),
            "requires_observable_convergence_check": True,
        }
        joint["top_k"] = joint["ranking"][:output_top_k]
        joint["top_k_status"] = {
            "requested": output_top_k,
            "available": len(joint["ranking"]),
            "complete": len(joint["ranking"]) >= output_top_k,
        }
    document = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ranking_scope": "same_declared_target_energy_window",
        "ensemble_mode": spec["ensemble_mode"],
        "requested_top_k": output_top_k,
        "structures": structures,
        "joint_common_basis": joint,
        "interpretation": {
            "trial_priority": "PROCAR/PyW90-like orbital coverage and model cost; determines run order only",
            "fit_ranking": "actual Wannier interpolation errors after hard gates",
            "optical_basis_recommendation": "complete-shell preference only within a small fit tolerance; must still converge the observable",
        },
    }
    output = root / "06_rank/candidate-ranking-v04.json"
    atomic_json(output, document)
    return {"ranking": str(output), **document}


def export_ranked_attempts(run_dir: str | Path) -> dict[str, Any]:
    """Export small, POTCAR-free top-K packages for downstream optics."""
    root = Path(run_dir).expanduser().resolve()
    physics: dict[str, Any] = {}
    structure_metadata: dict[str, Any] = {}
    snapshot = root / "project.snapshot.yaml"
    if snapshot.is_file():
        import yaml

        with snapshot.open("r", encoding="utf-8") as handle:
            project = yaml.safe_load(handle)
        if isinstance(project, dict) and isinstance(project.get("physics"), dict):
            source_physics = project["physics"]
            for key in ("soc", "dimensionality", "axes", "energy_reference", "target_energy_eV"):
                if key in source_physics:
                    physics[key] = source_physics[key]
            structure_metadata = {
                str(item["id"]): {
                    "structure_analysis": item.get("structure_analysis"),
                    "meshes_resolved": item.get("meshes_resolved", {}),
                }
                for item in project.get("structures", [])
                if isinstance(item, dict) and item.get("id")
            }
    ranking_path = root / "06_rank/candidate-ranking-v04.json"
    if not ranking_path.is_file():
        raise ValueError("请先运行 rank-attempts；缺少 candidate-ranking-v04.json。")
    ranking = read_json(ranking_path)
    ensemble_mode = str(ranking.get("ensemble_mode", ""))
    export_entries: list[dict[str, Any]] = []
    if ensemble_mode == "common_basis":
        joint = ranking.get("joint_common_basis")
        if not isinstance(joint, dict) or not joint.get("top_k"):
            raise ValueError("common_basis 排名没有共同通过项，不能导出。")
        output = root / "07_export/common-top-k-v04"
        for entry in joint["top_k"]:
            export_entries.append({
                "entry": entry,
                "members": entry["structures"],
                "rank_root": output / f"rank-{int(entry['rank']):02d}_{entry['candidate']}",
                "metric_key": "pair_worst_metrics",
                "metrics": entry["pair_worst_metrics"],
            })
    elif ensemble_mode == "independent":
        output = root / "07_export/independent-top-k-v04"
        for sid, structure in ranking.get("structures", {}).items():
            for entry in structure.get("top_k", []):
                export_entries.append({
                    "entry": entry,
                    "members": {sid: {"quality": entry["quality"], "metrics": entry["metrics"]}},
                    "rank_root": output / sid / f"rank-{int(entry['rank']):02d}_{entry['candidate']}",
                    "metric_key": "fit_metrics",
                    "metrics": entry["metrics"],
                })
        if not export_entries:
            raise ValueError("independent 排名没有通过项，不能导出。")
    else:
        raise ValueError(f"未知 ensemble_mode：{ensemble_mode}")
    output.mkdir(parents=True, exist_ok=True)
    packages: list[dict[str, Any]] = []

    def copy_verified(source: Path, target: Path) -> None:
        if not source.is_file():
            raise ValueError(f"交付源文件缺失：{source}")
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise FileExistsError(f"拒绝覆盖不同内容：{target}")
            return
        shutil.copy2(source, target)

    for item in export_entries:
        entry = item["entry"]
        rank = int(entry["rank"])
        candidate = str(entry["candidate"])
        attempt = str(entry["attempt"])
        rank_root = item["rank_root"]
        rank_root.mkdir(parents=True, exist_ok=True)
        package_physics = dict(physics)
        member_analyses = [
            structure_metadata.get(sid, {}).get("structure_analysis")
            for sid in item["members"]
        ]
        member_analyses = [value for value in member_analyses if isinstance(value, dict)]
        dimensions = {int(value["dimensionality"]) for value in member_analyses if "dimensionality" in value}
        periodic_axes = {tuple(value["periodic_axes"]) for value in member_analyses if "periodic_axes" in value}
        if len(dimensions) == 1:
            package_physics["dimensionality"] = next(iter(dimensions))
        if len(periodic_axes) == 1:
            package_physics["periodic_axes"] = list(next(iter(periodic_axes)))
        package = {
            "rank": rank,
            "candidate": candidate,
            "attempt": attempt,
            "projection": entry["projection"],
            "num_wann": entry["num_wann"],
            "nocc_model": entry["nocc_model"],
            "physics": package_physics,
            item["metric_key"]: item["metrics"],
            "structures": {},
        }
        for sid, member in item["members"].items():
            source = Path(member["quality"]).resolve().parent
            target = rank_root / sid
            target.mkdir(parents=True, exist_ok=True)
            mapping = {
                "POSCAR": "POSCAR",
                "wannier90_hr.dat": "wannier90_hr.dat",
                "VASP_W90_cmp.pdf": "VASP_W90_cmp.pdf",
                "quality.json": "quality.json",
                "wannier90.win": "wannier90.win",
            }
            for source_name, target_name in mapping.items():
                copy_verified(source / source_name, target / target_name)
            centres = source / "wannier90_centres.xyz"
            if not centres.is_file():
                raise ValueError(f"中心文件缺失：{centres}")
            centre_lines = [
                line.rstrip() for line in centres.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.split() and line.split()[0] == "X"
            ]
            if len(centre_lines) != int(entry["num_wann"]):
                raise ValueError(f"{sid}/{candidate} 的 Wannier 中心数不等于 NUM_WANN。")
            wcenter = target / "wannier90-wcenter.dat"
            centre_text = "\n".join(centre_lines) + "\n"
            if wcenter.exists() and wcenter.read_text(encoding="utf-8") != centre_text:
                raise FileExistsError(f"拒绝覆盖不同中心文件：{wcenter}")
            if not wcenter.exists():
                wcenter.write_text(centre_text, encoding="utf-8")
            files = sorted(path for path in target.iterdir() if path.is_file())
            package["structures"][sid] = {
                "metrics": member["metrics"],
                **structure_metadata.get(sid, {}),
                "files": {path.name: sha256_file(path) for path in files},
            }
        atomic_json(rank_root / "package-manifest.json", package)
        packages.append({"path": str(rank_root), **package})
    summary = {
        "schema_version": 1,
        "ranking": str(ranking_path),
        "ensemble_mode": ensemble_mode,
        "requested_top_k": ranking.get("requested_top_k"),
        "top_k_status": (
            ranking.get("joint_common_basis", {}).get("top_k_status")
            if ensemble_mode == "common_basis" else
            {sid: value.get("top_k_status") for sid, value in ranking.get("structures", {}).items()}
        ),
        "packages": packages,
        "contains_potcar": False,
    }
    atomic_json(output / "export-summary.json", summary)
    return {"export": str(output), **summary}
