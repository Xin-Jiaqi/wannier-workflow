from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr "
    "Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra "
    "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_Z = {symbol: index for index, symbol in enumerate(_ELEMENTS) if index}


def _formula_from_metadata(metadata: dict[str, Any]) -> str:
    olduid = str(metadata.get("olduid", ""))
    if olduid:
        formula = olduid.split("-", 1)[0]
        if _formula_elements(formula):
            return formula
    folder = str(metadata.get("folder", ""))
    parts = Path(folder).parts
    return parts[-2] if len(parts) >= 2 else str(metadata.get("uid", "unknown"))


def _formula_elements(formula: str) -> list[str]:
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return [symbol for symbol in symbols if symbol in _Z]


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _first_numeric(metadata: dict[str, Any], names: Iterable[str]) -> tuple[float | None, str | None]:
    for name in names:
        value = _as_float(metadata.get(name))
        if value is not None:
            return value, name
    return None, None


def _readonly_connection(database: str | Path) -> sqlite3.Connection:
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"C2DB 数据库不存在：{path}")
    # mode=ro prevents accidental writes to the source database.
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def screen_c2db(database: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read-only, provenance-preserving first-pass screen for NLO candidates."""
    gap_fields = list(config.get("gap_fields", ["gap_hse", "gap"]))
    gap_min = float(config.get("gap_min_eV", 0.3))
    gap_max = float(config.get("gap_max_eV", 3.5))
    ehull_max = float(config.get("ehull_max_eV_per_atom", 0.2))
    max_atoms = int(config.get("max_atoms", 16))
    require_noncentrosymmetric = bool(config.get("require_noncentrosymmetric", True))
    allowed_dyn = config.get("allowed_dyn_stab")
    allowed_magstate = config.get("allowed_magstate")
    limit = int(config.get("limit", 200))

    selected: list[dict[str, Any]] = []
    rejected_counts: dict[str, int] = {}
    total = 0
    with _readonly_connection(database) as connection:
        for row_id, unique_id, natoms, raw in connection.execute(
            "SELECT id, unique_id, natoms, key_value_pairs FROM systems"
        ):
            total += 1
            metadata = json.loads(raw or "{}")
            reasons: list[str] = []
            inversion = metadata.get("has_inversion_symmetry")
            if require_noncentrosymmetric and inversion is not False:
                reasons.append("not_verified_noncentrosymmetric")
            gap, gap_source = _first_numeric(metadata, gap_fields)
            if gap is None:
                reasons.append("missing_gap")
            elif not gap_min <= gap <= gap_max:
                reasons.append("gap_outside_range")
            ehull = _as_float(metadata.get("ehull"))
            if ehull is None:
                reasons.append("missing_ehull")
            elif ehull > ehull_max:
                reasons.append("ehull_too_high")
            if natoms is None or int(natoms) > max_atoms:
                reasons.append("too_many_atoms")
            dyn_stab = metadata.get("dyn_stab")
            if allowed_dyn is not None and dyn_stab not in set(allowed_dyn):
                reasons.append("dyn_stab_not_allowed")
            magstate = metadata.get("magstate")
            if allowed_magstate is not None and magstate not in set(allowed_magstate):
                reasons.append("magstate_not_allowed")
            if reasons:
                for reason in set(reasons):
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                continue

            formula = _formula_from_metadata(metadata)
            elements = _formula_elements(formula)
            max_z = max((_Z[symbol] for symbol in elements), default=0)
            # This is only a queue priority, never a scientific quality score.
            cost_proxy = int(natoms) * (1.0 + max_z / 100.0)
            selected.append({
                "database_row_id": int(row_id),
                "ase_unique_id": unique_id,
                "c2db_uid": metadata.get("uid"),
                "c2db_olduid": metadata.get("olduid"),
                "formula": formula,
                "natoms": int(natoms),
                "elements": elements,
                "max_atomic_number": max_z,
                "soc_recommended": max_z >= int(config.get("soc_atomic_number_threshold", 31)),
                "gap_eV": gap,
                "gap_source": gap_source,
                "ehull_eV_per_atom": ehull,
                "dyn_stab": dyn_stab,
                "magstate": magstate,
                "layergroup": metadata.get("layergroup"),
                "lgnum": metadata.get("lgnum"),
                "has_inversion_symmetry_database": inversion,
                "source_label": metadata.get("label"),
                "source_folder": metadata.get("folder"),
                "cost_proxy": round(cost_proxy, 3),
                "requires_structure_symmetry_recheck": True,
                "requires_dft_gap_recheck": True,
            })

    selected.sort(key=lambda item: (
        item["cost_proxy"],
        abs(float(item["gap_eV"]) - float(config.get("preferred_gap_eV", 1.5))),
        float(item["ehull_eV_per_atom"]),
        item["formula"],
    ))
    eligible_count = len(selected)
    selected = selected[:limit]
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(Path(database).expanduser().resolve()),
        "read_only": True,
        "config": config,
        "screening_semantics": {
            "purpose": "first-pass queue for Wannier/NLO robustness benchmark",
            "cost_proxy": "natoms * (1 + max_atomic_number / 100); queue ordering only",
            "noncentrosymmetry": "database flag only; must be recomputed from exported structure",
            "gap": "database prefilter only; must be recomputed with the production functional and SOC choice",
        },
        "database_rows_scanned": total,
        "eligible_count_before_limit": eligible_count,
        "selected_count": len(selected),
        "rejected_counts_nonexclusive": rejected_counts,
        "candidates": selected,
    }


def write_material_pool(document: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "material-pool.json"
    csv_path = output / "material-pool.csv"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = document["candidates"]
    fields = list(rows[0]) if rows else ["database_row_id", "c2db_uid", "formula"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "elements": ";".join(row.get("elements", []))})
    return {"json": str(json_path), "csv": str(csv_path)}
