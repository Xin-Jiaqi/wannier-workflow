"""Bounded mesh-refinement advice; never changes acceptance or starts a job.

Only a converged, well-reproduced training spectrum with an isolated maximum
held-out band error warrants this expensive direction. All diagnostics must
refer to the same attempt. Production integration remains explicitly opt-in.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def propose_mesh_retry(
    *,
    converged: bool,
    metrics: Mapping[str, float],
    gates: Mapping[str, float],
    training_max_mev: float,
    mesh: Sequence[int],
    periodic_axes: Sequence[bool],
    retries_used: int,
    retry_budget: int = 1,
    training_tolerance_mev: float = 1.0,
    maximum_mesh: int = 15,
) -> dict:
    """Return a proposal only; callers must build fresh matrices and revalidate.

    Metric keys follow quality.py; gate keys follow ranking.hard_gates.
    This rule does not identify the physical cause with certainty. A proposal
    is a controlled experiment with frozen projections/windows/density/path.
    """
    def stop(reason: str) -> dict:
        return {"action": "NO_MESH_RETRY", "reason": reason, "acceptance_changed": False}

    if len(mesh) != 3 or len(periodic_axes) != 3:
        raise ValueError("mesh and periodic_axes must have three entries")
    if any(type(n) is not int or n < 1 for n in mesh):
        raise ValueError("mesh entries must be positive integers")
    if any(type(v) is not bool for v in periodic_axes) or not any(periodic_axes):
        raise ValueError("periodic_axes must contain booleans and a periodic axis")
    if any(type(v) is not int or v < 0 for v in (retries_used, retry_budget)):
        raise ValueError("retry counts must be nonnegative integers")
    if type(maximum_mesh) is not int or maximum_mesh < 1:
        raise ValueError("maximum_mesh must be a positive integer")
    if type(converged) is not bool:
        raise ValueError("converged must be an explicit boolean")
    pairs = (("rmse_mev", "rmse_meV"), ("p95_mev", "p95_meV"),
             ("max_mev", "max_meV"), ("gap_error_mev", "gap_error_meV"))
    values = [float(metrics[m]) for m, _ in pairs]
    limits = [float(gates[g]) for _, g in pairs]
    if not all(math.isfinite(v) for v in values + limits +
               [training_max_mev, training_tolerance_mev]):
        return stop("NONFINITE_DIAGNOSTIC")
    if min(limits) <= 0 or training_tolerance_mev <= 0 or training_max_mev < 0:
        raise ValueError("thresholds must be positive and training error nonnegative")
    if any(v < 0 for v in values[:3]):
        raise ValueError("absolute band-error statistics must be nonnegative")
    if not converged:
        return stop("OPTIMIZATION_NOT_CONVERGED")
    if retries_used >= retry_budget:
        return stop("MESH_RETRY_BUDGET_EXHAUSTED")
    if training_max_mev > training_tolerance_mev:
        return stop("TRAINING_SPECTRUM_NOT_REPRODUCED")
    if any(abs(values[i]) > limits[i] for i in (0, 1, 3)):
        return stop("ERROR_NOT_ISOLATED_TO_MAXIMUM")
    if values[2] <= limits[2]:
        return stop("BAND_GATES_ALREADY_PASS")
    if values[2] > 2 * limits[2]:
        return stop("MAXIMUM_ERROR_TOO_LARGE_FOR_LOCAL_MESH_TRIAL")
    proposed = [n + 2 if periodic else n for n, periodic in zip(mesh, periodic_axes)]
    if any(n > maximum_mesh for n, periodic in zip(proposed, periodic_axes) if periodic):
        return stop("MESH_SIZE_BUDGET_EXHAUSTED")
    return {
        "action": "PROPOSE_FRESH_MATRIX_MESH_TRIAL",
        "mesh": proposed,
        "acceptance_changed": False,
        "preserve": ["projections", "energy_windows", "scf_density", "dft_reference",
                     "acceptance_gates", "original_failed_attempt"],
        "requires": ["fresh_vasp_matrices", "kpoint_order_validation",
                     "optimizer_convergence", "independent_band_quality"],
    }
