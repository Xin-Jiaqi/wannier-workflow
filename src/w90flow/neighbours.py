"""Select a reproducible neighbour-preprocessing proposal, never modify a run."""
from __future__ import annotations

import hashlib
import math
from typing import Any


def select_stable_neighbour_proposal(probes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Require two consecutive successful tolerances with identical topology.

    Input order is the recorded bounded sweep order. A failed probe breaks
    adjacency; merely obtaining one successful preprocessing is insufficient.
    Neighbour records are canonicalized as integers (k, k', Gx, Gy, Gz).
    """
    previous: tuple[float, int, tuple[int, ...]] | None = None
    for probe in probes:
        if not probe.get("passed"):
            previous = None
            continue
        tol = float(probe["kmesh_tol"])
        shells = int(probe["shells"])
        if not math.isfinite(tol) or tol <= 0 or shells < 1:
            raise ValueError("positive finite tolerance and shell count required")
        topology = tuple(int(token) for token in probe.get("nnkp_topology", []))
        if not topology or topology[0] < 1 or (len(topology) - 1) % (5 * topology[0]):
            raise ValueError("incomplete nnkp topology in successful probe")
        if len(topology) == 1:
            raise ValueError("successful probe contains no neighbour records")
        if previous is not None and previous[1:] == (shells, topology) and previous[0] != tol:
            return {
                "kmesh_tol": previous[0], "search_shells": shells,
                "verification_tolerance": tol,
                "nnkp_topology_sha256": hashlib.sha256(repr(topology).encode()).hexdigest(),
                "policy": "proposal_only; unchanged lattice, mesh, energy window and quality gates",
            }
        previous = (tol, shells, topology)
    return None
