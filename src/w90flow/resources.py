"""Runtime resource detection and safe candidate-search allocation.

This module deliberately treats CPU and memory discovery as operational
metadata, never as a scientific input.  It therefore changes neither the
POSCAR/POTCAR/INCAR hashes nor the finite candidate/quality contract.
"""

from __future__ import annotations

import os
from typing import Any


GiB = 1024 ** 3


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (ValueError, OSError, AttributeError):
        return None


def _affinity_cpus() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def detect_resources(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Return scheduler-aware CPU/RAM information without starting a process."""
    env = os.environ if environ is None else environ
    override = _positive_int(env.get("W90FLOW_AVAILABLE_CPUS"))
    slurm = _positive_int(env.get("SLURM_CPUS_ON_NODE"))
    slurm_task = _positive_int(env.get("SLURM_CPUS_PER_TASK"))
    affinity = _affinity_cpus()
    os_count = os.cpu_count() or 1
    # Scheduler allocation is authoritative.  Do not intersect it with the
    # login process affinity seen by a dry-run/test process; that would turn a
    # valid 32-core Slurm allocation into this Mac's local affinity count.
    if override:
        available = override
    elif slurm or slurm_task:
        candidates = [value for value in (slurm, slurm_task) if value]
        available = min(candidates)
    else:
        candidates = [value for value in (affinity, os_count) if value]
        available = min(candidates) if candidates else 1
    scheduler_memory_mib = _positive_int(env.get("SLURM_MEM_PER_NODE"))
    memory = scheduler_memory_mib * 1024 ** 2 if scheduler_memory_mib else _memory_bytes()
    scheduler = bool(env.get("SLURM_JOB_ID") or env.get("SLURM_CPUS_ON_NODE"))
    return {
        "available_cpus": available,
        "memory_bytes": memory,
        "memory_gib": round(memory / GiB, 3) if memory is not None else None,
        "scheduler_allocation": scheduler,
        "sources": {
            "W90FLOW_AVAILABLE_CPUS": override,
            "SLURM_CPUS_ON_NODE": slurm,
            "SLURM_CPUS_PER_TASK": slurm_task,
            "SLURM_MEM_PER_NODE_MiB": scheduler_memory_mib,
            "cpu_affinity": affinity,
            "os_cpu_count": os_count,
        },
    }


def candidate_search_plan(
    *,
    available: dict[str, Any],
    candidate_count: int,
    mpi_processes_per_candidate: int,
    policy: str = "serial",
    max_workers: int | str = "auto",
) -> dict[str, Any]:
    """Build a safe candidate allocation plan.

    ``serial`` is default production behavior.  ``candidate_parallel`` is an
    explicit opt-in: it is suitable only after one structure has benchmarked
    the requested MPI split on that specific machine/queue.
    """
    if candidate_count <= 0:
        raise ValueError("candidate_count 必须是正整数。")
    if mpi_processes_per_candidate <= 0:
        raise ValueError("mpi_processes_per_candidate 必须是正整数。")
    if policy not in {"serial", "candidate_parallel"}:
        raise ValueError("candidate_search.policy 只能是 serial 或 candidate_parallel。")
    cpus = int(available["available_cpus"])
    by_cpu = max(1, cpus // mpi_processes_per_candidate)
    memory_gib = available.get("memory_gib")
    # An 8-GiB laptop must not launch several VASP copies merely because its
    # logical CPU count is high.  Users may still explicitly request serial
    # benchmarking with a larger MPI count.
    by_memory = 1 if memory_gib is not None and float(memory_gib) < 12.0 else by_cpu
    if max_workers == "auto":
        requested = by_memory if policy == "candidate_parallel" else 1
    else:
        if isinstance(max_workers, bool):
            raise ValueError("candidate_search.max_workers 必须是 auto 或正整数。")
        requested = int(max_workers)
        if requested <= 0:
            raise ValueError("candidate_search.max_workers 必须是 auto 或正整数。")
    workers = min(candidate_count, by_cpu, by_memory, requested)
    reason = (
        "serial policy"
        if policy == "serial"
        else "candidate_parallel policy; worker count limited by CPU/RAM allocation"
    )
    return {
        "policy": policy,
        "candidate_count": candidate_count,
        "mpi_processes_per_candidate": mpi_processes_per_candidate,
        "workers": workers,
        "allocated_cpus": cpus,
        "allocated_cpus_used": workers * mpi_processes_per_candidate,
        "idle_cpus": max(0, cpus - workers * mpi_processes_per_candidate),
        "resource_detection": available,
        "reason": reason,
        "benchmark_required": policy == "candidate_parallel",
    }
