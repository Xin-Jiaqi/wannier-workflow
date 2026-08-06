"""Explicit resolution of machine-specific locations.

The workflow is developed against a private launcher workspace that is not part
of this repository.  All such locations are provided through environment
variables so the repository itself stays portable and auditable.  Directory
guessing through ``Path(__file__).parents[N]`` is intentionally not used here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def launcher_root() -> Path:
    """Root of the private launcher workspace holding ``00_命令入口/计算/*.command``.

    Falls back to the home directory to preserve the original single-machine
    layout; set ``W90FLOW_LAUNCHER_ROOT`` explicitly on any other machine.
    """
    value = os.environ.get("W90FLOW_LAUNCHER_ROOT")
    return Path(value).expanduser() if value else Path.home()


def pyw90_executable() -> str | None:
    """Path to the PyW90 executable, from ``W90FLOW_PYW90_EXECUTABLE``."""
    return os.environ.get("W90FLOW_PYW90_EXECUTABLE") or None


def matplotlib_cache_dir() -> Path:
    """Matplotlib cache directory, from ``W90FLOW_MPL_CACHE`` or a temp directory."""
    value = os.environ.get("W90FLOW_MPL_CACHE")
    return Path(value).expanduser() if value else Path(tempfile.gettempdir()) / "w90flow-matplotlib"
