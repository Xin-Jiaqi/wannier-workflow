"""Shared acceptance contract for selecting and preserving evaluated models."""
from pathlib import Path
from typing import Any, Mapping

from .state import sha256_file

REQUIRED_CHECKS = {'w90_completed', 'w90_converged', 'hr_centres_match', 'rmse', 'p95', 'max', 'gap'}


def strict_quality_passed(record: Mapping[str, Any]) -> bool:
    checks = record.get('checks')
    return (record.get('passed') is True and isinstance(checks, dict)
            and REQUIRED_CHECKS.issubset(checks)
            and all(value is True for value in checks.values()))


def require_audited_sources(record: Mapping[str, Any], sources: Mapping[str, Path]) -> None:
    if not strict_quality_passed(record):
        raise ValueError('Current convergence/quality audit required; re-evaluate historical accepted results.')
    audited = record.get('source_sha256', {})
    for name, path in sources.items():
        if not isinstance(audited, dict) or not audited.get(name) or not path.is_file():
            raise ValueError(f'Missing audited model source: {name}')
        if sha256_file(path) != audited[name]:
            raise ValueError(f'Audited model source changed: {name}')
