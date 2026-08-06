#!/usr/bin/env python3
"""Fail fast on files that must not enter the workflow repository."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {
    "POTCAR",
    "CHG",
    "CHGCAR",
    "WAVECAR",
    "WAVEDER",
}
FORBIDDEN_SUFFIXES = {".amn", ".mmn", ".unk"}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt"}
PERSONAL_PATH = re.compile(r"/(?:Users|Volumes|private|tmp)/[A-Za-z0-9_. -]+/")
MAX_TRACKABLE_BYTES = 5 * 1024 * 1024


def main() -> int:
    issues: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.name.startswith("._")
            or path.name == ".DS_Store"
        ):
            continue
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden runtime/licensed artifact: {relative}")
        if path.stat().st_size > MAX_TRACKABLE_BYTES:
            issues.append(f"file exceeds 5 MiB fixture limit: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                issues.append(f"declared text file is not UTF-8: {relative}")
                continue
            if PERSONAL_PATH.search(text):
                issues.append(f"personal absolute path in text: {relative}")
    if issues:
        print("Repository audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Repository audit passed: no forbidden artifacts, oversized files, or personal absolute paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
