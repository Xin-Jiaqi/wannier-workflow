from __future__ import annotations

from pathlib import Path
import re
import unittest

import w90flow


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_package_version_matches_project_metadata(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        project = re.search(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", pyproject)
        self.assertIsNotNone(project)
        version = re.search(r'^version\s*=\s*"([^"]+)"', project.group(1), re.MULTILINE)
        self.assertIsNotNone(version)
        self.assertEqual(w90flow.__version__, version.group(1))

    def test_readme_commands_and_local_links_are_checkout_safe(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("运行W90Flow.command", readme)
        for target in re.findall(r"\[[^]]+\]\(([^)]+\.md)\)", readme):
            self.assertFalse(target.startswith(".."), target)
            self.assertTrue((ROOT / target).is_file(), target)


if __name__ == "__main__":
    unittest.main()
